// hlbl_hypercube: connected HLbL kernel sum on a sliding-hypercube schedule.
//
// For each chain of the input (start, direction, stride, steps) it solves the
// propagators (tmLQCD/QUDA, or --random-fields), contracts every pair and
// writes one HDF5 file. Gauge field and multigrid are set up once.
//
// A production run checks only its input. The checks are in
// scripts/hlbl_mII_hypercube_schedule.py and two modes:
//   --dry-run    schedule counts, cache peak, chain overlaps. No GPU.
//   --dummy-run  the full setup, every check, and the memory peak of a run for
//                real (cache filled, one position solved, one pair contracted).
//                Writes no results. --tune saves QUDA's tune cache.

#include <mpi.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "hlbl/contract.hpp"
#include "hlbl/driver.hpp"
#include "hlbl/input.hpp"
#include "hlbl/lattice.hpp"
#include "hlbl/output.hpp"
#include "hlbl/physics.hpp"
#include "hlbl/propagator_cache.hpp"
#include "hlbl/provider.hpp"
#include "hlbl/schedule.hpp"
#ifdef HLBL_HAVE_SOLVER
#include "hlbl/solver.hpp"
#endif

namespace {

struct Options {
  hlbl::Input input;        // from --in, then overridden by the command line
  hlbl::Chain cli_chain;    // --direction --stride --steps --origin --window-range
  bool cli_chain_set = false;
  bool dry_run = false;
  bool dummy_run = false;
  bool tune = false;        // with --dummy-run
  int show_window = -1;     // -1: none
};

// The four values after option `i`
hlbl::Vec4 cli_vec4(int argc, char** argv, int& i) {
  hlbl::Vec4 v;
  for (int k = 0; k < 4; ++k) {
    if (++i >= argc) throw std::runtime_error(std::string(argv[i - k - 1]) + " wants 4 values");
    v[k] = std::atoi(argv[i]);
  }
  return v;
}

Options parse_args(int argc, char** argv) {
  Options o;

  // The input file is read first, so that the command line overrides it
  // wherever it is on the line.
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--in") {
      if (i + 1 >= argc) throw std::runtime_error("--in wants a file name");
      o.input = hlbl::parse_input_file(argv[i + 1]);
      break;
    }
  }

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (++i >= argc) throw std::runtime_error(a + " wants a value");
      return argv[i];
    };
    if (a == "--in") {
      next();  // read above
    } else if (a == "--direction") {
      o.cli_chain.direction = cli_vec4(argc, argv, i);
      o.cli_chain_set = true;
    } else if (a == "--stride") {
      o.cli_chain.stride = std::atoi(next());
      o.cli_chain_set = true;
    } else if (a == "--steps") {
      o.cli_chain.steps = std::atoi(next());
      o.cli_chain_set = true;
    } else if (a == "--origin") {
      o.cli_chain.start = cli_vec4(argc, argv, i);
      o.cli_chain_set = true;
    } else if (a == "--window-range") {
      hlbl::parse_window_range(next(), o.cli_chain.window_begin,
                               o.cli_chain.window_end);
      o.cli_chain_set = true;
    } else if (a == "--muon-mass") {
      o.input.muon_mass = hlbl::parse_positive(next(), a, "MeV");
    } else if (a == "--lattice-spacing") {
      o.input.lattice_spacing = hlbl::parse_positive(next(), a, "fm");
    } else if (a == "--flavors") {
      o.input.flavors = std::atoi(next());
    } else if (a == "--lattice") {
      o.input.lattice = cli_vec4(argc, argv, i);
    } else if (a == "--dry-run") {
      o.dry_run = true;
    } else if (a == "--dummy-run") {
      o.dummy_run = true;
    } else if (a == "--tune") {
      o.tune = true;
    } else if (a == "--show-window") {
      o.show_window = std::atoi(next());
    } else if (a == "--procs") {
      o.input.procs = cli_vec4(argc, argv, i);
    } else if (a == "--out") {
      o.input.out = next();
    } else if (a == "--timing") {
      o.input.timing = next();
    } else if (a == "--cache") {
      std::string v = next();
      if (v != "device" && v != "host") {
        throw std::runtime_error("--cache must be device or host");
      }
      o.input.device_cache = (v == "device");
    } else if (a == "--random-fields") {
      o.input.random_fields = true;
    } else {
      throw std::runtime_error("unrecognised option: " + a);
    }
  }

  if (o.cli_chain_set && !o.input.chains.empty()) {
    throw std::runtime_error(
        "--direction, --stride, --steps, --origin and --window-range describe one "
        "chain; they cannot be combined with chain lines in the input file");
  }
  if (o.cli_chain_set || o.input.chains.empty()) o.input.chains = {o.cli_chain};

  for (std::size_t c = 0; c < o.input.chains.size(); ++c) {
    try {
      hlbl::validate(o.input.chains[c]);
    } catch (const std::exception& e) {
      throw std::runtime_error("chain " + std::to_string(c) + ": " + e.what());
    }
  }

  if (o.dry_run && o.dummy_run) {
    throw std::runtime_error("--dry-run and --dummy-run cannot be combined");
  }
  if (o.tune && !o.dummy_run) throw std::runtime_error("--tune goes with --dummy-run");
  if (o.tune && o.input.random_fields) {
    throw std::runtime_error("--tune needs the solver, not random fields");
  }
  // checked here so that a run without them stops before the solver setup
  if (!o.dry_run && (o.input.muon_mass <= 0 || o.input.lattice_spacing <= 0)) {
    throw std::runtime_error(
        "the muon mass and the lattice spacing must be given: muon_mass = <MeV> "
        "and lattice_spacing = <fm> in the input file, or --muon-mass and "
        "--lattice-spacing");
  }
  return o;
}

void print_vec4(const hlbl::Vec4& v) {
  std::printf("(%d,%d,%d,%d)", v[0], v[1], v[2], v[3]);
}

const char* mode_name(hlbl::Mode mode) {
  return mode == hlbl::Mode::kSerial ? "serial" : "interleaved";
}

// Warns about sources and pairs in more than one chain (only the windows each
// chain runs count), and lists up to 20 pairs of chains, each with its first
// 10 shared sources and the window of each chain that solves them. Every line
// starts with `prefix`. Same lines as overlap_report() in
// scripts/hlbl_mII_hypercube_schedule.py.
void report_overlaps(const std::vector<hlbl::ChainTask>& tasks,
                     const hlbl::Vec4& global, const char* prefix) {
  using Pair = std::pair<hlbl::Vec4, hlbl::Vec4>;
  std::size_t const n = tasks.size();
  std::vector<std::map<hlbl::Vec4, int>> sources(n);  // position -> window
  std::vector<std::set<Pair>> pairs(n);
  std::map<hlbl::Vec4, int> source_count;
  std::map<Pair, int> pair_count;
  for (std::size_t c = 0; c < n; ++c) {
    const auto& task = tasks[c];
    int const end = task.window_end < 0
                        ? static_cast<int>(task.schedule.size())
                        : std::min(task.window_end, static_cast<int>(task.schedule.size()));
    for (int k = task.window_begin; k < end; ++k) {
      for (const auto& src : task.schedule[k].sources) {
        hlbl::Vec4 const p = hlbl::mod_pos(src.position, global);
        sources[c].emplace(p, k);
        for (const auto& q : src.partners) {
          hlbl::Vec4 const w = hlbl::mod_pos(q, global);
          pairs[c].insert(p < w ? std::make_pair(p, w) : std::make_pair(w, p));
        }
      }
    }
    for (const auto& [p, window] : sources[c]) source_count[p]++;
    for (const auto& p : pairs[c]) pair_count[p]++;
  }
  auto repeated = [](const auto& m) {
    return std::count_if(m.begin(), m.end(),
                         [](const auto& kv) { return kv.second > 1; });
  };
  auto common_pairs = [](const auto& a, const auto& b) {
    long k = 0;
    for (const auto& x : a) k += b.count(x);
    return k;
  };
  long const s = repeated(source_count), p = repeated(pair_count);
  if (s == 0 && p == 0) return;
  std::printf("%sWARNING: %ld source positions are in more than one chain "
              "(solved once per chain) and %ld pairs are computed in more than one "
              "chain\n", prefix, s, p);
  constexpr int kMaxLines = 20;     // pairs of chains listed
  constexpr long kMaxSources = 10;  // shared sources listed per pair of chains
  int shown = 0, more = 0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      long ns = 0;
      std::vector<std::tuple<hlbl::Vec4, int, int>> shared;  // position, windows
      for (const auto& [pos, window_i] : sources[i]) {
        auto const it = sources[j].find(pos);
        if (it == sources[j].end()) continue;
        if (ns++ < kMaxSources) shared.emplace_back(pos, window_i, it->second);
      }
      long const np = common_pairs(pairs[i], pairs[j]);
      if (ns == 0 && np == 0) continue;
      if (shown >= kMaxLines) {
        ++more;
        continue;
      }
      ++shown;
      std::printf("%schains %zu and %zu share %ld sources and %ld pairs\n", prefix,
                  i, j, ns, np);
      for (const auto& [pos, window_i, window_j] : shared) {
        std::printf("%s  source (%d,%d,%d,%d) in window %d of chain %zu and window "
                    "%d of chain %zu\n", prefix, pos[0], pos[1], pos[2], pos[3],
                    window_i, i, window_j, j);
      }
      if (ns > kMaxSources) {
        std::printf("%s  ... and %ld more sources\n", prefix, ns - kMaxSources);
      }
    }
  }
  if (more > 0) std::printf("%s... and %d more pairs of chains\n", prefix, more);
}

// Prints a report, including the lines scripts/compare_schedule.sh compares
// with the Python reference:
//   mode=<name> n_propagators=.. n_solves=.. n_pairs=.. n_kernel_sums=.. peak_cache=..
//   edge (a,b,c,d) (e,f,g,h)       one per pair of window --show-window
// With several chains, each block starts with "chain i of N", followed at the
// end by a "total ..." line (interleaved) and the overlap report. Counts are
// for whole chains; the overlap report uses only each chain's window range.
void run_dry_run(const Options& o) {
  auto const& chains = o.input.chains;
  hlbl::ScheduleStats total;
  std::vector<hlbl::ChainTask> tasks;
  int const n_chains = static_cast<int>(chains.size());

  for (int c = 0; c < n_chains; ++c) {
    const hlbl::Chain& chain = chains[c];
    int steps = chain.steps;
    if (steps < 0) {
      steps = hlbl::max_steps_without_wrap(chain.direction, chain.stride,
                                            o.input.lattice, chain.start);
    }
    if (chain.window_begin > steps) {
      throw std::runtime_error("chain " + std::to_string(c) +
                               ": window range starts after the last window");
    }

    if (n_chains > 1) std::printf("chain %d of %d\n", c, n_chains);
    std::printf("direction=");
    print_vec4(chain.direction);
    std::printf(" stride=%d origin=", chain.stride);
    print_vec4(chain.start);
    std::printf(" steps=%d windows=%d\n", steps, steps + 1);

    auto schedule =
        hlbl::build_schedule(chain.direction, chain.stride, steps, chain.start);
    hlbl::check_complete(schedule, chain.stride);
    std::printf("check_complete: ok\n");
    hlbl::ChainTask task;
    task.schedule = schedule;
    task.window_begin = chain.window_begin;
    task.window_end = chain.window_end;
    tasks.push_back(std::move(task));

    for (auto mode : {hlbl::Mode::kSerial, hlbl::Mode::kInterleaved}) {
      auto stats = hlbl::simulate(schedule, o.input.flavors, mode);
      std::printf(
          "mode=%s n_propagators=%lld n_solves=%lld n_pairs=%lld "
          "n_kernel_sums=%lld peak_cache=%d\n",
          mode_name(mode), stats.n_propagators, stats.n_solves, stats.n_pairs,
          stats.n_kernel_sums, stats.peak_cache);
      if (mode == hlbl::Mode::kInterleaved) {
        total.n_propagators += stats.n_propagators;
        total.n_solves += stats.n_solves;
        total.n_pairs += stats.n_pairs;
        total.n_kernel_sums += stats.n_kernel_sums;
        total.peak_cache = std::max(total.peak_cache, stats.peak_cache);
      }
    }

    if (o.show_window >= 0 &&
        o.show_window < static_cast<int>(schedule.size())) {
      const auto& step = schedule[o.show_window];
      std::printf("window=%d new_propagators=%d pairs=%d\n", step.index,
                  step.n_new(), step.n_pairs());

      auto rel = [&](const hlbl::Vec4& p) {
        hlbl::Vec4 r;
        for (int k = 0; k < 4; ++k) r[k] = (p[k] - chain.start[k]) / chain.stride;
        return r;
      };
      std::vector<std::pair<hlbl::Vec4, hlbl::Vec4>> edges;
      for (const auto& src : step.sources) {
        hlbl::Vec4 off = rel(src.position);
        for (const auto& q : src.partners) {
          hlbl::Vec4 poff = rel(q);
          edges.push_back(off < poff ? std::make_pair(off, poff)
                                      : std::make_pair(poff, off));
        }
      }
      std::sort(edges.begin(), edges.end());
      for (const auto& [a, b] : edges) {
        std::printf("edge ");
        print_vec4(a);
        std::printf(" ");
        print_vec4(b);
        std::printf("\n");
      }
    }
  }

  if (n_chains > 1) {
    std::printf("total chains=%d n_propagators=%lld n_solves=%lld n_pairs=%lld "
                "n_kernel_sums=%lld peak_cache=%d\n",
                n_chains, total.n_propagators, total.n_solves, total.n_pairs,
                total.n_kernel_sums, total.peak_cache);
    report_overlaps(tasks, o.input.lattice, "");
  }
}

double gib(std::size_t bytes) { return bytes / 1073741824.0; }

// The chains as they run (steps=auto resolved, window_end clamped to
// steps + 1) and their schedules.
struct Plan {
  std::vector<hlbl::Chain> chains;
  std::vector<hlbl::ChainTask> tasks;
};

Plan make_plan(const hlbl::Input& in, const hlbl::Vec4& global) {
  Plan plan;
  plan.chains = in.chains;
  for (std::size_t c = 0; c < plan.chains.size(); ++c) {
    hlbl::Chain& chain = plan.chains[c];
    if (chain.steps < 0) {
      chain.steps = hlbl::max_steps_without_wrap(chain.direction, chain.stride,
                                                 global, chain.start);
    }
    if (chain.window_begin > chain.steps) {
      throw std::runtime_error("chain " + std::to_string(c) +
                               ": window range starts after the last window");
    }
    chain.window_end = chain.window_end < 0
                           ? chain.steps + 1
                           : std::min(chain.window_end, chain.steps + 1);

    hlbl::ChainTask task;
    task.schedule = hlbl::build_schedule(chain.direction, chain.stride,
                                         chain.steps, chain.start);
    task.window_begin = chain.window_begin;
    task.window_end = chain.window_end;
    plan.tasks.push_back(std::move(task));
  }
  return plan;
}

// The timing log: `timing`, or the result file name with .timing.csv
std::string timing_path(const hlbl::Input& in) {
  if (!in.timing.empty()) return in.timing;
  return in.out.substr(0, in.out.rfind('.')) + ".timing.csv";
}

// What the run is, on rank 0
void print_run_info(const hlbl::Input& in, const hlbl::Lattice& lattice,
                    const Plan& plan) {
  auto const& global = lattice.global();
  auto const& procs = lattice.procs();
  std::printf("# [hlbl] lattice %dx%dx%dx%d on %dx%dx%dx%d ranks, %zu chains, %s cache\n",
              global[0], global[1], global[2], global[3], procs[0], procs[1],
              procs[2], procs[3], plan.chains.size(),
              in.device_cache ? "device" : "host");
  for (std::size_t c = 0; c < plan.chains.size(); ++c) {
    const hlbl::Chain& ch = plan.chains[c];
    std::printf("# [hlbl] chain %zu: start (%d,%d,%d,%d), direction (%d,%d,%d,%d), "
                "stride %d, %d steps, windows %d to %d\n", c, ch.start[0],
                ch.start[1], ch.start[2], ch.start[3], ch.direction[0],
                ch.direction[1], ch.direction[2], ch.direction[3], ch.stride,
                ch.steps, ch.window_begin, ch.window_end - 1);
  }
  std::printf("# [hlbl] muon mass %.10g MeV, lattice spacing %.10g fm: xunit = m_mu a = %.10g\n",
              in.muon_mass, in.lattice_spacing,
              hlbl::xunit(in.muon_mass, in.lattice_spacing));
  std::fflush(stdout);
}

// A production run. Collective; a failure aborts the whole job.
void run(const Options& o, const hlbl::Lattice& lattice, int nconf,
         hlbl::PropagatorProvider& provider) {
  hlbl::Input const& in = o.input;
  int rank = 0;
  MPI_Comm_rank(lattice.cart_comm(), &rank);
  Plan const plan = make_plan(in, lattice.global());
  if (rank == 0) print_run_info(in, lattice, plan);

  QED_kernel_temps kqed_t;
  if (initialise(&kqed_t) != 0) {
    std::fprintf(stderr, "[hlbl_hypercube] KQED initialise failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  hlbl::Contractor contractor(lattice, kqed_t);
  double const xunit = hlbl::xunit(in.muon_mass, in.lattice_spacing);

  std::unique_ptr<hlbl::ResultWriter> writer;
  std::unique_ptr<hlbl::TimingLog> timing;
  if (rank == 0 && !in.out.empty()) {
    hlbl::RunInfo info;
    info.lattice = lattice.global();
    info.procs = lattice.procs();
    info.nconf = nconf;
    info.chains = plan.chains;
    info.muon_mass = in.muon_mass;
    info.lattice_spacing = in.lattice_spacing;
    info.xunit = xunit;
    writer = std::make_unique<hlbl::ResultWriter>(in.out, info);
    timing = std::make_unique<hlbl::TimingLog>(timing_path(in));
  }

  hlbl::RunOptions ro;
  ro.residency = in.device_cache ? hlbl::PropagatorCache::Residency::kDevice
                                 : hlbl::PropagatorCache::Residency::kHost;
  ro.xunit = xunit;
  ro.verbose = true;
  hlbl::RunStats stats;
  double const t0 = MPI_Wtime();
  auto results = hlbl::run_chains(lattice, plan.tasks, ro, provider, contractor,
                                   writer.get(), timing.get(), &stats);
  double const wall = MPI_Wtime() - t0;

  if (rank == 0) {
    std::printf("# [hlbl] done: %d pairs (%zu results) in %zu chains, %d positions "
                "solved, %lld solver iterations, cache peak %d positions\n",
                stats.pairs, results.size(), plan.chains.size(), stats.positions_solved,
                stats.iterations, stats.cache_peak);
    std::printf("# [hlbl] wall %.1f s: solves %.1f s (%.2f s per position), "
                "kernel sums %.1f s (%.2f s per pair)\n", wall,
                stats.seconds_solve,
                stats.positions_solved ? stats.seconds_solve / stats.positions_solved : 0.,
                stats.seconds_kernel_sum,
                stats.pairs ? stats.seconds_kernel_sum / stats.pairs : 0.);
    if (writer) std::printf("# [hlbl] results in %s\n", in.out.c_str());
  }
  free_QED_temps(&kqed_t);
}

// The directory part of a path, "." if there is none
std::string directory_of(const std::string& path) {
  auto const slash = path.rfind('/');
  if (slash == std::string::npos) return ".";
  return slash == 0 ? "/" : path.substr(0, slash);
}

// --dummy-run: every check, then the memory peak of a run for real.
// Collective. Returns 0 if all passed (warnings do not count).
int dummy_run(const Options& o, const hlbl::Lattice& lattice,
              hlbl::PropagatorProvider& provider) {
  hlbl::Input const& in = o.input;
  MPI_Comm const comm = lattice.cart_comm();
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  auto const global = lattice.global();
  int const min_extent = *std::min_element(global.begin(), global.end());
  Plan const plan = make_plan(in, global);
  if (rank == 0) print_run_info(in, lattice, plan);
  int failed = 0;

  // ---- the chains: complete schedules, wrapping, stride, overlaps
  int peak_positions = 0;
  for (std::size_t c = 0; c < plan.chains.size(); ++c) {
    const hlbl::Chain& chain = plan.chains[c];
    int const max_steps = hlbl::max_steps_without_wrap(
        chain.direction, chain.stride, global, chain.start);
    if (rank == 0 && chain.steps > max_steps) {
      std::printf("# [hlbl] WARNING: chain %zu: %d steps wrap sources onto sites "
                  "already used (at most %d do not)\n", c, chain.steps, max_steps);
    }
    if (rank == 0 && 2 * chain.stride >= min_extent) {
      std::printf("# [hlbl] WARNING: chain %zu: stride %d is not below L/2: the "
                  "minimum-image separation is not (+-n,+-n,+-n,+-n)\n", c,
                  chain.stride);
    }
    hlbl::check_complete(plan.tasks[c].schedule, chain.stride);
    peak_positions = std::max(
        peak_positions,
        hlbl::simulate(plan.tasks[c].schedule, 2, hlbl::Mode::kInterleaved).peak_cache);
  }
  if (rank == 0) {
    std::printf("# [hlbl] schedules complete, cache peak %d positions\n", peak_positions);
    report_overlaps(plan.tasks, global, "# [hlbl] ");
  }

  // ---- the files a run writes (rank 0 writes them)
  if (rank == 0) {
    if (in.out.empty()) {
      std::printf("# [hlbl] WARNING: no result file (out = or --out): a run keeps "
                  "no results\n");
    }
    for (const std::string& path :
         in.out.empty() ? std::vector<std::string>{}
                        : std::vector<std::string>{in.out, timing_path(in)}) {
      if (access(directory_of(path).c_str(), W_OK) != 0) {
        std::printf("# [hlbl] FAILED: cannot write %s\n", path.c_str());
        failed = 1;
      } else if (access(path.c_str(), F_OK) == 0) {
        std::printf("# [hlbl] WARNING: %s exists; a run overwrites it\n", path.c_str());
      }
    }
    std::fflush(stdout);
  }

  QED_kernel_temps kqed_t;
  if (initialise(&kqed_t) != 0) throw std::runtime_error("KQED initialise failed");
  {
    hlbl::Contractor contractor(lattice, kqed_t);
    auto const residency = in.device_cache ? hlbl::PropagatorCache::Residency::kDevice
                                           : hlbl::PropagatorCache::Residency::kHost;

    // ---- memory estimate, after the contraction kernels took their stack
    unsigned long long reserved = contractor.reserve_kernel_memory();
    auto const mem = hlbl::estimate_memory(lattice, peak_positions, residency);
    unsigned long long device_free = mem.device_available;
    unsigned long long host_free = mem.host_available;
    MPI_Allreduce(MPI_IN_PLACE, &reserved, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &device_free, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &host_free, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm);
    bool const fits = mem.device_needed <= device_free && mem.host_needed <= host_free;
    if (rank == 0) {
      std::printf("# [hlbl] contraction kernels: %.2f GiB of device memory for their "
                  "stack (most on any rank)\n", gib(reserved));
      std::printf("# [hlbl] one position %.2f GiB per rank, peak %d positions, %s cache\n",
                  gib(mem.bytes_per_position), peak_positions,
                  in.device_cache ? "device" : "host");
      std::printf("# [hlbl] device: %.1f GiB needed, %.1f GiB free (least on any rank)\n",
                  gib(mem.device_needed), gib(device_free));
      std::printf("# [hlbl] host:   %.1f GiB needed, %.1f GiB available (least on any rank)\n",
                  gib(mem.host_needed), gib(host_free));
      std::printf("# [hlbl] memory estimate: %s\n", fits ? "fits" : "does NOT fit");
      std::fflush(stdout);
    }
    if (!fits) failed = 1;

    // ---- the real memory peak: peak - 1 cached positions (zeros) while the
    // next is solved, then one pair contracted. Unlike the estimate, this
    // includes QUDA's solver workspace.
    if (fits) {
      hlbl::Vec4 const site = hlbl::mod_pos(plan.chains[0].start, global);
      std::vector<hlbl::Vec4> filler;  // cache keys only; no site is negative
      std::map<hlbl::Vec4, int> pending{{site, 1}};
      for (int i = 0; i + 1 < peak_positions; ++i) {
        filler.push_back(hlbl::Vec4{-1, -1, -1, -1 - i});
        pending[filler.back()] = 1;
      }
      hlbl::PropagatorCache cache(residency, hlbl::block_length(lattice), pending);
      int allocated = 1;
      try {
        for (const hlbl::Vec4& key : filler) {
          cache.begin_insert(key).zero();
          cache.end_insert(key);
        }
      } catch (const std::exception& e) {
        std::fprintf(stderr, "[hlbl_hypercube] rank %d: filling the cache: %s\n",
                     rank, e.what());
        allocated = 0;
      }
      MPI_Allreduce(MPI_IN_PLACE, &allocated, 1, MPI_INT, MPI_MIN, comm);
      if (!allocated) {
        if (rank == 0) std::printf("# [hlbl] FAILED: the cache does not fit (see stderr)\n");
        failed = 1;
      } else {
        double t0 = MPI_Wtime();
        int const iterations = provider.compute(site, cache.begin_insert(site));
        cache.end_insert(site);
        double const t_solve = MPI_Wtime() - t0;
        hlbl::Vec4 const n = {plan.chains[0].stride, plan.chains[0].stride,
                              plan.chains[0].stride, plan.chains[0].stride};
        const double* d_y = cache.device_ptr(site, 0);
        const double* d_x = cache.device_ptr(filler.empty() ? site : filler[0], 1);
        t0 = MPI_Wtime();
        contractor.kernel_sum(d_x, d_y, site, n,
                              hlbl::xunit(in.muon_mass, in.lattice_spacing));
        double const t_pair = MPI_Wtime() - t0;
        if (rank == 0) {
          std::printf("# [hlbl] peak of a run: %d positions cached, one position solved "
                      "in %.1f s (%d iterations), one pair contracted in %.2f s: fits\n",
                      cache.size(), t_solve, iterations, t_pair);
          if (o.tune) {
            std::printf("# [hlbl] (the times include QUDA's tuning)\n");
          }
        }
      }
    }
  }
  free_QED_temps(&kqed_t);

  MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, comm);
  if (rank == 0) {
    std::printf("# [hlbl] dummy run: %s\n",
                failed ? "FAILED (see above)" : "all checks passed");
    std::fflush(stdout);
  }
  return failed;
}

}  // namespace

int main(int argc, char** argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Options o;
  try {
    o = parse_args(argc, argv);
  } catch (const std::exception& e) {
    if (rank == 0) std::fprintf(stderr, "[hlbl_hypercube] error: %s\n", e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int status = 0;
  if (o.dry_run) {
    // rank 0 only
    if (rank == 0) {
      try {
        run_dry_run(o);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "[hlbl_hypercube] error: %s\n", e.what());
        status = 1;
      }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
  }

  try {
    if (o.input.random_fields) {
      // random numbers on a lattice given in the input
      hlbl::Vec4 local;
      for (int i = 0; i < 4; ++i) {
        if (o.input.lattice[i] % o.input.procs[i] != 0) {
          throw std::runtime_error("procs does not divide lattice");
        }
        local[i] = o.input.lattice[i] / o.input.procs[i];
      }
      hlbl::Lattice lattice(MPI_COMM_WORLD, o.input.procs, local);
      hlbl::RandomProvider provider(lattice);
      if (o.dummy_run) {
        status = dummy_run(o, lattice, provider);
      } else {
        run(o, lattice, /*nconf=*/-1, provider);
      }
    } else {
#ifdef HLBL_HAVE_SOLVER
      if (o.tune) {
        // QUDA saves its tune cache in QUDA_RESOURCE_PATH when it ends
        const char* path = std::getenv("QUDA_RESOURCE_PATH");
        if (path == nullptr || *path == '\0' || access(path, W_OK) != 0) {
          throw std::runtime_error(
              "--tune: QUDA_RESOURCE_PATH must be a writable directory, or QUDA "
              "does not save its tune cache");
        }
        setenv("QUDA_ENABLE_TUNING", "1", 1);
        if (rank == 0) {
          std::printf("# [hlbl] QUDA tuning on; the tune cache is written to %s "
                      "when the program ends\n", path);
        }
      }
      // geometry, gauge field and operators come from invert.input in the
      // working directory
      double const t0 = MPI_Wtime();
      int stub_argc = 1;
      hlbl::Solver solver(stub_argc, argv);
      solver.read_gauge();
      solver.warm_up();  // QUDA gauge load and multigrid setup
      auto plaq = solver.plaquette();
      if (rank == 0) {
        std::printf("# [hlbl] configuration %d, plaquette %.12f, setup %.1f s\n",
                    solver.nconf(), plaq[0], MPI_Wtime() - t0);
      }
      hlbl::SolverProvider provider(solver);
      if (o.dummy_run) {
        status = dummy_run(o, solver.lattice(), provider);
      } else {
        run(o, solver.lattice(), solver.nconf(), provider);
      }
#else
      throw std::runtime_error(
          "built without tmLQCD (HLBL_WITH_TMLQCD=OFF): only --dry-run and "
          "--random-fields are available");
#endif
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[hlbl_hypercube] rank %d error: %s\n", rank, e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return status;
}
