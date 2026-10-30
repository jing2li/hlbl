// hlbl_hypercube: connected HLbL kernel sum on a sliding-hypercube schedule.
//
//   hlbl_hypercube --in FILE [--dummy-run [--tune]]
//
// For each chain of the input file (start, direction, stride, steps) it solves
// the propagators (tmLQCD/QUDA, or random numbers with random_fields), computes
// the 2+2 pieces of the 2p2 sources at their first solve, contracts every pair
// and writes one HDF5 file. Gauge field and multigrid are set up once.
// Everything else comes from the input file.
//
// A production run checks only its input. --dummy-run does the full setup,
// every check, and the memory peak of a run for real (cache filled, one
// position solved, one pair contracted); it writes no results. --tune saves
// QUDA's tune cache. scripts/hlbl_mII_hypercube_schedule.py plans a run
// without a GPU.

#include <mpi.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
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
#include "hlbl/solver.hpp"

namespace {

struct Options {
  hlbl::Input input;        // from --in
  bool dummy_run = false;
  bool tune = false;        // with --dummy-run
};

Options parse_args(int argc, char** argv) {
  Options o;
  std::string path;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--in") {
      if (++i >= argc) throw std::runtime_error("--in wants a file name");
      path = argv[i];
    } else if (a == "--dummy-run") {
      o.dummy_run = true;
    } else if (a == "--tune") {
      o.tune = true;
    } else {
      throw std::runtime_error("unrecognised option: " + a);
    }
  }
  if (path.empty()) {
    throw std::runtime_error("usage: hlbl_hypercube --in FILE [--dummy-run [--tune]]");
  }
  if (o.tune && !o.dummy_run) throw std::runtime_error("--tune goes with --dummy-run");

  // checked here so that a bad file stops before the solver setup
  o.input = hlbl::parse_input_file(path);
  hlbl::check_required(o.input, path);
  if (o.tune && o.input.random_fields) {
    throw std::runtime_error("--tune needs the solver, not random fields");
  }
  return o;
}

std::string str(const hlbl::Vec4& v) {
  return std::to_string(v[0]) + "," + std::to_string(v[1]) + "," +
         std::to_string(v[2]) + "," + std::to_string(v[3]);
}

// The input file's lattice and procs must be what the run uses (for a solver
// run, what tmLQCD read from invert.input).
void check_geometry(const hlbl::Input& in, const hlbl::Lattice& lattice) {
  if (*in.lattice != lattice.global() || *in.procs != lattice.procs()) {
    throw std::runtime_error(
        "the input file has lattice " + str(*in.lattice) + " on procs " +
        str(*in.procs) + ", but invert.input has lattice " + str(lattice.global()) +
        " on procs " + str(lattice.procs()));
  }
}

double gib(std::size_t bytes) { return bytes / 1073741824.0; }

hlbl::PropagatorCache::Residency residency(const hlbl::Input& in) {
  return in.cache == "device" ? hlbl::PropagatorCache::Residency::kDevice
                              : hlbl::PropagatorCache::Residency::kHost;
}

// The chains as they run (steps=auto resolved), their schedules, and which
// solves are followed by 2+2.
struct Plan {
  std::vector<hlbl::Chain> chains;
  std::vector<hlbl::Schedule> schedules;
  std::vector<hlbl::Vec4> sources_2p2;  // the 2p2 lines' sources
  hlbl::Plan2p2 plan_2p2;
  int max_targets_2p2 = 0;              // 0: no 2+2
};

Plan make_plan(const hlbl::Input& in, const hlbl::Vec4& global) {
  Plan plan;
  plan.chains = in.chains;
  for (hlbl::Chain& chain : plan.chains) {
    if (chain.steps < 0) {
      chain.steps = hlbl::max_steps_without_wrap(chain.direction, chain.stride,
                                                 global, chain.start);
    }
    plan.schedules.push_back(hlbl::build_schedule(chain.direction, chain.stride,
                                                  chain.steps, chain.start));
  }
  for (const hlbl::Source2p2& g : in.sources_2p2) {
    plan.sources_2p2.push_back(g.source);
    plan.max_targets_2p2 =
        std::max(plan.max_targets_2p2, static_cast<int>(g.targets.size()));
  }
  plan.plan_2p2 = hlbl::plan_2p2(plan.schedules, plan.sources_2p2, global);
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
              procs[2], procs[3], plan.chains.size(), in.cache.c_str());
  for (std::size_t c = 0; c < plan.chains.size(); ++c) {
    const hlbl::Chain& ch = plan.chains[c];
    std::printf("# [hlbl] chain %zu: start (%s), direction (%s), stride %d, %d steps\n",
                c, str(ch.start).c_str(), str(ch.direction).c_str(), ch.stride,
                ch.steps);
  }
  std::printf("# [hlbl] muon mass %.10g MeV, lattice spacing %.10g fm: xunit = m_mu a = %.10g\n",
              in.muon_mass, in.lattice_spacing,
              hlbl::xunit(in.muon_mass, in.lattice_spacing));
  if (!in.sources_2p2.empty()) {
    long long targets = 0;
    for (const hlbl::Source2p2& g : in.sources_2p2) targets += g.targets.size();
    std::printf("# [hlbl] 2+2: %zu sources (%lld targets), %zu solved by the chains\n",
                in.sources_2p2.size(), targets,
                in.sources_2p2.size() - plan.plan_2p2.unsolved.size());
  }
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
  if (rank == 0) {
    hlbl::RunInfo info;
    info.lattice = lattice.global();
    info.procs = lattice.procs();
    info.nconf = nconf;
    info.chains = plan.chains;
    info.muon_mass = in.muon_mass;
    info.lattice_spacing = in.lattice_spacing;
    info.xunit = xunit;
    info.with_2p2 = !in.sources_2p2.empty();
    writer = std::make_unique<hlbl::ResultWriter>(in.out, info);
    timing = std::make_unique<hlbl::TimingLog>(timing_path(in));
  }

  hlbl::RunOptions ro;
  ro.residency = residency(in);
  ro.xunit = xunit;
  ro.verbose = true;
  ro.sources_2p2 = in.sources_2p2;
  hlbl::RunStats stats;
  double const t0 = MPI_Wtime();
  auto results = hlbl::run_chains(lattice, plan.schedules, ro, provider,
                                   contractor, writer.get(), timing.get(), &stats);
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
    if (!in.sources_2p2.empty()) {
      std::printf("# [hlbl] 2+2: %d sources, %lld targets, %.1f s (%.2f s per source)\n",
                  stats.pieces_2p2, stats.targets_2p2, stats.seconds_2p2,
                  stats.pieces_2p2 ? stats.seconds_2p2 / stats.pieces_2p2 : 0.);
    }
    std::printf("# [hlbl] results in %s\n", in.out.c_str());
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
    hlbl::check_complete(plan.schedules[c], chain.stride);
    peak_positions = std::max(
        peak_positions,
        hlbl::simulate(plan.schedules[c], hlbl::Mode::kInterleaved).peak_cache);
  }
  if (rank == 0) {
    std::printf("# [hlbl] schedules complete, cache peak %d positions\n", peak_positions);
    for (const std::string& line : hlbl::overlap_report(plan.schedules, global)) {
      std::printf("# [hlbl] %s\n", line.c_str());
    }
    for (const std::string& line : hlbl::report_2p2(plan.plan_2p2, plan.sources_2p2)) {
      std::printf("# [hlbl] %s\n", line.c_str());
    }
  }

  // ---- the files a run writes (rank 0 writes them)
  if (rank == 0) {
    for (const std::string& path : {in.out, timing_path(in)}) {
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

    // ---- memory estimate, after the contraction kernels took their stack
    bool const with_2p2 = plan.max_targets_2p2 > 0;
    unsigned long long reserved = contractor.reserve_kernel_memory(with_2p2);
    auto const mem = hlbl::estimate_memory(lattice, peak_positions, residency(in),
                                           plan.max_targets_2p2);
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
                  gib(mem.bytes_per_position), peak_positions, in.cache.c_str());
      if (with_2p2) {
        std::printf("# [hlbl] 2+2 work buffers %.2f GiB per rank (up to %d targets)\n",
                    gib(mem.bytes_2p2), plan.max_targets_2p2);
      }
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
      hlbl::PropagatorCache cache(residency(in), hlbl::block_length(lattice), pending);
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
        hlbl::DeviceBuffer<double>& dest = cache.begin_insert(site);
        int const iterations = provider.compute(site, dest);
        double const t_solve = MPI_Wtime() - t0;
        // as in a run: the 2+2 of the largest target list, before the block
        // is stored
        double t_2p2 = 0;
        if (with_2p2) {
          auto const largest = std::max_element(
              in.sources_2p2.begin(), in.sources_2p2.end(),
              [](const auto& a, const auto& b) { return a.targets.size() < b.targets.size(); });
          t0 = MPI_Wtime();
          contractor.pieces_2p2(dest.get(), site, largest->targets,
                                hlbl::xunit(in.muon_mass, in.lattice_spacing));
          t_2p2 = MPI_Wtime() - t0;
        }
        cache.end_insert(site);
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
          if (with_2p2) {
            std::printf("# [hlbl] one 2+2 source (%d targets) in %.2f s\n",
                        plan.max_targets_2p2, t_2p2);
          }
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
  try {
    if (o.input.random_fields) {
      hlbl::Vec4 const& global = *o.input.lattice;
      hlbl::Vec4 const& procs = *o.input.procs;
      hlbl::Vec4 local;
      for (int i = 0; i < 4; ++i) {
        if (global[i] % procs[i] != 0) {
          throw std::runtime_error("procs does not divide lattice");
        }
        local[i] = global[i] / procs[i];
      }
      hlbl::Lattice lattice(MPI_COMM_WORLD, procs, local);
      hlbl::RandomProvider provider(lattice);
      if (o.dummy_run) {
        status = dummy_run(o, lattice, provider);
      } else {
        run(o, lattice, /*nconf=*/-1, provider);
      }
    } else {
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
      check_geometry(o.input, solver.lattice());
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
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[hlbl_hypercube] rank %d error: %s\n", rank, e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return status;
}
