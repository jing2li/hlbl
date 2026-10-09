// hlbl_hypercube: connected HLbL kernel sum on a sliding-hypercube schedule.
//
// For one separation |y| = 2n it solves the propagators (tmLQCD/QUDA, or
// random numbers with --random-fields), contracts every pair and writes the
// results to an HDF5 file. --dry-run only prints the schedule's counts and
// cache peak; it needs no GPU and no gauge field.

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hlbl/contract.hpp"
#include "hlbl/driver.hpp"
#include "hlbl/lattice.hpp"
#include "hlbl/output.hpp"
#include "hlbl/propagator_cache.hpp"
#include "hlbl/provider.hpp"
#include "hlbl/schedule.hpp"
#ifdef HLBL_HAVE_SOLVER
#include "hlbl/solver.hpp"
#endif

namespace {

struct Options {
  hlbl::Vec4 direction{1, 1, 1, 1};
  int stride = 4;
  int steps = -1;                       // -1: as many as fit before wrapping
  hlbl::Vec4 origin{0, 0, 0, 0};
  int flavors = 2;
  bool dry_run = false;
  int show_window = -1;                 // -1: none
  hlbl::Vec4 lattice{128, 64, 64, 64};   // global; bounds the default --steps
  hlbl::Vec4 procs{1, 1, 1, 1};          // --random-fields only
  std::string out;                      // result file; empty: do not write one
  std::string timing;                   // timing csv; default derived from out
  bool device_cache = true;
  int window_begin = 0;
  int window_end = -1;                  // half-open; -1: through the last
  bool random_fields = false;
  bool force = false;                   // run even if the cache will not fit
};

hlbl::Vec4 parse_vec4(char** argv, int& i) {
  hlbl::Vec4 v;
  for (int k = 0; k < 4; ++k) v[k] = std::atoi(argv[++i]);
  return v;
}

Options parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--direction") {
      o.direction = parse_vec4(argv, i);
    } else if (a == "--stride") {
      o.stride = std::atoi(argv[++i]);
    } else if (a == "--steps") {
      o.steps = std::atoi(argv[++i]);
    } else if (a == "--origin") {
      o.origin = parse_vec4(argv, i);
    } else if (a == "--flavors") {
      o.flavors = std::atoi(argv[++i]);
    } else if (a == "--lattice") {
      o.lattice = parse_vec4(argv, i);
    } else if (a == "--dry-run") {
      o.dry_run = true;
    } else if (a == "--show-window") {
      o.show_window = std::atoi(argv[++i]);
    } else if (a == "--procs") {
      o.procs = parse_vec4(argv, i);
    } else if (a == "--out") {
      o.out = argv[++i];
    } else if (a == "--timing") {
      o.timing = argv[++i];
    } else if (a == "--cache") {
      std::string v = argv[++i];
      if (v != "device" && v != "host") {
        throw std::runtime_error("--cache must be device or host");
      }
      o.device_cache = (v == "device");
    } else if (a == "--window-range") {
      // a:b, half-open; "a:" runs from a through the last window
      std::string v = argv[++i];
      auto colon = v.find(':');
      if (colon == std::string::npos) {
        throw std::runtime_error("--window-range wants a:b");
      }
      o.window_begin = std::atoi(v.substr(0, colon).c_str());
      std::string b = v.substr(colon + 1);
      o.window_end = b.empty() ? -1 : std::atoi(b.c_str());
    } else if (a == "--random-fields") {
      o.random_fields = true;
    } else if (a == "--force") {
      o.force = true;
    } else {
      throw std::runtime_error("unrecognised option: " + a);
    }
  }
  return o;
}

void print_vec4(const hlbl::Vec4& v) {
  std::printf("(%d,%d,%d,%d)", v[0], v[1], v[2], v[3]);
}

// Prints a report plus two kinds of lines that scripts/compare_schedule.sh
// compares with the Python reference:
//   mode=<name> n_propagators=.. n_solves=.. n_pairs=.. n_kernel_sums=.. peak_cache=..
//   edge (a,b,c,d) (e,f,g,h)       one per pair of the window given by --show-window
void run_dry_run(const Options& o) {
  for (int i = 0; i < 4; ++i) {
    if (o.direction[i] != 1 && o.direction[i] != -1) {
      throw std::runtime_error(
          "--direction must be a sign vector (+-1,+-1,+-1,+-1)");
    }
  }
  if (o.stride <= 0) throw std::runtime_error("--stride must be positive");

  int steps = o.steps;
  if (steps < 0) {
    steps = hlbl::max_steps_without_wrap(o.direction, o.stride, o.lattice,
                                          o.origin);
  }

  std::printf("direction=");
  print_vec4(o.direction);
  std::printf(" stride=%d origin=", o.stride);
  print_vec4(o.origin);
  std::printf(" steps=%d windows=%d\n", steps, steps + 1);

  auto schedule = hlbl::build_schedule(o.direction, o.stride, steps, o.origin);
  hlbl::check_complete(schedule, o.stride);
  std::printf("check_complete: ok\n");

  // Mode summary lines first, edges (if requested) last -- matching the
  // order scripts/compare_schedule.sh's Python side prints them in, so the
  // two can be diffed directly.
  for (auto mode : {hlbl::Mode::kSerial, hlbl::Mode::kPipelined,
                    hlbl::Mode::kInterleaved}) {
    auto stats = hlbl::simulate(schedule, o.flavors, mode);
    std::printf(
        "mode=%s n_propagators=%lld n_solves=%lld n_pairs=%lld "
        "n_kernel_sums=%lld peak_cache=%d\n",
        mode == hlbl::Mode::kSerial
            ? "serial"
            : (mode == hlbl::Mode::kPipelined ? "pipelined" : "interleaved"),
        stats.n_propagators, stats.n_solves, stats.n_pairs,
        stats.n_kernel_sums, stats.peak_cache);
  }

  if (o.show_window >= 0 &&
      o.show_window < static_cast<int>(schedule.size())) {
    const auto& step = schedule[o.show_window];
    std::printf("window=%d new_propagators=%d pairs=%d\n", step.index,
                step.n_new(), step.n_pairs());

    auto rel = [&](const hlbl::Vec4& p) {
      hlbl::Vec4 r;
      for (int k = 0; k < 4; ++k) r[k] = (p[k] - o.origin[k]) / o.stride;
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

double gib(std::size_t bytes) { return bytes / 1073741824.0; }

// Builds the schedule, checks memory, runs the loop and prints a summary.
// Collective; a failure aborts the whole job.
void run(const Options& o, const hlbl::Lattice& lattice, int nconf,
         hlbl::PropagatorProvider& provider) {
  int rank = 0;
  MPI_Comm_rank(lattice.cart_comm(), &rank);
  auto const global = lattice.global();

  int steps = o.steps;
  if (steps < 0) {
    steps = hlbl::max_steps_without_wrap(o.direction, o.stride, global, o.origin);
  } else if (steps > hlbl::max_steps_without_wrap(o.direction, o.stride, global,
                                                  o.origin) && rank == 0) {
    std::printf("# [hlbl] WARNING: %d steps wrap sources onto sites already used\n", steps);
  }
  if (2 * o.stride >= *std::min_element(global.begin(), global.end()) && rank == 0) {
    std::printf("# [hlbl] WARNING: stride %d is not below L/2: the minimum-image "
                "separation is not (+-n,+-n,+-n,+-n)\n", o.stride);
  }

  auto schedule = hlbl::build_schedule(o.direction, o.stride, steps, o.origin);
  hlbl::check_complete(schedule, o.stride);

  auto const residency = o.device_cache ? hlbl::PropagatorCache::Residency::kDevice
                                        : hlbl::PropagatorCache::Residency::kHost;
  auto const mem = hlbl::estimate_memory(lattice, schedule, residency);
  int fits = mem.fits() ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &fits, 1, MPI_INT, MPI_MIN, lattice.cart_comm());
  if (rank == 0) {
    std::printf("# [hlbl] lattice %dx%dx%dx%d on %dx%dx%dx%d ranks, direction (%d,%d,%d,%d), "
                "stride %d, %d steps (%d windows), %s cache\n",
                global[0], global[1], global[2], global[3], lattice.procs()[0],
                lattice.procs()[1], lattice.procs()[2], lattice.procs()[3],
                o.direction[0], o.direction[1], o.direction[2], o.direction[3],
                o.stride, steps, steps + 1, o.device_cache ? "device" : "host");
    std::printf("# [hlbl] one position %.2f GiB per rank, peak %d positions; "
                "device %.1f GiB needed / %.1f free, host %.1f GiB needed / %.1f available\n",
                gib(mem.bytes_per_position), mem.peak_positions,
                gib(mem.device_needed), gib(mem.device_available),
                gib(mem.host_needed), gib(mem.host_available));
    std::fflush(stdout);
  }
  if (!fits && !o.force) {
    if (rank == 0) {
      std::fprintf(stderr, "[hlbl_hypercube] the cache does not fit on this rank "
                   "(see above); use more ranks, --cache host, or --force\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  QED_kernel_temps kqed_t;
  if (initialise(&kqed_t) != 0) {
    std::fprintf(stderr, "[hlbl_hypercube] KQED initialise failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  hlbl::Contractor contractor(lattice, kqed_t);

  int const end = o.window_end < 0 ? steps + 1 : std::min(o.window_end, steps + 1);
  std::unique_ptr<hlbl::ResultWriter> writer;
  std::unique_ptr<hlbl::TimingLog> timing;
  if (rank == 0 && !o.out.empty()) {
    hlbl::RunInfo info;
    info.lattice = global;
    info.procs = lattice.procs();
    info.direction = o.direction;
    info.origin = o.origin;
    info.stride = o.stride;
    info.nconf = nconf;
    info.window_begin = o.window_begin;
    info.window_end = end;
    writer = std::make_unique<hlbl::ResultWriter>(o.out, info);
    std::string csv = o.timing;
    if (csv.empty()) {
      csv = o.out.substr(0, o.out.rfind('.')) + ".timing.csv";
    }
    timing = std::make_unique<hlbl::TimingLog>(csv);
  }

  hlbl::RunOptions ro;
  ro.window_begin = o.window_begin;
  ro.window_end = o.window_end;
  ro.residency = residency;
  ro.verbose = true;
  hlbl::RunStats stats;
  double const t0 = MPI_Wtime();
  auto results = hlbl::run_schedule(lattice, schedule, ro, provider, contractor,
                                     writer.get(), timing.get(), &stats);
  double const wall = MPI_Wtime() - t0;

  if (rank == 0) {
    std::printf("# [hlbl] done: %d pairs (%zu results), %d positions solved, "
                "%lld solver iterations, cache peak %d positions\n",
                stats.pairs, results.size(), stats.positions_solved,
                stats.iterations, stats.cache_peak);
    std::printf("# [hlbl] wall %.1f s: solves %.1f s (%.2f s per position), "
                "kernel sums %.1f s (%.2f s per pair)\n", wall,
                stats.seconds_solve,
                stats.positions_solved ? stats.seconds_solve / stats.positions_solved : 0.,
                stats.seconds_kernel_sum,
                stats.pairs ? stats.seconds_kernel_sum / stats.pairs : 0.);
    if (writer) std::printf("# [hlbl] results in %s\n", o.out.c_str());
  }
  free_QED_temps(&kqed_t);
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
  for (int i = 0; i < 4; ++i) {
    if (o.direction[i] != 1 && o.direction[i] != -1) {
      if (rank == 0) {
        std::fprintf(stderr, "[hlbl_hypercube] error: --direction must be a sign "
                     "vector (+-1,+-1,+-1,+-1)\n");
      }
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  if (o.stride <= 0) {
    if (rank == 0) std::fprintf(stderr, "[hlbl_hypercube] error: --stride must be positive\n");
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
    if (o.random_fields) {
      // random numbers on a lattice given on the command line
      hlbl::Vec4 local;
      for (int i = 0; i < 4; ++i) {
        if (o.lattice[i] % o.procs[i] != 0) {
          throw std::runtime_error("--procs does not divide --lattice");
        }
        local[i] = o.lattice[i] / o.procs[i];
      }
      hlbl::Lattice lattice(MPI_COMM_WORLD, o.procs, local);
      hlbl::RandomProvider provider(lattice);
      run(o, lattice, /*nconf=*/-1, provider);
    } else {
#ifdef HLBL_HAVE_SOLVER
      // geometry, gauge field and operators come from invert.input in the
      // working directory
      int stub_argc = 1;
      hlbl::Solver solver(stub_argc, argv);
      solver.read_gauge();
      solver.warm_up();  // QUDA gauge load and multigrid setup
      auto plaq = solver.plaquette();
      if (rank == 0) {
        std::printf("# [hlbl] configuration %d, plaquette %.12f\n", solver.nconf(),
                    plaq[0]);
      }
      hlbl::SolverProvider provider(solver);
      run(o, solver.lattice(), solver.nconf(), provider);
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
  return 0;
}
