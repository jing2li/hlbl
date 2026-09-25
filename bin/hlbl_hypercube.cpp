// hlbl_hypercube: connected HLbL kernel sum on a sliding-hypercube schedule.
//
// --dry-run builds the schedule and prints propagator/pair counts and the
// peak propagator-cache size for both execution modes; it needs no GPU, no
// tmLQCD and no gauge field. Everything else is not implemented yet.

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hlbl/schedule.hpp"

namespace {

struct Options {
  hlbl::Vec4 direction{1, 1, 1, 1};
  int stride = 4;
  int steps = -1;                       // -1: as many as fit before wrapping
  hlbl::Vec4 origin{0, 0, 0, 0};
  int flavors = 2;
  bool dry_run = false;
  int show_window = -1;                 // -1: none
  hlbl::Vec4 lattice{128, 64, 64, 64};   // only bounds the default --steps
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
    } else {
      throw std::runtime_error("unrecognised option: " + a);
    }
  }
  return o;
}

void print_vec4(const hlbl::Vec4& v) {
  std::printf("(%d,%d,%d,%d)", v[0], v[1], v[2], v[3]);
}

// Emits, on top of a human-readable report:
//   mode=<serial|pipelined> n_propagators=.. n_solves=.. n_pairs=.. n_kernel_sums=.. peak_cache=..
//   edge (a,b,c,d) (e,f,g,h)
// one line per mode and (if --show-window is given) per pair in that
// window, in a fixed, sorted format. scripts/compare_schedule.sh greps
// exactly these lines out to diff against the Python reference.
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
  for (auto mode : {hlbl::Mode::kSerial, hlbl::Mode::kPipelined}) {
    auto stats = hlbl::simulate(schedule, o.flavors, mode);
    std::printf(
        "mode=%s n_propagators=%lld n_solves=%lld n_pairs=%lld "
        "n_kernel_sums=%lld peak_cache=%d\n",
        mode == hlbl::Mode::kSerial ? "serial" : "pipelined",
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

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int status = 0;
  if (rank == 0) {
    try {
      Options o = parse_args(argc, argv);
      if (o.dry_run) {
        run_dry_run(o);
      } else {
        std::printf(
            "# [hlbl_hypercube] only --dry-run is implemented so far\n");
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[hlbl_hypercube] error: %s\n", e.what());
      status = 1;
    }
  }
  MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return status;
}
