// schedule_dump: prints the schedules of an input file, for
// scripts/compare_schedule.sh to compare with the Python reference. Test only.
//
//   schedule_dump FILE [window]
//
// Per chain ("chain i of N" first if there are several):
//   direction=(..) stride=n origin=(..) steps=K windows=K+1
//   mode=<name> n_propagators=.. n_solves=.. n_pairs=.. n_kernel_sums=.. peak_cache=..
//   edge (a,b,c,d) (e,f,g,h)      each pair of `window`, in units of the stride
// then, with several chains, a "total ..." line (interleaved) and the overlap
// report; then, with 2p2 lines, in run order
//   2p2 chain=c window=k source=(..) targets=n   each 2+2 computed
//   2p2 total sources=.. targets=..
// and the 2+2 warnings. Errors in the file go to stderr as
// "schedule_dump: error: ...".

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include "hlbl/input.hpp"
#include "hlbl/schedule.hpp"

using namespace hlbl;

namespace {

void print_vec4(const Vec4& v) { std::printf("(%d,%d,%d,%d)", v[0], v[1], v[2], v[3]); }

void print_edges(const Step& step, const Chain& chain) {
  auto rel = [&](const Vec4& p) {
    Vec4 r;
    for (int k = 0; k < 4; ++k) r[k] = (p[k] - chain.start[k]) / chain.stride;
    return r;
  };
  std::vector<std::pair<Vec4, Vec4>> edges;
  for (const Source& src : step.sources) {
    Vec4 const a = rel(src.position);
    for (const Vec4& q : src.partners) {
      Vec4 const b = rel(q);
      edges.push_back(a < b ? std::make_pair(a, b) : std::make_pair(b, a));
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::fprintf(stderr, "usage: schedule_dump FILE [window]\n");
    return 1;
  }
  int const window = argc > 2 ? std::atoi(argv[2]) : -1;
  try {
    Input const in = parse_input_file(argv[1]);
    check_required(in, argv[1]);
    int const n = static_cast<int>(in.chains.size());
    std::vector<Schedule> schedules;
    ScheduleStats total;
    for (int c = 0; c < n; ++c) {
      Chain chain = in.chains[c];
      if (chain.steps < 0) {
        chain.steps = max_steps_without_wrap(chain.direction, chain.stride,
                                             *in.lattice, chain.start);
      }
      if (n > 1) std::printf("chain %d of %d\n", c, n);
      std::printf("direction=");
      print_vec4(chain.direction);
      std::printf(" stride=%d origin=", chain.stride);
      print_vec4(chain.start);
      std::printf(" steps=%d windows=%d\n", chain.steps, chain.steps + 1);

      Schedule const schedule =
          build_schedule(chain.direction, chain.stride, chain.steps, chain.start);
      check_complete(schedule, chain.stride);
      for (auto mode : {Mode::kSerial, Mode::kInterleaved}) {
        ScheduleStats const s = simulate(schedule, mode);
        std::printf("mode=%s n_propagators=%lld n_solves=%lld n_pairs=%lld "
                    "n_kernel_sums=%lld peak_cache=%d\n",
                    mode == Mode::kSerial ? "serial" : "interleaved", s.n_propagators,
                    s.n_solves, s.n_pairs, s.n_kernel_sums, s.peak_cache);
        if (mode == Mode::kInterleaved) {
          total.n_propagators += s.n_propagators;
          total.n_solves += s.n_solves;
          total.n_pairs += s.n_pairs;
          total.n_kernel_sums += s.n_kernel_sums;
          total.peak_cache = std::max(total.peak_cache, s.peak_cache);
        }
      }
      if (window >= 0 && window < static_cast<int>(schedule.size())) {
        print_edges(schedule[window], chain);
      }
      schedules.push_back(schedule);
    }
    if (n > 1) {
      std::printf("total chains=%d n_propagators=%lld n_solves=%lld n_pairs=%lld "
                  "n_kernel_sums=%lld peak_cache=%d\n",
                  n, total.n_propagators, total.n_solves, total.n_pairs,
                  total.n_kernel_sums, total.peak_cache);
      for (const std::string& line : overlap_report(schedules, *in.lattice)) {
        std::printf("%s\n", line.c_str());
      }
    }
    if (!in.sources_2p2.empty()) {
      std::vector<Vec4> sources;
      for (const Source2p2& g : in.sources_2p2) sources.push_back(g.source);
      Plan2p2 const plan = plan_2p2(schedules, sources, *in.lattice);
      long long n_sources = 0, n_targets = 0;
      for (int c = 0; c < n; ++c) {
        for (const Step& step : schedules[c]) {
          for (const Source& src : step.sources) {
            auto const it = plan.at[c].find(src.position);
            if (it == plan.at[c].end()) continue;
            const Source2p2& g = in.sources_2p2[it->second];
            std::printf("2p2 chain=%d window=%d source=", c, step.index);
            print_vec4(g.source);
            std::printf(" targets=%zu\n", g.targets.size());
            ++n_sources;
            n_targets += static_cast<long long>(g.targets.size());
          }
        }
      }
      std::printf("2p2 total sources=%lld targets=%lld\n", n_sources, n_targets);
      for (const std::string& line : report_2p2(plan, sources)) {
        std::printf("%s\n", line.c_str());
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "schedule_dump: error: %s\n", e.what());
    return 1;
  }
  return 0;
}
