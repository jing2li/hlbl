// Checks the schedule against the worked examples in the README and against
// its own completeness/no-wrap invariants. scripts/compare_schedule.sh does
// the complementary check against the Python reference.

#include "hlbl/schedule.hpp"

#include <cstdio>

#include "check.hpp"

using namespace hlbl;

int main() {
  // window_offsets: 16 points, the zero vector plus the 15 sign vectors
  // other than h.
  {
    Vec4 h{1, 1, 1, 1};
    auto offsets = window_offsets(h);
    HLBL_CHECK(offsets.size() == 16);
    HLBL_CHECK(offsets[0] == (Vec4{0, 0, 0, 0}));
    for (std::size_t i = 1; i < offsets.size(); ++i) HLBL_CHECK(offsets[i] != h);
  }

  // README worked example: stride 1, 3 shifts. Window 0 is 16 props / 15
  // pairs; every later window is 15 props / 65 pairs (1+8+8+4+8+4+4+2+8+4+4+2+4+2+2).
  {
    Vec4 h{1, 1, 1, 1};
    auto steps = build_schedule(h, 1, 3);
    check_complete(steps, 1);
    HLBL_CHECK(steps.size() == 4);
    HLBL_CHECK(steps[0].n_new() == 16);
    HLBL_CHECK(steps[0].n_pairs() == 15);
    for (int k = 1; k < 4; ++k) {
      HLBL_CHECK(steps[k].n_new() == 15);
      HLBL_CHECK(steps[k].n_pairs() == 65);
    }
  }

  // K = 30 shifts: 466 propagators, 1965 pairs, peak cache 30 (serial) / 45
  // (pipelined) -- the numbers validated in the README and in the Python
  // prototype.
  {
    Vec4 h{1, 1, 1, 1};
    auto steps = build_schedule(h, 4, 30);
    check_complete(steps, 4);

    auto serial = simulate(steps, 2, Mode::kSerial);
    HLBL_CHECK(serial.n_propagators == 466);
    HLBL_CHECK(serial.n_solves == 466 * 12 * 2);
    HLBL_CHECK(serial.n_pairs == 1965);
    HLBL_CHECK(serial.n_kernel_sums == 3930);
    HLBL_CHECK(serial.peak_cache == 30);

    auto pipelined = simulate(steps, 2, Mode::kPipelined);
    HLBL_CHECK(pipelined.n_propagators == serial.n_propagators);
    HLBL_CHECK(pipelined.n_pairs == serial.n_pairs);
    HLBL_CHECK(pipelined.peak_cache == 45);
  }

  // max_steps_without_wrap: on a lattice too small to hold many windows
  // without collisions, the bound must actually engage (not hit the cap).
  {
    Vec4 h{1, 1, 1, 1};
    int m = max_steps_without_wrap(h, 4, Vec4{16, 16, 16, 16});
    HLBL_CHECK(m >= 0 && m < 100);
  }

  // check_complete must hold for other directions, strides and origins too.
  for (int n : {1, 2, 3}) {
    for (Vec4 h : {Vec4{1, -1, 1, 1}, Vec4{-1, -1, -1, -1}, Vec4{1, 1, -1, -1}}) {
      auto steps = build_schedule(h, n, 6, Vec4{5, 0, 0, 0});
      check_complete(steps, n);
    }
  }

  std::printf("test_schedule: all checks passed\n");
  return 0;
}
