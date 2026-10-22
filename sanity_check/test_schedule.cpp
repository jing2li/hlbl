// Window sizes, pair counts, completeness and cache peaks of the schedule.
// scripts/compare_schedule.sh checks it against the Python reference.

#include "hlbl/schedule.hpp"

#include <cstdio>
#include <set>
#include <utility>

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

  // K = 30 shifts: 466 propagators, 1965 pairs. Peak cache: 16 (interleaved),
  // 30 (serial).
  {
    Vec4 h{1, 1, 1, 1};
    auto steps = build_schedule(h, 4, 30);
    check_complete(steps, 4);

    auto serial = simulate(steps, Mode::kSerial);
    HLBL_CHECK(serial.n_propagators == 466);
    HLBL_CHECK(serial.n_solves == 466 * 12 * 2);
    HLBL_CHECK(serial.n_pairs == 1965);
    HLBL_CHECK(serial.n_kernel_sums == 3930);
    HLBL_CHECK(serial.peak_cache == 30);

    auto interleaved = simulate(steps, Mode::kInterleaved);
    HLBL_CHECK(interleaved.n_propagators == serial.n_propagators);
    HLBL_CHECK(interleaved.n_pairs == serial.n_pairs);
    HLBL_CHECK(interleaved.n_kernel_sums == serial.n_kernel_sums);
    HLBL_CHECK(interleaved.peak_cache == 16);
  }

  // Every direction has the same counts and the same cache peak of 16 as
  // (1,1,1,1): the solve order of a window is that of (1,1,1,1), reflected by h.
  for (const Vec4& h : sign_vectors()) {
    auto offsets = window_offsets(h);
    HLBL_CHECK(offsets.size() == 16);
    HLBL_CHECK(std::set<Vec4>(offsets.begin(), offsets.end()).size() == 16);
    for (int k : {1, 2, 5, 30}) {
      auto steps = build_schedule(h, 4, k);
      check_complete(steps, 4);
      auto st = simulate(steps, Mode::kInterleaved);
      HLBL_CHECK(st.n_propagators == 16 + 15 * k);
      HLBL_CHECK(st.n_pairs == 15 + 65 * k);
      HLBL_CHECK(st.peak_cache == 16);
    }
  }

  // Resuming after window j (README, "Resume"): a new chain from window j's
  // centre with K - j steps has windows j+1..K as its windows 1.., and repeats
  // only the 15 pairs of window j's centre.
  for (Vec4 h : {Vec4{1, 1, 1, 1}, Vec4{1, -1, 1, -1}}) {
    int const n = 3, K = 8;
    Vec4 const origin{2, 0, 5, 1};
    auto const full = build_schedule(h, n, K, origin);
    for (int j : {0, 1, 5}) {
      auto const resumed =
          build_schedule(h, n, K - j, add(origin, scale(n * j, h)));
      for (int k = 1; k <= K - j; ++k) {
        const Step& a = full[j + k];
        const Step& b = resumed[k];
        HLBL_CHECK(a.sources.size() == b.sources.size());
        for (std::size_t i = 0; i < a.sources.size(); ++i) {
          HLBL_CHECK(a.sources[i].position == b.sources[i].position);
          HLBL_CHECK(a.sources[i].partners == b.sources[i].partners);
        }
      }
      // the pairs of windows 0..j and of the resumed chain cover the full
      // chain; they share 15
      std::set<std::pair<Vec4, Vec4>> done, again;
      auto unordered = [](Vec4 p, Vec4 q) {
        return p < q ? std::make_pair(p, q) : std::make_pair(q, p);
      };
      for (int k = 0; k <= j; ++k) {
        for (const auto& s : full[k].sources) {
          for (const auto& q : s.partners) done.insert(unordered(q, s.position));
        }
      }
      for (const auto& [q, p] : all_pairs(resumed)) again.insert(unordered(q, p));
      int shared = 0;
      for (const auto& pr : again) shared += static_cast<int>(done.count(pr));
      HLBL_CHECK(shared == 15);
      HLBL_CHECK(done.size() + again.size() - shared == all_pairs(full).size());
    }
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
