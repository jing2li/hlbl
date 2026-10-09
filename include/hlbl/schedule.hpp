#pragma once

// The sliding-hypercube schedule: which sources to solve and which pairs to
// contract. It must stay in step with scripts/hlbl_mII_hypercube_schedule.py
// (checked by scripts/compare_schedule.sh). See the README for the scheme.

#include <array>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hlbl/vec4.hpp"

namespace hlbl {

// The 16 sign vectors (+-1,+-1,+-1,+-1).
std::array<Vec4, 16> sign_vectors();

// A0 = {0} u (G \ {h}), in units of the stride.
std::vector<Vec4> window_offsets(const Vec4& h);

struct Source {
  Vec4 position;                // unwrapped lattice coordinates
  std::vector<Vec4> partners;   // earlier sources it forms a pair with
};

// One window: the sources that are new in it, in solve order.
struct Step {
  int index = 0;
  std::vector<Source> sources;

  int n_new() const { return static_cast<int>(sources.size()); }
  int n_pairs() const {
    int n = 0;
    for (const auto& s : sources) n += static_cast<int>(s.partners.size());
    return n;
  }
};

using Schedule = std::vector<Step>;

// Windows 0..n_steps. A new source is paired with every earlier source (from
// the previous window or earlier in its own window) that is at a separation
// of stride*(+-1,+-1,+-1,+-1).
Schedule build_schedule(const Vec4& h, int stride, int n_steps,
                         const Vec4& origin = {0, 0, 0, 0});

std::vector<Vec4> all_positions(const Schedule& steps);
std::vector<std::pair<Vec4, Vec4>> all_pairs(const Schedule& steps);

// Checks that every pair of scheduled sources at separation
// stride*(+-1,+-1,+-1,+-1) appears exactly once, and no source twice.
// Throws std::runtime_error on failure.
void check_complete(const Schedule& steps, int stride);

// Largest n_steps for which no two sources land on the same site once the
// coordinates are wrapped onto `lattice` (about one lap around it).
int max_steps_without_wrap(const Vec4& h, int stride, const Vec4& lattice,
                            const Vec4& origin = {0, 0, 0, 0},
                            int cap = 10000);

// Order of solves and contractions. A propagator is freed as soon as its last
// pair is done in every mode; the modes differ in how many are alive at once.
//   kInterleaved  solve a source, contract its pairs at once. Peak: 16.
//                 This is what the driver does.
//   kSerial       solve a whole window, then contract it. Peak: 30.
//   kPipelined    solve window k+1 while window k is contracted. Peak: 45.
enum class Mode { kSerial, kPipelined, kInterleaved };

struct ScheduleStats {
  long long n_propagators = 0;
  long long n_solves = 0;       // n_propagators * 12 * n_flavors
  long long n_pairs = 0;
  long long n_kernel_sums = 0;  // n_pairs * n_flavors
  int peak_cache = 0;           // max source positions held at once
};

// Counts solves, pairs and the peak cache size for the given mode, without
// computing anything. Throws std::runtime_error if a propagator is still
// cached after its last pair.
ScheduleStats simulate(const Schedule& steps, int n_flavors, Mode mode);

}  // namespace hlbl
