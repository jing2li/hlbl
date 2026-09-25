#pragma once

// The sliding-hypercube source schedule. A direct C++ port of
// scripts/hlbl_mII_hypercube_schedule.py; see the README "The schedule" for
// the scheme, and that script for the reference this must stay in step with.

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
  Vec4 position;                // absolute, unwrapped, lattice units
  std::vector<Vec4> partners;   // already-available positions it pairs with
};

struct Step {
  int index = 0;
  std::vector<Source> sources;  // new in this window, in inversion order

  int n_new() const { return static_cast<int>(sources.size()); }
  int n_pairs() const {
    int n = 0;
    for (const auto& s : sources) n += static_cast<int>(s.partners.size());
    return n;
  }
};

using Schedule = std::vector<Step>;

// Windows 0..n_steps. Each new source is paired with every source already
// available (the previous window plus the earlier sources of this window)
// at a separation of stride*(+-1,+-1,+-1,+-1).
Schedule build_schedule(const Vec4& h, int stride, int n_steps,
                         const Vec4& origin = {0, 0, 0, 0});

std::vector<Vec4> all_positions(const Schedule& steps);
std::vector<std::pair<Vec4, Vec4>> all_pairs(const Schedule& steps);

// Brute-force check that every pair of scheduled sources at separation
// stride*(+-1,+-1,+-1,+-1) is scheduled exactly once, and that no position
// or pair is scheduled twice. Throws std::runtime_error on failure.
void check_complete(const Schedule& steps, int stride);

// Largest n_steps for which no two sources land on the same lattice site
// after periodic wrapping onto `lattice`, i.e. roughly one lap around it.
int max_steps_without_wrap(const Vec4& h, int stride, const Vec4& lattice,
                            const Vec4& origin = {0, 0, 0, 0},
                            int cap = 10000);

// Execution modes: how many windows' propagators are resident at once.
//   serial     invert all of window k, then run its kernel sums. Two
//              windows live at once, 30 source positions at the peak.
//   pipelined  window k+1 is inverted while window k's kernel sums run.
//              Three windows live at once, 45 at the peak.
// In every mode a propagator is evicted as soon as its last pair is done.
enum class Mode { kSerial, kPipelined };

struct ScheduleStats {
  long long n_propagators = 0;
  long long n_solves = 0;       // n_propagators * 12 * n_flavors
  long long n_pairs = 0;
  long long n_kernel_sums = 0;  // n_pairs * n_flavors
  int peak_cache = 0;           // max source positions held at once
};

// Simulates the propagator cache under the given execution mode. Throws
// std::runtime_error if a propagator is still cached after its last pair
// (a bug in the schedule, not something that should happen here).
ScheduleStats simulate(const Schedule& steps, int n_flavors, Mode mode);

}  // namespace hlbl
