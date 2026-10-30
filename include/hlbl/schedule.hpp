#pragma once

// The sliding-hypercube schedule (see the README): which sources to solve and
// which pairs to contract. Must match scripts/hlbl_mII_hypercube_schedule.py
// (scripts/compare_schedule.sh checks).

#include <array>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "hlbl/vec4.hpp"

namespace hlbl {

// The 16 sign vectors (+-1,+-1,+-1,+-1).
std::array<Vec4, 16> sign_vectors();

// A0 = {0} u (G \ {h}) in units of the stride, in solve order: that of
// h = (1,1,1,1), reflected by h, which gives a cache peak of 16 for every h.
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

// Order of solves and contractions; a propagator is freed after its last pair.
//   kInterleaved  contract each source right after its solve. Peak 16. The
//                 driver's order.
//   kSerial       solve a whole window, then contract it. Peak 30.
enum class Mode { kSerial, kInterleaved };

struct ScheduleStats {
  long long n_propagators = 0;
  long long n_solves = 0;       // n_propagators * 12 spin-colours * 2 flavours
  long long n_pairs = 0;
  long long n_kernel_sums = 0;  // n_pairs * 2 flavours
  int peak_cache = 0;           // max source positions held at once
};

// Counts solves, pairs and the cache peak of `mode` without computing
// anything. Throws std::runtime_error if a propagator outlives its last pair.
ScheduleStats simulate(const Schedule& steps, Mode mode);

// The warning about sources and pairs in more than one chain (positions
// wrapped onto `lattice`): a summary line, then up to 20 pairs of chains, each
// with its first 10 shared sources and the window of each chain that solves
// them. Empty if no chains overlap. Same lines as overlap_report() in
// scripts/hlbl_mII_hypercube_schedule.py.
std::vector<std::string> overlap_report(const std::vector<Schedule>& chains,
                                        const Vec4& lattice);

// Which solves are followed by the 2+2 pieces: the first solve, in run order
// (chains in input order, windows, solve order), of each 2+2 source. Sites
// are compared wrapped onto `lattice`, so a site solved again (by another
// chain, or by a chain that wraps) gets no second 2+2.
struct Plan2p2 {
  // per chain: solve position (unwrapped, as in the schedule) -> index into
  // the 2+2 sources
  std::vector<std::map<Vec4, int>> at;
  // first solves of sites that are no 2+2 source: wrapped site, chain, window
  std::vector<std::tuple<Vec4, int, int>> not_sources;
  std::vector<int> unsolved;  // indices of 2+2 sources that no chain solves
};

Plan2p2 plan_2p2(const std::vector<Schedule>& chains,
                 const std::vector<Vec4>& sources, const Vec4& lattice);

// The warnings about a Plan2p2 of the 2+2 sources `sources`: solved sites
// that are no 2+2 source, and 2+2 sources that no chain solves, each with its
// first 10 sites. Empty if `sources` is empty or there is nothing to warn
// about. Same lines as report_2p2() in scripts/hlbl_mII_hypercube_schedule.py.
std::vector<std::string> report_2p2(const Plan2p2& plan,
                                    const std::vector<Vec4>& sources);

}  // namespace hlbl
