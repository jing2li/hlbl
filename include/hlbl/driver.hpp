#pragma once

// The schedule loop: for each window, obtain the propagators of its new
// sources (a solve), then contract every pair the schedule lists for it,
// evicting a source from the cache as soon as its last pair is done.

#include <cstddef>
#include <vector>

#include "hlbl/contract.hpp"
#include "hlbl/lattice.hpp"
#include "hlbl/output.hpp"
#include "hlbl/physics.hpp"
#include "hlbl/propagator_cache.hpp"
#include "hlbl/provider.hpp"
#include "hlbl/schedule.hpp"

namespace hlbl {

struct RunOptions {
  // Windows [window_begin, window_end) of the schedule; window_end < 0 means
  // through the last one. Starting after window 0 first re-solves the
  // sources of earlier windows that the range's first pairs need.
  int window_begin = 0;
  int window_end = -1;
  PropagatorCache::Residency residency = PropagatorCache::Residency::kDevice;
  double xunit = hlbl::xunit();
  bool verbose = false;  // one line per window on rank 0
};

struct RunStats {
  int positions_solved = 0;    // includes the re-solved ones of a restart
  long long iterations = 0;
  int pairs = 0;
  int cache_peak = 0;          // source positions held at once
  double seconds_solve = 0;
  double seconds_kernel_sum = 0;
};

// Runs `schedule` (from build_schedule) and returns every pair's result,
// identical on all ranks. `writer` and `timing` may be null (pass them on
// rank 0 only); the writer gets each window's rows as soon as it is done.
std::vector<PairResult> run_schedule(const Lattice& lattice,
                                      const Schedule& schedule,
                                      const RunOptions& options,
                                      PropagatorProvider& provider,
                                      Contractor& contractor,
                                      ResultWriter* writer, TimingLog* timing,
                                      RunStats* stats = nullptr);

// What the run needs to hold in memory, against what this rank has. Device
// residency keeps the schedule's peak number of source positions on the
// GPU; host residency keeps them in pinned host memory, and puts three
// blocks (a solve target and two staging slots) on the device.
struct MemoryEstimate {
  std::size_t bytes_per_position = 0;  // one position's block on this rank
  int peak_positions = 0;
  std::size_t device_needed = 0, device_available = 0;  // this rank
  std::size_t host_needed = 0, host_available = 0;      // this rank's share
  bool fits() const {
    return device_needed <= device_available && host_needed <= host_available;
  }
};

// Collective (it counts the ranks sharing this rank's node). Call after the
// solver is set up: QUDA's multigrid takes most of the device memory, and
// the free memory reported here is what it left.
MemoryEstimate estimate_memory(const Lattice& lattice, const Schedule& schedule,
                               PropagatorCache::Residency residency);

}  // namespace hlbl
