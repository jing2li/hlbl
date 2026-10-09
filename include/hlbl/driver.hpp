#pragma once

// The schedule loop. For each new source: solve it, contract it with its
// partners, and free any propagator whose last pair is done.

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
  // Run windows [window_begin, window_end); window_end < 0 means to the end.
  // If window_begin > 0, the earlier sources it needs are solved again first.
  int window_begin = 0;
  int window_end = -1;
  PropagatorCache::Residency residency = PropagatorCache::Residency::kDevice;
  double xunit = hlbl::xunit();
  bool verbose = false;  // print one line per window on rank 0
};

struct RunStats {
  int positions_solved = 0;    // includes the ones re-solved at a restart
  long long iterations = 0;
  int pairs = 0;
  int cache_peak = 0;          // most source positions held at once
  double seconds_solve = 0;
  double seconds_kernel_sum = 0;
};

// Runs `schedule` (from build_schedule) and returns every pair's result,
// the same on all ranks. `writer` and `timing` may be null; pass them on
// rank 0 only. The writer gets each window's rows when the window is done.
std::vector<PairResult> run_schedule(const Lattice& lattice,
                                      const Schedule& schedule,
                                      const RunOptions& options,
                                      PropagatorProvider& provider,
                                      Contractor& contractor,
                                      ResultWriter* writer, TimingLog* timing,
                                      RunStats* stats = nullptr);

// Memory the run needs on this rank, against what is available.
//   device cache: the peak number of positions on the GPU.
//   host cache:   the peak number in pinned host memory, plus 3 blocks on the
//                 GPU (one solve target, two staging slots).
struct MemoryEstimate {
  std::size_t bytes_per_position = 0;  // one position, on this rank
  int peak_positions = 0;
  std::size_t device_needed = 0, device_available = 0;  // this rank
  std::size_t host_needed = 0, host_available = 0;      // this rank's share
  bool fits() const {
    return device_needed <= device_available && host_needed <= host_available;
  }
};

// Collective over all ranks. Call after the solver is set up, so the free
// device memory is what is left after QUDA's multigrid.
MemoryEstimate estimate_memory(const Lattice& lattice, const Schedule& schedule,
                               PropagatorCache::Residency residency);

}  // namespace hlbl
