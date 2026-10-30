#pragma once

// The schedule loop: solve each new source, compute its 2+2 pieces if it is
// a 2+2 source solved for the first time, contract it with its partners, free
// propagators whose last pair is done.

#include <cstddef>
#include <vector>

#include "hlbl/contract.hpp"
#include "hlbl/lattice.hpp"
#include "hlbl/output.hpp"
#include "hlbl/propagator_cache.hpp"
#include "hlbl/provider.hpp"
#include "hlbl/schedule.hpp"

namespace hlbl {

struct RunOptions {
  PropagatorCache::Residency residency = PropagatorCache::Residency::kDevice;
  double xunit = 0;      // m_mu a (hlbl::xunit()); required
  bool verbose = false;  // print one line per chain and window on rank 0
  // the 2p2 lines of the input file; empty: no 2+2
  std::vector<Source2p2> sources_2p2;
};

struct RunStats {
  int positions_solved = 0;
  long long iterations = 0;
  int pairs = 0;
  int cache_peak = 0;          // most source positions held at once
  double seconds_solve = 0;
  double seconds_kernel_sum = 0;
  int pieces_2p2 = 0;          // 2+2 sources computed
  long long targets_2p2 = 0;   // their targets
  double seconds_2p2 = 0;
};

// Runs the chains (schedules from build_schedule) in order with one shared
// cache (memory of the largest chain) and returns every pair's result on all
// ranks. `writer` and `timing` may be null and are for rank 0 only; the writer
// gets each window's rows when it is done. `stats` are totals, with cache_peak
// the largest. With options.sources_2p2, the first solve of each 2+2 source
// (plan_2p2) is followed by its 2+2 pieces, before the pairs; they go to the
// writer with the window's pairs and, if not null, to `results_2p2`.
std::vector<PairResult> run_chains(const Lattice& lattice,
                                    const std::vector<Schedule>& chains,
                                    const RunOptions& options,
                                    PropagatorProvider& provider,
                                    Contractor& contractor,
                                    ResultWriter* writer, TimingLog* timing,
                                    RunStats* stats = nullptr,
                                    std::vector<Result2p2>* results_2p2 = nullptr);

// Memory a run needs on this rank, and what is available.
//   device cache: peak positions on the GPU.
//   host cache:   peak positions in pinned host memory, plus 3 blocks on the
//                 GPU (solve target, two staging slots).
// With 2+2, the device also holds the work buffers of compute_2p2_gpu.
struct MemoryEstimate {
  std::size_t bytes_per_position = 0;  // one position, on this rank
  int peak_positions = 0;
  std::size_t bytes_2p2 = 0;           // 2+2 work buffers, on the device
  std::size_t device_needed = 0, device_available = 0;  // this rank
  std::size_t host_needed = 0, host_available = 0;      // this rank's share
  bool fits() const {
    return device_needed <= device_available && host_needed <= host_available;
  }
};

// `peak_positions`: the largest cache peak of the chains (simulate(),
// kInterleaved). `max_targets_2p2`: the most targets of a 2+2 source, 0
// without 2+2. Collective. Call after the solver setup and
// Contractor::reserve_kernel_memory(), so free device memory excludes QUDA's
// multigrid and the kernels' stack. Used by --dummy-run only.
MemoryEstimate estimate_memory(const Lattice& lattice, int peak_positions,
                               PropagatorCache::Residency residency,
                               int max_targets_2p2 = 0);

}  // namespace hlbl
