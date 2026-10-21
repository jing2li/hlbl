#pragma once

// The schedule loop: solve each new source, contract it with its partners,
// free propagators whose last pair is done.

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
  // Window range as in ChainTask; run_schedule only (run_chains uses each
  // task's own).
  int window_begin = 0;
  int window_end = -1;
  PropagatorCache::Residency residency = PropagatorCache::Residency::kDevice;
  double xunit = 0;      // m_mu a (hlbl::xunit()); required
  bool verbose = false;  // print one line per chain and window on rank 0
};

struct RunStats {
  int positions_solved = 0;    // includes the ones re-solved at a restart
  long long iterations = 0;
  int pairs = 0;
  int cache_peak = 0;          // most source positions held at once
  double seconds_solve = 0;
  double seconds_kernel_sum = 0;
};

// One chain of a run_chains call.
struct ChainTask {
  Schedule schedule;        // from build_schedule
  // Windows [window_begin, window_end); window_end < 0: to the end. The
  // earlier sources window_begin needs are solved again first.
  int window_begin = 0;
  int window_end = -1;
};

// Runs the chains in order with one shared cache (memory of the largest chain)
// and returns every pair's result on all ranks. `writer` and `timing` may be
// null and are for rank 0 only; the writer gets each window's rows when it is
// done. `stats` are totals, with cache_peak the largest.
std::vector<PairResult> run_chains(const Lattice& lattice,
                                    const std::vector<ChainTask>& chains,
                                    const RunOptions& options,
                                    PropagatorProvider& provider,
                                    Contractor& contractor,
                                    ResultWriter* writer, TimingLog* timing,
                                    RunStats* stats = nullptr);

// One chain (chain 0) with the window range of `options`.
std::vector<PairResult> run_schedule(const Lattice& lattice,
                                      const Schedule& schedule,
                                      const RunOptions& options,
                                      PropagatorProvider& provider,
                                      Contractor& contractor,
                                      ResultWriter* writer, TimingLog* timing,
                                      RunStats* stats = nullptr);

// Memory a run needs on this rank, and what is available.
//   device cache: peak positions on the GPU.
//   host cache:   peak positions in pinned host memory, plus 3 blocks on the
//                 GPU (solve target, two staging slots).
struct MemoryEstimate {
  std::size_t bytes_per_position = 0;  // one position, on this rank
  int peak_positions = 0;
  std::size_t device_needed = 0, device_available = 0;  // this rank
  std::size_t host_needed = 0, host_available = 0;      // this rank's share
  bool fits() const {
    return device_needed <= device_available && host_needed <= host_available;
  }
};

// `peak_positions`: the largest cache peak of the chains (simulate(),
// kInterleaved). Collective. Call after the solver setup and
// Contractor::reserve_kernel_memory(), so free device memory excludes QUDA's
// multigrid and the kernels' stack. Used by --dummy-run only.
MemoryEstimate estimate_memory(const Lattice& lattice, int peak_positions,
                               PropagatorCache::Residency residency);

}  // namespace hlbl
