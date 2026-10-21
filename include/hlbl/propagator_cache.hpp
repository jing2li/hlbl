#pragma once

// The propagators the schedule still needs. A position is inserted once, after
// its solve, and freed after its last pair (`pending` counts them). One cache
// serves a list of chains: reset() starts the next one and keeps the buffers.
//
//   kDevice  blocks on the GPU, read in place.
//   kHost    blocks in pinned host memory. A contraction copies its two blocks
//            to two device staging slots; a third device block is the solve
//            target.

#include <cstddef>
#include <map>
#include <optional>
#include <vector>

#include "hlbl/mdarray.hpp"
#include "hlbl/vec4.hpp"

namespace hlbl {

class PropagatorCache {
 public:
  enum class Residency { kDevice, kHost };

  // block_len: doubles per position (see block_length()).
  // pending: for each position, the number of pairs it takes part in.
  PropagatorCache(Residency residency, std::size_t block_len,
                  std::map<Vec4, int> pending = {});

  // Starts a new run with a new `pending`; the cache must be empty. Resets
  // the peak.
  void reset(std::map<Vec4, int> pending);

  // Insert in two steps: begin_insert returns a device buffer, the solve
  // fills it, end_insert stores it.
  DeviceBuffer<double>& begin_insert(const Vec4& position);
  void end_insert(const Vec4& position);

  // The position's block on the device. kHost copies it into staging slot 0
  // or 1 unless already there; valid until the next call with that slot.
  const double* device_ptr(const Vec4& position, int slot);

  // A pair is done; frees either position if this was its last pair.
  void pair_done(const Vec4& older, const Vec4& newer);

  bool contains(const Vec4& position) const;
  int size() const { return static_cast<int>(entries_.size()); }
  int peak() const { return peak_; }
  Residency residency() const { return residency_; }

 private:
  struct Entry {
    DeviceBuffer<double> device;   // kDevice
    PinnedBuffer<double> host;     // kHost
  };

  void evict(const Vec4& position);

  Residency residency_;
  std::size_t block_len_;
  std::map<Vec4, int> pending_;
  std::map<Vec4, Entry> entries_;
  int peak_ = 0;

  // kHost only: freed pinned blocks, reused because cudaMallocHost is slow
  std::vector<PinnedBuffer<double>> pool_;

  // kHost only: allocated on first use
  DeviceBuffer<double> scratch_;
  DeviceBuffer<double> staging_[2];
  std::optional<Vec4> staged_[2];
};

}  // namespace hlbl
