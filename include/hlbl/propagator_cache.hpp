#pragma once

// The propagators of the sources the schedule still needs.
//
// A position is inserted once (after its solve) and freed when the last pair
// that uses it is done. The number of pairs per position is known from the
// schedule and given to the constructor (`pending`).
//
// Where the blocks live:
//   kDevice  on the GPU; contractions read them in place.
//   kHost    in pinned host memory. A contraction first copies the two blocks
//            it needs into two device staging slots. A third device block is
//            the target of the solve.

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
                  std::map<Vec4, int> pending);

  // Insert in two steps: begin_insert returns a device buffer, the solve
  // fills it, end_insert stores it.
  DeviceBuffer<double>& begin_insert(const Vec4& position);
  void end_insert(const Vec4& position);

  // The position's block on the device. For kHost it is copied into staging
  // slot 0 or 1 (skipped if the slot already holds it); the pointer is valid
  // until the next call with the same slot.
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

  // kHost only: freed pinned blocks are kept and reused, because
  // cudaMallocHost is slow.
  std::vector<PinnedBuffer<double>> pool_;

  // kHost only: allocated on first use
  DeviceBuffer<double> scratch_;
  DeviceBuffer<double> staging_[2];
  std::optional<Vec4> staged_[2];
};

}  // namespace hlbl
