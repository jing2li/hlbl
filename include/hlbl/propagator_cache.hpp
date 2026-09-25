#pragma once

// The propagators of the source positions the schedule currently needs.
//
// A position is inserted once (its solve), taken part in some number of
// pair contractions, and evicted as soon as the last of them is done. That
// number is known up front from the schedule, so the cache is told it at
// construction (`pending`) and needs no eviction policy of its own.
//
// Two residencies:
//   kDevice  every cached block lives on the GPU. Contractions read it in
//            place. Fastest, and the cache is limited by device memory.
//   kHost    blocks live in pinned host memory. A contraction stages the
//            two blocks it needs into device buffers first (one per
//            slot); the device only ever holds those two blocks, plus a
//            third used as the target of a solve.

#include <cstddef>
#include <map>
#include <optional>

#include "hlbl/mdarray.hpp"
#include "hlbl/vec4.hpp"

namespace hlbl {

class PropagatorCache {
 public:
  enum class Residency { kDevice, kHost };

  // block_len: doubles per position (hlbl::block_length()). pending: for
  // every position, the number of pair contractions it takes part in.
  PropagatorCache(Residency residency, std::size_t block_len,
                  std::map<Vec4, int> pending);

  // Inserting a position is two steps so that a solve can write straight
  // into the device block the cache will keep (kDevice) or stage through
  // (kHost): fill the returned buffer, then commit it.
  DeviceBuffer<double>& begin_insert(const Vec4& position);
  void end_insert(const Vec4& position);

  // The position's block on the device. kDevice: the cached block itself.
  // kHost: staged into slot 0 or 1, unless that slot already holds it. The
  // pointer stays valid until the next call with the same slot (or, for
  // kDevice, until the position is evicted).
  const double* device_ptr(const Vec4& position, int slot);

  // One contraction involving both positions is done; evicts either whose
  // last contraction this was.
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

  // kHost only, allocated on first use
  DeviceBuffer<double> scratch_;
  DeviceBuffer<double> staging_[2];
  std::optional<Vec4> staged_[2];
};

}  // namespace hlbl
