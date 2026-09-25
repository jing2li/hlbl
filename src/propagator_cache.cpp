#include "hlbl/propagator_cache.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace hlbl {

PropagatorCache::PropagatorCache(Residency residency, std::size_t block_len,
                                 std::map<Vec4, int> pending)
    : residency_(residency),
      block_len_(block_len),
      pending_(std::move(pending)) {}

DeviceBuffer<double>& PropagatorCache::begin_insert(const Vec4& position) {
  if (entries_.count(position)) {
    throw std::logic_error("PropagatorCache: position inserted twice");
  }
  if (residency_ == Residency::kDevice) {
    Entry& e = entries_[position];
    e.device = DeviceBuffer<double>(block_len_);
    return e.device;
  }
  if (scratch_.size() == 0) scratch_ = DeviceBuffer<double>(block_len_);
  return scratch_;
}

void PropagatorCache::end_insert(const Vec4& position) {
  if (residency_ == Residency::kHost) {
    Entry& e = entries_[position];
    e.host = PinnedBuffer<double>(block_len_);
    scratch_.copy_to_host(e.host.data(), block_len_);
  }
  peak_ = std::max(peak_, size());
  // a position that takes part in no contraction is never needed
  if (pending_[position] == 0) evict(position);
}

const double* PropagatorCache::device_ptr(const Vec4& position, int slot) {
  auto it = entries_.find(position);
  if (it == entries_.end()) {
    throw std::logic_error("PropagatorCache: position is not cached");
  }
  if (residency_ == Residency::kDevice) return it->second.device.get();

  if (slot < 0 || slot > 1) throw std::logic_error("PropagatorCache: bad slot");
  if (staging_[slot].size() == 0) {
    staging_[slot] = DeviceBuffer<double>(block_len_);
  }
  if (!staged_[slot] || *staged_[slot] != position) {
    staging_[slot].copy_from_host(it->second.host.data(), block_len_);
    staged_[slot] = position;
  }
  return staging_[slot].get();
}

void PropagatorCache::pair_done(const Vec4& older, const Vec4& newer) {
  for (const Vec4& p : {older, newer}) {
    if (--pending_[p] == 0) evict(p);
  }
}

bool PropagatorCache::contains(const Vec4& position) const {
  return entries_.count(position) != 0;
}

void PropagatorCache::evict(const Vec4& position) {
  entries_.erase(position);
  for (auto& s : staged_) {
    if (s && *s == position) s.reset();
  }
}

}  // namespace hlbl
