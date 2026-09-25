#pragma once

// RAII wrappers around the three kinds of storage the driver needs for a
// propagator cache: plain host memory, pinned (page-locked) host memory,
// and device memory. Deliberately thin -- indexing into a block is the
// caller's responsibility, matching the flat-pointer layout the vendored
// GPU kernels expect.

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace hlbl {

inline void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                              cudaGetErrorString(err));
  }
}

// Plain host-side buffer.
template <typename T>
class HostArray {
 public:
  explicit HostArray(std::size_t n) : data_(n) {}
  T* data() { return data_.data(); }
  const T* data() const { return data_.data(); }
  std::size_t size() const { return data_.size(); }
  T& operator[](std::size_t i) { return data_[i]; }
  const T& operator[](std::size_t i) const { return data_[i]; }

 private:
  std::vector<T> data_;
};

// Owning device buffer.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t n) : size_(n) {
    check_cuda(cudaMalloc(&ptr_, n * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() { reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
    o.ptr_ = nullptr;
    o.size_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      reset();
      ptr_ = o.ptr_;
      size_ = o.size_;
      o.ptr_ = nullptr;
      o.size_ = 0;
    }
    return *this;
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  std::size_t size() const { return size_; }

  void copy_from_host(const T* src, std::size_t n) {
    copy_from_host(0, src, n);
  }
  void copy_from_host(std::size_t offset, const T* src, std::size_t n) {
    check_cuda(cudaMemcpy(ptr_ + offset, src, n * sizeof(T),
                           cudaMemcpyHostToDevice),
               "cudaMemcpy H2D");
  }
  void copy_to_host(T* dst, std::size_t n) const {
    copy_to_host(0, dst, n);
  }
  void copy_to_host(std::size_t offset, T* dst, std::size_t n) const {
    check_cuda(cudaMemcpy(dst, ptr_ + offset, n * sizeof(T),
                           cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H");
  }
  void zero() {
    check_cuda(cudaMemset(ptr_, 0, size_ * sizeof(T)), "cudaMemset");
  }

 private:
  void reset() {
    if (ptr_) cudaFree(ptr_);
    ptr_ = nullptr;
    size_ = 0;
  }
  T* ptr_ = nullptr;
  std::size_t size_ = 0;
};

// Owning pinned (page-locked) host buffer, for fast H2D/D2H staging of a
// host-resident propagator cache (see hlbl::PropagatorCache, phase 6).
template <typename T>
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  explicit PinnedBuffer(std::size_t n) : size_(n) {
    check_cuda(cudaMallocHost(&ptr_, n * sizeof(T)), "cudaMallocHost");
  }
  ~PinnedBuffer() { reset(); }

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  PinnedBuffer(PinnedBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
    o.ptr_ = nullptr;
    o.size_ = 0;
  }
  PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
    if (this != &o) {
      reset();
      ptr_ = o.ptr_;
      size_ = o.size_;
      o.ptr_ = nullptr;
      o.size_ = 0;
    }
    return *this;
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  std::size_t size() const { return size_; }

 private:
  void reset() {
    if (ptr_) cudaFreeHost(ptr_);
    ptr_ = nullptr;
    size_ = 0;
  }
  T* ptr_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace hlbl
