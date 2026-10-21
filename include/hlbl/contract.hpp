#pragma once

// The connected kernel sum of one pair of point sources, using the GPU kernels
// of kernels/cuda_lattice.h (from cvc's hlbl_mII_invert_contract).

#include <array>

#include "cuda_lattice.h"  // vendored: Coord, Geom, Pair, cu_dzu_dzsu, cu_4pt_contraction
#include "hlbl/lattice.hpp"
#include "hlbl/mdarray.hpp"
#include "hlbl/vec4.hpp"

namespace hlbl {

// Buffer sizes the kernels expect: dzu[6][12][24], dzsu[4][12][24],
// g_dzu[6][4][12][24], g_dzsu[4][4][12][24].
inline constexpr int kDzuLen = 6 * 12 * 24;
inline constexpr int kDzsuLen = 4 * 12 * 24;
inline constexpr int kGDzuLen = 6 * 4 * 12 * 24;
inline constexpr int kGDzsuLen = 4 * 4 * 12 * 24;
inline constexpr int kKernelN = CUDA_N_QED_KERNEL;  // 1: LLambda0.4 only

// fwd_src and fwd_y are device propagator blocks (block_length()) of the
// sources x = gsx and y = gsx + yv. Create one Contractor per run: it keeps
// its work buffers.
class Contractor {
 public:
  Contractor(const Lattice& lattice, QED_kernel_temps kqed_t);

  // The kernel sum of both flavours, MPI-summed on every rank. xunit = m_mu a.
  std::array<double, 2> kernel_sum(const double* d_fwd_src,
                                    const double* d_fwd_y, const Vec4& gsx,
                                    const Vec4& yv, double xunit);

  // The 4-point kernel needs ~40 kB of stack per thread (the QED kernel), and
  // the CUDA driver reserves it for every thread slot at the first launch:
  // ~10.4 GiB on a GH200. This launches both kernels once on a 2^4 lattice so
  // the reservation happens before free memory is measured. Returns the
  // device memory taken on this rank.
  std::size_t reserve_kernel_memory();

 private:
  const Lattice& lattice_;
  QED_kernel_temps kqed_t_;

  DeviceBuffer<double> d_dzu_{kDzuLen};
  DeviceBuffer<double> d_dzsu_{kDzsuLen};
  DeviceBuffer<double> d_g_dzu_{kGDzuLen};
  DeviceBuffer<double> d_g_dzsu_{kGDzsuLen};
  DeviceBuffer<double> d_kernel_sum_{kKernelN};

  // The z-sum of the two propagators over all ranks, with gammas applied.
  // Fills d_g_dzu_ and d_g_dzsu_.
  void compute_dzu_dzsu(const double* d_fwd_src, const double* d_fwd_y,
                         const Vec4& gsx, int iflavor);

  // Sum over x of the 4-point contraction times the QED kernel. Run
  // compute_dzu_dzsu for the same flavour first.
  double compute_4pt(const double* d_fwd_src, const double* d_fwd_y,
                      const Vec4& gsx, int iflavor, double xunit,
                      const Vec4& yv);
};

}  // namespace hlbl
