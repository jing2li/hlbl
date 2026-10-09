#pragma once

// The connected kernel sum for one pair of point sources. It calls the GPU
// kernels in kernels/cuda_lattice.h (taken from cvc's hlbl_mII_invert_contract).

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

// fwd_src and fwd_y are device propagator blocks (see block_length()) for the
// sources at x = gsx and y = gsx + yv. The Contractor keeps its work buffers
// between calls, so create one for the whole run.
class Contractor {
 public:
  Contractor(const Lattice& lattice, QED_kernel_temps kqed_t);

  // The kernel sum for both flavours. xunit = m_mu * a converts lattice units
  // to units of the muon mass. Every rank gets the full (MPI-summed) result.
  std::array<double, 2> kernel_sum(const double* d_fwd_src,
                                    const double* d_fwd_y, const Vec4& gsx,
                                    const Vec4& yv, double xunit);

 private:
  const Lattice& lattice_;
  QED_kernel_temps kqed_t_;

  DeviceBuffer<double> d_dzu_{kDzuLen};
  DeviceBuffer<double> d_dzsu_{kDzsuLen};
  DeviceBuffer<double> d_g_dzu_{kGDzuLen};
  DeviceBuffer<double> d_g_dzsu_{kGDzsuLen};
  DeviceBuffer<double> d_kernel_sum_{kKernelN};

  // The z-sum of the two propagators, summed over all ranks, with gamma
  // matrices applied. Fills d_g_dzu_ and d_g_dzsu_.
  void compute_dzu_dzsu(const double* d_fwd_src, const double* d_fwd_y,
                         const Vec4& gsx, int iflavor);

  // The sum over x of the 4-point contraction times the QED kernel. Needs
  // compute_dzu_dzsu for the same flavour to have run first.
  double compute_4pt(const double* d_fwd_src, const double* d_fwd_y,
                      const Vec4& gsx, int iflavor, double xunit,
                      const Vec4& yv);
};

}  // namespace hlbl
