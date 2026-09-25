#pragma once

// The connected kernel sum for one point-source pair. Ports
// hlbl_mII_invert_contract.cpp's compute_dzu_dzsu + compute_4pt_contraction
// (CUDA path) onto this project's Lattice and gamma tables, calling the
// vendored device kernels (kernels/cuda_lattice.h) directly.

#include <array>

#include "cuda_lattice.h"  // vendored: Coord, Geom, Pair, cu_dzu_dzsu, cu_4pt_contraction
#include "hlbl/lattice.hpp"
#include "hlbl/mdarray.hpp"
#include "hlbl/vec4.hpp"

namespace hlbl {

// Buffer sizes cu_dzu_dzsu/cu_4pt_contraction assume (production's
// dzu[6][12][24], dzsu[4][12][24], g_dzu[6][4][12][24], g_dzsu[4][4][12][24]);
// the vendored header does not expose them as constants.
inline constexpr int kDzuLen = 6 * 12 * 24;
inline constexpr int kDzsuLen = 4 * 12 * 24;
inline constexpr int kGDzuLen = 6 * 4 * 12 * 24;
inline constexpr int kGDzsuLen = 4 * 4 * 12 * 24;
inline constexpr int kKernelN = CUDA_N_QED_KERNEL;  // 1: LLambda0.4 only

// fwd_src/fwd_y are device propagator blocks, [2 flavours][12 source
// spin-colour][24 * local volume] doubles each (README "Propagators"), for
// the source at x = gsx and at y = gsx + yv respectively.
//
// Persistent device buffers are reused across calls, so one Contractor is
// meant to live for the whole run, not be rebuilt per pair.
class Contractor {
 public:
  Contractor(const Lattice& lattice, QED_kernel_temps kqed_t);

  // The kernel sum for both flavours, xunit = m_mu * a in the same lattice
  // units as gsx and yv. Every rank gets the full (MPI-reduced) result.
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

  // D_y^+ z g5 gsigma U_src, volume- and rank-summed, then g5/gamma_mu
  // inserted -- fills d_g_dzu_/d_g_dzsu_ for the given flavour.
  void compute_dzu_dzsu(const double* d_fwd_src, const double* d_fwd_y,
                         const Vec4& gsx, int iflavor);

  // The 4-point contraction summed against the QED kernel, for the given
  // flavour. Requires compute_dzu_dzsu for the same flavour to have run
  // first.
  double compute_4pt(const double* d_fwd_src, const double* d_fwd_y,
                      const Vec4& gsx, int iflavor, double xunit,
                      const Vec4& yv);
};

}  // namespace hlbl
