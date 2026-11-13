#pragma once

// The connected kernel sum of one pair of point sources, using the GPU kernels
// of kernels/cuda_lattice.h (from cvc's hlbl_mII_invert_contract), and the
// 2+2 pieces of one source (kernels/hlbl_kernels, compute_2p2_gpu).

#include <array>
#include <cstddef>
#include <vector>

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
inline constexpr int kGeom2p2 = CUDA_N_QED_GEOM;    // 3: P2_0, P2_1, P3

// The 2+2 pieces of one source w, MPI-summed on every rank:
//   p1   [4][4][4][Lmax]        P1[rho][sigma][nu][zeta]: Pi_{sigma nu}(z)
//                               summed over the sites z with
//                               (z - w)_rho mod L_rho = zeta (0 for zeta >= L_rho)
//   p23  [targets][6][4][4][4]  P2_0, P2_1, P3 minimum image, then P2_0, P2_1,
//                               P3 zerohalf; each [rho][sigma][nu], rho < sigma
// with Lmax the largest global extent.
struct Pieces2p2 {
  std::vector<double> p1;
  std::vector<double> p23;
};

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

  // The 2+2 pieces of the source w = gsw for `targets` (global sites in
  // [0, L)), from the device block of w. Flavour 0 only, as in cvc: the
  // two-point function is the same for both flavours. Collective.
  Pieces2p2 pieces_2p2(const double* d_fwd, const Vec4& gsw,
                       const std::vector<Vec4>& targets, double xunit);

  // The 4-point kernel needs ~27 kB of stack per thread (the QED kernel), and
  // the CUDA driver reserves it for every thread slot at the first launch:
  // ~6.8 GiB on a GH200. This launches the connected kernels (and with
  // with_2p2 the 2+2 kernels, whose stack is smaller now) once on a 2^4
  // lattice, so the reservation happens before free memory is measured.
  // Returns the device memory taken on this rank.
  std::size_t reserve_kernel_memory(bool with_2p2 = false);

 private:
  const Lattice& lattice_;
  QED_kernel_temps kqed_t_;

  DeviceBuffer<double> d_dzu_{kDzuLen};
  DeviceBuffer<double> d_dzsu_{kDzsuLen};
  DeviceBuffer<double> d_g_dzu_{kGDzuLen};
  DeviceBuffer<double> d_g_dzsu_{kGDzsuLen};
  DeviceBuffer<double> d_kernel_sum_{kKernelN};

  // compute_2p2_gpu reads gsw, xunit and the process coordinates in device
  // code, so they live on the device
  DeviceBuffer<int> d_gsw_{4};
  DeviceBuffer<int> d_proc_coords_{4};
  DeviceBuffer<double> d_xunit_{2};
  // it copies P1 and P23 to the host asynchronously and sums them there in
  // place, so they are pinned; grown on first use
  PinnedBuffer<double> h_p1_, h_p23_;

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
