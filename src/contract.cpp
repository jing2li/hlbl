#include "hlbl/contract.hpp"

#include <algorithm>
#include <array>

#include "hlbl/gamma.hpp"
// after contract.hpp: cuda_runtime.h defines __global__ for the declarations
// in kernels.cuh, and KQED.h is already included
#include "hlbl_kernels/kernels.cuh"

namespace hlbl {

namespace {

Coord to_coord(const Vec4& v) { return Coord{v[0], v[1], v[2], v[3]}; }
Geom to_geom(const Vec4& v) { return Geom{v[0], v[1], v[2], v[3]}; }

// out = gamma_mu * gamma_5 * in
void insert_g5_gamma(double* out, int mu, const double* in) {
  Spinor24 t;
  std::copy(in, in + 24, t.begin());
  Spinor24 s = apply_gamma(mu, apply_gamma(5, t));
  std::copy(s.begin(), s.end(), out);
}

}  // namespace

Contractor::Contractor(const Lattice& lattice, QED_kernel_temps kqed_t)
    : lattice_(lattice), kqed_t_(kqed_t) {
  d_proc_coords_.copy_from_host(lattice_.proc_coords().data(), 4);
}

void Contractor::compute_dzu_dzsu(const double* d_fwd_src,
                                   const double* d_fwd_y, const Vec4& gsx,
                                   int iflavor) {
  d_dzu_.zero();
  d_dzsu_.zero();
  cu_dzu_dzsu(d_dzu_.get(), d_dzsu_.get(), d_fwd_src, d_fwd_y, iflavor,
              to_coord(lattice_.proc_coords()), to_coord(gsx),
              to_geom(lattice_.global()), to_geom(lattice_.local()));
  // a failed launch is only reported here; errors while the kernel runs are
  // reported by the copy below
  check_cuda(cudaGetLastError(), "cu_dzu_dzsu launch");

  std::array<double, kDzuLen> local_dzu;
  std::array<double, kDzsuLen> local_dzsu;
  d_dzu_.copy_to_host(local_dzu.data(), kDzuLen);
  d_dzsu_.copy_to_host(local_dzsu.data(), kDzsuLen);

  // every rank needs the full sum over all ranks
  std::array<double, kDzuLen> dzu;
  std::array<double, kDzsuLen> dzsu;
  MPI_Allreduce(local_dzu.data(), dzu.data(), kDzuLen, MPI_DOUBLE, MPI_SUM,
                lattice_.cart_comm());
  MPI_Allreduce(local_dzsu.data(), dzsu.data(), kDzsuLen, MPI_DOUBLE, MPI_SUM,
                lattice_.cart_comm());

  // g_dzu[k][mu][ia] = gamma_mu * gamma_5 * dzu[k][ia]
  std::array<double, kGDzuLen> g_dzu;
  for (int k = 0; k < 6; ++k) {
    for (int ia = 0; ia < 12; ++ia) {
      const double* src = &dzu[(k * 12 + ia) * 24];
      for (int mu = 0; mu < 4; ++mu) {
        insert_g5_gamma(&g_dzu[((k * 4 + mu) * 12 + ia) * 24], mu, src);
      }
    }
  }
  // g_dzsu[s][mu][ia], the same for dzsu
  std::array<double, kGDzsuLen> g_dzsu;
  for (int s = 0; s < 4; ++s) {
    for (int ia = 0; ia < 12; ++ia) {
      const double* src = &dzsu[(s * 12 + ia) * 24];
      for (int mu = 0; mu < 4; ++mu) {
        insert_g5_gamma(&g_dzsu[((s * 4 + mu) * 12 + ia) * 24], mu, src);
      }
    }
  }

  d_g_dzu_.copy_from_host(g_dzu.data(), kGDzuLen);
  d_g_dzsu_.copy_from_host(g_dzsu.data(), kGDzsuLen);
}

double Contractor::compute_4pt(const double* d_fwd_src, const double* d_fwd_y,
                                const Vec4& gsx, int iflavor, double xunit,
                                const Vec4& yv) {
  // the kernels take {value, uncertainty}; only the value is used
  Pair d_xunit{xunit, 0.0};

  d_kernel_sum_.zero();
  cu_4pt_contraction(d_kernel_sum_.get(), d_g_dzu_.get(), d_g_dzsu_.get(),
                      d_fwd_src, d_fwd_y, iflavor,
                      to_coord(lattice_.proc_coords()), to_coord(gsx),
                      d_xunit, to_coord(yv), kqed_t_,
                      to_geom(lattice_.global()), to_geom(lattice_.local()));
  check_cuda(cudaGetLastError(), "cu_4pt_contraction launch");

  double local = 0.0;
  d_kernel_sum_.copy_to_host(&local, 1);
  double total = 0.0;
  MPI_Allreduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, lattice_.cart_comm());
  return total;
}

std::array<double, 2> Contractor::kernel_sum(const double* d_fwd_src,
                                              const double* d_fwd_y,
                                              const Vec4& gsx, const Vec4& yv,
                                              double xunit) {
  std::array<double, 2> result;
  for (int iflavor = 0; iflavor < 2; ++iflavor) {
    compute_dzu_dzsu(d_fwd_src, d_fwd_y, gsx, iflavor);
    result[iflavor] = compute_4pt(d_fwd_src, d_fwd_y, gsx, iflavor, xunit, yv);
  }
  return result;
}

Pieces2p2 Contractor::pieces_2p2(const double* d_fwd, const Vec4& gsw,
                                 const std::vector<Vec4>& targets,
                                 double xunit) {
  static_assert(kKernelN == 1, "the P23 reordering below assumes one QED kernel");
  const Vec4& g = lattice_.global();
  const Vec4& l = lattice_.local();
  int const lmax = *std::max_element(g.begin(), g.end());
  int const n_y = static_cast<int>(targets.size());
  std::size_t const n_p1 = 4 * 4 * 4 * static_cast<std::size_t>(lmax);
  std::size_t const n_p23 = static_cast<std::size_t>(n_y) * kGeom2p2 * 64;
  if (h_p1_.size() < n_p1) h_p1_ = PinnedBuffer<double>(n_p1);
  if (h_p23_.size() < n_p23) h_p23_ = PinnedBuffer<double>(n_p23);

  std::vector<int> gycoords;
  for (const Vec4& t : targets) gycoords.insert(gycoords.end(), t.begin(), t.end());
  // the kernels take {value, uncertainty}; only the value is used
  double const xunit_pair[2] = {xunit, 0.0};
  d_gsw_.copy_from_host(gsw.data(), 4);
  d_xunit_.copy_from_host(xunit_pair, 2);

  // compute_2p2_gpu only reads fwd_y
  compute_2p2_gpu(const_cast<double*>(d_fwd), h_p1_.data(), h_p23_.data(),
                  /*iflavor=*/0, d_gsw_.get(), gycoords.data(), n_y,
                  d_xunit_.get(), kqed_t_,
                  static_cast<unsigned>(lattice_.volume()), d_proc_coords_.get(),
                  lattice_.cart_comm(), l[0], l[1], l[2], l[3], g[0], g[1], g[2],
                  g[3]);
  check_cuda(cudaGetLastError(), "compute_2p2_gpu");

  Pieces2p2 out;
  out.p1.assign(h_p1_.data(), h_p1_.data() + n_p1);
  // compute_2p2_gpu gives [geometry][target][64]; one row per target here
  out.p23.resize(n_p23);
  for (int y = 0; y < n_y; ++y) {
    for (int geom = 0; geom < kGeom2p2; ++geom) {
      std::copy_n(h_p23_.data() + (static_cast<std::size_t>(geom) * n_y + y) * 64, 64,
                  out.p23.data() + (static_cast<std::size_t>(y) * kGeom2p2 + geom) * 64);
    }
  }
  return out;
}

std::size_t Contractor::reserve_kernel_memory(bool with_2p2) {
  std::size_t free_before = 0, free_after = 0, total = 0;
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  check_cuda(cudaMemGetInfo(&free_before, &total), "cudaMemGetInfo");
  {
    // the values do not matter; one block serves as both propagators
    Geom const tiny{2, 2, 2, 2};
    Coord const zero{0, 0, 0, 0}, yv{1, 1, 1, 1};
    DeviceBuffer<double> block(2 * 12 * 24 * 16);
    block.zero();
    d_dzu_.zero();
    d_dzsu_.zero();
    d_g_dzu_.zero();
    d_g_dzsu_.zero();
    d_kernel_sum_.zero();
    cu_dzu_dzsu(d_dzu_.get(), d_dzsu_.get(), block.get(), block.get(), 0, zero,
                zero, tiny, tiny);
    check_cuda(cudaGetLastError(), "cu_dzu_dzsu launch");
    cu_4pt_contraction(d_kernel_sum_.get(), d_g_dzu_.get(), d_g_dzsu_.get(),
                        block.get(), block.get(), 0, zero, zero, Pair{1.0, 0.0},
                        yv, kqed_t_, tiny, tiny);
    check_cuda(cudaGetLastError(), "cu_4pt_contraction launch");
    if (with_2p2) {
      // one target, this rank alone (MPI_COMM_SELF): not collective
      int const zeros[4] = {0, 0, 0, 0}, target[4] = {1, 1, 1, 1};
      double const xunit_pair[2] = {1.0, 0.0};
      DeviceBuffer<int> d_zeros(4);
      d_zeros.copy_from_host(zeros, 4);
      d_xunit_.copy_from_host(xunit_pair, 2);
      PinnedBuffer<double> p1(4 * 4 * 4 * 2), p23(kGeom2p2 * 64);
      compute_2p2_gpu(block.get(), p1.data(), p23.data(), 0, d_zeros.get(), target,
                      1, d_xunit_.get(), kqed_t_, 16, d_zeros.get(), MPI_COMM_SELF,
                      2, 2, 2, 2, 2, 2, 2, 2);
      check_cuda(cudaGetLastError(), "compute_2p2_gpu");
    }
    check_cuda(cudaDeviceSynchronize(), "contraction kernels");
  }
  check_cuda(cudaMemGetInfo(&free_after, &total), "cudaMemGetInfo");
  return free_before > free_after ? free_before - free_after : 0;
}

}  // namespace hlbl
