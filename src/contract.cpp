#include "hlbl/contract.hpp"

#include <algorithm>
#include <array>

#include "hlbl/gamma.hpp"

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
    : lattice_(lattice), kqed_t_(kqed_t) {}

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
  static_assert(kKernelN == 1, "the P234 rows below assume one QED kernel");
  const Vec4& g = lattice_.global();
  int const lmax = *std::max_element(g.begin(), g.end());
  int const n_y = static_cast<int>(targets.size());
  std::size_t const n_p1 = 4 * 4 * 4 * static_cast<std::size_t>(lmax);
  std::size_t const n_p234 = static_cast<std::size_t>(n_y) * kGeom2p2 * 64;
  if (d_p1_.size() < n_p1) d_p1_ = DeviceBuffer<double>(n_p1);
  if (d_p234_.size() < n_p234) d_p234_ = DeviceBuffer<double>(n_p234);
  if (d_targets_.size() < targets.size()) d_targets_ = DeviceBuffer<Coord>(targets.size());

  std::vector<Coord> ycoords;
  for (const Vec4& t : targets) ycoords.push_back(to_coord(t));
  d_targets_.copy_from_host(ycoords.data(), ycoords.size());
  // cu_2p2_pieces adds to P1 and P234
  d_p1_.zero();
  d_p234_.zero();
  // the kernels take {value, uncertainty}; only the value is used
  cu_2p2_pieces(d_p1_.get(), d_p234_.get(), d_fwd, /*iflavor=*/0,
                to_coord(lattice_.proc_coords()), to_coord(gsw), n_y, d_targets_.get(),
                Pair{xunit, 0.0}, kqed_t_, to_geom(g), to_geom(lattice_.local()));
  check_cuda(cudaGetLastError(), "cu_2p2_pieces launch");

  // this rank's sums, P234 [target][geometry][64]: one row per target
  Pieces2p2 out;
  out.p1.resize(n_p1);
  out.p234.resize(n_p234);
  d_p1_.copy_to_host(out.p1.data(), n_p1);
  d_p234_.copy_to_host(out.p234.data(), n_p234);
  MPI_Allreduce(MPI_IN_PLACE, out.p1.data(), static_cast<int>(n_p1), MPI_DOUBLE, MPI_SUM,
                lattice_.cart_comm());
  MPI_Allreduce(MPI_IN_PLACE, out.p234.data(), static_cast<int>(n_p234), MPI_DOUBLE,
                MPI_SUM, lattice_.cart_comm());
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
      // one target, no MPI: not collective
      Coord const target{1, 1, 1, 1};
      DeviceBuffer<Coord> d_target(1);
      d_target.copy_from_host(&target, 1);
      DeviceBuffer<double> p1(4 * 4 * 4 * 2), p234(kGeom2p2 * 64);
      p1.zero();
      p234.zero();
      cu_2p2_pieces(p1.get(), p234.get(), block.get(), 0, zero, zero, 1, d_target.get(),
                    Pair{1.0, 0.0}, kqed_t_, tiny, tiny);
      check_cuda(cudaGetLastError(), "cu_2p2_pieces launch");
    }
    check_cuda(cudaDeviceSynchronize(), "contraction kernels");
  }
  check_cuda(cudaMemGetInfo(&free_after, &total), "cudaMemGetInfo");
  return free_before > free_after ? free_before - free_after : 0;
}

}  // namespace hlbl
