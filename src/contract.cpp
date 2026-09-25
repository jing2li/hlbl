#include "hlbl/contract.hpp"

#include <algorithm>
#include <array>

#include "hlbl/gamma.hpp"

namespace hlbl {

namespace {

Coord to_coord(const Vec4& v) { return Coord{v[0], v[1], v[2], v[3]}; }
Geom to_geom(const Vec4& v) { return Geom{v[0], v[1], v[2], v[3]}; }

// out = gamma_mu . gamma_5 . in -- cvc's "insert g5, then gamma_mu"
// (_fv_eq_gamma_ti_fv(s, 5, in) followed by _fv_eq_gamma_ti_fv(out, mu, s)).
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

  std::array<double, kDzuLen> local_dzu;
  std::array<double, kDzsuLen> local_dzsu;
  d_dzu_.copy_to_host(local_dzu.data(), kDzuLen);
  d_dzsu_.copy_to_host(local_dzsu.data(), kDzsuLen);

  // every rank needs the full z-sum before it can do its own x-sum
  std::array<double, kDzuLen> dzu;
  std::array<double, kDzsuLen> dzsu;
  MPI_Allreduce(local_dzu.data(), dzu.data(), kDzuLen, MPI_DOUBLE, MPI_SUM,
                lattice_.cart_comm());
  MPI_Allreduce(local_dzsu.data(), dzsu.data(), kDzsuLen, MPI_DOUBLE, MPI_SUM,
                lattice_.cart_comm());

  // g_dzu[k][mu][ia] = insert_g5_gamma(mu, dzu[k][ia]); layout matches
  // production's init_4level_dtable(6,4,12,24): ((k*4+mu)*12+ia)*24
  std::array<double, kGDzuLen> g_dzu;
  for (int k = 0; k < 6; ++k) {
    for (int ia = 0; ia < 12; ++ia) {
      const double* src = &dzu[(k * 12 + ia) * 24];
      for (int mu = 0; mu < 4; ++mu) {
        insert_g5_gamma(&g_dzu[((k * 4 + mu) * 12 + ia) * 24], mu, src);
      }
    }
  }
  // g_dzsu[s][mu][ia], layout ((s*4+mu)*12+ia)*24
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
  // cvc's xunit[2] = {value, its uncertainty}; only the value feeds the
  // kernel (see cuda_lattice.cu, every use is xunit.a).
  Pair d_xunit{xunit, 0.0};

  d_kernel_sum_.zero();
  cu_4pt_contraction(d_kernel_sum_.get(), d_g_dzu_.get(), d_g_dzsu_.get(),
                      d_fwd_src, d_fwd_y, iflavor,
                      to_coord(lattice_.proc_coords()), to_coord(gsx),
                      d_xunit, to_coord(yv), kqed_t_,
                      to_geom(lattice_.global()), to_geom(lattice_.local()));

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

}  // namespace hlbl
