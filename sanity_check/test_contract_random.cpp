// Regression test, not a physics test: the Contractor on random fields (one
// rank) gives a finite result that matches a stored value, for the kernel sum
// and the 2+2 pieces; plus the index structure of P1 and P234.

#include "hlbl/contract.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "check.hpp"
#include "hlbl/lattice.hpp"
#include "hlbl/mdarray.hpp"
#include "hlbl/provider.hpp"

using namespace hlbl;

namespace {

std::vector<double> random_propagator(const Lattice& lat, uint64_t seed) {
  std::vector<double> buf(block_length(lat));
  fill_random_block(lat, seed, buf.data());
  return buf;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  HLBL_CHECK(size == 1);  // single-rank test; see file header

  Lattice lat(MPI_COMM_WORLD, Vec4{1, 1, 1, 1}, Vec4{8, 4, 4, 4});

  QED_kernel_temps kqed_t;
  HLBL_CHECK(initialise(&kqed_t) == 0);

  auto h_src = random_propagator(lat, 1);
  auto h_y = random_propagator(lat, 2);
  DeviceBuffer<double> d_src(h_src.size()), d_y(h_y.size());
  d_src.copy_from_host(h_src.data(), h_src.size());
  d_y.copy_from_host(h_y.data(), h_y.size());

  Contractor contractor(lat, kqed_t);
  Vec4 gsx{0, 0, 0, 0};
  Vec4 yv{1, 1, 1, 1};
  auto result = contractor.kernel_sum(d_src.get(), d_y.get(), gsx, yv, 1.0);

  for (double v : result) HLBL_CHECK(std::isfinite(v));

  // Stored values, from the kernels of b677727 (before: 6.5973514693254831,
  // -6.9614442514884631). The tolerance allows for the order of the atomicAdd
  // sums changing from run to run.
  const double golden[2] = {6.1108971307943145e+00, -4.9351727205043510e+00};
  for (int f = 0; f < 2; ++f) {
    double rel =
        std::fabs(result[f] - golden[f]) / std::max(1.0, std::fabs(golden[f]));
    if (rel > 1e-10) {
      std::fprintf(stderr, "flavor %d: got %.17g, golden %.17g (rel diff %.3e)\n",
                   f, result[f], golden[f], rel);
    }
    HLBL_CHECK(rel <= 1e-10);
  }

  // 2+2 pieces on 6^4: 1296 sites, not a multiple of the 256 threads of a
  // P2/P3/P4 block, as the kernel_p21 fix needs
  {
    Lattice lat6(MPI_COMM_WORLD, Vec4{1, 1, 1, 1}, Vec4{6, 6, 6, 6});
    Contractor c6(lat6, kqed_t);
    auto h_w = random_propagator(lat6, 3);
    DeviceBuffer<double> d_w(h_w.size());
    d_w.copy_from_host(h_w.data(), h_w.size());
    int const lmax = 6;
    Vec4 const w{1, 2, 3, 4}, a{3, 3, 3, 3}, b{0, 5, 1, 2};
    Pieces2p2 const ab = c6.pieces_2p2(d_w.get(), w, {a, b}, 1.0);
    HLBL_CHECK(ab.p1.size() == 64u * lmax && ab.p234.size() == 2u * kGeom2p2 * 64);
    double sum_p1 = 0, sum_p23 = 0, sum_p4 = 0;
    for (double v : ab.p1) {
      HLBL_CHECK(std::isfinite(v));
      sum_p1 += v;
    }
    for (std::size_t i = 0; i < ab.p234.size(); ++i) {
      HLBL_CHECK(std::isfinite(ab.p234[i]));
      // P2_0, P2_1, P3 or P4_0, P4_1
      (i / 64 % kGeom2p2 < 3 ? sum_p23 : sum_p4) += ab.p234[i];
    }

    auto near = [](double x, double ref) {
      return std::fabs(x - ref) <= 1e-10 * std::max(1.0, std::fabs(ref));
    };
    // sum_zeta P1[rho][sigma][nu][zeta] is Pi_{sigma nu} summed over all
    // sites, whatever rho the sites were binned along
    for (int sn = 0; sn < 16; ++sn) {
      double total[4] = {0, 0, 0, 0};
      for (int rho = 0; rho < 4; ++rho) {
        for (int z = 0; z < lmax; ++z) total[rho] += ab.p1[(rho * 16 + sn) * lmax + z];
      }
      for (int rho = 1; rho < 4; ++rho) HLBL_CHECK(near(total[rho], total[0]));
    }
    // P234[target][geometry][rho][sigma][nu]: only rho < sigma is filled,
    // except P4_1 (geometry 4): rho != sigma
    for (int t = 0; t < 2; ++t) {
      for (int geom = 0; geom < kGeom2p2; ++geom) {
        for (int rho = 0; rho < 4; ++rho) {
          for (int sigma = (geom == 4 ? rho : 0); sigma <= rho; ++sigma) {
            for (int nu = 0; nu < 4; ++nu) {
              HLBL_CHECK(ab.p234[(t * kGeom2p2 + geom) * 64 + rho * 16 + sigma * 4 + nu] == 0.0);
            }
          }
        }
      }
    }
    // a target's row does not depend on the other targets (and the rows are
    // in target order)
    Pieces2p2 const only_b = c6.pieces_2p2(d_w.get(), w, {b}, 1.0);
    for (int i = 0; i < kGeom2p2 * 64; ++i) {
      HLBL_CHECK(near(only_b.p234[i], ab.p234[kGeom2p2 * 64 + i]));
    }
    for (int i = 0; i < 64 * lmax; ++i) HLBL_CHECK(near(only_b.p1[i], ab.p1[i]));

    // stored values from the first run (P4: from the run that added it)
    const double golden_2p2[3] = {1.9010328152022312e+03, -9.3714854604502837e-02,
                                  8.2386837480270175e-02};
    double const got[3] = {sum_p1, sum_p23, sum_p4};
    const char* const name[3] = {"sum P1", "sum P2_0, P2_1, P3", "sum P4_0, P4_1"};
    for (int k = 0; k < 3; ++k) {
      if (!near(got[k], golden_2p2[k])) {
        std::fprintf(stderr, "2+2 %s: got %.17g, golden %.17g\n", name[k], got[k],
                     golden_2p2[k]);
      }
      HLBL_CHECK(near(got[k], golden_2p2[k]));
    }
  }

  free_QED_temps(&kqed_t);
  std::printf("test_contract_random: all checks passed\n");
  MPI_Finalize();
  return 0;
}
