// End-to-end check of hlbl::Contractor on random propagator fields.
//
// Fields are seeded by each site's GLOBAL lattice index (not the local
// one), so the same field values appear under any process grid that tiles
// the same global lattice -- letting a human check kernel_sum agreement
// across process grids by running this binary under different -np. This
// (single-rank) test does not do that itself, only a single-rank run.
//
// No physics: the fields are structureless noise, so what is checked here
// is that the contraction pipeline (dzu/dzsu -> g5/gamma_mu -> 4-point ->
// QED kernel -> MPI reduce) runs to completion, gives a finite result, and
// reproduces a stored golden value -- a regression guard, not a physics
// check.

#include "hlbl/contract.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "check.hpp"
#include "hlbl/lattice.hpp"
#include "hlbl/mdarray.hpp"

using namespace hlbl;

namespace {

// splitmix64 finalizer: cheap, well-mixed, deterministic.
uint64_t hash64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

double hash_to_unit(uint64_t x) {
  return static_cast<double>(hash64(x) >> 11) * (1.0 / 9007199254740992.0) *
             2.0 -
         1.0;
}

// [2 flavours][12 source components][24 * local volume], filled from each
// site's global lattice index.
std::vector<double> random_propagator(const Lattice& lat, uint64_t seed) {
  std::size_t vol = lat.volume();
  std::vector<double> buf(2 * 12 * 24 * vol);
  const Vec4& local = lat.local();
  const Vec4& pc = lat.proc_coords();
  for (int t = 0; t < local[0]; ++t) {
    for (int x = 0; x < local[1]; ++x) {
      for (int y = 0; y < local[2]; ++y) {
        for (int z = 0; z < local[3]; ++z) {
          std::size_t site = lat.local_index({t, x, y, z});
          Vec4 gc = {pc[0] * local[0] + t, pc[1] * local[1] + x,
                     pc[2] * local[2] + y, pc[3] * local[3] + z};
          uint64_t g = (((static_cast<uint64_t>(gc[0]) * 1000003u + gc[1]) *
                             1000003u +
                         gc[2]) *
                            1000003u +
                        gc[3]);
          for (int f = 0; f < 2; ++f) {
            for (int ia = 0; ia < 12; ++ia) {
              for (int c = 0; c < 24; ++c) {
                std::size_t idx = ((f * 12 + ia) * vol + site) * 24 + c;
                uint64_t key =
                    ((g * 2 + f) * 12 + ia) * 24 + c + seed * 0x2545F4914F6CDD1DULL;
                buf[idx] = hash_to_unit(key);
              }
            }
          }
        }
      }
    }
  }
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

  // Regression guard: the value the pipeline produced when this test was
  // written. The fields are structureless noise, so this is not a physics
  // check -- only a check that the pipeline still reproduces itself. The
  // tolerance is looser than double precision because kernel_sum is an
  // atomicAdd reduction over threads whose completion order (and hence
  // summation order) is not fixed from run to run; observed drift is
  // ~1e-15 relative, so 1e-10 leaves ample margin.
  const double golden[2] = {6.5973514693254831e+00, -6.9614442514884631e+00};
  for (int f = 0; f < 2; ++f) {
    double rel =
        std::fabs(result[f] - golden[f]) / std::max(1.0, std::fabs(golden[f]));
    if (rel > 1e-10) {
      std::fprintf(stderr, "flavor %d: got %.17g, golden %.17g (rel diff %.3e)\n",
                   f, result[f], golden[f], rel);
    }
    HLBL_CHECK(rel <= 1e-10);
  }

  free_QED_temps(&kqed_t);
  std::printf("test_contract_random: all checks passed\n");
  MPI_Finalize();
  return 0;
}
