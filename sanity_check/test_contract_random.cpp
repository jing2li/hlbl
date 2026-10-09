// Runs the Contractor on random propagator fields (one rank). The fields are
// noise, so this checks only that the pipeline gives a finite result and
// reproduces a stored value. It is a regression test, not a physics test.

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

  // Stored values from the first run. The tolerance allows for the order of
  // the atomicAdd sums changing from run to run.
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
