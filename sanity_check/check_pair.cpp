// check_pair: solves two point sources on a real gauge configuration with
// tmLQCD/QUDA, contracts the pair they form, and prints kernel_sum for each
// flavour. A manual check, not part of ctest: it needs a gauge configuration
// and an invert.input in the working directory (see scripts/gen_invert_input.sh).
//
//   check_pair [n [plaquette]]
//
// The pair is (origin, origin + n*(1,1,1,1)); n defaults to 1. If plaquette
// is given, the run fails unless QUDA's mean plaquette agrees to 1e-9.

#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "hlbl/contract.hpp"
#include "hlbl/mdarray.hpp"
#include "hlbl/physics.hpp"
#include "hlbl/solver.hpp"
#include "hlbl/vec4.hpp"

using namespace hlbl;

namespace {

// Solves both flavours into one block, in the layout the Contractor expects.
DeviceBuffer<double> invert_pair(Solver& solver, const Vec4& gsx) {
  std::size_t const per_flavor = 12 * 24 * solver.lattice().volume();
  DeviceBuffer<double> d_block(2 * per_flavor);
  for (int flavor = 0; flavor < 2; ++flavor) {
    solver.invert_point(gsx, flavor, d_block, flavor * per_flavor);
  }
  return d_block;
}

}  // namespace

int main(int argc, char** argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int status = 0;

  try {
    int const stride = (argc > 1) ? std::atoi(argv[1]) : 1;
    bool const have_expected = (argc > 2);
    double const expected_plaquette = have_expected ? std::atof(argv[2]) : 0.0;

    Solver solver(argc, argv);
    if (rank == 0) {
      std::printf("# [check_pair] configuration %d, local %d,%d,%d,%d, global %d,%d,%d,%d\n",
                  solver.nconf(), solver.lattice().local()[0],
                  solver.lattice().local()[1], solver.lattice().local()[2],
                  solver.lattice().local()[3], solver.lattice().global()[0],
                  solver.lattice().global()[1], solver.lattice().global()[2],
                  solver.lattice().global()[3]);
    }

    solver.read_gauge();
    solver.warm_up();  // QUDA gauge load and multigrid setup

    // Print which GPU each rank uses (QUDA picks it during warm_up). Each
    // rank of a node should show a different PCI bus id.
    {
      int dev = -1;
      cudaGetDevice(&dev);
      char busid[32] = "?";
      cudaDeviceGetPCIBusId(busid, sizeof(busid), dev);
      std::printf("# [check_pair] rank %d: cuda device %d, pci %s\n", rank, dev,
                  busid);
      std::fflush(stdout);
    }

    auto plaq = solver.plaquette();
    if (rank == 0) {
      std::printf("# [check_pair] plaqQuda {mean, spatial, temporal} = %.12f %.12f %.12f\n",
                  plaq[0], plaq[1], plaq[2]);
    }
    if (have_expected && std::fabs(plaq[0] - expected_plaquette) > 1e-9) {
      throw std::runtime_error("check_pair: QUDA plaquette " +
                               std::to_string(plaq[0]) + " does not match the "
                               "expected " + std::to_string(expected_plaquette));
    }

    QED_kernel_temps kqed_t;
    if (initialise(&kqed_t) != 0) {
      throw std::runtime_error("check_pair: KQED initialise failed");
    }

    Vec4 gsx{0, 0, 0, 0};
    Vec4 yv{stride, stride, stride, stride};
    Vec4 gsy = add(gsx, yv);
    if (rank == 0) {
      std::printf("# [check_pair] pair x=(%d,%d,%d,%d) y=(%d,%d,%d,%d)\n", gsx[0],
                  gsx[1], gsx[2], gsx[3], gsy[0], gsy[1], gsy[2], gsy[3]);
    }

    DeviceBuffer<double> d_fwd_src = invert_pair(solver, gsx);
    DeviceBuffer<double> d_fwd_y = invert_pair(solver, gsy);

    Contractor contractor(solver.lattice(), kqed_t);
    auto result = contractor.kernel_sum(d_fwd_src.get(), d_fwd_y.get(), gsx,
                                         yv, xunit());

    if (rank == 0) {
      std::printf("# [check_pair] kernel_sum flavor0=%.18g flavor1=%.18g\n",
                  result[0], result[1]);
    }

    free_QED_temps(&kqed_t);
  } catch (const std::exception& e) {
    if (rank == 0) std::fprintf(stderr, "[check_pair] error: %s\n", e.what());
    status = 1;
  }

  MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return status;
}
