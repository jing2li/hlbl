// check_pair: inverts two point sources on a real gauge configuration
// through tmLQCD/QUDA, contracts the one pair they form, and prints
// kernel_sum per flavour.
//
// A manual sanity check, not a ctest case: it needs a real gauge
// configuration and a working invert.input in the current directory (see
// the cvc time_hlbl input templates), and a multigrid solve can take
// minutes to set up. Options are tmLQCD's own (mpirun/srun -np N ...
// check_pair, reading invert.input from cwd) plus:
//
//   check_pair [n]
//
// n is the stride along the (1,1,1,1) direction (default 1): the pair is
// (origin, origin + n*(1,1,1,1)), the same convention the hypercube
// schedule uses.

#include <mpi.h>

#include <array>
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

// [2 flavours][12 source components][24 * local volume] doubles: the
// layout hlbl::Contractor expects, and what invert_point's two calls (one
// per flavour) fill.
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
    solver.warm_up();  // pays for the QUDA gauge load / multigrid setup

    auto plaq = solver.plaquette();
    if (rank == 0) {
      std::printf("# [check_pair] plaqQuda {spatial, temporal, mean} = %.12f %.12f %.12f\n",
                  plaq[0], plaq[1], plaq[2]);
      std::printf("# [check_pair] compare against the configuration's recorded plaquette "
                  "(printed in the gauge file's xlf-info record)\n");
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
