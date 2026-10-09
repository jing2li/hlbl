// Checks Lattice on one rank: local index, site owner, periodic difference.

#include "hlbl/lattice.hpp"

#include <mpi.h>

#include <cstdio>

#include "check.hpp"

using namespace hlbl;

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  HLBL_CHECK(size == 1);  // single-rank test

  Lattice lat(MPI_COMM_WORLD, Vec4{1, 1, 1, 1}, Vec4{8, 4, 4, 4});
  HLBL_CHECK(lat.global() == (Vec4{8, 4, 4, 4}));
  HLBL_CHECK(lat.volume() == 8u * 4u * 4u * 4u);

  // local_index: ((t*LX+x)*LY+y)*LZ+z, t slowest, z fastest.
  HLBL_CHECK(lat.local_index({0, 0, 0, 0}) == 0u);
  HLBL_CHECK(lat.local_index({0, 0, 0, 1}) == 1u);
  HLBL_CHECK(lat.local_index({1, 0, 0, 0}) == 4u * 4u * 4u);

  // Single rank: every global site is local, with local == global.
  auto own = lat.owner({3, 2, 1, 0});
  HLBL_CHECK(own.rank == 0);
  HLBL_CHECK(own.local_coord == (Vec4{3, 2, 1, 0}));

  // site_map_diff: minimum image on an L=8 axis. (6-1) mod 8 = 5 ->
  // site_map(5,8) = 5-8 = -3.
  Vec4 d = lat.site_map_diff({6, 0, 0, 0}, {1, 0, 0, 0});
  HLBL_CHECK(d[0] == -3);

  std::printf("test_lattice: all checks passed\n");
  MPI_Finalize();
  return 0;
}
