// hlbl_hypercube: connected HLbL kernel sum on a sliding-hypercube schedule.

#include <mpi.h>

#include <cstdio>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) std::printf("# [hlbl_hypercube] not implemented yet\n");
  MPI_Finalize();
  return 0;
}
