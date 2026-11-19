#ifndef _KERNELS_CUH_
#define _KERNELS_CUH_

extern "C" {
#include "KQED.h"
}
#include <mpi.h>

/* CUDA 2P2 BREAKDOWN */
__host__ void compute_2p2_gpu(double *fwd_y, double *P1, double *P234, int const iflavor, int const gsw[4],
     const int *gycoords, int const n_y, const double xunit[2], QED_kernel_temps kqed_t,  unsigned const VOLUME,
     int const g_proc_coords[4], MPI_Comm g_cart_grid, unsigned const T, unsigned const LX, unsigned const LY, unsigned const LZ,
     unsigned const T_global, unsigned const LX_global, unsigned const LY_global, unsigned const LZ_global);
__global__ void kernel_pi(double* fwd_y, double * Pi, int iflavor, unsigned const VOLUME);
__global__ void kernel_p1(double *Pi, double *P1, int iflavor, int Lmax, int const * gsw, unsigned const VOLUME,
     int const g_proc_coords[4], unsigned const T, unsigned const LX, unsigned const LY, unsigned const LZ,
     unsigned const T_global, unsigned const LX_global, unsigned const LY_global, unsigned const LZ_global);

#endif /* _KERNELS_CUH_ */
