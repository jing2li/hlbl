#ifndef _KERNELS_CUH_
#define _KERNELS_CUH_

//#include "cuda_lattice.h"
extern "C" {
#include "KQED.h"
}
#include <mpi.h>

/* CUDA 2P2 BREAKDOWN */
__host__ void compute_2p2_gpu(double *fwd_y, double *P1, double *P23, int const iflavor, int const gsw[4], 
     const int *gycoords, int const n_y, const double xunit[2], QED_kernel_temps kqed_t,  unsigned const VOLUME, 
     int const g_proc_coords[4], MPI_Comm g_cart_grid, unsigned const T, unsigned const LX, unsigned const LY, unsigned const LZ,
     unsigned const T_global, unsigned const LX_global, unsigned const LY_global, unsigned const LZ_global);
__global__ void kernel_pi(double* fwd_y, double * Pi, int iflavor, unsigned const VOLUME);
__global__ void kernel_p1(double *Pi, double *P1, int iflavor, int Lmax, int const * gsw, unsigned const VOLUME, 
     int const g_proc_coords[4], unsigned const T, unsigned const LX, unsigned const LY, unsigned const LZ, 
     unsigned const T_global, unsigned const LX_global, unsigned const LY_global, unsigned const LZ_global);
__global__ void kernel_p23(double *pi, double *P23, int n_y, const int*gsw, 
     const int *gycoords, const double xunit[2], QED_kernel_temps kqed_t, unsigned const VOLUME, 
     int const g_proc_coords[4], unsigned const T, unsigned const LX, unsigned const LY, unsigned const LZ, 
     unsigned const T_global, unsigned const LX_global, unsigned const LY_global, unsigned const LZ_global);

__host__ void record_pi_cuda(double *fwd_y, int VOLUME, int iflavor, 
     unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global);
__host__ void record_p1_cuda(double *Pi, int iflavor, int const * gsw, int VOLUME, 
     unsigned T, unsigned LX, unsigned LY, unsigned LZ, 
     unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global);
__host__ void record_p23_cuda(double *Pi, int n_y, const int gsw[4], 
     const int *gycoords, const double xunit[2], QED_kernel_temps kqed_t, 
     unsigned VOLUME, unsigned T, unsigned LX, unsigned LY, unsigned LZ, 
     unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global);
__host__ void record_2p2_cuda(double *fwd_y, double *P1, double *P23, int iflavor, 
     int const gsw[4], const int *gycoords, int n_y, const double xunit[2], 
     QED_kernel_temps kqed_t,  unsigned VOLUME, int const g_proc_coords[4], MPI_Comm g_cart_grid,
     unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global, 
     unsigned T, unsigned LX, unsigned LY, unsigned LZ);

__host__ void compute_4pt_gpu(
    const double * fwd_src, const double * fwd_y,
    double const *g_dzu, double const *g_dzsu,
    const int* gsx, int iflavor, const double xunit[2], const int yv[4],
    double* kernel_sum, QED_kernel_temps kqed_t, unsigned VOLUME,
    int const g_proc_coords[4], MPI_Comm g_cart_grid, unsigned T, unsigned LX, unsigned LY, unsigned LZ, 
    unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global);

__host__ void test_KQED_on_gpu(const double xv[4], const double yv[4], QED_kernel_temps kqed_t);
#endif /* _KERNELS_CUH_ */
