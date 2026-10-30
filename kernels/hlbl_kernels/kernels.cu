/* GPU kernels of 2p2
    JingJing Li, 2025
*/
#include <iostream>
#include <stdio.h> 
#include <stdlib.h>
#include <mpi.h>
#include "kernels.cuh"

typedef struct {
  double re, im;
} complex;

#define kernel_n 1 // LLambda0.4 only; see hlbl_mII_invert_contract.cpp
#define kernel_n_geom 3 // P2_0, P2_1, P3

#define check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ __constant__ int idx_comb_d[6][2] ={
    {0,1},
    {0,2},
    {0,3},
    {1,2},
    {1,3},
    {2,3}
  };

#define _co_eq_fv_dag_ti_fv(c,s,t) {\
  (c)->re = \
    (s)[ 0]*(t)[ 0] + (s)[ 1]*(t)[ 1] +\
    (s)[ 2]*(t)[ 2] + (s)[ 3]*(t)[ 3] +\
    (s)[ 4]*(t)[ 4] + (s)[ 5]*(t)[ 5] +\
    (s)[ 6]*(t)[ 6] + (s)[ 7]*(t)[ 7] +\
    (s)[ 8]*(t)[ 8] + (s)[ 9]*(t)[ 9] +\
    (s)[10]*(t)[10] + (s)[11]*(t)[11] +\
    (s)[12]*(t)[12] + (s)[13]*(t)[13] +\
    (s)[14]*(t)[14] + (s)[15]*(t)[15] +\
    (s)[16]*(t)[16] + (s)[17]*(t)[17] +\
    (s)[18]*(t)[18] + (s)[19]*(t)[19] +\
    (s)[20]*(t)[20] + (s)[21]*(t)[21] +\
    (s)[22]*(t)[22] + (s)[23]*(t)[23];\
  (c)->im =\
    (s)[ 0]*(t)[ 1] - (s)[ 1]*(t)[ 0] +\
    (s)[ 2]*(t)[ 3] - (s)[ 3]*(t)[ 2] +\
    (s)[ 4]*(t)[ 5] - (s)[ 5]*(t)[ 4] +\
    (s)[ 6]*(t)[ 7] - (s)[ 7]*(t)[ 6] +\
    (s)[ 8]*(t)[ 9] - (s)[ 9]*(t)[ 8] +\
    (s)[10]*(t)[11] - (s)[11]*(t)[10] +\
    (s)[12]*(t)[13] - (s)[13]*(t)[12] +\
    (s)[14]*(t)[15] - (s)[15]*(t)[14] +\
    (s)[16]*(t)[17] - (s)[17]*(t)[16] +\
    (s)[18]*(t)[19] - (s)[19]*(t)[18] +\
    (s)[20]*(t)[21] - (s)[21]*(t)[20] +\
    (s)[22]*(t)[23] - (s)[23]*(t)[22];}

__device__ inline static void _fv_eq_gamma_ti_fv(double* out, int gamma_index, double const* in) {
    switch (gamma_index)
    {
    case 0: // gamma_0
        out[0]  = -in[12]; 
        out[1]  = -in[13];
        out[2]  = -in[14];
        out[3]  = -in[15];
        out[4]  = -in[16];
        out[5]  = -in[17];
        out[6]  = -in[18];
        out[7]  = -in[19];
        out[8]  = -in[20];
        out[9]  = -in[21];
        out[10] = -in[22];
        out[11] = -in[23];
        out[12] = -in[0];
        out[13] = -in[1];
        out[14] = -in[2];
        out[15] = -in[3];
        out[16] = -in[4];
        out[17] = -in[5];
        out[18] = -in[6];
        out[19] = -in[7];
        out[20] = -in[8];
        out[21] = -in[9];
        out[22] = -in[10];
        out[23] = -in[11];
        break;
    case 1: // gamma_1
        out[0]  =  in[19];
        out[1]  = -in[18];
        out[2]  =  in[21];
        out[3]  = -in[20];
        out[4]  =  in[23];
        out[5]  = -in[22];
        out[6]  =  in[13];
        out[7]  = -in[12];
        out[8]  =  in[15];
        out[9]  = -in[14];
        out[10] =  in[17];
        out[11] = -in[16];
        out[12] = -in[7];
        out[13] =  in[6];
        out[14] = -in[9];
        out[15] =  in[8];
        out[16] = -in[11];
        out[17] =  in[10];
        out[18] = -in[1];
        out[19] =  in[0];
        out[20] = -in[3];
        out[21] =  in[2];
        out[22] = -in[5];
        out[23] =  in[4];
        break;
    case 2: // gamma_2
        out[0]  = -in[18];
        out[1]  = -in[19];
        out[2]  = -in[20];
        out[3]  = -in[21];
        out[4]  = -in[22];
        out[5]  = -in[23];
        out[6]  =  in[12];
        out[7]  =  in[13];
        out[8]  =  in[14];
        out[9]  =  in[15];
        out[10] =  in[16];
        out[11] =  in[17];
        out[12] =  in[6];
        out[13] =  in[7];
        out[14] =  in[8];
        out[15] =  in[9];
        out[16] =  in[10];
        out[17] =  in[11];
        out[18] = -in[0];
        out[19] = -in[1];
        out[20] = -in[2];
        out[21] = -in[3];
        out[22] = -in[4];
        out[23] = -in[5];
        break;
    case 3: // gamma_3
        out[0]  =  in[13];
        out[1]  = -in[12];
        out[2]  =  in[15];
        out[3]  = -in[14];
        out[4]  =  in[17];
        out[5]  = -in[16];
        out[6]  = -in[19];
        out[7]  =  in[18];
        out[8]  = -in[21];
        out[9]  =  in[20];
        out[10] = -in[23];
        out[11] =  in[22];
        out[12] = -in[1];
        out[13] =  in[0];
        out[14] = -in[3];
        out[15] =  in[2];
        out[16] = -in[5];
        out[17] =  in[4];
        out[18] =  in[7];
        out[19] = -in[6];
        out[20] =  in[9];
        out[21] = -in[8];
        out[22] =  in[11];
        out[23] = -in[10];
        break;
    case 5:
        out[0]  =  in[0];
        out[1]  =  in[1];
        out[2]  =  in[2];
        out[3]  =  in[3];
        out[4]  =  in[4];
        out[5]  =  in[5];
        out[6]  =  in[6];
        out[7]  =  in[7];
        out[8]  =  in[8];
        out[9]  =  in[9];
        out[10] =  in[10];
        out[11] =  in[11];
        out[12] = -in[12];
        out[13] = -in[13];
        out[14] = -in[14];
        out[15] = -in[15];
        out[16] = -in[16];
        out[17] = -in[17];
        out[18] = -in[18];
        out[19] = -in[19];
        out[20] = -in[20];
        out[21] = -in[21];
        out[22] = -in[22];
        out[23] = -in[23];
        break;
    default:
        break;
    }
}

__device__ inline static void _fv_ti_eq_g5(double* in_out) {
    #pragma unroll
    for (int i = 12; i < 24; ++i) {
        in_out[i] *= -1;
    }
}

__host__ inline static int get_max(unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global)
{
    int Lmax = T_global;
    Lmax = (LX_global < Lmax) ? Lmax : LX_global;
    Lmax = (LY_global < Lmax) ? Lmax : LY_global;
    Lmax = (LZ_global < Lmax) ? Lmax : LZ_global;
    return Lmax;
}

__device__ inline static size_t prop_idx(int iflavour, int ia, int x, int ib, unsigned VOL) {
    return (size_t)iflavour * 12 * 24 * VOL
         + (size_t)ia * 24 * VOL
         + (size_t)x * 24 + ib;
}

/* x in [0, L) -> [-L/2, L/2), L/2 going to -L/2. Every QED kernel argument
 * uses it. */
__device__ inline static void site_map (int xv[4], int const x[4], unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global)
{
  xv[0] = ( x[0] >= T_global   / 2 ) ? (x[0] - T_global )  : x[0];
  xv[1] = ( x[1] >= LX_global  / 2 ) ? (x[1] - LX_global)  : x[1];
  xv[2] = ( x[2] >= LY_global  / 2 ) ? (x[2] - LY_global)  : x[2];
  xv[3] = ( x[3] >= LZ_global  / 2 ) ? (x[3] - LZ_global)  : x[3];

  return;
}

/* As site_map, but L/2 goes to 0 (range (-L/2, L/2), even L). Only for the
 * coordinate weights x_rho: they are odd, so the images +-L/2 average to 0. */
__device__ inline static void site_map_zerohalf (int xv[4], int const x[4], unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global)
{
  xv[0] = ( x[0] > T_global   / 2 ) ? x[0] - T_global   : (  ( x[0] < T_global   / 2 ) ? x[0] : 0 );
  xv[1] = ( x[1] > LX_global  / 2 ) ? x[1] - LX_global  : (  ( x[1] < LX_global  / 2 ) ? x[1] : 0 );
  xv[2] = ( x[2] > LY_global  / 2 ) ? x[2] - LY_global  : (  ( x[2] < LY_global  / 2 ) ? x[2] : 0 );
  xv[3] = ( x[3] > LZ_global  / 2 ) ? x[3] - LZ_global  : (  ( x[3] < LZ_global  / 2 ) ? x[3] : 0 );

  return;
}

__device__ static void KQED_LX(int const ikernel, const double xm[4], const double ym[4],
    const struct QED_kernel_temps kqed_t, double kerv[6][4][4][4]) {
    /* only LLambda0.4; see the note on kernel_n in hlbl_mII_invert_contract.cpp */
    if (ikernel != 0) {
        printf("Error: kernel index %d out of range (0-%d)\n", ikernel, kernel_n-1);
        return;
    }
    QED_Mkernel_L2( 0.4, xm, ym, kqed_t, kerv );
}

__global__ void kernel_pi(double* fwd_y, double * Pi, int iflavor, unsigned const VOLUME){
    __shared__ double u[12 * 24];
    __shared__ double d[12 * 24];
    __shared__ double gu_diag[4][4][12];

    //for (int ix = tid; ix < VOLUME; ix += gridDim.x * blockDim.x) {
    for (int ix = blockIdx.x; ix < VOLUME; ix += gridDim.x) {
        // preload d = fwd_y[1-iflavor]
        int const tid = threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z;
        //int const off1 = prop_idx_(1-iflavor, 0, ix, 0, VOLUME);
        for (int i = tid; i < 12 * 24; i+=blockDim.x*blockDim.y*blockDim.z) {
            u[i] = fwd_y[prop_idx(iflavor, i/24, ix, i%24, VOLUME)];
            d[i] = fwd_y[prop_idx(1-iflavor, i/24, ix, i%24, VOLUME)];
        }
        __syncthreads();

        // operation at every lattice site
        int const mu = threadIdx.x;
        int const nu = threadIdx.y;
        //for (int ia = threadIdx.z; ia < 12; ia += blockDim.z) {
        int const ia = threadIdx.z;
        {
            //double dot_prod[12 * 24];
            double gu[24];
            double dot_prod[24];
            /* apply gammas: gu = g_5 g_mu u */
            _fv_eq_gamma_ti_fv(gu, mu, u + ia * 24);
            //_fv_eq_gamma_ti_fv(gu, mu, fwd_y + prop_idx(iflavor, ia, ix, 0, VOLUME));
            _fv_ti_eq_g5(gu);


            /* compute <d, u>, dot_prod = <d, gmu> */
            for (int ib=0; ib<12; ib++ ) {
                complex w;
                _co_eq_fv_dag_ti_fv(&w, d + ib * 24, gu);
                dot_prod[2*ib] = w.re;
                dot_prod[2*ib + 1] = w.im;
            }

            /* apply gammas: gu = g_nu g_5 dot_prod */
            _fv_ti_eq_g5(dot_prod);
            _fv_eq_gamma_ti_fv(gu, nu, dot_prod);
                
            gu_diag[mu][nu][ia] = gu[2 * ia]; // only real part needed for trace
        }

        __syncthreads();
        if (threadIdx.z == 0) {
            /* take trace over ia: pi[x][mu][nu] = Tr[gu] */
            double trace = 0.0;
            #pragma unroll
            for (int ia=0; ia<12; ia++) {
                //trace += gu[ia * 24 + 2 * ia]; 
                trace += gu_diag[mu][nu][ia];
            }
            Pi[ix*16+mu*4+nu] = trace;
        }
    }
}

// integrate over Pi to get P1, *_proc is the MPI proc's coordinate in each direction
__global__ void kernel_p1(double *Pi, double *P1, int iflavor, int Lmax, int const gsw[4], unsigned const VOLUME, int const g_proc_coords[4], 
    unsigned const T, unsigned const LX, unsigned const LY, unsigned const LZ, unsigned const T_global, unsigned const LX_global, unsigned const LY_global, unsigned const LZ_global){
   //const int Lmax = T_global;
    const int local_dim[4] = {static_cast<int>(T), static_cast<int>(LX), static_cast<int>(LY), static_cast<int>(LZ)};
    const int global_dim[4] = {static_cast<int>(T_global), static_cast<int>(LX_global), static_cast<int>(LY_global), static_cast<int>(LZ_global)};

    // P1[rho][sigma][nu][z[rho]] += Pi[z][sigma][nu]
    // i.e. P1[rho][sigma][nu][i] += sum over all z with z[rho]=i of Pi[z][sigma][nu]
    //for (int rho=blockIdx.x; rho<4; rho+=gridDim.x)
    //for (int sigma=blockIdx.y; sigma<4; sigma+=gridDim.y)
    //for (int nu=blockIdx.z; nu<4; nu+=gridDim.z)
    int rho = blockIdx.x;
    int sigma = blockIdx.y;
    int nu = blockIdx.z;
    for (int zr=threadIdx.x; zr<local_dim[rho]; zr+=blockDim.x) {
        double sum = 0.0;
        // the index of the three non-rho directions
        int dir[3];
        int cnt = 0;
        for (int d=0; d<4; d++){
            if (d!=rho){
                dir[cnt] = d;
                cnt++;
            }
        }
        // construct local z[4], rho direction fixed
        int z[4];
        z[rho]=zr;
        // loop over the rest VOLUME/local_dim[rho] points in 3D
        for (int iz = 0; iz < VOLUME/local_dim[rho]; iz++) {
            // the 3 other local z directions
            z[dir[0]] = iz / (local_dim[dir[1]] * local_dim[dir[2]]);
            z[dir[1]] = iz / local_dim[dir[2]] % local_dim[dir[1]];
            z[dir[2]] = iz % local_dim[dir[2]];

            // now machine address z_lex
            const int z_lex = z[0] * LX * LY * LZ + z[1] * LY * LZ + z[2] * LZ + z[3];

            // accumulate sum
            sum += Pi[z_lex*16 + sigma*4 + nu];
        }

        // write to P1
        // global z[rho] - w[rho]
        const unsigned z_w = (zr + local_dim[rho] * g_proc_coords[rho]  + global_dim[rho] - gsw[rho]) % global_dim[rho];
        atomicAdd_system(&P1[rho*16*Lmax + sigma*4*Lmax + nu*Lmax + z_w], sum);
    }
}

//warp-level reduction
__device__ inline static double warpReduceSum(double val) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

//block-level reduction
__device__ inline static double blockReduceSum(double val) {
    static __shared__ double shared[32]; // Shared memory for at most 32 warps
    int const lane = threadIdx.x % 32;
    int const wid = threadIdx.x / 32;
    int const nwarps = (blockDim.x + 31) / 32;

    val = warpReduceSum(val); // Each warp performs partial reduction

    if (lane == 0) {
        shared[wid] = val; // Write reduced value to shared memory
    }
    __syncthreads();

    // Final reduction within the first warp
    val = (threadIdx.x < nwarps) ? shared[lane] : 0;
    __syncthreads();
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

__global__ void /* __launch_bounds__(128,2) */ kernel_p20(double *pi, double *P23, int n_y, const int gsw[4], const int *gycoords, const double xunit[2],
QED_kernel_temps kqed_t, unsigned const VOLUME, const int g_proc_coords[4], unsigned const T, unsigned const LX, unsigned const LY, unsigned const LZ, 
unsigned const T_global, unsigned const LX_global, unsigned const LY_global, unsigned const LZ_global){

    for ( int yi = blockIdx.x; yi < n_y; yi+=gridDim.x){
        // For P2: y = (gsy - gsw)
        // For P3: y' = (gsw - gsy)
        // We define y = (gsy - gsw) and use -y as input for P3.
        int const * gsy = &gycoords[4*yi];
        int const y[4] = {
            ( gsy[0] - gsw[0] + static_cast<int>(T_global) ) % static_cast<int>(T_global),
            ( gsy[1] - gsw[1] + static_cast<int>(LX_global) ) % static_cast<int>(LX_global),
            ( gsy[2] - gsw[2] + static_cast<int>(LY_global) ) % static_cast<int>(LY_global),
            ( gsy[3] - gsw[3] + static_cast<int>(LZ_global) ) % static_cast<int>(LZ_global)
        };
        int yv[4];
        site_map ( yv, y, T_global, LX_global, LY_global, LZ_global );
        double const ym[4] = {yv[0] * xunit[0], yv[1] * xunit[0], yv[2] * xunit[0], yv[3] * xunit[0] };
        
        // parallelise over ikernel
        int const ikernel = blockIdx.y;

        double local_p2_0[64]={0};
        
        //for (int sweep = 0; sweep < (VOLUME + blockDim.x -1) / blockDim.x; sweep++){
        for (int ix = threadIdx.x; ix<VOLUME; ix +=blockDim.x){
            //int const ix = sweep * blockDim.x + threadIdx.x;
            /* a different local copy of P2/3 for each kernel */
            //if (ix >= VOLUME) continue;
            const double pix[16] = {pi[ix*16 +0], pi[ix*16 +1], pi[ix*16 +2], pi[ix*16 +3],
                            pi[ix*16 +4], pi[ix*16 +5], pi[ix*16 +6], pi[ix*16 +7],
                            pi[ix*16 +8], pi[ix*16 +9], pi[ix*16 +10],pi[ix*16 +11],
                            pi[ix*16 +12],pi[ix*16 +13],pi[ix*16 +14],pi[ix*16 +15]};

            int const x[4] = {
            (ix / (static_cast<int>(LX) * static_cast<int>(LY) * static_cast<int>(LZ)) - gsw[0] + g_proc_coords[0] * static_cast<int>(T) + static_cast<int>(T_global)) % static_cast<int>(T_global),
            (ix / (static_cast<int>(LY) * static_cast<int>(LZ)) % static_cast<int>(LX) - gsw[1] + g_proc_coords[1] * static_cast<int>(LX) + static_cast<int>(LX_global)) % static_cast<int>(LX_global),
            ((ix / static_cast<int>(LZ)) % static_cast<int>(LY) - gsw[2]  + g_proc_coords[2] * static_cast<int>(LY) + static_cast<int>(LY_global)) % static_cast<int>(LY_global),
            (ix % static_cast<int>(LZ) - gsw[3] + g_proc_coords[3] * static_cast<int>(LZ) + static_cast<int>(LZ_global)) % static_cast<int>(LZ_global)};

            int xv[4];
            site_map ( xv, x, T_global, LX_global, LY_global, LZ_global );
            double const xm[4] = {xv[0] * xunit[0], xv[1] * xunit[0], xv[2] * xunit[0], xv[3] * xunit[0] };

            double kerv1[6][4][4][4] KQED_ALIGN ;
            KQED_LX(ikernel, xm, ym, kqed_t, kerv1);
            /* unroll k (too much register pressure)*/
            #pragma unroll
            for (int mu=0; mu<4; mu++)
            for (int nu=0; nu<4; nu++)
            for (int lambda=0; lambda<4; lambda++){
                // k=0: {0,1}
                local_p2_0[0*16 + 1*4 + nu] += kerv1[0][mu][nu][lambda] * pix[mu*4 +lambda];
                        
                // k=1: {0,2}
                local_p2_0[0*16 + 2*4 + nu] += kerv1[1][mu][nu][lambda] * pix[mu*4 +lambda];
                            
                // k=2: {0,3}
                local_p2_0[0*16 + 3*4 + nu] += kerv1[2][mu][nu][lambda] * pix[mu*4 +lambda];
                
                // k=3: {1,2}
                local_p2_0[1*16 + 2*4 + nu] += kerv1[3][mu][nu][lambda] * pix[mu*4 +lambda];
                                
                // k=4: {1,3}
                local_p2_0[1*16 + 3*4 + nu] += kerv1[4][mu][nu][lambda] * pix[mu*4 +lambda];
                
                // k=5: {2,3}
                local_p2_0[2*16 + 3*4 + nu] += kerv1[5][mu][nu][lambda] * pix[mu*4 +lambda];
            
            }
        }
        
        

        #pragma unroll
        for (int rho = 0; rho < 4; rho++)
        for (int sigma = 0; sigma < 4; sigma++)
        for (int nu = 0; nu < 4; nu++) {
            int idx = rho * 16 + sigma * 4 + nu;

            double sum_p2_0 = blockReduceSum(local_p2_0[idx]);

            if (threadIdx.x == 0) {
                double *P23_y = P23 + 0*n_y*kernel_n*64 + yi*kernel_n*64 + ikernel*64;
                P23_y[idx] = sum_p2_0;
            }
        } 
    }
    
}

__global__ void /* __launch_bounds__(128,2) */ kernel_p21(double *pi, double *P23, int n_y, const int gsw[4], const int *gycoords, const double xunit[2],
QED_kernel_temps kqed_t, unsigned const VOLUME, const int g_proc_coords[4], unsigned const T, unsigned const LX, unsigned const LY, unsigned const LZ, 
unsigned const T_global, unsigned const LX_global, unsigned const LY_global, unsigned const LZ_global){
    //set_zero(P23, n_p23);
    
    for ( int yi = blockIdx.x; yi < n_y; yi+=gridDim.x){
        // For P2: y = (gsy - gsw)
        // For P3: y' = (gsw - gsy)
        // We define y = (gsy - gsw) and use -y as input for P3.
        int const * gsy = &gycoords[4*yi];
        int const y[4] = {
            ( gsy[0] - gsw[0] + static_cast<int>(T_global) ) % static_cast<int>(T_global),
            ( gsy[1] - gsw[1] + static_cast<int>(LX_global) ) % static_cast<int>(LX_global),
            ( gsy[2] - gsw[2] + static_cast<int>(LY_global) ) % static_cast<int>(LY_global),
            ( gsy[3] - gsw[3] + static_cast<int>(LZ_global) ) % static_cast<int>(LZ_global)
        };
        int yv[4];
        site_map ( yv, y, T_global, LX_global, LY_global, LZ_global );

        double const ym[4] = {yv[0] * xunit[0], yv[1] * xunit[0], yv[2] * xunit[0], yv[3] * xunit[0] };
        
        // parallelise over ikernel
        int ikernel = blockIdx.y;

        double local_p2_1[64]={0};

        //for (int sweep = 0; sweep < (VOLUME + blockDim.x -1) / blockDim.x; sweep++){
        for (int ix = threadIdx.x; ix<VOLUME; ix +=blockDim.x){
            //int const ix = sweep * blockDim.x + threadIdx.x;
            /* a different local copy of P2/3 for each kernel */
            //if (ix >= VOLUME) continue;
            const double pix[16] = {pi[ix*16 +0], pi[ix*16 +1], pi[ix*16 +2], pi[ix*16 +3],
                            pi[ix*16 +4], pi[ix*16 +5], pi[ix*16 +6], pi[ix*16 +7],
                            pi[ix*16 +8], pi[ix*16 +9], pi[ix*16 +10],pi[ix*16 +11],
                            pi[ix*16 +12],pi[ix*16 +13],pi[ix*16 +14],pi[ix*16 +15]};

            int const x[4] = {
            (ix / (static_cast<int>(LX) * static_cast<int>(LY) * static_cast<int>(LZ)) - gsw[0] + g_proc_coords[0] * static_cast<int>(T) + static_cast<int>(T_global)) % static_cast<int>(T_global),
            (ix / (static_cast<int>(LY) * static_cast<int>(LZ)) % static_cast<int>(LX) - gsw[1] + g_proc_coords[1] * static_cast<int>(LX) + static_cast<int>(LX_global)) % static_cast<int>(LX_global),
            ((ix / static_cast<int>(LZ)) % static_cast<int>(LY) - gsw[2]  + g_proc_coords[2] * static_cast<int>(LY) + static_cast<int>(LY_global)) % static_cast<int>(LY_global),
            (ix % static_cast<int>(LZ) - gsw[3] + g_proc_coords[3] * static_cast<int>(LZ) + static_cast<int>(LZ_global)) % static_cast<int>(LZ_global)};

            int xv[4];
            site_map ( xv, x, T_global, LX_global, LY_global, LZ_global );

            double const xm[4] = {xv[0] * xunit[0], xv[1] * xunit[0], xv[2] * xunit[0], xv[3] * xunit[0] };
            
            double kerv2[6][4][4][4] KQED_ALIGN ;
            KQED_LX(ikernel, ym, xm, kqed_t, kerv2);
            #pragma unroll
            for (int mu=0; mu<4; mu++)
            for (int nu=0; nu<4; nu++)
            for (int lambda=0; lambda<4; lambda++){
                // k=0: {0,1}
                local_p2_1[0*16 + 1*4 + nu] += kerv2[0][nu][mu][lambda] * pix[mu*4+lambda];
            
                // k=1: {0,2}
                local_p2_1[0*16 + 2*4 + nu] += kerv2[1][nu][mu][lambda] * pix[mu*4+lambda];
            
                // k=2: {0,3}
                local_p2_1[0*16 + 3*4 + nu] += kerv2[2][nu][mu][lambda] * pix[mu*4+lambda];
 
                // k=3: {1,2}
                local_p2_1[1*16 + 2*4 + nu] += kerv2[3][nu][mu][lambda] * pix[mu*4+lambda];

                // k=4: {1,3}
                local_p2_1[1*16 + 3*4 + nu] += kerv2[4][nu][mu][lambda] * pix[mu*4+lambda];
            
                // k=5: {2,3}
                local_p2_1[2*16 + 3*4 + nu] += kerv2[5][nu][mu][lambda] * pix[mu*4+lambda];
            }
        }

        /* after the ix loop, as in kernel_p20 and kernel_p3: blockReduceSum
         * syncs the block, so every thread must call it equally often */
        #pragma unroll
        for (int rho = 0; rho < 4; rho++)
        for (int sigma = 0; sigma < 4; sigma++)
        for (int nu = 0; nu < 4; nu++) {
            int idx = rho * 16 + sigma * 4 + nu;

            double sum_p2_1 = blockReduceSum(local_p2_1[idx]);

            if (threadIdx.x == 0) {
                double *P23_y = P23 + n_y*kernel_n*64 + yi*kernel_n*64 + ikernel*64;
                P23_y[idx] = sum_p2_1;
            }
        }
    }

}

__global__ void /* __launch_bounds__(128,2) */ kernel_p3(double *pi, double *P23, int n_y, const int gsw[4], const int *gycoords, const double xunit[2],
QED_kernel_temps kqed_t, unsigned const VOLUME, const int g_proc_coords[4], unsigned const T, unsigned const LX, unsigned const LY, unsigned const LZ, 
unsigned const T_global, unsigned const LX_global, unsigned const LY_global, unsigned const LZ_global){
    //set_zero(P23, n_p23);
    
    for ( int yi = blockIdx.x; yi < n_y; yi+=gridDim.x){
        // For P2: y = (gsy - gsw)
        // For P3: y' = (gsw - gsy)
        // We define y = (gsy - gsw) and use -y as input for P3.
        int const * gsy = &gycoords[4*yi];
        int const y[4] = {
            ( gsy[0] - gsw[0] + static_cast<int>(T_global) ) % static_cast<int>(T_global),
            ( gsy[1] - gsw[1] + static_cast<int>(LX_global) ) % static_cast<int>(LX_global),
            ( gsy[2] - gsw[2] + static_cast<int>(LY_global) ) % static_cast<int>(LY_global),
            ( gsy[3] - gsw[3] + static_cast<int>(LZ_global) ) % static_cast<int>(LZ_global)
        };
        // -y = gsw - gsy mapped on its own, not -yv: -(-L/2) would leave [-L/2, L/2)
        int const y_minus[4] = {
            ( static_cast<int>(T_global) - y[0] ) % static_cast<int>(T_global),
            ( static_cast<int>(LX_global) - y[1] ) % static_cast<int>(LX_global),
            ( static_cast<int>(LY_global) - y[2] ) % static_cast<int>(LY_global),
            ( static_cast<int>(LZ_global) - y[3] ) % static_cast<int>(LZ_global)
        };
        int yv_minus[4];
        site_map ( yv_minus, y_minus, T_global, LX_global, LY_global, LZ_global );

        double const ym_minus[4] = { yv_minus[0] * xunit[0], yv_minus[1] * xunit[0], yv_minus[2] * xunit[0], yv_minus[3] * xunit[0] };
        
        // parallelise over ikernel
        int ikernel = blockIdx.y;

        double local_p3[64]={0};

        //for (int sweep = 0; sweep < (VOLUME + blockDim.x -1) / blockDim.x; sweep++){
        for (int ix = threadIdx.x; ix<VOLUME; ix +=blockDim.x){
            //int const ix = sweep * blockDim.x + threadIdx.x;
            /* a different local copy of P2/3 for each kernel */
            //if (ix >= VOLUME) continue;
            const double pix[16] = {pi[ix*16 +0], pi[ix*16 +1], pi[ix*16 +2], pi[ix*16 +3],
                            pi[ix*16 +4], pi[ix*16 +5], pi[ix*16 +6], pi[ix*16 +7],
                            pi[ix*16 +8], pi[ix*16 +9], pi[ix*16 +10],pi[ix*16 +11],
                            pi[ix*16 +12],pi[ix*16 +13],pi[ix*16 +14],pi[ix*16 +15]};

            int const x[4] = {
            (ix / (static_cast<int>(LX) * static_cast<int>(LY) * static_cast<int>(LZ)) - gsw[0] + g_proc_coords[0] * static_cast<int>(T) + static_cast<int>(T_global)) % static_cast<int>(T_global),
            (ix / (static_cast<int>(LY) * static_cast<int>(LZ)) % static_cast<int>(LX) - gsw[1] + g_proc_coords[1] * static_cast<int>(LX) + static_cast<int>(LX_global)) % static_cast<int>(LX_global),
            ((ix / static_cast<int>(LZ)) % static_cast<int>(LY) - gsw[2]  + g_proc_coords[2] * static_cast<int>(LY) + static_cast<int>(LY_global)) % static_cast<int>(LY_global),
            (ix % static_cast<int>(LZ) - gsw[3] + g_proc_coords[3] * static_cast<int>(LZ) + static_cast<int>(LZ_global)) % static_cast<int>(LZ_global)};

            // wrapped (minimum image)
            int const x_mi_y[4] = {
                (x[0] - y[0] + static_cast<int>(T_global)) % static_cast<int>(T_global), 
                (x[1] - y[1] + static_cast<int>(LX_global)) % LX_global, 
                (x[2] - y[2] + static_cast<int>(LY_global)) % LY_global, 
                (x[3] - y[3] + static_cast<int>(LZ_global)) % LZ_global};
            int xmyv[4];
            site_map(xmyv, x_mi_y, T_global, LX_global, LY_global, LZ_global);
            double xm_mi_ym[4] = {xmyv[0] * xunit[0], xmyv[1] * xunit[0], xmyv[2] * xunit[0], xmyv[3] * xunit[0]};
            

            double kerv3[6][4][4][4] KQED_ALIGN ;
            KQED_LX(ikernel, xm_mi_ym, ym_minus, kqed_t, kerv3);
            #pragma unroll
            for (int mu=0; mu<4; mu++)
            for (int lambda=0; lambda<4; lambda++)
            for (int nu=0; nu<4; nu++){
                // k=0: {0,1}
                local_p3[0*16 + 1*4 + nu] += kerv3[0][mu][lambda][nu] * pix[mu*4+lambda];
            
                // k=1: {0,2}
                local_p3[0*16 + 2*4 + nu] += kerv3[1][mu][lambda][nu] * pix[mu*4+lambda];
            
                // k=2: {0,3}
                local_p3[0*16 + 3*4 + nu] += kerv3[2][mu][lambda][nu] * pix[mu*4+lambda];
            
                // k=3: {1,2}
                local_p3[1*16 + 2*4 + nu] += kerv3[3][mu][lambda][nu] * pix[mu*4+lambda];
            
                // k=4: {1,3}
                local_p3[1*16 + 3*4 + nu] += kerv3[4][mu][lambda][nu] * pix[mu*4+lambda];
            
                // k=5: {2,3}
                local_p3[2*16 + 3*4 + nu] += kerv3[5][mu][lambda][nu] * pix[mu*4+lambda];
            }
        }

        #pragma unroll
        for (int rho = 0; rho < 4; rho++)
        for (int sigma = 0; sigma < 4; sigma++)
        for (int nu = 0; nu < 4; nu++) {
            int idx = rho * 16 + sigma * 4 + nu;

            double sum_p3 = blockReduceSum(local_p3[idx]);

            if (threadIdx.x == 0) {
                double *P23_y = P23 + 2*n_y*kernel_n*64 + yi*kernel_n*64 + ikernel*64;
                P23_y[idx] = sum_p3;
            }
        } 
    }
    
}

__host__ void compute_2p2_gpu(double *fwd_y, double *P1, double *P23, int iflavor, 
     int const gsw[4], const int *gycoords, int const n_y, const double xunit[2], 
     QED_kernel_temps kqed_t,  unsigned const VOLUME, int const g_proc_coords[4], MPI_Comm g_cart_grid,
     unsigned T, unsigned const LX, unsigned const LY, unsigned const LZ,
     unsigned const T_global, unsigned const LX_global, unsigned const LY_global, unsigned const LZ_global) {
    /* --- 1. SETUP & ALLOCATION --- */
    double *Pi_d, *P1_d, *P23_d/* , *fwd_y_d */;
    int *gycoords_d;
    int const Lmax = get_max(T_global, LX_global, LY_global, LZ_global);
    
    // Create Streams for concurrent execution
    cudaStream_t /* stream_p23,  */stream_p1, stream_p20, stream_p21, stream_p3;
    cudaStreamCreate(&stream_p20);
    cudaStreamCreate(&stream_p21);
    cudaStreamCreate(&stream_p3);
    cudaStreamCreate(&stream_p1);

    // Allocation
    size_t const size_pi = 4 * 4 * VOLUME;
    size_t const size_p23 = n_y * kernel_n * kernel_n_geom * 4 * 4 * 4;
    size_t const size_p1  = 4 * 4 * 4 * Lmax;

    check(cudaMalloc((void **)&Pi_d, size_pi * sizeof(double)));
    check(cudaMalloc((void **)&P23_d, size_p23 * sizeof(double)));
    check(cudaMalloc((void **)&P1_d, size_p1 * sizeof(double)));
    check(cudaMalloc((void **)&gycoords_d, sizeof(int) * 4 * n_y));

    // Async copies (using streams to overlap transfer if needed, though usually fast)
    //check(cudaMemcpy(fwd_y_d, fwd_y, size_fwd * sizeof(double), cudaMemcpyHostToDevice));
    check(cudaMemset(P1_d, 0, size_p1 * sizeof(double)));
    check(cudaMemset(P23_d, 0, size_p23 * sizeof(double)));
    check(cudaMemcpy(gycoords_d, gycoords, sizeof(int) * 4 * n_y, cudaMemcpyHostToDevice));

    /* --- 2. COMPUTE PI (Common Dependency) --- */
    // We launch in the DEFAULT stream. This creates an implicit barrier.
    // P23 and P1 streams will not start until this kernel finishes.
    dim3 gridPi(264);
    dim3 blockPi(4, 4, 12);
    kernel_pi<<<gridPi, blockPi>>>(fwd_y, Pi_d, iflavor, VOLUME);
    //cudaFree(fwd_y_d);
    
    cudaError_t err_pi = cudaGetLastError();
    if (err_pi != cudaSuccess) {
        printf("CUDA Error at Pi: %s\n", cudaGetErrorString(err_pi));
    }
    cudaDeviceSynchronize(); // Force check

    /* --- 3. CONCURRENT KERNEL LAUNCH --- */

    
    // --- Stream x3: P23 ---
    dim3 gridP23(88, kernel_n);
    dim3 blockP23(256);
    kernel_p20<<<gridP23, blockP23, 0, stream_p20>>>(Pi_d, P23_d, n_y, gsw, gycoords_d, xunit, kqed_t, VOLUME, g_proc_coords, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);
    kernel_p21<<<gridP23, blockP23, 0, stream_p21>>>(Pi_d, P23_d, n_y, gsw, gycoords_d, xunit, kqed_t, VOLUME, g_proc_coords, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);
    kernel_p3<<<gridP23, blockP23, 0, stream_p3>>>(Pi_d,P23_d,n_y,gsw ,gycoords_d,xunit,kqed_t,VOLUME,g_proc_coords,T,LX ,LY,LZ,T_global,LX_global ,LY_global,LZ_global);
    cudaMemcpyAsync(P23, P23_d, size_p23/3 * sizeof(double), cudaMemcpyDeviceToHost, stream_p20);
    cudaMemcpyAsync(P23 + size_p23/3, P23_d + size_p23/3, size_p23/3 * sizeof(double), cudaMemcpyDeviceToHost, stream_p21);
    cudaMemcpyAsync(P23 + 2*size_p23/3, P23_d + 2*size_p23/3, size_p23/3 * sizeof(double), cudaMemcpyDeviceToHost, stream_p3);    

        // --- Stream x1: P1 ---
    dim3 gridP1(4, 4, 4);
    dim3 blockP1(128);
    kernel_p1<<<gridP1, blockP1, 0, stream_p1>>>(Pi_d, P1_d, iflavor, Lmax, gsw, VOLUME, g_proc_coords, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);
    cudaMemcpyAsync(P1, P1_d, size_p1 * sizeof(double), cudaMemcpyDeviceToHost, stream_p1);


    /* --- 4. MPI REDUCTION --- */
    MPI_Request request[4];    

    cudaStreamSynchronize(stream_p20);
    if(MPI_Iallreduce(MPI_IN_PLACE, P23, size_p23/3, MPI_DOUBLE, MPI_SUM, g_cart_grid, &request[0]) != MPI_SUCCESS) {
        fprintf(stderr, "[] Error from MPI_Iallreduce %s %d\n", __FILE__, __LINE__ );
        MPI_Abort(g_cart_grid, -1);
    }

    cudaStreamSynchronize(stream_p21);
    if(MPI_Iallreduce(MPI_IN_PLACE, P23+size_p23/3, size_p23/3, MPI_DOUBLE, MPI_SUM, g_cart_grid, &request[1]) != MPI_SUCCESS) {
        fprintf(stderr, "[] Error from MPI_Iallreduce %s %d\n", __FILE__, __LINE__ );
        MPI_Abort(g_cart_grid, -1);
    }

    cudaStreamSynchronize(stream_p3);
    if(MPI_Iallreduce(MPI_IN_PLACE, P23+2*size_p23/3, size_p23/3, MPI_DOUBLE, MPI_SUM, g_cart_grid, &request[2]) != MPI_SUCCESS) {
        fprintf(stderr, "[] Error from MPI_Iallreduce %s %d\n", __FILE__, __LINE__ );
        MPI_Abort(g_cart_grid, -1);
    }

    cudaStreamSynchronize(stream_p1);
    if(MPI_Iallreduce(MPI_IN_PLACE, P1, size_p1, MPI_DOUBLE, MPI_SUM, g_cart_grid, &request[3]) != MPI_SUCCESS) {
        fprintf(stderr, "[] Error from MPI_Iallreduce %s %d\n", __FILE__, __LINE__ );
        MPI_Abort(g_cart_grid, -1);
    }

    MPI_Waitall(4, request, MPI_STATUSES_IGNORE);

    cudaFree(Pi_d);
    cudaFree(P23_d);
    cudaFree(P1_d);
    cudaFree(gycoords_d);
    cudaStreamDestroy(stream_p1);
    cudaStreamDestroy(stream_p20);
    cudaStreamDestroy(stream_p21);
    cudaStreamDestroy(stream_p3);
}

// compare pi computed on cpu and gpu 
// write pi into file named pi_cuda.dat
__host__ void record_pi_cuda(double *fwd_y, int VOLUME, int iflavor, unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global) {
    double *pi_gpu = (double *)malloc(4 * 4 * VOLUME * sizeof(double));

    /* set up problem on gpu */
    double* pi_d, * fwd_y_d;
    check(cudaMalloc((void **) &pi_d, sizeof(double) * 4 * 4 * VOLUME));
    check(cudaMalloc((void **) &fwd_y_d, sizeof(double)* 2 * 12 * 24 * VOLUME));
    check(cudaMemcpy(fwd_y_d, fwd_y, sizeof(double)* 2 * 12 * 24 * VOLUME, cudaMemcpyHostToDevice));

    dim3 gridDim(264);
    dim3 blockDim(4, 4, 4);

    // call gpu code
    kernel_pi<<<gridDim, blockDim>>>(fwd_y_d, pi_d, iflavor, VOLUME);
    check(cudaMemcpy(pi_gpu, pi_d, 4 * 4 * VOLUME * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(fwd_y_d);
    cudaFree(pi_d);

    // write to file
    /*FILE *file;
    for (int i=0; i< 16 * VOLUME; i++) {
        file = fopen("pi_cuda.dat", "a");
        fprintf(file, "%.10e\n", pi_gpu[i]);
        fclose(file);
    }*/
    free(pi_gpu);
    return;
}

// write p1 into file named p1_cuda.dat
__host__ void record_p1_cuda(double *Pi, int iflavor, int const * gsw, int VOLUME, 
    unsigned T, unsigned LX, unsigned LY, unsigned LZ, 
    unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global) {
    // set up on gpu
    int const Lmax = get_max(T_global, LX_global, LY_global, LZ_global);
    double *Pi_d, *P1_d;
    cudaMalloc((void **)&Pi_d, sizeof(double) * 16 * VOLUME);
    cudaMemcpy(Pi_d, Pi, sizeof(double) * 16 * VOLUME, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&P1_d, sizeof(double) * 64 * T_global);
    cudaMemset(P1_d, 0, 64 * T_global * sizeof(double));

    dim3 gridDim(4, 4, 4);
    dim3 blockDim(64);

    // compute p1 on gpu
    int const g_proc_coords[4] = {0,0,0,0}; // dummy value for testing
    kernel_p1<<<gridDim, blockDim>>>(Pi_d, P1_d, iflavor, Lmax, gsw, VOLUME, g_proc_coords, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);
    // copy back to host
    double *P1 = (double *)malloc(sizeof(double) * 64 * T_global);
    cudaMemcpy(P1, P1_d, sizeof(double) * 64 * T_global, cudaMemcpyDeviceToHost);
    cudaFree(Pi_d);
    cudaFree(P1_d);

    // write to file
    /*FILE *file;
    for (int i=0; i< 64 * T_global; i++) {
        file = fopen("p1_cuda.dat", "a");
        fprintf(file, "%.10e\n", P1[i]);
        fclose(file);
    }*/
    free(P1);
    return;
}

__host__ void record_p23_cuda(double *Pi, int n_y, const int gsw[4], const int *gycoords, const double xunit[2], QED_kernel_temps kqed_t, unsigned VOLUME, unsigned T, unsigned LX, unsigned LY, unsigned LZ, unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global) {
    // set up on gpu
    double *Pi_d, *P23_d;
    cudaMalloc((void **)&Pi_d, sizeof(double) * 16 * VOLUME);
    cudaMemcpy(Pi_d, Pi, sizeof(double) * 16 * VOLUME, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&P23_d, sizeof(double) * n_y * kernel_n * kernel_n_geom * 4 * 4 *4);
    cudaMemset(P23_d, 0, n_y * kernel_n * kernel_n_geom * 4 * 4 *4 * sizeof(double));
    int *gycoords_d;
    cudaMalloc((void **)&gycoords_d, sizeof(int) * 4 * n_y);
    cudaMemcpy(gycoords_d, gycoords, sizeof(int) * 4 * n_y, cudaMemcpyHostToDevice);


    dim3 gridDim(88, 3);
    dim3 blockDim(256);

    // compute p23 on gpu
    int const g_proc_coords[4] = {0,0,0,0}; // dummy value for testing
    kernel_p20<<<gridDim, blockDim>>>(Pi_d, P23_d, n_y, gsw, gycoords_d, xunit, kqed_t, VOLUME, g_proc_coords, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);
    kernel_p21<<<gridDim, blockDim>>>(Pi_d, P23_d, n_y, gsw, gycoords_d, xunit, kqed_t, VOLUME, g_proc_coords, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);
    kernel_p3<<<gridDim, blockDim>>>(Pi_d, P23_d, n_y, gsw, gycoords_d, xunit, kqed_t, VOLUME, g_proc_coords, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);

    // copy back to host
    double *P23 = (double *)malloc(sizeof(double) * n_y * kernel_n * kernel_n_geom * 4 * 4 *4);
    cudaMemcpy(P23, P23_d, sizeof(double) * n_y * kernel_n * kernel_n_geom * 4 * 4 *4, cudaMemcpyDeviceToHost);
    cudaFree(Pi_d);
    cudaFree(P23_d);
    cudaFree(gycoords_d);

    // write to file
    FILE *file;
    for (int i=0; i< n_y * kernel_n * kernel_n_geom * 4 * 4 *4; i++) {
        file = fopen("p23_cuda.dat", "a");
        fprintf(file, "%.10e\n", P23[i]);
        fclose(file);
    }
    free(P23);
    return;
}


__host__ void record_2p2_cuda(double *fwd_y, double *P1, double *P23, int iflavor, 
     int const gsw[4], const int *gycoords, int n_y, const double xunit[2], 
     QED_kernel_temps kqed_t,  unsigned VOLUME, int const g_proc_coords[4], MPI_Comm g_cart_grid,
     unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global, 
     unsigned T, unsigned LX, unsigned LY, unsigned LZ) {
    
    compute_2p2_gpu(fwd_y, P1, P23, iflavor, gsw, gycoords, n_y, xunit, kqed_t, VOLUME, g_proc_coords, g_cart_grid, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);

    // write to file
    if (g_proc_coords[0]==0 && g_proc_coords[1]==0 && g_proc_coords[2]==0 && g_proc_coords[3]==0) {

        FILE *file23;
        for (int i=0; i< n_y * kernel_n * kernel_n_geom * 4 * 4 *4; i++) {
            file23 = fopen("2p2_p23_cuda.dat", "a");
            fprintf(file23, "%.10e\n", P23[i]);
            fclose(file23);
        }
        FILE *file1;
        for (int i=0; i< 64 * T_global; i++) {
            file1 = fopen("2p2_p1_cuda.dat", "a");
            fprintf(file1, "%.10e\n", P1[i]);
            fclose(file1);
        }
    }
    return;
}

union array {
    double dxu[4][12][24];
    double kerv[3][3][6][4][4][4] KQED_ALIGN;
};

__device__ inline void kqed_set_zero(double kerv[6][4][4][4]) {
    for (int i=0; i<6; i++)
    for (int mu=0; mu<4; mu++)
    for (int nu=0; nu<4; nu++)
    for (int lambda=0; lambda<4; lambda++)
        kerv[i][mu][nu][lambda] = 0.;
}

__global__ void kernel_4pt(const double * fwd_src, const double * fwd_y,
    double const* g_dzu, double const* g_dzsu,
    const int* gsx, int iflavor, const double xunit[2], const int yv[4],
    double* kernel_sum, QED_kernel_temps kqed_t, unsigned VOLUME,
    int const g_proc_coords[4], MPI_Comm g_cart_grid, unsigned T, unsigned LX, unsigned LY, unsigned LZ, 
    unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global)
{
    // Shared memory for intermediate results. Total size: 23KB < 48KB limit
    __shared__ double dxu[4][12][24];
    __shared__ double kerv[3][3][6][4][4][4] KQED_ALIGN;
    double sum[3] = {0.};

    // One lattice point per block
    for ( unsigned int ix = blockIdx.x; ix < VOLUME; ix+=gridDim.x ){      
        int const x[4] = {(ix / (LX * LY * LZ)  + g_proc_coords[0] * T - gsx[0] + T_global) % T_global,
        (ix / (LY * LZ) % LX + g_proc_coords[1] * LX - gsx[1] + LX_global) % LX_global,
        ((ix / LZ) % LY + g_proc_coords[2] * LY - gsx[2] + LY_global) % LY_global,
        (ix % LZ + g_proc_coords[3] * LZ - gsx[3] + LZ_global) % LZ_global};

        int xv[4], xvzh[4];
        site_map ( xv, x, T_global, LX_global, LY_global, LZ_global);
        site_map_zerohalf ( xvzh, x, T_global, LX_global, LY_global, LZ_global);

        double const xm[4] = {
        xv[0] * xunit[0],
        xv[1] * xunit[0],
        xv[2] * xunit[0],
        xv[3] * xunit[0] };

        /* yv comes from the caller: fold it into [0, L) and map it here, so ym
         * is in [-L/2, L/2) whatever the caller passed */
        int const L[4] = { static_cast<int>(T_global), static_cast<int>(LX_global),
                           static_cast<int>(LY_global), static_cast<int>(LZ_global) };
        int const y[4] = {
            (yv[0] % L[0] + L[0]) % L[0],
            (yv[1] % L[1] + L[1]) % L[1],
            (yv[2] % L[2] + L[2]) % L[2],
            (yv[3] % L[3] + L[3]) % L[3] };
        int yv_map[4];
        site_map ( yv_map, y, T_global, LX_global, LY_global, LZ_global);

        double const ym[4] = {
        yv_map[0] * xunit[0],
        yv_map[1] * xunit[0],
        yv_map[2] * xunit[0],
        yv_map[3] * xunit[0] };

        /* double const xm_mi_ym[4] = {
        xm[0] - ym[0],
        xm[1] - ym[1],
        xm[2] - ym[2],
        xm[3] - ym[3] }; */

        int const x_mi_y[4] = {
            (x[0] - y[0] + T_global) % T_global,
            (x[1] - y[1] + LX_global) % LX_global,
            (x[2] - y[2] + LY_global) % LY_global,
            (x[3] - y[3] + LZ_global) % LZ_global };
        int xv_mi_yv[4];
        site_map(xv_mi_yv, x_mi_y, T_global, LX_global, LY_global, LZ_global);

        double const xm_mi_ym[4] = {
            xv_mi_yv[0] * xunit[0],
            xv_mi_yv[1] * xunit[0],
            xv_mi_yv[2] * xunit[0],
            xv_mi_yv[3] * xunit[0] };
            
        if (threadIdx.x < 48){
            int const mu = threadIdx.x / 12;
            int const ia = threadIdx.x % 12;
            double u[24]; //u[mu][ia]
            for (int i=0; i<24; i++) u[i] = fwd_y[prop_idx(iflavor, ia, ix, i, VOLUME)];
            _fv_ti_eq_g5(u);
            double gu[24];
            _fv_eq_gamma_ti_fv(gu, mu, u);

            for (int ib=0; ib<12; ib++) {
                complex w;
                _co_eq_fv_dag_ti_fv( &w, fwd_src + prop_idx(1-iflavor, ib, ix, 0, VOLUME), gu);
                dxu[mu][ia][ib*2] = w.re;
                dxu[mu][ia][ib*2+1] = w.im;
            }
            _fv_ti_eq_g5(dxu[mu][ia]);
        }

        else if (threadIdx.x < 48 + 9) {
            int const ikernel = (threadIdx.x - 48) / 3;
            int const idx = (threadIdx.x - 48) % 3;
            if (idx == 0) {
                //set zero
                KQED_LX(ikernel, xm, ym, kqed_t, kerv[ikernel][0]);
            }
            else if (idx == 1) {
                KQED_LX(ikernel,  ym, xm, kqed_t, kerv[ikernel][1]);
            }
            else {
                KQED_LX(ikernel,  xm, xm_mi_ym, kqed_t, kerv[ikernel][2]);
            }
        }

        __syncthreads();
        int const k = threadIdx.x / 16;
        int const nu = (threadIdx.x % 16) / 4;
        int const lambda = threadIdx.x % 4;
        for (int mu=0; mu<4; mu++) {
            double g_dxu[12][24];
            for (int ia=0; ia<12; ia++) {
                _fv_eq_gamma_ti_fv(g_dxu[ia], lambda, dxu[mu][ia]);
            }
            
            double tr1=0.; // the real trace
            for (int ia=0; ia<12; ia++)
            for (int ib=0; ib<12; ib++) {
                double const u[2] = {g_dxu[ia][ib*2], g_dxu[ia][ib*2+1]};
                double const v[2] = {g_dzu[k*4*12*24 + nu*12*24 + ib*24 + ia*2], g_dzu[k*4*12*24 + nu*12*24 + ib*24 + ia*2+1]};
                tr1 += u[0] * v[0] - u[1] * v[1];
            }
            double corr_I = tr1;
        
            double tr2 = 0.; // the real trace
            int const rho = idx_comb_d[k][0];
            int const sigma = idx_comb_d[k][1];
            for (int ia=0; ia<12; ia++)
            for (int ib=0; ib<12; ib++) {
                complex const factor = {xvzh[rho] * g_dzsu[sigma*4*12*24 + nu *12*24 + ib*24 + 2*ia] - xvzh[sigma] * g_dzsu[rho*4*12*24 + nu *12*24 + ib*24 + 2*ia],
                            xvzh[rho] * g_dzsu[sigma*4*12*24 + nu *12*24 + ib*24 + 2*ia+1] - xvzh[sigma] * g_dzsu[rho*4*12*24 + nu*12*24 + ib*24 + 2*ia+1]};
                tr2 += g_dxu[ia][ib*2] * factor.re - g_dxu[ia][ib*2+1] * factor.im;
            }
            double corr_II = tr2;

            //contract with QED kernel
            for (int ikernel=0; ikernel<kernel_n; ikernel++){
                sum[ikernel] += (kerv[ikernel][0][k][mu][nu][lambda] + kerv[ikernel][1][k][nu][mu][lambda] - kerv[ikernel][2][k][lambda][nu][mu]) * corr_I
                    + kerv[ikernel][2][k][lambda][nu][mu] * corr_II;
            }
        }
        __syncthreads();
    }

    // reduce sum across threads
    for (int i=0; i<kernel_n; i++) {
        double const sum_block = blockReduceSum(sum[i]);
        if (threadIdx.x == 0) atomicAdd_system(&kernel_sum[i], sum_block);
        /* blockReduceSum reuses one static __shared__ buffer: the next call
         * must not write it while warp 0 is still reading it */
    }
}

__host__ void compute_4pt_gpu(
    const double * fwd_src, const double * fwd_y,
    double const *g_dzu, double const *g_dzsu,
    const int* gsx, int iflavor, const double xunit[2], const int yv[4],
    double* kernel_sum, QED_kernel_temps kqed_t, unsigned VOLUME,
    int const g_proc_coords[4], MPI_Comm g_cart_grid, unsigned T, unsigned LX, unsigned LY, unsigned LZ, 
    unsigned T_global, unsigned LX_global, unsigned LY_global, unsigned LZ_global)
{    
    //for (int i=0; i<kernel_n; i++) kernel_sum[i] = 0.;

    double *g_dxu_d, *kernel_sum_d, *g_dzu_d, *g_dzsu_d;
    cudaMalloc((void **)&g_dxu_d, sizeof(double) * 4*4*12*24);
    cudaMalloc((void **)&g_dzu_d, sizeof(double) *6*4*12*24);
    cudaMalloc((void **)&g_dzsu_d, sizeof(double) *6*4*12*24);
    cudaMalloc((void **)&kernel_sum_d, sizeof(double) * kernel_n);
    cudaMemcpy(g_dzu_d, g_dzu, sizeof(double)*6*4*12*24, cudaMemcpyHostToDevice);
    cudaMemcpy(g_dzsu_d, g_dzsu, sizeof(double)*6*4*12*24, cudaMemcpyHostToDevice);
    cudaMemset(kernel_sum_d, 0, sizeof(double) * kernel_n);

    dim3 gridDim(264);
    dim3 blockDim(96);

    kernel_4pt<<<gridDim, blockDim>>>(fwd_src, fwd_y, g_dzu_d, g_dzsu_d, gsx, iflavor, xunit, yv, kernel_sum_d, kqed_t, VOLUME,
    g_proc_coords, g_cart_grid, T, LX, LY, LZ, T_global, LX_global, LY_global, LZ_global);

    cudaDeviceSynchronize();
    cudaMemcpy(kernel_sum, kernel_sum_d, sizeof(double) * kernel_n, cudaMemcpyDeviceToHost);
    cudaFree(g_dxu_d);
    cudaFree(g_dzu_d);
    cudaFree(g_dzsu_d);
    cudaFree(kernel_sum_d);
    return;
  }


__global__ void static call_KQED(const double xv[4], const double yv[4], QED_kernel_temps kqed_t, double kerv[4][4][4][4]) {
    if (threadIdx.x == 0) {
         QED_Mkernel_L2( 0.4, xv, yv, kqed_t, kerv);
    }
    return;
}

__host__ void test_KQED_on_gpu(const double xv[4], const double yv[4], QED_kernel_temps kqed_t) {
    double (*kerv)[4][4][4];
    cudaMalloc((void **)&kerv, sizeof(double)*4*4*4*4);
    call_KQED<<<1,1>>>(xv, yv, kqed_t, kerv);
    double h_kerv[4][4][4][4];
    cudaMemcpy(h_kerv, kerv, sizeof(double)*4*4*4*4, cudaMemcpyDeviceToHost);
    cudaFree(kerv);
    // print norm
    double norm = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                for (int l = 0; l < 4; l++) {
                    norm += h_kerv[i][j][k][l] * h_kerv[i][j][k][l];
                }
            }
        }
    }
    printf("Norm of kerv = %e\n", norm);
}