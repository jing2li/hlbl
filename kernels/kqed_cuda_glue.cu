/****************************************************************************
 * time_hlbl/kqed_cuda_glue.cu
 *
 * External definition of __checkCudaErrors for libKQED_cuda.a.
 *
 * deps/KQED/src_cuda/HEADERS/cu_util.h defines it as a plain C99 `inline`,
 * which by C99 6.7.4 gives init.c / io.c an inline definition only and no
 * external one.  Whenever the compiler does not inline the call -- always at
 * -O0, i.e. an empty CMAKE_BUILD_TYPE -- init.c.o and io.c.o are left with an
 * unresolved __checkCudaErrors and nothing in the tree provides it.  Link this
 * into every binary that pulls in KQED_cuda.
 ****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

extern "C" void __checkCudaErrors ( cudaError_t err, const char * file, const int line )
{
  if ( err != cudaSuccess )
  {
    fprintf ( stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
              (int)err, cudaGetErrorString ( err ) );
    exit ( EXIT_FAILURE );
  }
}
