# ---------------------------------------------------------------------------
# KQED_cuda: the GPU build of the position-space QED kernel library
# (https://github.com/RJHudspith/KQED), which only ships an autotools build.
#
# KQED_SOURCE_DIR points at a KQED checkout. When it is empty the pinned
# revision is cloned into the build tree. The checkout must stay in place
# after the build: KQED reads its precomputed tables from
# ${KQED_SOURCE_DIR}/PRECOMP at run time, and that path is compiled in.
# ---------------------------------------------------------------------------

include(CheckIncludeFile)
include(TestBigEndian)
include(FetchContent)

set(KQED_GIT_TAG deb4d488e4394d735d74e5deb689995b76053cd2)

set(KQED_SOURCE_DIR "" CACHE PATH "KQED checkout (empty: clone ${KQED_GIT_TAG})")
if (NOT KQED_SOURCE_DIR)
  FetchContent_Declare(kqed
    GIT_REPOSITORY https://github.com/RJHudspith/KQED.git
    GIT_TAG ${KQED_GIT_TAG}
    # single-threaded pack handling: the default thread pool fails under
    # the per-user thread limit of the login nodes
    GIT_CONFIG pack.threads=1 index.threads=1)
  FetchContent_MakeAvailable(kqed)
  set(KQED_SOURCE_DIR ${kqed_SOURCE_DIR} CACHE PATH "" FORCE)
endif()

if (NOT EXISTS "${KQED_SOURCE_DIR}/src_cuda/LIBFILES/KQED.cu")
  message(FATAL_ERROR "KQED_SOURCE_DIR=${KQED_SOURCE_DIR} is not a KQED checkout")
endif()

# KQED.h includes "config.h", normally produced by autotools
check_include_file(immintrin.h HAVE_IMMINTRIN_H)
check_include_file(omp.h HAVE_OMP_H)
check_include_file(sys/time.h HAVE_SYS_TIME_H)
check_include_file(time.h HAVE_TIME_H)
test_big_endian(WORDS_BIGENDIAN)
configure_file(config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h @ONLY)

# KQED.cu is a unity build that #includes the per-kernel .cu files itself,
# so those must not be compiled separately.
add_library(KQED_cuda STATIC
  ${KQED_SOURCE_DIR}/src_cuda/LIBFILES/KQED.cu
  ${KQED_SOURCE_DIR}/src_cuda/LIBFILES/init.c
  ${KQED_SOURCE_DIR}/src_cuda/LIBFILES/io.c
  ${KQED_SOURCE_DIR}/src_cuda/UTILS/GLU_timer.c
  ${KQED_SOURCE_DIR}/src_cuda/UTILS/GLU_bswap.c
  ${KQED_SOURCE_DIR}/src_cuda/UTILS/crc32c.c)

target_include_directories(KQED_cuda PUBLIC
  ${KQED_SOURCE_DIR}/src_cuda/HEADERS
  ${CMAKE_CURRENT_BINARY_DIR})

# KQED annotates __device__ functions with KQED_PRIVATE without defining it;
# the relocatable device code needs them non-static, so it expands to nothing
target_compile_definitions(KQED_cuda PRIVATE KQED_PRIVATE=)

# io.c passes double** where void** is expected, which GCC >= 14 rejects
target_compile_options(KQED_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:C>:-Wno-error=incompatible-pointer-types>)

target_link_libraries(KQED_cuda PUBLIC CUDA::cudart CUDA::cuda_driver)
set_target_properties(KQED_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON)
