cmake_minimum_required(VERSION 3.24)

# must precede project(), which otherwise sets the compiler's default
set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "CUDA architectures (90 = GH200)")

project(hlbl
  DESCRIPTION "Connected HLbL kernel sum on a sliding-hypercube schedule"
  LANGUAGES C CXX CUDA)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(HLBL_WITH_TMLQCD "Build the solver interface (needs tmLQCD and QUDA)" ON)

enable_testing()

find_package(MPI REQUIRED COMPONENTS C CXX)
find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

add_subdirectory(deps/kqed)

# ---------------------------------------------------------------------------
# hlbl_core: lattice geometry, gamma tables and the hypercube schedule.
# No CUDA or KQED, only MPI (Lattice's Cartesian communicator), so this
# builds and its tests run even with -DHLBL_WITH_TMLQCD=OFF and without a
# GPU.
# ---------------------------------------------------------------------------
add_library(hlbl_core STATIC
  src/gamma.cpp
  src/schedule.cpp
  src/geometry.cpp)
target_include_directories(hlbl_core PUBLIC include)
target_link_libraries(hlbl_core PUBLIC MPI::MPI_CXX)

function(hlbl_core_test name)
  add_executable(${name} sanity_check/${name}.cpp)
  target_link_libraries(${name} PRIVATE hlbl_core)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

hlbl_core_test(test_schedule)
hlbl_core_test(test_gamma)
hlbl_core_test(test_lattice)

# ---------------------------------------------------------------------------
# Vendored production kernels (see kernels/VENDORED.md)
# ---------------------------------------------------------------------------
add_library(hlbl_kernels STATIC
  kernels/cuda_lattice.cu
  kernels/hlbl_kernels/kernels.cu
  kernels/kqed_cuda_glue.cu)
target_include_directories(hlbl_kernels PUBLIC kernels kernels/hlbl_kernels)
target_link_libraries(hlbl_kernels PUBLIC KQED_cuda MPI::MPI_CXX CUDA::cudart)
set_target_properties(hlbl_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# tmLQCD, which brings QUDA, lemon and c-lime
# ---------------------------------------------------------------------------
if (HLBL_WITH_TMLQCD)
  # tmLQCDConfig.cmake calls find_dependency() without including this
  include(CMakeFindDependencyMacro)
  find_package(tmLQCD REQUIRED)
endif()

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
function(hlbl_executable name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE hlbl_core hlbl_kernels)
  # the kernels call __device__ functions in KQED_cuda's relocatable device
  # code, so the device link happens in every executable
  set_target_properties(${name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endfunction()

hlbl_executable(hlbl_hypercube bin/hlbl_hypercube.cpp)

add_test(NAME compare_schedule
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_schedule.sh
          $<TARGET_FILE:hlbl_hypercube>)
