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
# Vendored production kernels (see kernels/VENDORED.md), as an OBJECT
# library rather than a STATIC one.
#
# cuda_lattice.cu calls device functions defined in KQED_cuda's own
# relocatable device code, so at the final executable's device-link step
# nvlink must see both. nvlink does not reliably extract device code across
# two cross-referencing STATIC archives -- it silently drops it, leaving
# __cudaRegisterLinkedBinary_* undefined at the host link, and only once
# something actually pulls the affected archive member in. An OBJECT
# library sidesteps this: its .o files are added directly to whichever
# executable links it, never archived, so nvlink again sees exactly what
# cvc's own time_hlbl/CMakeLists.txt sees for the same kernels: plain
# object files plus the one self-contained device archive, KQED_cuda.
#
# Must be linked directly by each final executable (see hlbl_executable()),
# never through an intermediate STATIC library -- that would recreate the
# same archive-in-archive problem one level up.
# ---------------------------------------------------------------------------
add_library(hlbl_kernels_obj OBJECT
  kernels/cuda_lattice.cu
  kernels/hlbl_kernels/kernels.cu
  kernels/kqed_cuda_glue.cu)
target_include_directories(hlbl_kernels_obj PUBLIC kernels kernels/hlbl_kernels)
target_link_libraries(hlbl_kernels_obj PUBLIC KQED_cuda MPI::MPI_CXX CUDA::cudart)
set_target_properties(hlbl_kernels_obj PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# hlbl_core: lattice geometry, gamma tables, the hypercube schedule, and the
# connected contraction. contract.cpp calls the vendored kernels but
# contains no device code of its own, so this stays a plain host archive --
# see hlbl_kernels_obj above for why it must NOT also embed that OBJECT
# library.
#
# Needs CUDA, MPI, KQED (for cuda_lattice.h/contract.cpp) and HDF5 (once
# output.cpp is added), but not tmLQCD -- unlike hlbl_solver, none of it
# needs a gauge field or a real solver.
# ---------------------------------------------------------------------------
add_library(hlbl_core STATIC
  src/gamma.cpp
  src/schedule.cpp
  src/geometry.cpp
  src/contract.cpp)
target_include_directories(hlbl_core PUBLIC include)
target_link_libraries(hlbl_core PUBLIC hlbl_kernels_obj)

# ---------------------------------------------------------------------------
# hlbl_solver: the tmLQCD/QUDA interface (gauge read, plaquette, point-source
# solves). Needs tmLQCD, which brings QUDA, lemon and c-lime; nothing here
# is needed by hlbl_core or the tests that don't solve.
# ---------------------------------------------------------------------------
if (HLBL_WITH_TMLQCD)
  # tmLQCDConfig.cmake calls find_dependency() without including this
  include(CMakeFindDependencyMacro)
  find_package(tmLQCD REQUIRED)

  add_library(hlbl_solver STATIC src/solver.cpp)
  target_link_libraries(hlbl_solver PUBLIC hlbl_core tmlqcd::tmlqcd)
endif()

# ---------------------------------------------------------------------------
# Executables. Every one links hlbl_kernels_obj directly (not only via
# hlbl_core) so its .o files are added to the link rather than archived --
# harmless if hlbl_core already pulls it in publicly, but explicit here
# because that propagation is exactly the property the comment above says
# not to rely on for a STATIC consumer.
# ---------------------------------------------------------------------------
function(hlbl_executable name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE hlbl_core hlbl_kernels_obj)
  set_target_properties(${name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endfunction()

function(hlbl_test name)
  hlbl_executable(${name} sanity_check/${name}.cpp)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

hlbl_executable(hlbl_hypercube bin/hlbl_hypercube.cpp)

hlbl_test(test_schedule)
hlbl_test(test_gamma)
hlbl_test(test_lattice)
hlbl_test(test_contract_random)

# check_pair needs a real gauge configuration and a solve that can take
# minutes to set up, so it is a manual tool (see its own top-of-file
# comment), not a ctest case.
if (HLBL_WITH_TMLQCD)
  add_executable(check_pair sanity_check/check_pair.cpp)
  target_link_libraries(check_pair PRIVATE hlbl_solver hlbl_kernels_obj)
  set_target_properties(check_pair PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

add_test(NAME compare_schedule
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_schedule.sh
          $<TARGET_FILE:hlbl_hypercube>)
