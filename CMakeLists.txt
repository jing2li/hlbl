cmake_minimum_required(VERSION 3.24)

# must come before project()
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

enable_testing()

find_package(MPI REQUIRED COMPONENTS C CXX)
find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

# tmLQCDConfig.cmake calls find_dependency() without including this
include(CMakeFindDependencyMacro)
# tmlqcd::tmlqcd brings lemon, c-lime and QUDA, but tmLQCDConfig.cmake looks
# them up again instead of recording where tmLQCD found them, so
# CMAKE_PREFIX_PATH must hold the prefixes tmLQCD was configured with
find_package(tmLQCD REQUIRED)

# ---------------------------------------------------------------------------
# KQED_cuda: an installed GPU build of KQED (https://github.com/RJHudspith/KQED),
# the prefix of its `configure --with-cuda && make install`. The library has
# relocatable device code and reads its tables from LIB_PATH/PRECOMP at run
# time, where LIB_PATH is compiled in (include/config.h).
# ---------------------------------------------------------------------------
set(KQED_ROOT "" CACHE PATH "KQED GPU install prefix (lib/libKQED_cuda.a, include/KQED.h)")
find_library(KQED_CUDA_LIBRARY KQED_cuda PATHS ${KQED_ROOT}/lib NO_DEFAULT_PATH REQUIRED)
find_path(KQED_INCLUDE_DIR KQED.h PATHS ${KQED_ROOT}/include NO_DEFAULT_PATH REQUIRED)

file(STRINGS ${KQED_INCLUDE_DIR}/config.h KQED_LIB_PATH REGEX "^#define LIB_PATH ")
string(REGEX REPLACE "^#define LIB_PATH \"(.*)\"$" "\\1" KQED_LIB_PATH "${KQED_LIB_PATH}")
if (NOT EXISTS "${KQED_LIB_PATH}/PRECOMP/FFxy_single_cksum.bin")
  message(FATAL_ERROR "KQED reads its tables from ${KQED_LIB_PATH}/PRECOMP, which is missing")
endif()

add_library(KQED_cuda STATIC IMPORTED)
set_target_properties(KQED_cuda PROPERTIES
  IMPORTED_LOCATION ${KQED_CUDA_LIBRARY}
  INTERFACE_INCLUDE_DIRECTORIES ${KQED_INCLUDE_DIR}
  INTERFACE_LINK_LIBRARIES "CUDA::cudart;CUDA::cuda_driver")

# ---------------------------------------------------------------------------
# GPU kernels from cvc (kernels/VENDORED.md). Must be OBJECT, not STATIC: the
# kernels call device code in KQED_cuda, and nvlink can lose device code split
# over two static archives (undefined __cudaRegisterLinkedBinary_* symbols).
# Every executable links it directly (hlbl_executable()). hlbl_core must not
# link it: that also archives the objects into libhlbl_core.a, and nvlink
# then finds them twice (multiple definition).
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
# hlbl_core: everything but the solver. Needs CUDA, MPI, KQED and HDF5.
# ---------------------------------------------------------------------------
add_library(hlbl_core STATIC
  src/gamma.cpp
  src/input.cpp
  src/schedule.cpp
  src/geometry.cpp
  src/contract.cpp
  src/provider.cpp
  src/propagator_cache.cpp
  src/output.cpp
  src/driver.cpp)
target_include_directories(hlbl_core PUBLIC include kernels)
target_link_libraries(hlbl_core PUBLIC KQED_cuda MPI::MPI_CXX CUDA::cudart HDF5::HDF5)

# ---------------------------------------------------------------------------
# hlbl_solver: the tmLQCD/QUDA interface.
# ---------------------------------------------------------------------------
add_library(hlbl_solver STATIC src/solver.cpp)
target_link_libraries(hlbl_solver PUBLIC hlbl_core tmlqcd::tmlqcd)

# ---------------------------------------------------------------------------
# Executables
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
target_link_libraries(hlbl_hypercube PRIVATE hlbl_solver)

hlbl_test(test_schedule)
hlbl_test(test_input)
hlbl_test(test_gamma)
hlbl_test(test_lattice)
hlbl_test(test_contract_random)
hlbl_test(test_driver_random)

# check_pair needs a real gauge configuration, so it is not a ctest case.
add_executable(check_pair sanity_check/check_pair.cpp)
target_link_libraries(check_pair PRIVATE hlbl_solver hlbl_kernels_obj)
set_target_properties(check_pair PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# The whole program on random fields: one chain (1 shift: 15 + 65 = 80 pairs,
# device cache) and two chains (80 + 145 pairs, host cache, 2+2 at 3 sources
# with 6 targets)
set(HLBL_TEST_INPUTS ${CMAKE_CURRENT_SOURCE_DIR}/sanity_check)
add_test(NAME hypercube_random_fields
  COMMAND hlbl_hypercube --in ${HLBL_TEST_INPUTS}/one_chain.input)
set_tests_properties(hypercube_random_fields PROPERTIES
  PASS_REGULAR_EXPRESSION "done: 80 pairs")

add_test(NAME hypercube_random_fields_multi
  COMMAND hlbl_hypercube --in ${HLBL_TEST_INPUTS}/multi_chain.input)
set_tests_properties(hypercube_random_fields_multi PROPERTIES
  PASS_REGULAR_EXPRESSION "done: 225 pairs \\(225 results\\) in 2 chains.*2\\+2: 3 sources, 6 targets")

add_test(NAME hypercube_dummy_run
  COMMAND hlbl_hypercube --dummy-run --in ${HLBL_TEST_INPUTS}/multi_chain.input)
set_tests_properties(hypercube_dummy_run PROPERTIES
  PASS_REGULAR_EXPRESSION "one 2\\+2 source \\(3 targets\\).*dummy run: all checks passed")

# The C++ schedules against the Python reference
hlbl_executable(schedule_dump sanity_check/schedule_dump.cpp)
add_test(NAME compare_schedule
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_schedule.sh
          $<TARGET_FILE:schedule_dump>)
