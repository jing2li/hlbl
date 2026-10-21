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

option(HLBL_WITH_TMLQCD "Build the solver interface (needs tmLQCD and QUDA)" ON)

enable_testing()

find_package(MPI REQUIRED COMPONENTS C CXX)
find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

add_subdirectory(deps/kqed)

# ---------------------------------------------------------------------------
# GPU kernels from cvc (kernels/VENDORED.md). Must be OBJECT, not STATIC: the
# kernels call device code in KQED_cuda, and nvlink can lose device code split
# over two static archives (undefined __cudaRegisterLinkedBinary_* symbols).
# Every executable links it directly (hlbl_executable()).
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
target_include_directories(hlbl_core PUBLIC include)
target_link_libraries(hlbl_core PUBLIC hlbl_kernels_obj HDF5::HDF5)

# Git revision for the result files; re-run cmake after committing.
execute_process(
  COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} describe --always --dirty --abbrev=12
  OUTPUT_VARIABLE HLBL_GIT_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if (NOT HLBL_GIT_REVISION)
  set(HLBL_GIT_REVISION unknown)
endif()
set_source_files_properties(src/output.cpp PROPERTIES
  COMPILE_DEFINITIONS HLBL_GIT_REVISION="${HLBL_GIT_REVISION}")

# ---------------------------------------------------------------------------
# hlbl_solver: the tmLQCD/QUDA interface.
# ---------------------------------------------------------------------------
if (HLBL_WITH_TMLQCD)
  # tmLQCDConfig.cmake calls find_dependency() without including this
  include(CMakeFindDependencyMacro)
  find_package(tmLQCD REQUIRED)

  add_library(hlbl_solver STATIC src/solver.cpp)
  target_link_libraries(hlbl_solver PUBLIC hlbl_core tmlqcd::tmlqcd)
endif()

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
if (HLBL_WITH_TMLQCD)
  target_link_libraries(hlbl_hypercube PRIVATE hlbl_solver)
  target_compile_definitions(hlbl_hypercube PRIVATE HLBL_HAVE_SOLVER)
endif()

hlbl_test(test_schedule)
hlbl_test(test_input)
hlbl_test(test_gamma)
hlbl_test(test_lattice)
hlbl_test(test_contract_random)
hlbl_test(test_driver_random)

# check_pair needs a real gauge configuration, so it is not a ctest case.
if (HLBL_WITH_TMLQCD)
  add_executable(check_pair sanity_check/check_pair.cpp)
  target_link_libraries(check_pair PRIVATE hlbl_solver hlbl_kernels_obj)
  set_target_properties(check_pair PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# The whole program on random fields (1 shift: 15 + 65 = 80 pairs)
add_test(NAME hypercube_random_fields
  COMMAND hlbl_hypercube --random-fields --lattice 6 6 6 6 --stride 1 --steps 1
          --muon-mass 105.6583745 --lattice-spacing 0.07957
          --out hypercube_random_fields.h5)
set_tests_properties(hypercube_random_fields PROPERTIES
  PASS_REGULAR_EXPRESSION "done: 80 pairs")

# The checks of a dummy run, on random fields
add_test(NAME hypercube_dummy_run
  COMMAND hlbl_hypercube --dummy-run --in ${CMAKE_CURRENT_SOURCE_DIR}/sanity_check/multi_chain.input)
set_tests_properties(hypercube_dummy_run PROPERTIES
  PASS_REGULAR_EXPRESSION "dummy run: all checks passed")

# Two chains from an input file (80 + 130 pairs; the second starts at window 1)
add_test(NAME hypercube_random_fields_multi
  COMMAND hlbl_hypercube --in ${CMAKE_CURRENT_SOURCE_DIR}/sanity_check/multi_chain.input
          --out hypercube_random_fields_multi.h5)
set_tests_properties(hypercube_random_fields_multi PROPERTIES
  PASS_REGULAR_EXPRESSION "done: 210 pairs \\(210 results\\) in 2 chains")

add_test(NAME compare_schedule
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_schedule.sh
          $<TARGET_FILE:hlbl_hypercube>)
