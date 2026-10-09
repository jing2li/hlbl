#!/bin/bash
# Configure and build hlbl. Run inside the programming environment:
#   uenv start --view=default prgenv-gnu/25.06:rc5
#   scripts/do_cmake.sh [build-dir]
#
# Libraries that end up linked into hlbl_hypercube:
#   tmLQCD, lemon, c-lime   tmLQCD_DIR, lemon_DIR, CMAKE_PREFIX_PATH
#   QUDA                    QUDA_DIR
#   KQED_cuda               built from KQED_SOURCE_DIR (revision deb4d488)
#   HDF5, MPI, CUDA         from the uenv
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build=${1:-$here/build}
S=/capstor/store/cscs/userlab/lp142

cmake -S "$here" -B "$build" \
  -DCMAKE_PREFIX_PATH=$S/software/install_dir/clime_pic \
  -DtmLQCD_DIR=$S/ljingjin/tmlqcd_install/lib64/cmake/tmlqcd \
  -DQUDA_DIR=$S/ljingjin/quda_install/lib/cmake/QUDA \
  -Dlemon_DIR=$S/ljingjin/tmlqcd_install/lib/cmake/lemon \
  -DKQED_SOURCE_DIR=/users/ljingjin/cvc_cmake/cvc/deps/KQED
cmake --build "$build" -j 16
