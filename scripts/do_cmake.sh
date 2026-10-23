#!/bin/bash
# Configure and build hlbl. Run inside the programming environment:
#   uenv start --view=default prgenv-gnu/25.06:rc5
#   scripts/do_cmake.sh [build-dir]
#
# hlbl links two external packages itself:
#   tmLQCD      TMLQCD, its install prefix; brings lemon, c-lime and QUDA
#   KQED_cuda   KQED, the install prefix of KQED's GPU build
# HDF5, MPI and CUDA come from the uenv.
#
# tmLQCD's package config looks up lemon, c-lime and QUDA again, so
# CMAKE_PREFIX_PATH also holds TMLQCD_DEPS: the prefixes tmLQCD itself was
# configured with (tmLQCD/build/do_cmake.sh). lemon is installed with tmLQCD.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build=${1:-$here/build}
S=/capstor/store/cscs/userlab/lp142

TMLQCD=$S/ljingjin/tmlqcd_install
TMLQCD_DEPS="$S/software/install_dir/clime_pic;$S/ljingjin/quda_install"
KQED=$S/ljingjin/build_kqed_gpu

cmake -S "$here" -B "$build" \
  -DCMAKE_PREFIX_PATH="$TMLQCD;$TMLQCD_DEPS" \
  -DKQED_ROOT=$KQED
cmake --build "$build" -j 16
