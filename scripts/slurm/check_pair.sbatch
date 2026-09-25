#!/bin/bash
#SBATCH --job-name=hlbl-check-pair
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --account=lp142
#SBATCH --uenv=prgenv-gnu/25.06:rc5
#SBATCH --view=default
#
# check_pair on the production configuration: 16 ranks, 2x2x2x2, local
# 64x32x32x32. Submit from a run directory that holds invert.input
# (scripts/gen_invert_input.sh -o . -p 2,2,2,2), naming the repository and the
# build's binary. HLBL_ROOT is needed because Slurm runs a copy of this script
# from its spool directory, so the script cannot find its siblings itself:
#
#   sbatch --export=ALL,HLBL_ROOT=/path/to/hlbl,HLBL_BIN=/path/to/build/check_pair \
#          /path/to/hlbl/scripts/slurm/check_pair.sbatch
#
# Measured on nid006544-6565: ~7.5 min wall, of which the QUDA setup solve
# is ~50 s and each of the 48 point-source solves ~2.3 s (46-47 iterations).
set -eu
ulimit -c 0   # a crash must not leave a >100 GB core dump behind
: "${HLBL_ROOT:?set HLBL_ROOT to the hlbl repository}"
: "${HLBL_BIN:?set HLBL_BIN to the check_pair binary}"
cd "$SLURM_SUBMIT_DIR"

# QUDA / MPI settings, as in the production job script
export QUDA_RESOURCE_PATH=$PWD/quda_cache
mkdir -p "$QUDA_RESOURCE_PATH"
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0
export QUDA_ENABLE_DSLASH_COARSE_POLICY=0
export QUDA_ENABLE_GDR=0
export QUDA_ENABLE_P2P=0
export QUDA_ENABLE_TUNING=1
export QUDA_REORDER_LOCATION=GPU
export CRAY_CUDA_MPS=1
export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_MAX_THREAD_SAFETY=multiple
export OMP_NUM_THREADS=16

# per_rank.sh gives each rank its own GPU: without it every rank of a node
# lands on GPU 0 (see that file). The plaquette argument is the value in the
# gauge file's xlf-info record.
srun "$HLBL_ROOT/scripts/slurm/per_rank.sh" "$HLBL_BIN" 2 0.554279822834
