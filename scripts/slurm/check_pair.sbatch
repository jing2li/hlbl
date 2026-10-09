#!/bin/bash
#SBATCH --job-name=hlbl-check-pair
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --uenv=prgenv-gnu/25.06:rc5
#SBATCH --view=default
#
# check_pair on 4 nodes (16 ranks, grid 2x2x2x2). Submit from a directory that
# holds invert.input (scripts/gen_invert_input.sh -o . -p 2,2,2,2):
#
#   sbatch -A <account> -p <partition> --export=ALL,HLBL_BIN=/path/to/build/check_pair \
#          /path/to/hlbl/scripts/slurm/check_pair.sbatch
#
# Takes about 8 minutes.
set -eu
ulimit -c 0   # no core dumps: they can be very large
: "${HLBL_BIN:?set HLBL_BIN to the check_pair binary}"
cd "$SLURM_SUBMIT_DIR"

# QUDA and MPI settings
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

# The second argument is the plaquette recorded in the gauge file.
srun "$HLBL_BIN" 2 0.554279822834
