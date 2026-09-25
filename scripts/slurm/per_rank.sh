#!/bin/bash
# One GPU per rank, chosen before the process starts.
#
# Two constraints meet here. tmLQCD asks QUDA for device 0 on every rank (its
# node-local rank resolves to 0 on this system), so left alone all ranks of a
# node land on GPU 0 and run out of memory. But QUDA's communicator setup
# requires at least as many *visible* devices as ranks on the node, so each
# rank cannot be given just one. The way through is to show every rank all
# GPUs of the node, rotated so that this rank's own GPU comes first: device 0
# is then physical GPU $SLURM_LOCALID (tmLQCD's own, commented-out ptbc code
# does the same rotation).
n=${SLURM_NTASKS_PER_NODE:-4}
list=""
for ((i = 0; i < n; i++)); do
  list+="${list:+,}$(( (SLURM_LOCALID + i) % n ))"
done
export CUDA_VISIBLE_DEVICES=$list
exec "$@"
