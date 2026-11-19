# Timing hint
2+2 pieces, seconds per call on one GH200, median, targets 16 / 64 / 128 (2026-11-19):
- 65,536 local sites: `compute_2p2_gpu` 1.04 / 2.12 / 4.23, `cu_2p2_pieces` 0.54 / 2.31 / 4.59
- 262,144 local sites: `compute_2p2_gpu` 4.63 / 9.49 / 18.98, `cu_2p2_pieces` 2.24 / 9.48 / 19.09
- `compute_2p2_gpu` runs one block per target and geometry, one block per SM: below 33 targets part of the GPU is idle.
