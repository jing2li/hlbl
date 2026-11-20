# Timing hint
2+2 pieces, 8 geometries, seconds per call on one GH200, 262,144 local sites
(32x16x16x32 of 128x64x64x64), 2026-11-20:

| targets | `cu_2p2_pieces` | `compute_2p2_gpu` | difference |
|---|---|---|---|
| 16 | 5.02 | 6.33 | cu 21% faster |
| 44 | 13.96 | 14.40 | cu 3% faster |
| 64 | 20.39 | 20.16 | compute 1% faster |
| 88 | 27.99 | 26.62 | compute 5% faster |
| 128 | 40.95 | 38.69 | compute 6% faster |

- `cu_2p2_pieces` (used): median of 5 calls; 0.32 s per target, linear.
- `compute_2p2_gpu`: 88 x 256 threads, `p3`, `p4`, `p3_zh` launched first; one call,
  2-3% run-to-run noise. Never faster than one pass of its slowest kernel (~6.3 s).
