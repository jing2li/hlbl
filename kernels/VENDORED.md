# Vendored GPU kernels

These files are copies of the cvc production kernels. They are kept verbatim
so that results can be compared with cvc's `hlbl_mII_invert_contract`. Fix
bugs here deliberately and record each change below.

## Origin

- Repository: cvc, branch `hlbl_runtime`
- Base commit: `ce6080a77cc0aa53e2f61fbd55e9d76102a00caa` (2026-07-20)
- The copies include work that was not yet committed in cvc at copy time.
  `cvc_uncommitted.patch` is the diff from the base commit to the copied
  state of `cuda_lattice.{cu,h}` and `hlbl_kernels/kernels.{cu,cuh}`.
- `kqed_cuda_glue.cu` comes from cvc's `time_hlbl/`, which was untracked at
  copy time.
- The `yi` loop of `ker_2p2_pieces` (`cuda_lattice.cu`) comes from
  LeoFilipovic/cvc, branch `dev-cpff-zerohalfwrap`, commit
  `dc38e2b6d3ec1df2a1d19795782b48a4765af9b3` (see the table below).

| file | cvc path | sha256 as copied |
|---|---|---|
| `cuda_lattice.cu` | `cuda_lattice.cu` | `ba787bc869d18e8e8f76bf88f16b3a8ac1ac679f8b147d225ccac45e82e8f74b` |
| `cuda_lattice.h` | `cuda_lattice.h` | `7f3af22c66bd50c7db0470883e56de1db5d15282752887b292e1cdb82bbf7bdd` |
| `hlbl_kernels/kernels.cu` | `hlbl_kernels/kernels.cu` | `4b37029ba59432278eeb021dd433fb141983da7f2480c4830d4e91b66ba97fad` |
| `hlbl_kernels/kernels.cuh` | `hlbl_kernels/kernels.cuh` | `e5338444971e113d7cd6700d40db42ff7639b6d3b46cdd26eb7d1d030536061a` |
| `kqed_cuda_glue.cu` | `time_hlbl/kqed_cuda_glue.cu` | `6b8b677b0b2c9e67ff02ca4344358979c3997752229b9bdc67748ed1dd86f825` |

## What is used

- `cuda_lattice.cu`: `cu_dzu_dzsu` and `cu_4pt_contraction`, the connected
  kernel sum.
- `hlbl_kernels/kernels.cu`: the 2+2 pieces (`compute_2p2_gpu`), called by
  `Contractor::pieces_2p2` for the sources of the input file's `2p2` lines.
  `cu_2p2_pieces` of `cuda_lattice.cu` is not called; it is the reference
  `compute_2p2_gpu` is checked against.
- `kqed_cuda_glue.cu`: the external definition of `__checkCudaErrors`, which
  `libKQED_cuda.a` references but does not define.

Both kernel files evaluate a single QED kernel, LΛ with Λ = 0.4
(`QED_Mkernel_L2(0.4, ...)`).

## Local modifications

| date | commit | file | change |
|---|---|---|---|
| 2026-10-16 | `f7e4a90` | `hlbl_kernels/kernels.cu` | `prop_idx()` returns `size_t` and casts each term before multiplying. It returned `int` and multiplied in 32 bits; its largest value, 576 × the local volume, passes 2³¹ above 3.7 million local sites. Only the 2+2 code (`kernel_pi`, `kernel_4pt` of this file) uses it, and hlbl calls neither yet. |
| 2026-10-23 | `b677727` | `cuda_lattice.cu` | One rule for every QED kernel argument: `coord_map`, [−L/2, L/2). Zerohalf (L/2 → 0) only for the odd weights x_ρ, z_ρ. Differs from cvc in 4pt x − y (cvc: zerohalf) and in 2p2 x, y, −y, x − y (cvc: zerohalf); 4pt also maps y itself. Results change only where a component of such a separation is exactly L/2. `coord_map_zerohalf_diff` removed. |
| 2026-10-23 | `b677727` | `hlbl_kernels/kernels.cu` | The same rule: `site_map` for the QED kernel arguments of `kernel_4pt` (x − y, and y mapped in the kernel) and `kernel_p20`/`p21`/`p3` (cvc: zerohalf); `site_map_zerohalf` only for x_ρ. Removed `kernel_p23` (never launched), the `HAVE_KQED_V2` path (`KQED_LX_INIT`/`QUERY`, `kernel_p2{0,1}_mt`, `kernel_p3_mt` and their launch branch) and a dead commented block in `kernel_p3`. |
| 2026-10-23 | `b677727` | `hlbl_kernels/kernels.cuh` | `kernel_p23` declaration removed. |
| 2026-10-30 | (uncommitted) | `hlbl_kernels/kernels.cu` | `kernel_p21`: the block reduction and the P23 write moved after the `ix` loop, as in `kernel_p20` and `kernel_p3`. Inside the loop, threads with one iteration fewer (local volume not a multiple of the 256 threads of a block) skipped the `__syncthreads` of `blockReduceSum`: undefined behaviour and stale shared memory in the sum. Local volumes that are multiples of 256 give the same values as before. |
| 2026-11-19 | (uncommitted) | `cuda_lattice.cu` | `ker_2p2_pieces`: the `yi` loop of Leo's `dev-cpff-zerohalfwrap` (`dc38e2b`), which adds P4_0 and P4_1, without its zerohalf parts (`P23x_wrap`, `*_old`, `need_wrap`). Two changes to it: −y and −x mapped on their own (`coord_map_diff`, [−L/2, L/2); Leo: −yv, −xv), and the z component of x − y and y − x taken mod L_z (Leo adds L_y). |
| 2026-11-19 | (uncommitted) | `cuda_lattice.h` | `CUDA_N_QED_GEOM` 3 → 5. |
| 2026-11-19 | (uncommitted) | `hlbl_kernels/kernels.cu` | `kernel_p4`: P4_0 and P4_1 as in `ker_2p2_pieces`; `kernel_n_geom` 5, an `#error` unless `kernel_n`, `kernel_n_geom` match `cuda_lattice.h`; `MPI_Waitall` of all five requests. Removed unused code: `kernel_4pt`, `compute_4pt_gpu`, `record_*_cuda`, `call_KQED`, `test_KQED_on_gpu`, `kqed_set_zero`, `union array`, `idx_comb_d`, `site_map_zerohalf` and commented-out lines. P23 renamed P234. |
| 2026-11-19 | (uncommitted) | `hlbl_kernels/kernels.cuh` | Declarations of the removed functions dropped; P234. |

sha256 now:

| file | sha256 |
|---|---|
| `cuda_lattice.cu` | `1d90760f70441b2250a135f1f1847e01cb8231cd2cdf2ef1da3d5b4c391d769f` |
| `cuda_lattice.h` | `a1784c14b948c11522e419459b6767b59d2874016d2d345d3593c34eae4ca308` |
| `hlbl_kernels/kernels.cu` | `4a5750b4497c4474708517665d889d16c90037bbb84cd50d8fd83120039cb991` |
| `hlbl_kernels/kernels.cuh` | `94c3a430db23a6bdedf291ff8616165ddc4e169304a06be51292772168ca98db` |

The other files are as copied.

## Changes in cvc since the copy (not taken)

cvc's working tree, checked on 2026-10-20:

- `cuda_lattice.h`: `CUDA_THREAD_DIM_1D` 32 → 128, the block size of the
  `cuda_lattice.cu` launches.
- `hlbl_kernels/kernels.cu`: the same `prop_idx` fix, plus two changes to the
  2+2 code: the P2/P3 grid is clamped to `n_y` (`n_y > 88 ? 88 : n_y`), and the
  shared `kerv` array of `kernel_4pt` is sized by `kernel_n` instead of 3.
- `cuda_lattice.cu`: unchanged.
