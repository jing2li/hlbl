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
- `hlbl_kernels/kernels.cu`: the 2+2 pieces (`compute_2p2_gpu`). It is
  compiled but not called yet. `HAVE_KQED_V2` stays undefined, so the
  elementwise KQED path, which needs a newer KQED, is left out.
- `kqed_cuda_glue.cu`: the external definition of `__checkCudaErrors`, which
  `libKQED_cuda.a` references but does not define.

Both kernel files evaluate a single QED kernel, LΛ with Λ = 0.4
(`QED_Mkernel_L2(0.4, ...)`).

## Local modifications

| date | commit | file | change |
|---|---|---|---|
| 2026-10-16 | `f7e4a90` | `hlbl_kernels/kernels.cu` | `prop_idx()` returns `size_t` and casts each term before multiplying. It returned `int` and multiplied in 32 bits; its largest value, 576 × the local volume, passes 2³¹ above 3.7 million local sites. Only the 2+2 code (`kernel_pi`, `kernel_4pt` of this file) uses it, and hlbl calls neither yet. |

sha256 of `hlbl_kernels/kernels.cu` now:
`e52f47dcbffab8d5bde5b21e00a3703e1def26bb995bce37b9573c82f958ea82`. The other
files are as copied.

## Changes in cvc since the copy (not taken)

cvc's working tree, checked on 2026-10-20:

- `cuda_lattice.h`: `CUDA_THREAD_DIM_1D` 32 → 128, the block size of the
  `cuda_lattice.cu` launches.
- `hlbl_kernels/kernels.cu`: the same `prop_idx` fix, plus two changes to the
  2+2 code: the P2/P3 grid is clamped to `n_y` (`n_y > 88 ? 88 : n_y`), and the
  shared `kerv` array of `kernel_4pt` is sized by `kernel_n` instead of 3.
- `cuda_lattice.cu`: unchanged.
