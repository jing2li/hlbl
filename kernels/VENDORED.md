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

| file | cvc path | sha256 |
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

None.
