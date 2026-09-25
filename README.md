# hlbl — connected HLbL kernel sum on a sliding-hypercube schedule

`hlbl_hypercube` computes the connected contribution to the hadronic
light-by-light scattering in the muon g−2 in position space (method II, QED
kernel LΛ with Λ = 0.4). It covers every point-source pair at one separation
|y| = 2n. The pairs come from a sliding-hypercube schedule: each step
reuses most propagators of the previous one and keeps a small cache.

The program runs on GPUs only. Propagators are solved with tmLQCD/QUDA, and
the contraction uses the GPU kernels of cvc's `hlbl_mII_invert_contract`,
vendored in `kernels/`. Nothing else is taken from cvc.

## Plan and status

| phase | deliverable | state |
|---|---|---|
| 1 | repository, vendored kernels, KQED build wrapper | done |
| 2 | CMake build: KQED_cuda, `hlbl_kernels`, tmLQCD found, `hlbl_hypercube` stub links | done |
| 3 | host core: `lattice`, `mdarray`, `gamma`, `schedule`; `hlbl_hypercube --dry-run`; tests `test_schedule`, `test_gamma`, `test_lattice`, `scripts/compare_schedule.sh` | done |
| 4 | `contract` (dzu/dzsu → γ products → 4-pt kernel sum, persistent device buffers); `test_contract_random` | next |
| 5 | `solver` (tmLQCD init, gauge read, plaquette check, MG setup solve, point-source solves); `check_pair` | — |
| 6 | driver: `propagator_cache` (host or device residency), HDF5 output, timing CSV, `--window-range` | — |
| 7 | validation against cvc, timing runs at 16 and 32 nodes on 128×64³ | — |

The core library is split in two. `hlbl_core` needs only CUDA, MPI, KQED
and HDF5. `hlbl_solver` adds tmLQCD. Tests that don't solve link only
`hlbl_core`.

## Validation

| check | pass criterion | state |
|---|---|---|
| schedule vs `scripts/hlbl_mII_hypercube_schedule.py` for several h, n, steps | identical pair tables and counts; `--stride 4 --steps 30` gives 466 propagators, 1965 pairs, 3930 kernel sums, peak cache 30 | done, `scripts/compare_schedule.sh` |
| gamma tables | Clifford algebra; γ5 = γ0γ1γ2γ3 in the cvc convention | done, `test_gamma` (γμ²=γ5²=1, pairwise anticommutation) |
| random-field contraction on 1×1×1×1 vs 2×1×1×1 ranks | agreement to rounding; golden value stored | — |
| free field (identity gauge, 16×8³, CG solver) vs cvc `hlbl_mII_invert_contract` built with `-DDO_CONNECTED=1`, source at origin, y = n·(1,1,1,1) | `check_pair` and the driver's corresponding pair agree to ~1e-10 relative | — |
| same comparison on cB211a.072.64 configuration 500, production layout | agreement at solver precision | — |

## The schedule

G is the set of the 16 sign vectors (±1,±1,±1,±1), h ∈ G is the step
direction, and n is the stride. In units of n:

    A_0 = {0} ∪ (G \ {h})        16 sites
    A_k = k·h + A_0              window k

A pair (x, y) is contracted iff y − x ∈ n·G, so every pair has |y − x| = 2n.
Consecutive windows share one site (k·h − h = (k−1)·h). Window 0 therefore
costs 16 propagators and 15 pairs, and every later window costs 15
propagators and yields 65 pairs: 15 inside the window and 50 with the
previous one. No pair reaches back two windows, so a window's propagators are
evicted once the next window has been contracted. The cache peaks at 30
source positions.

For K shifts (K + 1 windows):

    propagators  = 16 + 15·K           (×12 spin-colour × 2 flavours solves)
    pairs        = 15 + 65·K           (×2 flavours kernel sums)

`scripts/hlbl_mII_hypercube_schedule.py` is the reference implementation of
the schedule. It also reports counts, memory and cost estimates. It needs
Python ≥ 3.8 (`python3.11` on Alps).

## Conventions

- **Lattice sites.** Local index `((t·LX + x)·LY + y)·LZ + z`, the tmLQCD
  lexicographic order. The local extents and the process grid come from
  tmLQCD. Global extents are local × process grid.
- **Propagators.** One source position is a block of `2 × 12 × 24·V` doubles:
  `[flavour][source spin-colour][site][sink spin-colour, complex]`. This is
  the layout of the `fwd_src` and `fwd_y` arguments of the kernels.
- **Flavour.** The tmLQCD operator index: operator 0 has twisted mass +μ,
  operator 1 has −μ.
- **Pairs.** A pair (q, p), where q is the earlier of the two in the
  schedule, is contracted with q as the kernel origin, and
  `y = site_map_diff(p − q)` is the minimum-image separation. Pairs are
  unordered: the reversed order is a correlated sample of the same quantity,
  since the x ↔ y exchange is already part of every contraction.
- **Units.** Kernel arguments are in units of the muon mass,
  `xunit = m_μ · a` with a = 0.07957 fm.
- **Gamma matrices.** The cvc basis and sign convention.

## Layout

```
bin/            hlbl_hypercube (driver)
include/hlbl/   core library headers
src/            core library sources
kernels/        vendored GPU kernels, see kernels/VENDORED.md
deps/kqed/      CMake build of KQED_cuda
sanity_check/   tests and validation programs
scripts/        reference schedule, comparison scripts, input templates
```

## Build

Build inside the programming environment on Alps:

```bash
uenv start --view=default prgenv-gnu/25.06:rc5

S=/capstor/store/cscs/userlab/lp142
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=$S/software/install_dir/clime_pic \
  -DtmLQCD_DIR=$S/ljingjin/tmlqcd_install/lib64/cmake/tmlqcd \
  -DQUDA_DIR=$S/ljingjin/quda_install/lib/cmake/QUDA \
  -Dlemon_DIR=$S/ljingjin/tmlqcd_install/lib/cmake/lemon \
  -DKQED_SOURCE_DIR=/users/ljingjin/cvc_cmake/cvc/deps/KQED
cmake --build build -j 16
```

Notes:

- **`tmLQCD_DIR`** must point at the *installed* package config
  (`lib64/cmake/tmlqcd`). A tmLQCD build tree also contains a
  `tmLQCDConfig.cmake`, but that one lacks the c-lime finder and configure
  fails.
- **`KQED_SOURCE_DIR`** names a KQED checkout at revision `deb4d488`. If it is
  left empty, that revision is cloned into the build tree. Either way the
  checkout must stay in place, because KQED reads its precomputed tables
  from `KQED_SOURCE_DIR/PRECOMP` at run time.
- `CMAKE_BUILD_TYPE` defaults to `Release` and `CMAKE_CUDA_ARCHITECTURES` to
  `90` (GH200).
- `-DHLBL_WITH_TMLQCD=OFF` builds everything except the solver interface.

### Testing

```bash
ctest --test-dir build --output-on-failure
```

`test_schedule`, `test_gamma` and `test_lattice` are single-rank, GPU-free
unit tests. `compare_schedule` runs `scripts/compare_schedule.sh`, which
checks `hlbl_hypercube --dry-run`'s schedule against the Python reference for
several step directions, strides and step counts by calling the reference's
own functions -- not by parsing its prose output.

## Future work

Not implemented:

- **Sparsened propagators.** Keep each cached propagator only on a fixed,
  stochastically chosen subset of sites, which would cut the cache by the
  sparsening factor. This is unverified: it needs kernels that loop over a
  site list, and a validated estimator for the sparsened z-sum.
- **Pipelined execution.** Solve window k+1 while window k is contracted.
  The cache peak rises from 30 to 45 source positions.
- **2+2 contribution.** The kernels (`compute_2p2_gpu`) are vendored and
  compiled. A schedule for them needs ordered pairs, because the two
  orderings carry different terms of the master formula.
- Single-precision propagator cache.
- Several chains or step directions per job.
- Restart from inside a window.
