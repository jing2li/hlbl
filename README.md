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
| 4 | `contract` (dzu/dzsu → γ products → 4-pt kernel sum, persistent device buffers); `test_contract_random` | done |
| 5 | `solver` (tmLQCD init, gauge read, plaquette check, MG setup solve, point-source solves); `check_pair` | done: runs end to end on the production configuration (see "Running on Alps") |
| 6 | driver: `provider`, `propagator_cache` (host or device residency), HDF5 output, timing CSV, `--window-range`, `hlbl_hypercube` | done on random fields (`test_driver_random`, `hypercube_random_fields`); the solver path of the binary builds but has not been run on the cluster yet |
| 7 | validation against cvc, timing runs at 16 and 32 nodes on 128×64³ | — |

The core library is split in two. `hlbl_core` needs only CUDA, MPI, KQED
and HDF5. `hlbl_solver` adds tmLQCD. Tests that don't solve link only
`hlbl_core`.

## Validation

| check | pass criterion | state |
|---|---|---|
| schedule vs `scripts/hlbl_mII_hypercube_schedule.py` for several h, n, steps | identical pair tables and counts; `--stride 4 --steps 30` gives 466 propagators, 1965 pairs, 3930 kernel sums, peak cache 30 | done, `scripts/compare_schedule.sh` |
| gamma tables | Clifford algebra; γ5 = γ0γ1γ2γ3 in the cvc convention | done, `test_gamma` (γμ²=γ5²=1, pairwise anticommutation) |
| random-field contraction, single rank | finite result, reproduces a stored golden value (~1e-10 rel., atomicAdd order is not fixed run to run) | done, `test_contract_random` |
| random-field contraction on 1×1×1×1 vs 2×1×1×1 ranks | agreement to rounding | not automated: the field is seeded by each site's *global* index for exactly this, but running it under two process grids and diffing is a manual step, not yet in ctest |
| free field (identity gauge, 16×8³, CG solver) vs cvc `hlbl_mII_invert_contract` built with `-DDO_CONNECTED=1`, source at origin, y = n·(1,1,1,1) | `check_pair` and the driver's corresponding pair agree to solver precision (expect ~1e-8 relative with the multigrid solver, see above; tighter with CG) | — |
| gauge field seen by QUDA, cB211a.072.64 #500, 16 GPUs | QUDA's mean plaquette equals the file's recorded 0.554279822834 | done, 12 digits (`check_pair`, `plaqQuda` order is {mean, spatial, temporal}) |
| real-propagator `kernel_sum` for the pair (origin, origin + 2·(1,1,1,1)) | finite, reproducible run to run | done, two independent 16-GPU runs: flavour 0 = −4.61833559e-08 / −4.61833559e-08 (1.5e-9 rel. apart), flavour 1 = −2.10504892e-08 / −2.10504891e-08 (5.2e-9 apart). The spread is the solver's (random multigrid setup vectors, finite tolerance), not the contraction's (~1e-15 in `test_contract_random`). **Not yet compared with cvc**, so correctness against production is still open |
| same comparison on cB211a.072.64 configuration 500, production layout | agreement at solver precision | — |
| schedule loop on random propagators: pair count, solves and cache peak equal `simulate()`; device and host-staged cache agree; a pair equals a direct `Contractor` call on regenerated fields; window ranges `[0,1)` + `[1,3)` reproduce the whole run; the HDF5 file holds exactly the computed rows | ~1e-10 relative (atomicAdd order) | done, `test_driver_random`. Checked to be able to fail: swapping the source and sink propagators in the loop, and breaking the host cache's re-staging, each make it fail |
| whole binary on random fields | exits 0, 80 pairs for one shift | done, `hypercube_random_fields` |
| `hlbl_hypercube` with the solver on the production configuration | runs, and its pairs reproduce `check_pair` | — (needs enough GPUs for the cache, see "Running the driver") |

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
- **The vendored kernels are an OBJECT library (`hlbl_kernels_obj`), not a
  STATIC one, and must stay that way.** `cuda_lattice.cu` calls device
  functions defined in KQED_cuda's own relocatable device code; nvlink does
  not reliably extract device code across two cross-referencing STATIC
  archives, and silently drops it rather than erroring, leaving
  `__cudaRegisterLinkedBinary_*` undefined at the host link -- only once
  something actually pulls the affected archive member in (`test_schedule`
  built and linked fine; `test_contract_random`, the first target to
  actually call into `cuda_lattice.cu`, failed). An OBJECT library's `.o`
  files are added directly to whichever executable links it instead of
  being archived, avoiding the problem; cvc's own `time_hlbl/CMakeLists.txt`
  hits the same constraint and works around it the same way.

### Testing

```bash
ctest --test-dir build --output-on-failure
```

`test_schedule`, `test_gamma` and `test_lattice` are single-rank tests that
never touch the GPU at run time (see the build note above for why they still
link CUDA). `test_contract_random` runs the full contraction pipeline on
random propagator fields on one GPU and checks the result against a stored
golden value. `compare_schedule` runs `scripts/compare_schedule.sh`, which
checks `hlbl_hypercube --dry-run`'s schedule against the Python reference for
several step directions, strides and step counts by calling the reference's
own functions -- not by parsing its prose output.

## Running the driver

`hlbl_hypercube` runs a schedule and writes the kernel sums of all its pairs.

```
hlbl_hypercube [--direction HT HX HY HZ] [--stride n] [--steps K] [--origin T X Y Z]
               [--out results.h5] [--timing timing.csv]
               [--cache device|host] [--window-range a:b] [--force]
               [--dry-run [--show-window k] [--flavors 1|2] [--lattice T X Y Z]]
               [--random-fields --lattice T X Y Z --procs PT PX PY PZ]
```

- **Modes.** By default the propagators come from tmLQCD/QUDA: geometry, gauge
  field and operators are read from `invert.input` in the working directory,
  exactly as for `check_pair`, and `--lattice`/`--procs` are not used.
  `--random-fields` replaces the solves by seeded noise on a lattice given on
  the command line, which exercises everything else with no gauge field and no
  solver. `--dry-run` builds the schedule and prints its counts and the cache
  peak, on rank 0, without touching the GPU.
- **Steps.** `--steps` is the number of shifts K (K + 1 windows, 16 + 15·K
  solves, 15 + 65·K pairs). The default is as many as fit before sources wrap
  onto sites already used.
- **Cache.** `--cache device` keeps every cached propagator on the GPU;
  `--cache host` keeps them in pinned host memory and stages the two a
  contraction needs, so the device holds three blocks (two staging slots and a
  solve target). Either way a source is dropped as soon as its last pair is
  done.
- **Memory.** One source position is 2 flavours × 12 × 24 doubles per site:
  4608 B per local site, 9 GiB per rank at the production layout (local
  64×32×32×32). Window 0 alone holds 16 positions and the peak is 30, so
  **the full-volume cache does not fit at 16 GPUs** (270 GiB per rank). It fits
  from 32 nodes (~34 GiB per rank, device cache), or with sparsened
  propagators (future work). The driver prints the estimate against the free
  device memory and this rank's share of the node's host memory before doing
  anything expensive, and stops if it will not fit; `--force` overrides that.
  `scripts/hlbl_mII_hypercube_schedule.py` gives the same numbers for any
  lattice and process grid.
- **Restarting.** `--window-range a:b` runs windows a to b-1 (`a:` runs to the
  end). Before window a it re-solves the sources of earlier windows that
  window a's pairs need (15 of them for a > 0), so a chain can be split over
  several jobs at window boundaries; each job writes its own file.

### Output

One HDF5 file per run, rows appended and flushed after every window, so a job
that is killed keeps the windows it finished:

| dataset | shape | content |
|---|---|---|
| `x`, `y` | N × 4 int | the pair's two sources (global sites); `x` is the earlier one and the kernel origin |
| `yv` | N × 4 int | minimum-image separation `y − x`, always (±n, ±n, ±n, ±n) |
| `window` | N int | window that closed the pair |
| `kernel_sum` | N × 2 double | flavour 0, flavour 1 |

Attributes: `lattice`, `procs`, `direction`, `origin`, `stride`, `nconf`,
`window_begin`, `window_end`, `kernel` (`LLambda0.4`), `flavour_order` and
`hlbl_revision` (the git revision of this tree when CMake last configured).

`--timing` (default: the result file's name with `.timing.csv`) records one CSV
row per solve (`kind,window,x0..x3,y0..y3,seconds,iterations`) and one per
kernel sum, which is where per-solve and per-kernel-sum times for a resource
estimate come from.

## Running on Alps

`check_pair` (`sanity_check/check_pair.cpp`) inverts two point sources through
tmLQCD/QUDA and contracts the one pair they form. It is a manual tool: it
needs a real gauge configuration, and the multigrid setup takes about a
minute.

```bash
mkdir run && cd run
../scripts/gen_invert_input.sh -o . -T 128 -L 64 -p 2,2,2,2     # writes invert.input
sbatch --export=ALL,HLBL_ROOT=$PWD/..,HLBL_BIN=/path/to/build/check_pair ../scripts/slurm/check_pair.sbatch
```

Everything the program needs comes from `invert.input` in the working
directory: tmLQCD reads the lattice, the process grid, the gauge file and the
operators from it (`tmLQCD_invert_init`), and `hlbl::Solver` builds its
geometry from tmLQCD's own parameters. Measured on 4 nodes (16 GPUs,
128×64³, local volume 64×32×32×32): the setup solve takes ~50 s and each
point-source solve ~2.3 s (46-47 iterations), i.e. ~55 s per source position
for both flavours.

Three things that are easy to get wrong, each of which cost a failed run:

- **Local volume per GPU.** The multigrid setup of `invert.input.in` (2048
  coarse-level deflation vectors) fits in GPU memory at a local volume of
  64×32×32×32, the production layout. 1 GPU (16× that volume) and 4 GPUs (4×)
  failed, with an illegal-address error, a "failed to allocate device memory"
  abort, or a crash. Use 16 GPUs for 128×64³, and check
  `gen_invert_input.sh`'s multigrid-blocking constraint for other lattices.
- **GPU placement.** tmLQCD asks QUDA for device 0 on *every* rank, and on
  this system that is what it gets: left alone, all four ranks of a node share
  GPU 0 (94 GB of 98 GB in use, GPUs 1-3 idle). Giving each rank a single
  visible GPU does not work either, since QUDA's communicator setup requires
  as many visible devices as ranks on the node ("Too few GPUs available").
  `scripts/slurm/per_rank.sh` shows every rank all the node's GPUs, rotated so
  its own comes first; `check_pair` prints each rank's PCI bus id after the
  setup solve, and a node's four ranks must show four different ids.
- **Core dumps.** A crashed job can leave a core file larger than the 50 GB
  home quota. The job scripts set `ulimit -c 0`.

QUDA also refuses a tuning cache written by a different QUDA version and
aborts, so point `QUDA_RESOURCE_PATH` at a fresh directory when switching
builds.

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
- Restart from inside a window (`--window-range` restarts at window boundaries only).
