# hlbl: connected HLbL kernel sum on a sliding-hypercube schedule

`hlbl_hypercube` computes the connected hadronic light-by-light contribution
to the muon g−2 in position space (QED kernel LΛ, Λ = 0.4). It contracts all
pairs of point sources at one separation |y| = 2n, taken from a
sliding-hypercube schedule that reuses most propagators from step to step. One
job runs a list of such chains; the gauge field is read and the multigrid set
up once. With `2p2` lines it also computes the 2+2 pieces (P1, P2_0, P2_1, P3)
of chosen sources, right after their first solve.

Propagators are solved with tmLQCD/QUDA. The contraction uses the GPU kernels
of cvc's `hlbl_mII_invert_contract`, copied to `kernels/`.

## Schedule

G is the 16 sign vectors (±1,±1,±1,±1), h ∈ G the step direction, n the
stride. In units of n:

```text
A_0 = {0} ∪ (G \ {h})        16 sources
A_k = k·h + A_0              window k
```

A pair (x, y) is contracted if y − x ∈ n·G, i.e. |y − x| = 2n. Neighbouring
windows share one source and no pair reaches back two windows. Window 0 has 16
propagators and 15 pairs; each later window adds 15 propagators and 65 pairs
(15 inside, 50 with the previous window). For K shifts:

```text
propagators = 16 + 15·K       (24 solves each: 12 spin-colours × 2 flavours)
pairs       = 15 + 65·K       (2 kernel sums each, one per flavour)
```

Each source is contracted right after its solve and a propagator is freed
after its last pair, so at most 16 are alive, for every h.

`scripts/hlbl_mII_hypercube_schedule.py` (Python ≥ 3.8) is the reference
implementation. It plans a run from its input file: schedules, overlaps,
memory and run time.

## Conventions

- **Site index:** `((t·LX + x)·LY + y)·LZ + z`, as in tmLQCD, which also sets
  the local extents and process grid.
- **Propagator block** of one source: [flavour][source spin-colour][site][sink
  spin-colour, complex], 2 × 12 × 24·V doubles (the kernels' `fwd_src` and
  `fwd_y`).
- **Flavour:** tmLQCD operator 0 has twisted mass +μ, operator 1 has −μ.
- **Pairs:** (q, p) is contracted with the earlier source q as kernel origin
  and y = p − q (minimum image). Pairs are unordered: the reversed pair is a
  correlated copy.
- **Units:** kernel arguments are in units of the muon mass, `xunit = m_μ a`,
  from the run inputs `muon_mass` (MeV) and `lattice_spacing` (fm). Only
  ħc = 197.3269804 MeV fm is built in.
- **Gamma matrices:** cvc basis and signs.

## Layout

```text
bin/            hlbl_hypercube
include/hlbl/   headers
src/            sources
kernels/        GPU kernels from cvc (kernels/VENDORED.md)
sanity_check/   tests and check_pair
scripts/        Python reference, build, input generator, Slurm script
```

`hlbl_core` needs CUDA, MPI, KQED and HDF5; `hlbl_solver` adds tmLQCD/QUDA.

## Build

Run `scripts/do_cmake.sh`. It configures and builds in `build/`.

- hlbl links tmLQCD, which brings lemon, c-lime and QUDA. tmLQCD's package
  config looks these up again, so `CMAKE_PREFIX_PATH` holds the tmLQCD install
  prefix (not its build tree, which lacks the c-lime finder) and the prefixes
  tmLQCD was configured with. Change them together when tmLQCD is rebuilt.
- `KQED_ROOT` is the install prefix of KQED's GPU build
  (`lib/libKQED_cuda.a`, `include/`). Keep it in place: KQED reads `PRECOMP/`
  from it at run time. The library in `do_cmake.sh` is a bug-fixed KQED: its
  kernel is 0 beyond |x| ≈ 12 fm, where KQED `deb4d488` (used before) gave
  values ≲ 2·10⁻⁸. Below that the two agree to rounding.
- Defaults: `Release`, CUDA architecture 90 (GH200).
- `hlbl_kernels_obj` must stay an OBJECT library, or nvlink can drop device
  code (see `CMakeLists.txt`).
- After changing the tmLQCD/QUDA build, use a new `QUDA_RESOURCE_PATH`: QUDA
  aborts on a tune cache from another version.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

| test | checks |
|---|---|
| `test_schedule` | window sizes, pair counts, completeness, cache peaks; peak 16 for all 16 directions; which solves get 2+2 |
| `compare_schedule` | C++ schedules (printed by the test program `schedule_dump`) = Python reference, also for multi-chain files (counts, totals, overlaps), the 2+2 plan and warnings, and bad-file errors |
| `test_input` | input parser: required keys, comments, `2p2` lines, every error |
| `test_gamma`, `test_lattice` | gamma algebra; site indexing and ownership |
| `test_contract_random` | contraction and 2+2 pieces on random fields: finite, match stored values; index structure of P1 and P23 |
| `test_driver_random` | schedule loop on random fields: pair count, cache peak, device vs host cache, two chains vs independent runs, 2+2 at first solves vs direct calls, HDF5 content |
| `hypercube_random_fields` | whole program on `sanity_check/one_chain.input` (80 pairs) |
| `hypercube_random_fields_multi` | whole program on `sanity_check/multi_chain.input` (2 chains, 225 pairs, 2+2 at 3 sources) |
| `hypercube_dummy_run` | `--dummy-run` on the same file, with one 2+2 |

`test_contract_random`, `test_driver_random` and `hypercube_*` need one GPU.
`check_pair` (not in ctest) solves two real point sources and contracts their
pair; see "Running on Alps".

## Running

```text
hlbl_hypercube --in hlbl.input [--dummy-run [--tune]]
```

Everything else is in the input file, and a missing key stops the run at
once.

- **Propagators** come from tmLQCD/QUDA, which reads lattice, process grid,
  gauge file and operators from `invert.input` in the working directory
  (written by `scripts/gen_invert_input.sh`). The run stops if `lattice` and
  `procs` of the input file differ from it. `random_fields = yes` uses random
  numbers on `lattice` and `procs` instead.
- **Cache.** `device` keeps propagators on the GPU. `host` keeps them in
  pinned host memory, and the GPU holds only 3 blocks (solve target, two
  staging slots). Use `host` when the cache does not fit on the GPU.
- **Resume.** Rows are written after every window. To continue a chain whose
  windows 0..j are in the result file, run a new chain from window j's centre,
  `start + j·stride·dir`, with `steps = K − j`. It gives windows j+1..K
  exactly and repeats only window j's 15 inner pairs (reported by the overlap
  warning). Starting at window j+1 instead would miss the 50 pairs between
  windows j and j+1. Window j's sources are solved again: drop them from the
  `2p2` lines, or their 2+2 is computed again (in the new result file).

### Checks

A production run checks only its input. Before submitting:

- `scripts/hlbl_mII_hypercube_schedule.py --in hlbl.input` (no GPU):
  schedules, overlaps, memory per rank on the file's process grid, cost. Ends
  with a PRECHECK block; exit status 1 on errors.
- `--dummy-run` (on the job's resources): the full setup (gauge field, QUDA
  multigrid, QED kernel), every check, the memory estimate on the worst rank,
  then the memory peak for real: cache filled, one position solved (24
  solves), one pair contracted and, with `2p2` lines, the 2+2 of the largest
  target list (`one 2+2 source (n targets) in t s`, a value for `--t-2p2`).
  Writes no results; ends with `dummy run: all checks passed` (status 0) or
  `FAILED` (status 1). `--tune` makes QUDA tune
  and save its tune cache to `QUDA_RESOURCE_PATH`, so a production run with
  the same build, node count and path starts tuned.

```bash
python3.11 scripts/hlbl_mII_hypercube_schedule.py --in hlbl.input \
    --t-invert 40.6 --t-kernel-sum 2.8 --t-setup 275       # no GPU
srun build/hlbl_hypercube --in hlbl.input --dummy-run --tune  # in the job's allocation
```

### Input file

`#` starts a comment. Global lines are `key = value`, chain lines
`chain key=value ...` with vectors written `t,x,y,z`:

```text
lattice         = 128,64,64,64
procs           = 4,2,2,2
muon_mass       = 105.6583745   # MeV
lattice_spacing = 0.07957       # fm
cache           = host
out             = result.h5
chain start=0,0,0,0   dir=1,1,1,1   stride=4 steps=30
chain start=0,16,0,0  dir=1,1,1,1   stride=6
2p2 source=4,4,4,4 targets=(8,8,8,8),(0,0,0,0)
```

| key | meaning |
|---|---|
| `lattice`, `procs` | global extents and process grid; must match `invert.input` |
| `muon_mass`, `lattice_spacing` | MeV and fm; give `xunit`, stored in the result file |
| `cache` | `device` or `host` |
| `out` | result file |
| `timing` (optional) | timing log; default `out` with `.timing.csv` |
| `random_fields` (optional) | `yes`: random numbers instead of solves |
| `chain` | `start=t,x,y,z dir=±1,±1,±1,±1 stride=n [steps=K]`: first source, step direction, every pair at \|y\| = 2n, windows 0..K (default `auto`: as many as fit before sources wrap) |
| `2p2` (optional) | `source=t,x,y,z targets=(t,x,y,z),(t,x,y,z),...`: the 2+2 pieces of `source` for every target, at its first solve (see 2+2). One line per source; sites in [0, L) |

`scripts/input/hlbl.input.example` is a commented template.

Chains run one after the other with one cache, so memory is that of the
largest chain. They are independent: a shared source is solved, and a shared
pair computed and stored, once per chain. `--dummy-run` and the Python script
warn about this:

```text
WARNING: 39 source positions are in more than one chain (solved once per chain) and 119 pairs are computed in more than one chain
chains 0 and 2 share 31 sources and 80 pairs
  source (4,4,4,4) in window 2 of chain 0 and window 0 of chain 2
  source (4,4,4,8) in window 3 of chain 0 and window 0 of chain 2
  ... and 29 more sources
```

Sources are in wrapped global coordinates, with the window of each chain that
solves them. At most 20 pairs of chains are listed, each with its first 10
shared sources.

### 2+2

A `2p2` line names a source and its targets. Right after the first solve of
the source in the run (chains in input order, then windows, then solve
order), `compute_2p2_gpu` computes from that block its P1 and, for every
target, P2_0, P2_1 and P3; then the source's pairs are contracted as usual.
Only flavour 0 is computed, as in cvc: the two-point function is the same for
both. A site solved again, by another chain or by a chain that wraps, gets no
second 2+2. The source must be solved by a chain; targets need no propagator.

`--dummy-run` and the Python script warn about solved sites that are no 2+2
source and about 2+2 sources that no chain solves (no 2+2 for them); the first
10 of each are listed. A run prints how many 2+2 sources the chains solve.

### Memory

One source position is 4608 B per local site (2 flavours × 12 × 24 doubles).
The cache holds at most 16, for every direction. 128×64³ on 8 nodes (32 GPUs,
1,048,576 local sites):

| per rank | formula | size |
|---|---|---|
| one position | 4608 B × local sites | 4.5 GiB |
| host (`cache = host`) | 16 × 4.5 GiB + solver process (≈ 15 GiB) | ≈ 87 GiB |
| device (`cache = host`) | QUDA 11.4 + kernel stack 6.8 + 3 × 4.5 | ≈ 32 GiB |

Measured before `b677727` (kernel stack then 10.4 GiB): 83 GiB host (Slurm
`MaxRSS`), 36–38 GiB per GPU (`nvidia-smi`). On 4 nodes a position is 9 GiB
and 16 do not fit in host memory.

The kernel stack: the 4-point kernel uses 27,112 B of stack per thread (STACK
of `ker_4pt_contraction` in `cuobjdump --dump-resource-usage
build/hlbl_hypercube`; 41,280 B before `b677727`), the 2+2 kernels
(`kernel_p20`, `p21`, `p3`) 16,552 B. The CUDA driver reserves the largest
stack of the kernels launched for every thread slot (132 SMs × 2048), so 2+2
adds none: 6.8 GiB per GPU (6.64 GiB measured by `--dummy-run`). With 2+2 the
GPU also holds the buffers of `compute_2p2_gpu`: Π, 128 B per local site
(0.13 GiB at 32⁴), and P1, P23. The Python script tabulates memory per node
count from these numbers (`--kernel-stack-bytes`, `--kernel-stack-bytes-2p2`,
`--solver-gb`, `--solver-host-gb`, ...).

### Run time

```text
time = setup + invert-y × n_propagators + kernel_sum × 2 × n_pairs + 2p2 × n_2p2
```

At 8 nodes on 128×64³: setup ≈ 275 s (gauge read and multigrid, once per
job), invert-y ≈ 40.6 s (24 solves of one position), kernel_sum ≈ 2.8 s per
flavour. K = 1 (31 propagators, 80 pairs) took about 2000 s. Each further
window adds 15 × 40.6 s + 65 × 5.7 s ≈ 980 s, or 2.2 node-hours.

2+2: about 52 s per source with 16 targets (cvc `time_2p2` at 8 nodes,
`~/cvc_timing/cB/strong-2p2`), so a 2+2 at every new source adds about 780 s
per window. Its P2/P3 kernels run one block per target and geometry, at one
block per SM, so on a GH200 (132 SMs) up to 44 targets should cost about the
same as 16 (not measured). `--t-2p2` of the Python script takes this time.

### Output

One HDF5 file per run, written after every window, so a killed job keeps its
finished windows. The pairs are in the group `kernel_sum` (at the root in
files from before 2+2), the 2+2 pieces, with `2p2` lines, in the group `2p2`.

| dataset | shape | content |
|---|---|---|
| `kernel_sum/x`, `kernel_sum/y` | N × 4 int | the pair's sources (global sites, in [0, L)); `x` is the earlier one and the kernel origin |
| `kernel_sum/yv` | N × 4 int | y − x, minimum image in [−L/2, L/2); always (±n, ±n, ±n, ±n) |
| `kernel_sum/chain` | N int | chain index, in input order |
| `kernel_sum/window` | N int | window of that chain |
| `kernel_sum/kernel_sum` | N × 2 double | flavour 0, flavour 1 |
| `2p2/w` | M × 4 int | 2+2 source, global site in [0, L) |
| `2p2/chain`, `2p2/window` | M int | chain and window of its first solve |
| `2p2/P1` | M × 4 × 4 × 4 × Lmax double | P1[ρ][σ][ν][ζ]: Π_σν(z) summed over the sites with (z − w)_ρ mod L_ρ = ζ; 0 for ζ ≥ L_ρ |
| `2p2/pair_w`, `2p2/pair_y` | R × 4 int | source and target of each (source, target) row |
| `2p2/pair_yv` | R × 4 int | pair_y − pair_w, minimum image in [−L/2, L/2) |
| `2p2/P23` | R × 3 × 4 × 4 × 4 double | P2_0, P2_1, P3, each [ρ][σ][ν], only ρ < σ filled |
| `chains` | C × 10 int | chains as run: start (4), dir (4), stride, steps (resolved) |

Attributes: `lattice`, `procs`, `nconf`, `chains_columns`, `kernel`,
`flavour_order`, `coordinate_ranges`, `muon_mass_MeV`, `lattice_spacing_fm`,
`xunit`. With exactly
one chain also `direction`, `origin` and `stride`, as in files from before
chain lists. The group `2p2` has `rows`, `P1_index`, `P23_index` and
`flavour`.

The timing log has one CSV row per solve, per kernel sum and per 2+2 source:
`kind,chain,window,x0..x3,y0..y3,seconds,iterations`. A 2+2 row (kind `2p2`)
has its source as x.

## Running on Alps

The process grid in `invert.input` must give one rank per GPU, 4 per node:

```bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --uenv=prgenv-gnu/25.06:rc5
#SBATCH --view=default

ulimit -c 0                                  # no core dumps (they are huge)
export QUDA_RESOURCE_PATH=$PWD/quda_cache    # must be writable
mkdir -p $QUDA_RESOURCE_PATH
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0 QUDA_ENABLE_DSLASH_COARSE_POLICY=0
export QUDA_ENABLE_GDR=0 QUDA_ENABLE_P2P=0 QUDA_ENABLE_TUNING=1
export QUDA_REORDER_LOCATION=GPU CRAY_CUDA_MPS=1
export MPICH_RDMA_ENABLED_CUDA=1 MPICH_MAX_THREAD_SAFETY=multiple
export OMP_NUM_THREADS=72

srun hlbl_hypercube --in hlbl.input
```

Before the first production job at a node count, submit the same script with
`srun hlbl_hypercube --in hlbl.input --dummy-run --tune` (a few minutes). It
checks everything on the real resources and leaves QUDA's tune cache for the
production job.

- **GPUs.** Do not change `CUDA_VISIBLE_DEVICES`. Every rank sees its node's
  four GPUs and tmLQCD/QUDA picks one by node-local rank. A reordered list can
  make ranks share GPUs (about 2× slower); one GPU per rank fails in QUDA
  ("Too few GPUs available").
- **Local volume.** The multigrid in `invert.input.in` fits a local volume of
  at most 64×32×32×32, so use at least 16 GPUs for 128×64³.
- **check_pair** on 4 nodes:

  ```bash
  mkdir run && cd run
  ../scripts/gen_invert_input.sh -o . -T 128 -L 64 -p 2,2,2,2
  sbatch -A <account> -p <partition> --export=ALL,HLBL_BIN=/path/to/build/check_pair \
         ../scripts/slurm/check_pair.sbatch
  ```

## Status

- Schedule, contraction, cache, output and driver work and are tested on
  random fields.
- Real-solver run: 8 nodes, cB211a.072.64 configuration 500, windows 0 and 1
  (31 propagators, 80 pairs), `cache = host`. 34 minutes, cache peak 16 as
  predicted, QUDA plaquette matches the file (0.554279822834), and a rerun
  gives the same kernel sums.
- 2+2 pieces: `compute_2p2_gpu` at the first solve of each `2p2` source,
  tested on random fields.
- **Not done:** comparing the kernel sums and the 2+2 pieces with cvc's
  `hlbl_mII_invert_contract`. Correctness against production is still open.

## Possible extensions

- Keep each cached propagator on a random subset of sites (needs kernels that
  loop over a site list, and a check of the statistical error).
- Share propagators between overlapping chains; restart inside a window.
