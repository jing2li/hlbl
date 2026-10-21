# hlbl: connected HLbL kernel sum on a sliding-hypercube schedule

`hlbl_hypercube` computes the connected hadronic light-by-light contribution
to the muon g−2 in position space (QED kernel LΛ, Λ = 0.4). It contracts all
pairs of point sources at one separation |y| = 2n, taken from a
sliding-hypercube schedule that reuses most propagators from step to step. One
job runs a list of such chains; the gauge field is read and the multigrid set
up once.

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
after its last pair, so at most 16 are alive, for every h. That needs the
solve order of h = (1,1,1,1), reflected by h; other orders peak at up to 25,
and solving a whole window first needs 30.

`scripts/hlbl_mII_hypercube_schedule.py` (Python ≥ 3.8) is the reference
implementation. It also estimates memory and run time, for one chain or for an
input file (`--in`).

## Conventions

- **Site index:** `((t·LX + x)·LY + y)·LZ + z`, as in tmLQCD, which also sets
  the local extents and process grid.
- **Propagator block** of one source: `[flavour][source spin-colour][site][sink
  spin-colour, complex]`, 2 × 12 × 24·V doubles (the kernels' `fwd_src` and
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
deps/kqed/      CMake build of KQED_cuda
sanity_check/   tests and check_pair
scripts/        Python reference, build, input generator, Slurm script
```

`hlbl_core` needs CUDA, MPI, KQED and HDF5; `hlbl_solver` adds tmLQCD/QUDA.
Reference runs are in `/users/ljingjin/hlbl_run` (not in the repository).

## Build

On Alps, inside `uenv start --view=default prgenv-gnu/25.06:rc5`, run
`scripts/do_cmake.sh`. It configures and builds in `build/`.

- `tmLQCD_DIR` must be the *installed* package config; the build tree's lacks
  the c-lime finder.
- `KQED_SOURCE_DIR` is a KQED checkout at `deb4d488` (cloned if empty). Keep
  it in place: KQED reads `PRECOMP/` from it at run time.
- Defaults: `Release`, CUDA architecture 90 (GH200). `-DHLBL_WITH_TMLQCD=OFF`
  builds without the solver.
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
| `test_schedule` | window sizes, pair counts, completeness, cache peaks; peak 16 for all 16 directions |
| `compare_schedule` | C++ schedule = Python reference, also for multi-chain input files (counts, totals, overlaps) and bad-file errors |
| `test_input` | input parser: defaults, comments, every error |
| `test_gamma`, `test_lattice` | gamma algebra; site indexing and ownership |
| `test_contract_random` | contraction on random fields: finite, matches a stored value |
| `test_driver_random` | schedule loop on random fields: pair count, cache peak, device vs host cache, window ranges, two chains vs independent runs, HDF5 content |
| `hypercube_random_fields` | whole program on random fields (80 pairs) |
| `hypercube_random_fields_multi` | whole program on `sanity_check/multi_chain.input` (2 chains, 210 pairs) |
| `hypercube_dummy_run` | `--dummy-run` on the same file |

`test_contract_random`, `test_driver_random` and `hypercube_*` need one GPU.
`check_pair` (not in ctest) solves two real point sources and contracts their
pair; see "Running on Alps".

## Running

```text
hlbl_hypercube --in hlbl.input [--out results.h5] [options below]

hlbl_hypercube [--direction HT HX HY HZ] [--stride n] [--steps K] [--origin T X Y Z]
               [--window-range a:b]                  # one chain
               [--muon-mass MeV --lattice-spacing fm]
               [--out results.h5] [--timing timing.csv]
               [--cache device|host]
               [--dummy-run [--tune]]
               [--dry-run [--show-window k] [--flavors 1|2] [--lattice T X Y Z]]
               [--random-fields --lattice T X Y Z --procs PT PX PY PZ]
```

- **Input file.** `--in` reads chains and options; the command line overrides
  it. Without `--in`, `--direction --stride --steps --origin --window-range`
  describe one chain (not combinable with `chain` lines).
- **Physics inputs.** `muon_mass` and `lattice_spacing` (or `--muon-mass`,
  `--lattice-spacing`) are required except for `--dry-run`, and are stored in
  the result file.
- **Propagators** come from tmLQCD/QUDA, which reads lattice, process grid,
  gauge file and operators from `invert.input` in the working directory
  (written by `scripts/gen_invert_input.sh`). `--random-fields` uses random
  numbers on `--lattice` instead.
- **Steps.** `--steps K` runs K + 1 windows; the default is as many as fit
  before sources wrap around the lattice.
- **Cache.** `device` keeps propagators on the GPU. `host` keeps them in
  pinned host memory, and the GPU holds only 3 blocks (solve target, two
  staging slots). Use `host` when the cache does not fit on the GPU.
- **Restart.** `--window-range a:b` (`windows=a:b` on a chain line) runs
  windows a to b−1, first solving again the earlier sources window a needs.
  Each job writes its own file. To resume a killed multi-chain job, drop the
  finished chains and set `windows=` on the one that was cut.

### Checks

A production run checks only its input. Before submitting:

- `scripts/hlbl_mII_hypercube_schedule.py --in hlbl.input --procs ...` (no
  GPU): schedules, overlaps, physics inputs, memory per rank, cost. Ends with
  a PRECHECK block; exit status 1 on errors.
- `--dry-run` (no GPU): counts and cache peak per chain and in total, and the
  overlap warnings. Counts are for whole chains, ignoring window ranges.
- `--dummy-run` (on the job's resources): the full setup (gauge field, QUDA
  multigrid, QED kernel), every check, the memory estimate on the worst rank,
  then the memory peak for real: cache filled, one position solved (24
  solves), one pair contracted. Writes no results; ends with `dummy run: all
  checks passed` (status 0) or `FAILED` (status 1). `--tune` makes QUDA tune
  and save its tune cache to `QUDA_RESOURCE_PATH`, so a production run with
  the same build, node count and path starts tuned.

```bash
python3.11 scripts/hlbl_mII_hypercube_schedule.py --in hlbl.input --procs 4 2 2 2 \
    --t-invert 40.6 --t-kernel-sum 2.8 --t-setup 275       # no GPU
srun build/hlbl_hypercube --in hlbl.input --dummy-run --tune  # in the job's allocation
```

### Input file

`#` starts a comment. Global lines are `key = value`, chain lines
`chain key=value ...` with vectors written `t,x,y,z`:

```text
muon_mass       = 105.6583745   # MeV
lattice_spacing = 0.07957       # fm
cache = host
chain start=0,0,0,0   dir=1,1,1,1   stride=4 steps=30
chain start=32,0,0,0  dir=1,-1,1,1  stride=4 steps=30 windows=10:20
chain start=0,16,0,0  dir=1,1,1,1   stride=6
```

| chain key | meaning |
|---|---|
| `start` (required) | first source, global coordinates (`--origin`) |
| `dir` (required) | step direction, a sign vector (`--direction`) |
| `stride` (required) | n; every pair has \|y\| = 2n |
| `steps` | K, windows 0..K; `auto` (default): as many as fit before wrapping |
| `windows` | `a:b` runs windows a to b−1; `a:` to the end, `:b` from 0 |

Global keys: `muon_mass`, `lattice_spacing`, `cache`, `out`, `timing`, and
for dry runs and random fields `random_fields`, `lattice`, `procs`,
`flavors`. `scripts/input/hlbl.input.example` lists them all.

Chains run one after the other with one cache, so memory is that of the
largest chain. They are independent: a shared source is solved, and a shared
pair computed and stored, once per chain. `--dry-run`, `--dummy-run` and the
Python script warn about this:

```text
WARNING: 39 source positions are in more than one chain (solved once per chain) and 119 pairs are computed in more than one chain
chains 0 and 2 share 31 sources and 80 pairs
  source (4,4,4,4) in window 2 of chain 0 and window 0 of chain 2
  source (4,4,4,8) in window 3 of chain 0 and window 0 of chain 2
  ... and 29 more sources
```

Sources are in wrapped global coordinates, with the window of each chain that
solves them, so `windows=` can cut the overlap. Only the windows a chain runs
count. At most 20 pairs of chains are listed, each with its first 10 shared
sources.

### Memory

One source position is 4608 B per local site (2 flavours × 12 × 24 doubles).
The cache holds at most 16, for every direction and after a `windows=`
restart. 128×64³ on 8 nodes (32 GPUs, 1,048,576 local sites):

| per rank | formula | size |
|---|---|---|
| one position | 4608 B × local sites | 4.5 GiB |
| host (`--cache host`) | 16 × 4.5 GiB + solver process (≈ 15 GiB) | ≈ 87 GiB |
| device (`--cache host`) | QUDA 11.4 + kernel stack 10.4 + 3 × 4.5 | ≈ 35 GiB |

Measured: 83 GiB host (Slurm `MaxRSS`), 36–38 GiB per GPU (`nvidia-smi`). On
4 nodes a position is 9 GiB and 16 do not fit in host memory.

The kernel stack: the 4-point kernel uses 41,280 B of stack per thread (STACK
of `ker_4pt_contraction` in `cuobjdump --dump-resource-usage
build/hlbl_hypercube`), and the CUDA driver reserves that for every thread
slot (132 SMs × 2048) at the first launch. The Python script tabulates memory
per node count from these numbers (`--kernel-stack-bytes`, `--solver-gb`,
`--solver-host-gb`, ...).

### Run time

```text
time = setup + invert-y × n_propagators + kernel_sum × 2 × n_pairs
```

At 8 nodes on 128×64³: setup ≈ 275 s (gauge read and multigrid, once per
job), invert-y ≈ 40.6 s (24 solves of one position), kernel_sum ≈ 2.8 s per
flavour. K = 1 (31 propagators, 80 pairs) took about 2000 s. Each further
window adds 15 × 40.6 s + 65 × 5.7 s ≈ 980 s, or 2.2 node-hours.

### Output

One HDF5 file per run, written after every window, so a killed job keeps its
finished windows.

| dataset | shape | content |
|---|---|---|
| `x`, `y` | N × 4 int | the pair's sources (global sites); `x` is the earlier one and the kernel origin |
| `yv` | N × 4 int | y − x, always (±n, ±n, ±n, ±n) |
| `chain` | N int | chain index, in input order |
| `window` | N int | window of that chain |
| `kernel_sum` | N × 2 double | flavour 0, flavour 1 |
| `chains` | C × 12 int | chains as run: start (4), dir (4), stride, steps, window_begin, window_end (half-open, steps resolved) |

Attributes: `lattice`, `procs`, `nconf`, `chains_columns`, `kernel`,
`flavour_order`, `hlbl_revision`, `muon_mass_MeV`, `lattice_spacing_fm`,
`xunit`. With exactly one chain also `direction`, `origin`, `stride`,
`window_begin`, `window_end`, as in files from before chain lists.

`--timing` (default: the result file name with `.timing.csv`) logs one CSV row
per solve and per kernel sum: `kind,chain,window,x0..x3,y0..y3,seconds,iterations`.

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

srun hlbl_hypercube --in hlbl.input --out result.h5
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
  (31 propagators, 80 pairs), `--cache host`. 34 minutes, cache peak 16 as
  predicted, QUDA plaquette matches the file (0.554279822834), and a rerun
  gives the same kernel sums.
- **Not done:** comparing the kernel sums with cvc's
  `hlbl_mII_invert_contract`. Correctness against production is still open.

## Possible extensions

- Keep each cached propagator on a random subset of sites (needs kernels that
  loop over a site list, and a check of the statistical error).
- The 2+2 contribution: the kernels are compiled, but it needs ordered pairs.
- Share propagators between overlapping chains; restart inside a window.
