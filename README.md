# hlbl: connected HLbL kernel sum on a sliding-hypercube schedule

`hlbl_hypercube` computes the connected hadronic light-by-light contribution
to the muon g-2 in position space (QED kernel LΛ, Λ = 0.4). It contracts all
pairs of point sources at one separation |y| = 2n. The pairs come from a
sliding-hypercube schedule that reuses most propagators from one step to the
next.

Everything runs on GPUs. Propagators are solved with tmLQCD/QUDA. The
contraction uses the GPU kernels from cvc's `hlbl_mII_invert_contract`, copied
to `kernels/`.

## The schedule

Let G be the 16 sign vectors (±1,±1,±1,±1), h ∈ G the step direction and n
the stride. Positions below are in units of n:

    A_0 = {0} ∪ (G \ {h})        16 sources
    A_k = k·h + A_0              window k

A pair (x, y) is contracted if y − x ∈ n·G, so every pair has |y − x| = 2n.
Neighbouring windows share one source. Window 0 needs 16 propagators and has
15 pairs. Every later window needs 15 new propagators and adds 65 pairs: 15
inside the window and 50 with the previous window. No pair reaches back two
windows. For K shifts (K + 1 windows):

    propagators = 16 + 15·K       (each: 12 spin-colours × 2 flavours = 24 solves)
    pairs       = 15 + 65·K       (each: 2 kernel sums, one per flavour)

**Order.** A new source is contracted with its partners right after it is
solved, and a propagator is freed when its last pair is done. At most 16
propagators are alive at once. (Solving a whole window before contracting it
would need 30.)

`scripts/hlbl_mII_hypercube_schedule.py` is the Python reference for the
schedule. It also estimates memory and run time (needs Python ≥ 3.8).

## Conventions

- **Site index:** `((t·LX + x)·LY + y)·LZ + z`, as in tmLQCD. Local extents and
  the process grid come from tmLQCD.
- **Propagator block** of one source: `[flavour][source spin-colour][site][sink
  spin-colour, complex]`, i.e. 2 × 12 × 24·V doubles. This is the layout of the
  `fwd_src` and `fwd_y` arguments of the kernels.
- **Flavour:** tmLQCD operator 0 has twisted mass +μ, operator 1 has −μ.
- **Pairs:** a pair (q, p) is contracted with q, the earlier source, as the
  kernel origin, and y = p − q (shortest way around the lattice). Pairs are
  unordered: the reversed pair gives a correlated copy of the same number.
- **Units:** kernel arguments are in units of the muon mass, `xunit = m_μ · a`
  with a = 0.07957 fm.
- **Gamma matrices:** cvc basis and signs.

## Layout

```
bin/            hlbl_hypercube (the program)
include/hlbl/   headers
src/            sources
kernels/        GPU kernels from cvc, see kernels/VENDORED.md
deps/kqed/      CMake build of KQED_cuda
sanity_check/   tests and check_pair
scripts/        Python reference, do_cmake.sh, input generator, Slurm script
```

Reference runs and their results are in `/users/ljingjin/hlbl_run` (not part
of the repository).

`hlbl_core` needs CUDA, MPI, KQED and HDF5. `hlbl_solver` adds tmLQCD/QUDA.
Tests that do not solve link only `hlbl_core`.

## Build

On Alps, inside the programming environment (`uenv start --view=default
prgenv-gnu/25.06:rc5`), run `scripts/do_cmake.sh`. It configures and builds in
`build/` and lists the tmLQCD, QUDA, lemon, c-lime and KQED paths that are
linked.

- `tmLQCD_DIR` must be the *installed* package config. The one in a tmLQCD
  build tree is missing the c-lime finder and configure fails.
- `KQED_SOURCE_DIR` is a KQED checkout at revision `deb4d488` (cloned into the
  build tree if empty). Keep it in place: KQED reads its tables from
  `KQED_SOURCE_DIR/PRECOMP` at run time.
- The build type is `Release` and the CUDA architecture is 90 (GH200) by
  default. `-DHLBL_WITH_TMLQCD=OFF` builds without the solver.
- The kernels in `hlbl_kernels_obj` must stay an OBJECT library. As a static
  library, nvlink can drop device code (see the comment in `CMakeLists.txt`).
- The tmLQCD/QUDA build must match the one that wrote the QUDA tune cache.
  QUDA aborts on a cache from another version, so use a new
  `QUDA_RESOURCE_PATH` after changing builds.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

| test | what it checks |
|---|---|
| `test_schedule` | window sizes, pair counts, completeness, cache peak of each mode |
| `compare_schedule` | the C++ schedule equals the Python reference |
| `test_gamma`, `test_lattice` | gamma algebra, site indexing and ownership |
| `test_contract_random` | the contraction on random fields (finite, matches a stored value) |
| `test_driver_random` | schedule loop on random fields: pair count, cache peak, device vs host cache, window ranges, HDF5 content |
| `hypercube_random_fields` | the whole program on random fields (80 pairs) |

`test_contract_random`, `test_driver_random` and `hypercube_random_fields`
need one GPU.

`check_pair` (not in ctest) solves two real point sources and contracts their
pair. It needs a gauge field, see "Running".

## Running

```
hlbl_hypercube [--direction HT HX HY HZ] [--stride n] [--steps K] [--origin T X Y Z]
               [--out results.h5] [--timing timing.csv]
               [--cache device|host] [--window-range a:b] [--force]
               [--dry-run [--show-window k] [--flavors 1|2] [--lattice T X Y Z]]
               [--random-fields --lattice T X Y Z --procs PT PX PY PZ]
```

- **Propagators.** By default they come from tmLQCD/QUDA. The lattice, process
  grid, gauge file and operators are read from `invert.input` in the working
  directory (create it with `scripts/gen_invert_input.sh`).
  `--random-fields` uses random numbers instead, on the lattice given in the
  command line. `--dry-run` only prints the schedule's counts and cache peak.
- **Steps.** `--steps K` runs K + 1 windows. The default is as many as fit
  before sources wrap around the lattice.
- **Cache.** `--cache device` keeps propagators on the GPU. `--cache host`
  keeps them in pinned host memory and copies the two needed for a pair to the
  GPU. The device then holds 3 blocks: the target of the solve and two staging
  slots (one for each propagator of the pair). Use `host` when the cache does
  not fit on the GPU.
- **Restart.** `--window-range a:b` runs windows a to b−1. The sources of
  earlier windows that window a needs are solved again first. Each job writes
  its own file.
- **Memory check.** At start the program compares the estimated memory with
  what is free and stops if it does not fit. `--force` skips this.

### Memory

One source position takes 4608 B per local site (2 flavours × 12 × 24 doubles).
The cache holds at most 16 positions. Example, 128×64³ on 8 nodes (32 GPUs,
1,048,576 local sites):

| per rank | formula | size |
|---|---|---|
| one position | 4608 B × local sites | 4.5 GiB |
| host (`--cache host`) | 16 × 4.5 GiB, plus the solver process (about 15 GiB) | about 87 GiB |
| device (`--cache host`) | QUDA 11.4 + kernel stack 10.5 + 3 × 4.5 | about 35 GiB |

Measured at 8 nodes: 83 GiB peak host memory (Slurm `MaxRSS`), 36 to 38 GiB
peak per GPU (`nvidia-smi`). On 4 nodes a position is 9 GiB and 16 of them do
not fit in host memory.

### Run time

    time = setup + invert-y × n_propagators + kernel_sum × 2 × n_pairs

with, at 8 nodes on 128×64³: setup ≈ 275 s (gauge read, multigrid setup),
invert-y ≈ 40.6 s (24 solves of one source position), kernel_sum ≈ 2.8 s per
flavour. For K = 1 (31 propagators, 80 pairs) that is about 2000 s measured. Each
further window adds about 15 × 40.6 s + 65 × 5.7 s ≈ 980 s, which is 2.2
node-hours.

### Output

One HDF5 file per run. Rows are written after every window, so a killed job
keeps the finished windows.

| dataset | shape | content |
|---|---|---|
| `x`, `y` | N × 4 int | the two sources of the pair (global sites); `x` is the earlier one and the kernel origin |
| `yv` | N × 4 int | separation y − x, always (±n, ±n, ±n, ±n) |
| `window` | N int | window in which the pair was contracted |
| `kernel_sum` | N × 2 double | flavour 0, flavour 1 |

Attributes: `lattice`, `procs`, `direction`, `origin`, `stride`, `nconf`,
`window_begin`, `window_end`, `kernel`, `flavour_order`, `hlbl_revision`.

`--timing` (default: the result file name with `.timing.csv`) writes one CSV
row per solve and per kernel sum (`kind,window,x0..x3,y0..y3,seconds,iterations`).

## Running on Alps

Slurm job (adapt nodes and times; the process grid in `invert.input` must give
one rank per GPU, 4 per node):

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

srun hlbl_hypercube --stride 4 --cache host --window-range 0:2 --out result.h5
```

**GPUs.** Do not change `CUDA_VISIBLE_DEVICES`. Every rank sees all four GPUs
of its node and tmLQCD/QUDA picks one by node-local rank. If the list is
reordered, ranks can end up sharing GPUs and everything runs about twice as
slow. Giving each rank a single GPU fails in QUDA ("Too few GPUs available").

**Local volume.** The multigrid setup in `invert.input.in` fits on a GPU for
a local volume of up to 64×32×32×32. Larger volumes fail with out-of-memory
errors, so use at least 16 GPUs for 128×64³.

**check_pair.** `scripts/slurm/check_pair.sbatch` runs it on 4 nodes:

```bash
mkdir run && cd run
../scripts/gen_invert_input.sh -o . -T 128 -L 64 -p 2,2,2,2
sbatch -A <account> -p <partition> --export=ALL,HLBL_BIN=/path/to/build/check_pair \
       ../scripts/slurm/check_pair.sbatch
```

## Status

- The schedule, contraction, cache, output and driver work, and are tested on
  random fields.
- Real-solver run: 8 nodes, cB211a.072.64 configuration 500, windows 0 and 1
  (31 propagators, 80 pairs) with `--cache host`. It ran in 34 minutes and
  reached the predicted peak of 16 positions. The QUDA plaquette matches the
  file (0.554279822834). Repeating the run gives the same kernel sums.
- **Not done:** comparison of the kernel sums with cvc's
  `hlbl_mII_invert_contract`. Correctness against production is still open.

## Possible extensions

- Keep each cached propagator on a random subset of sites only (needs kernels
  that loop over a site list, and a check of the statistical error).
- The 2+2 contribution. The kernels are compiled, but it needs ordered pairs.
- Several chains or directions per job, and restart from inside a window.
