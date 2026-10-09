#!/usr/bin/env python3.11
"""
hlbl_mII_hypercube_schedule.py

Prototype of the "sliding hypercube" source schedule for the connected
(kernel-sum) part of hlbl method II, targeting ONE separation |y| = 2n.

No physics is computed. Inversion and kernel sum are placeholders that do
nothing; what the script produces is

  * the schedule itself (which propagators, which pairs, in which order),
  * a check that the schedule finds every |y| = 2n pair among its sources,
  * the number of inversions and kernel-sum calls   -> node hours,
  * the peak propagator-cache size, and with the solver and contraction
    reserve the GPU memory per rank                 -> memory.

The scheme
----------
Let G be the 16 sign vectors (+-1,+-1,+-1,+-1), h in G the step direction and
n the stride. In units of n:

    A0 = {0} u {g in G : g != h}             16 source points
    A_k = k*h + A0                           window k

A_k and A_{k-1} share exactly one point: k*h + (-h) = (k-1)*h. So window 0
costs 16 propagators and every later window costs 15.

A pair (x, y) is contracted iff y - x in n*G, i.e. y - x = (+-n,+-n,+-n,+-n),
|y - x| = 2n. Window 0 gives 15 pairs (0 with each g). Each further window
gives 65: 15 inside the window and 50 with window k-1. Nothing ever pairs
with window k-2, so a window can be dropped once the next one is contracted.
`check_complete` verifies all of this by brute force for the schedule you ask
for.

Pairs are unordered: (x, y) and (y, x) are the same pair. See
hlbl_mII_xy_exchange_note.tex: the reversed order is a strongly correlated
duplicate, not new physics.

Execution modes (they differ only in when propagators are held)
---------------------------------------------------------------
  serial       invert all of window k, then do all its kernel sums. Two
               windows are live, 30 propagators.
  pipelined    window k+1 is inverted *while* window k's kernel sums run
               (e.g. solver and contraction on different resources), so the
               next window's propagators are already in the cache. Three
               windows are live, 45 propagators.
  interleaved  each source is contracted with its partners right after its
               own solve, in the order of Step.sources. 16 propagators at the
               peak (steady state). This is what the driver does.

In every mode a propagator is evicted as soon as its last pair is done.

Sparsening
----------
--sparsen F keeps each cached propagator only on a fixed, predetermined set of
V/F sites (the stochastically chosen x and z of the kernel sum), so the cache
shrinks by F. The set must be the same for every propagator: the contraction
at x needs both propagators at x. The solver still works on the full volume,
but each of the 12 solutions can be sparsened as soon as it is done, so the
only full-volume fields are the solver's own. The kernel-sum time also
changes with F, so time it with the same sparsening.

Memory model (per rank, GiB)
----------------------------
    peak positions x size of one position + QUDA solver overhead <= GPU memory

  cache        peak source positions x 4608 B (2 flavours x 12 x 24 doubles)
               x local sites
  solver       QUDA's peak device allocation of the production multigrid solver
               (time_hlbl/input/invert.input.in), measured with time_invert at
               8/16/32/64 nodes (QUDA_DEV_GIB, from strong-invert/memory.txt).
               Only those node counts have data; other rows print "no data".
               --solver-gb replaces the measurement with one number.

  Not modelled: the CUDA context, cuBLAS/MPI buffers and the driver's stack
  reservation of the contraction kernel (about 10 GiB per rank for the 4-point
  kernel of cuda_lattice.cu). QUDA's counters do not show them; --reserve-gb
  adds a flat amount for them.

Usage (Python >= 3.8; the system python3 on the login node is 3.6)
-----------------------------------------------------------------
    python3.11 hlbl_mII_hypercube_schedule.py --stride 4 --steps 10
    python3.11 hlbl_mII_hypercube_schedule.py --stride 4 --show-pairs 1
    python3.11 hlbl_mII_hypercube_schedule.py --stride 8 --lattice 160 80 80 80 \\
        --procs 4 2 2 2 --t-invert 300 --t-kernel-sum 40 --configs 100
"""

import argparse
import itertools
import math
from collections import Counter
from dataclasses import dataclass, field

SIGN_VECTORS = tuple(itertools.product((-1, 1), repeat=4))
ZERO = (0, 0, 0, 0)
N_SPIN_COLOR = 12          # point-source components per flavor
GB = 1024.0 ** 3
MODES = ("serial", "pipelined", "interleaved")

# QUDA's peak device allocation per rank (GiB) with the production multigrid
# solver on cB211a.072.64 (128x64^3), 4 ranks per node, rank 0 high-water mark from
# strong-invert/memory.txt: (local sites, GiB) at 8, 16, 32, 64 nodes.
QUDA_DEV_GIB = ((1048576, 11.38), (524288, 5.89), (262144, 3.04), (131072, 1.56))

GPU_MEM_GB = 95.6         # nvidia-smi memory.total of a GH200: 97871 MiB


# =====================================================================
# 4-vector helpers (plain tuples, lattice units)
# =====================================================================

def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def scale(s, a):
    return tuple(s * x for x in a)


def wrap(a, lattice):
    return tuple(x % L for x, L in zip(a, lattice))


def unordered(p, q):
    return frozenset((p, q))


# =====================================================================
# The schedule
# =====================================================================

def window_offsets(h):
    """A0 = {0} u {g in G : g != h}, in units of the stride."""
    return [ZERO] + [g for g in SIGN_VECTORS if g != h]


@dataclass
class Source:
    """One new propagator and the already-available propagators it pairs with."""
    position: tuple                               # unwrapped, lattice units
    partners: list = field(default_factory=list)  # earlier positions


@dataclass
class Step:
    index: int
    sources: list  # the propagators new in this window, in inversion order

    @property
    def n_new(self):
        return len(self.sources)

    @property
    def n_pairs(self):
        return sum(len(s.partners) for s in self.sources)


def build_schedule(h, stride, n_steps, origin=ZERO):
    """Windows 0..n_steps. Each new source is paired with every source already
    available (previous window + earlier sources of this window) that sits at a
    separation in stride*G."""
    offsets = window_offsets(h)
    separations = {scale(stride, g) for g in SIGN_VECTORS}

    steps = []
    previous_window = []
    for k in range(n_steps + 1):
        window = [add(origin, scale(stride, add(scale(k, h), a))) for a in offsets]
        available = list(previous_window)
        new_sources = []
        for p in window:
            if p in available:           # the one point shared with window k-1
                continue
            partners = [q for q in available if sub(p, q) in separations]
            new_sources.append(Source(p, partners))
            available.append(p)
        steps.append(Step(k, new_sources))
        previous_window = window
    return steps


def all_positions(steps):
    return [s.position for step in steps for s in step.sources]


def all_pairs(steps):
    return [(q, s.position) for step in steps for s in step.sources for q in s.partners]


def check_complete(steps, stride):
    """Brute force: every pair of scheduled sources at separation stride*G must be
    in the schedule exactly once. Fails if the window logic ever needs a
    propagator that was already dropped."""
    separations = {scale(stride, g) for g in SIGN_VECTORS}
    positions = all_positions(steps)
    assert len(set(positions)) == len(positions), "a propagator is scheduled twice"

    expected = {unordered(p, q) for p, q in itertools.combinations(positions, 2)
                if sub(p, q) in separations}
    scheduled = [unordered(p, q) for p, q in all_pairs(steps)]
    assert len(set(scheduled)) == len(scheduled), "a pair is scheduled twice"
    missing = expected - set(scheduled)
    assert not missing, f"{len(missing)} pairs at |y|=2n are not scheduled"
    assert set(scheduled) == expected


def max_steps_without_wrap(h, stride, lattice, origin=ZERO, cap=10000):
    """Largest n_steps for which no two sources land on the same lattice site
    after periodic wrapping, i.e. roughly one lap around the lattice."""
    offsets = window_offsets(h)
    seen = set()
    for k in range(cap):
        window = {wrap(add(origin, scale(stride, add(scale(k, h), a))), lattice)
                  for a in offsets}
        new = window if k == 0 else window - {wrap(add(origin, scale(stride * (k - 1), h)), lattice)}
        if new & seen:
            return k - 1
        seen |= new
    return cap


# =====================================================================
# Placeholders for the expensive parts
# =====================================================================

@dataclass(frozen=True)
class Propagator:
    """Stand-in. A real one holds, per flavor, (12, volume, 12) complex:
    12 spin-color point sources x sink spinor-color at every site."""
    site: tuple


def invert_propagator(site, n_flavors):
    """Placeholder for 12 * n_flavors solves from a point source at `site`
    (the production `invert-y` block)."""
    return Propagator(site)


def kernel_sum(prop_x, prop_y, yv, flavor):
    """Placeholder for dzu/dzsu + the 4-point contraction summed against the QED
    kernel (the production `kernel-sum` region), source at x, y = yv."""
    return 0.0


# =====================================================================
# Running the schedule, with a propagator cache
# =====================================================================

class PropagatorCache:
    """Holds propagators; evicts one as soon as its last pair has been done."""

    def __init__(self, pending):
        self.pending = Counter(pending)  # position -> pairs still to do
        self.props = {}
        self.peak = 0

    def add(self, position, prop):
        self.props[position] = prop
        self.peak = max(self.peak, len(self.props))

    def __getitem__(self, position):
        return self.props[position]

    def pair_done(self, p, q):
        for pos in (p, q):
            self.pending[pos] -= 1
            if self.pending[pos] == 0:
                del self.props[pos]


@dataclass
class RunStats:
    mode: str
    n_propagators: int = 0
    n_solves: int = 0
    n_pairs: int = 0
    n_kernel_sums: int = 0
    peak_cache: int = 0
    per_step: list = field(default_factory=list)  # (step, new, pairs, cache after)


def run(steps, lattice, n_flavors, mode):
    pending = Counter()
    for older, newer in all_pairs(steps):
        pending[older] += 1
        pending[newer] += 1
    cache = PropagatorCache(pending)
    stats = RunStats(mode)

    def invert(position):
        cache.add(position, invert_propagator(wrap(position, lattice), n_flavors))
        stats.n_propagators += 1
        stats.n_solves += N_SPIN_COLOR * n_flavors

    def contract(older, newer):
        yv = sub(newer, older)  # (+-n, +-n, +-n, +-n)
        for flavor in range(n_flavors):
            kernel_sum(cache[older], cache[newer], yv, flavor)
            stats.n_kernel_sums += 1
        stats.n_pairs += 1
        cache.pair_done(older, newer)

    for k, step in enumerate(steps):
        if mode == "interleaved":
            for src in step.sources:
                invert(src.position)
                for q in src.partners:
                    contract(q, src.position)
            stats.per_step.append((k, step.n_new, step.n_pairs, len(cache.props)))
            continue
        if mode == "serial" or k == 0:
            for src in step.sources:
                invert(src.position)
        if mode == "pipelined" and k + 1 < len(steps):
            # next window inverts concurrently with this window's kernel sums
            for src in steps[k + 1].sources:
                invert(src.position)
        for src in step.sources:
            for q in src.partners:
                contract(q, src.position)
        stats.per_step.append((k, step.n_new, step.n_pairs, len(cache.props)))

    assert not cache.props, "propagators left in the cache after the last pair"
    stats.peak_cache = cache.peak
    return stats


# =====================================================================
# Reports
# =====================================================================

def print_pairs(steps, k, origin, stride):
    """The pairs closed in window k, in units of the stride relative to origin
    (same layout as the hand-written table)."""
    def rel(p):
        return tuple((x - o) // stride for x, o in zip(p, origin))

    step = steps[k]
    print(f"\nwindow {k}: {step.n_new} new propagators, {step.n_pairs} pairs (units of stride)")
    print(f"  {'new offset':<20} pairs with already available offsets")
    for src in step.sources:
        partners = ", ".join(str(rel(q)) for q in sorted(src.partners))
        print(f"  {str(rel(src.position)):<20} [{len(src.partners)}] {partners}")


def print_orientations(steps):
    """How often each of the 8 orientations +-g of y is sampled."""
    counts = Counter()
    for older, newer in all_pairs(steps):
        g = tuple(1 if c > 0 else -1 for c in sub(newer, older))
        counts[max(g, scale(-1, g))] += 1
    print("\norientation balance, y ~ +-g:")
    for g in sorted(counts):
        print(f"  {str(g):<20} {counts[g]}")


def find_process_grid(lattice, ranks, local_multiple):
    """An MPI grid (PT,PX,PY,PZ) with product `ranks` such that every local extent
    is a multiple of local_multiple = (T, X/Y/Z). Among the valid grids, the one
    with the most even local box. None if there is none."""
    mt, mx = local_multiple
    best = None
    divisors = [d for d in range(1, ranks + 1) if ranks % d == 0]
    for grid in itertools.product(divisors, repeat=4):
        if math.prod(grid) != ranks:
            continue
        local = [L // P for L, P in zip(lattice, grid)]
        if any(L % P for L, P in zip(lattice, grid)):
            continue
        if local[0] % mt or any(l % mx for l in local[1:]):
            continue
        spread = max(local) / min(local)
        if best is None or spread < best[0]:
            best = (spread, grid)
    return None if best is None else best[1]


def quda_dev_gb(local_sites, points=QUDA_DEV_GIB):
    """Measured QUDA peak device allocation per rank [GiB] at exactly this local
    volume, or None if there is no measurement."""
    for sites, gib in points:
        if sites == local_sites:
            return gib
    return None


def memory_report(stats_by_mode, lattice, procs, n_flavors, precision, sparsen,
                  gpu_mem_gb, host_mem_gb, gpus_per_node, cache_limit,
                  nodes_scan, local_multiple, solver_gb=None, reserve_gb=0.0):
    volume = math.prod(lattice)
    bytes_complex = 16 if precision == "double" else 8
    kept_sites = volume / sparsen
    per_flavor = N_SPIN_COLOR * N_SPIN_COLOR * bytes_complex * kept_sites
    per_position = n_flavors * per_flavor

    def fit(cache, block, res):
        """GPU: cache + reserve on the device. host: cache in pinned host memory,
        the device holds the reserve and three blocks (two staging slots and a
        solve target). -: neither."""
        if cache + res <= gpu_mem_gb * GB:
            return "GPU"
        if block * 3 + res <= gpu_mem_gb * GB and cache <= host_mem_gb * GB:
            return "host"
        return "-"

    print("\n" + "=" * 78)
    print("MEMORY  (peak positions x position size + QUDA solver)")
    print("=" * 78)
    print(f"lattice {tuple(lattice)} = {volume:,} sites, {precision} precision")
    if sparsen > 1:
        print(f"sparsened: 1 site in {sparsen:g} kept -> {kept_sites:,.0f} sites per propagator")
    print(f"one flavor propagator  12x12 x {bytes_complex} B x sites = {per_flavor / GB:10.2f} GB")
    print(f"one source position    x {n_flavors} flavors            = {per_position / GB:10.2f} GB")
    print(f"budget per rank: {gpu_mem_gb:g} GB GPU, {host_mem_gb:g} GB host")
    if solver_gb is None:
        print("solver: QUDA peak measured with time_invert (8/16/32/64 nodes, 128x64^3)")
    else:
        print(f"solver: {solver_gb:g} GB (--solver-gb)")
    if reserve_gb:
        print(f"plus {reserve_gb:g} GB flat reserve (--reserve-gb)")
    print()
    print(f"  {'mode':<12} {'positions':>9} {'flavor-props':>12} {'total GB':>10} {'<= limit':>8}")
    for mode, st in stats_by_mode.items():
        print(f"  {mode:<12} {st.peak_cache:>9} {st.peak_cache * n_flavors:>12} "
              f"{st.peak_cache * per_position / GB:>10.1f} "
              f"{'yes' if st.peak_cache <= cache_limit else 'no':>8}")
    print(f"  'positions' = source points held at once (all flavors each);")
    print(f"  '<= limit' compares positions with --cache-limit {cache_limit}.")

    print(f"\n  GB per rank by node count ({gpus_per_node} GPUs = ranks per node): "
          f"cache, and cache + solver = total")
    print(f"  fits: GPU = everything on the device, host = cache in pinned host memory "
          f"(device keeps solver + 3 blocks), - = neither")
    print(f"  grid: local extents multiples of T {local_multiple[0]}, "
          f"X/Y/Z {local_multiple[1]} (--local-multiple)")
    header = f"  {'nodes':>5} {'ranks':>5}  {'grid':<10} {'solver':>7}"
    for mode in stats_by_mode:
        header += f"  {mode:>9}: {'cache':>6} {'total':>6} {'fits':>4}"
    print(header)
    rows = sorted(set(nodes_scan) | {math.prod(procs) // gpus_per_node})
    for nodes in rows:
        ranks = nodes * gpus_per_node
        grid = find_process_grid(lattice, ranks, local_multiple)
        solver = solver_gb if solver_gb is not None else quda_dev_gb(volume // ranks)
        grid_s = "none" if grid is None else "x".join(map(str, grid))
        if solver is None:
            print(f"  {nodes:>5} {ranks:>5}  {grid_s:<10} {'no data':>7}")
            continue
        res_b = (solver + reserve_gb) * GB
        line = f"  {nodes:>5} {ranks:>5}  {grid_s:<10} {solver:>7.2f}"
        for st in stats_by_mode.values():
            cache = st.peak_cache * per_position / ranks
            # no process grid -> the solver cannot run there, whatever the memory says
            verdict = "-" if grid is None else fit(cache, per_position / ranks, res_b)
            line += f"  {'':>9}  {cache / GB:>6.1f} {(cache + res_b) / GB:>6.1f} {verdict:>4}"
        print(line)
    print("  'none': no process grid of that size fits the lattice and the MG blocking.")
    print("  'no data': QUDA was not measured at this local volume.")


def cost_report(stats, chains, configs, procs, gpus_per_node,
                t_invert, t_kernel_sum, t_setup):
    ranks = math.prod(procs)
    nodes = ranks / gpus_per_node
    n_prop = stats.n_propagators * chains
    n_ks = stats.n_kernel_sums * chains

    print("\n" + "=" * 78)
    print("CALL COUNTS AND NODE HOURS")
    print("=" * 78)
    print(f"per chain:  {stats.n_propagators} propagators ({stats.n_solves} solves), "
          f"{stats.n_pairs} pairs, {stats.n_kernel_sums} kernel-sum calls")
    print(f"per config: {chains} chain(s) -> {n_prop} propagators, {n_ks} kernel-sum calls")
    print(f"all:        {configs} config(s) -> {n_prop * configs} propagators, "
          f"{n_ks * configs} kernel-sum calls")

    if t_invert is None or t_kernel_sum is None:
        print("\n(pass --t-invert and --t-kernel-sum for node hours)")
        print("node_hours = nodes * configs * (t_setup + N_prop*t_invert + N_ks*t_kernel_sum) / 3600")
        return

    t_inv = n_prop * t_invert
    t_ks = n_ks * t_kernel_sum
    t_cfg = t_setup + t_inv + t_ks
    print(f"\nper config on {nodes:g} nodes (serial wall time):")
    print(f"  setup         {t_setup:>12.0f} s")
    print(f"  inversions    {t_inv:>12.0f} s   ({n_prop} x {t_invert:g} s)")
    print(f"  kernel sums   {t_ks:>12.0f} s   ({n_ks} x {t_kernel_sum:g} s)")
    print(f"  total         {t_cfg:>12.0f} s = {t_cfg / 3600:.2f} h "
          f"-> {nodes * t_cfg / 3600:.1f} node-h")
    print(f"all {configs} configs: {configs * nodes * t_cfg / 3600:,.0f} node-h")


# =====================================================================
# main
# =====================================================================

def main():
    p = argparse.ArgumentParser(
        description="Sliding-hypercube source schedule for the hlbl connected kernel sum.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("schedule")
    g.add_argument("--direction", type=int, nargs=4, default=[1, 1, 1, 1],
                   metavar=("HT", "HX", "HY", "HZ"), help="step direction h, a sign vector")
    g.add_argument("--stride", type=int, default=4, help="n; target separation |y| = 2n")
    g.add_argument("--steps", type=int, default=None,
                   help="number of shifts (windows = steps+1); default: as many as fit "
                        "before sources wrap onto each other")
    g.add_argument("--origin", type=int, nargs=4, default=[0, 0, 0, 0], metavar=("T", "X", "Y", "Z"))
    g.add_argument("--flavors", type=int, default=2, choices=(1, 2),
                   help="flavors inverted per position, and kernel-sum calls per pair")
    g.add_argument("--mode", choices=MODES, default="interleaved",
                   help="execution order used for the per-step table")

    g = p.add_argument_group("lattice and machine")
    g.add_argument("--lattice", type=int, nargs=4, default=[128, 64, 64, 64],
                   metavar=("T", "LX", "LY", "LZ"))
    g.add_argument("--procs", type=int, nargs=4, default=[2, 2, 2, 2],
                   metavar=("PT", "PX", "PY", "PZ"), help="MPI grid, one GPU per rank")
    g.add_argument("--gpus-per-node", type=int, default=4)
    g.add_argument("--gpu-mem-gb", type=float, default=GPU_MEM_GB,
                   help="per GPU, as nvidia-smi memory.total (GH200: 97871 MiB)")
    g.add_argument("--host-mem-gb", type=float, default=120.0, help="per rank (GH200 Grace)")
    g.add_argument("--precision", choices=("double", "single"), default="double",
                   help="storage precision of cached propagators")
    g.add_argument("--cache-limit", type=int, default=64,
                   help="max source positions the cache may hold")
    g.add_argument("--sparsen", type=float, default=1.0,
                   help="keep 1 site in this many in every cached propagator (e.g. 64)")
    g.add_argument("--nodes-scan", type=int, nargs="*", default=[8, 16, 32, 64],
                   help="node counts for the GB-per-rank table")
    g.add_argument("--local-multiple", type=int, nargs=2, default=[32, 16], metavar=("T", "XYZ"),
                   help="local extents must be multiples of these (production MG blocking); "
                        "1 1 to disable")

    g = p.add_argument_group("solver memory beside the cache (per rank, see 'Memory model')")
    g.add_argument("--solver-gb", type=float, default=None,
                   help="QUDA's peak allocation per rank; default: measured values per node count")
    g.add_argument("--reserve-gb", type=float, default=0.0,
                   help="flat extra per rank (CUDA context, kernel stack reservation, ...)")

    g = p.add_argument_group("cost (seconds, from time_hlbl at the same --procs)")
    g.add_argument("--t-invert", type=float, default=None,
                   help="per source position, all flavors (the invert-y timer)")
    g.add_argument("--t-kernel-sum", type=float, default=None,
                   help="per kernel-sum call, one flavor (kernel-sum + dzu-dzsu); "
                        "measure it with the same sparsening")
    g.add_argument("--t-setup", type=float, default=0.0, help="per config (gauge read, MG setup)")
    g.add_argument("--chains", type=int, default=1, help="independent chains per config")
    g.add_argument("--configs", type=int, default=1)

    g = p.add_argument_group("output")
    g.add_argument("--show-pairs", type=int, nargs="*", default=None, metavar="K",
                   help="print the pair table of these windows (default window 1)")
    g.add_argument("--show-steps", type=int, default=5, help="rows of the per-step table")
    args = p.parse_args()

    h = tuple(args.direction)
    if h not in SIGN_VECTORS:
        p.error(f"--direction must be a sign vector (+-1,+-1,+-1,+-1), got {h}")
    lattice, origin, stride = tuple(args.lattice), tuple(args.origin), args.stride
    if any(L % P for L, P in zip(lattice, args.procs)):
        p.error(f"--procs {tuple(args.procs)} does not divide --lattice {lattice}")

    max_steps = max_steps_without_wrap(h, stride, lattice, origin)
    n_steps = max_steps if args.steps is None else args.steps

    print(f"direction h = {h}, stride n = {stride}, |y| = 2n = {2 * stride}, "
          f"y = (+-{stride},+-{stride},+-{stride},+-{stride})")
    print(f"lattice {lattice}, origin {origin}, {n_steps} steps ({n_steps + 1} windows)")
    if 2 * stride >= min(lattice):
        print(f"WARNING: stride {stride} >= L/2 = {min(lattice) // 2}: the minimum-image "
              f"separation is not (+-n,+-n,+-n,+-n).")
    if n_steps > max_steps:
        print(f"WARNING: beyond {max_steps} steps sources wrap onto sites already used.")

    steps = build_schedule(h, stride, n_steps, origin)
    check_complete(steps, stride)
    print("check: every |y| = 2n pair among the scheduled sources is scheduled exactly once")

    if args.show_pairs is not None:
        for k in (args.show_pairs or [min(1, n_steps)]):
            if 0 <= k <= n_steps:
                print_pairs(steps, k, origin, stride)
    print_orientations(steps)

    stats_by_mode = {mode: run(steps, lattice, args.flavors, mode) for mode in MODES}
    reference = stats_by_mode["serial"]
    for st in stats_by_mode.values():
        assert (st.n_propagators, st.n_kernel_sums) == (reference.n_propagators, reference.n_kernel_sums)

    chosen = stats_by_mode[args.mode]
    print(f"\nper-step table, mode {args.mode} (cache = positions held after the step):")
    print(f"  {'step':>5} {'new props':>9} {'pairs':>6} {'cache':>6}")
    rows = chosen.per_step
    shown = rows if len(rows) <= args.show_steps + 1 else rows[:args.show_steps] + [None] + rows[-1:]
    for row in shown:
        print("  " + ("   ..." if row is None else f"{row[0]:>5} {row[1]:>9} {row[2]:>6} {row[3]:>6}"))
    print(f"  total: {chosen.n_propagators} propagators, {chosen.n_pairs} pairs "
          f"(16 + 15*K and 15 + 65*K for K = {n_steps})")

    memory_report(stats_by_mode, lattice, args.procs, args.flavors, args.precision, args.sparsen,
                  args.gpu_mem_gb, args.host_mem_gb, args.gpus_per_node, args.cache_limit,
                  args.nodes_scan, tuple(args.local_multiple), args.solver_gb, args.reserve_gb)
    cost_report(reference, args.chains, args.configs, args.procs, args.gpus_per_node,
                args.t_invert, args.t_kernel_sum, args.t_setup)


if __name__ == "__main__":
    main()
