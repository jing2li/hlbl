#!/usr/bin/env python3.11
"""
hlbl_mII_hypercube_schedule.py

Plans an hlbl_hypercube run from its input file, without a GPU, using a
Python reference of the sliding-hypercube schedule (scheme in the README).
Nothing physical is computed: inversion and kernel sum are placeholders. The
input file must be complete (as for hlbl_hypercube); its chains run one after
the other with one cache. It prints

  * each chain's schedule, with a brute-force check that it has every
    |y| = 2n pair among its sources exactly once, and the chain overlaps,
  * the number of solves and kernel sums, and node hours,
  * the peak propagator cache and the memory per rank,
  * a PRECHECK block: warnings and errors (a cache that does not fit on the
    file's process grid). Exit status 1 on errors.

Pairs are unordered; see hlbl_mII_xy_exchange_note.tex.

Modes (each propagator is freed after its last pair):
  serial       solve a whole window, then contract it. Peak 30.
  interleaved  contract each source right after its solve. Peak 16. This is
               what the driver does.

Memory model (per rank, GiB)
    GPU:   cache (3 blocks with cache = host) + solver + kernel stack + reserve
    host:  cache (cache = host only) + solver host memory

  cache         peak positions x 4608 B (2 flavours x 12 x 24 doubles) x local
                sites
  solver        QUDA's peak device memory, measured at 4-64 nodes
                (QUDA_DEV_GIB); other node counts print "no data". --solver-gb
                replaces it.
  kernel stack  --kernel-stack-bytes x --gpu-sms x --gpu-threads-per-sm, which
                the CUDA driver reserves at the 4-point kernel's first launch
  reserve       --reserve-gb: CUDA context, cuBLAS, MPI buffers. 0 by default.
  solver host   --solver-host-gb: host memory beside the cache, the MaxRSS of
                `hlbl_hypercube --dummy-run` with cache = device. 15 GiB by
                default.

--sparsen F keeps every cached propagator on the same fixed V/F sites (the
kernel sum's x and z), so the cache shrinks by F. Time the kernel sum with the
same F.

Usage (Python >= 3.8; the login node's python3 is 3.6)
    python3.11 hlbl_mII_hypercube_schedule.py --in hlbl.input
    python3.11 hlbl_mII_hypercube_schedule.py --in hlbl.input --show-pairs 1
    python3.11 hlbl_mII_hypercube_schedule.py --in hlbl.input \\
        --t-invert 40.6 --t-kernel-sum 2.8 --t-setup 275 --configs 100
"""

import argparse
import itertools
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

SIGN_VECTORS = tuple(itertools.product((-1, 1), repeat=4))
ZERO = (0, 0, 0, 0)
ONES = (1, 1, 1, 1)
N_SPIN_COLOR = 12          # point-source components per flavor
N_FLAVORS = 2              # twisted mass +mu and -mu, as hlbl_hypercube
GB = 1024.0 ** 3
HBARC_MEV_FM = 197.3269804  # as in include/hlbl/physics.hpp
MODES = ("serial", "interleaved")

# QUDA's peak device allocation per rank (GiB) with the production multigrid
# solver on cB211a.072.64 (128x64^3), 4 ranks per node, rank 0 high-water mark
# ("Device memory used" in ~/cvc_timing/cB/strong-invert/n*/out.*):
# (local sites, GiB) at 4, 8, 16, 32, 64 nodes.
QUDA_DEV_GIB = ((2097152, 22.07), (1048576, 11.38), (524288, 5.89), (262144, 3.04),
                (131072, 1.56))

GPU_MEM_GB = 95.6         # nvidia-smi memory.total of a GH200: 97871 MiB

# Kernel stack (see "Memory model"): STACK of ker_4pt_contraction in
# `cuobjdump --dump-resource-usage build/hlbl_hypercube`, and a GH200's SMs and
# threads per SM.
KERNEL_STACK_BYTES = 41280
GPU_SMS = 132
GPU_THREADS_PER_SM = 2048
SOLVER_HOST_GB = 15.0     # host memory beside the cache, per rank (README)


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
    """A0 = {0} u {g in G : g != h}, in units of the stride, in solve order:
    that of h = (1,1,1,1), reflected by h. A reflection keeps the separations,
    so every h has the cache peak (16) of (1,1,1,1); G \\ {h} in SIGN_VECTORS
    order would peak at 17 to 25."""
    return [ZERO] + [tuple(hi * gi for hi, gi in zip(h, g))
                     for g in SIGN_VECTORS if g != ONES]


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
# Several chains: the input file of hlbl_hypercube
# =====================================================================
# Same grammar and error messages as `hlbl_hypercube --in`. timing and
# random_fields are checked and ignored.

@dataclass
class Chain:
    """Windows 0..steps of the schedule from `start` in direction `direction`.
    start, dir and stride are required."""
    start: tuple = ZERO
    direction: tuple = ZERO
    stride: int = 0
    steps: int = -1            # -1: as many as fit before sources wrap


@dataclass
class Input:
    """Required keys are None (or empty) when not given; see check_required()."""
    chains: list = field(default_factory=list)
    lattice: tuple = None
    procs: tuple = None
    cache: str = None
    out: str = None
    muon_mass: float = None        # MeV
    lattice_spacing: float = None  # fm


class InputError(ValueError):
    """A bad input file; the message starts with `file:line:`."""


def _int(text):
    t = text.strip()
    if not re.fullmatch(r"[+-]?[0-9]+", t):
        raise ValueError(f"not an integer: '{t}'")
    return int(t)


def _float(text):
    """Decimal notation only, like the C++ parser: no inf, nan, hex or '_'."""
    t = text.strip()
    if not re.fullmatch(r"[0-9+\-.eE]+", t):
        raise ValueError(f"not a number: '{t}'")
    try:
        return float(t)
    except ValueError:
        raise ValueError(f"not a number: '{t}'") from None


def _positive(text, name, unit):
    v = _float(text)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be a positive number ({unit})")
    return v


def _bool(text):
    t = text.strip().lower()
    if t in ("yes", "on", "true", "1"):
        return True
    if t in ("no", "off", "false", "0"):
        return False
    raise ValueError(f"not a yes/no value: '{text.strip()}'")


def parse_vec4(text):
    """'a,b,c,d' (commas or blanks)"""
    words = text.replace(",", " ").split()
    if len(words) != 4:
        raise ValueError(f"expected 4 integers: '{text}'")
    return tuple(_int(w) for w in words)


def validate_chain(c):
    """Raises ValueError unless the direction is a sign vector, the stride is
    positive and steps >= -1."""
    if any(d not in (1, -1) for d in c.direction):
        raise ValueError("dir must be a sign vector (+-1,+-1,+-1,+-1)")
    if c.stride <= 0:
        raise ValueError("stride must be positive")
    if c.steps < -1:
        raise ValueError("steps must be >= 0 (or auto)")


def _parse_chain(tokens):
    c = Chain()
    seen = set()
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"expected key=value, got '{token}'")
        key, value = token.split("=", 1)
        if key in seen:
            raise ValueError(f"chain key '{key}' given twice")
        seen.add(key)
        if key == "start":
            c.start = parse_vec4(value)
        elif key == "dir":
            c.direction = parse_vec4(value)
        elif key == "stride":
            c.stride = _int(value)
        elif key == "steps":
            c.steps = -1 if value == "auto" else _int(value)
        else:
            raise ValueError(f"unknown chain key '{key}'")
    for required in ("start", "dir", "stride"):
        if required not in seen:
            raise ValueError(f"chain needs {required}=")
    validate_chain(c)
    return c


def _set_global(inp, key, value):
    v = value.strip()
    if key == "cache":
        if v not in ("device", "host"):
            raise ValueError("cache must be device or host")
        inp.cache = v
    elif key == "random_fields":
        _bool(v)
    elif key == "out":
        inp.out = v or None
    elif key == "timing":
        pass
    elif key == "muon_mass":
        inp.muon_mass = _positive(v, key, "MeV")
    elif key == "lattice_spacing":
        inp.lattice_spacing = _positive(v, key, "fm")
    elif key == "lattice":
        inp.lattice = parse_vec4(v)
    elif key == "procs":
        inp.procs = parse_vec4(v)
    else:
        raise ValueError(f"unknown option '{key}'")


def parse_input(text, name="input"):
    """The chains and options of an input file. Raises InputError."""
    inp = Input()
    for number, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            words = line.split()
            if words[0] == "chain":
                inp.chains.append(_parse_chain(words[1:]))
            elif "=" in line:   # `key = value`, `key=value`
                key, value = line.split("=", 1)
                _set_global(inp, key.strip(), value)
            else:               # `key value`
                _set_global(inp, words[0], line[len(words[0]):])
        except ValueError as e:
            raise InputError(f"{name}:{number}: {e}") from None
    return inp


def parse_input_file(path):
    with open(path) as f:
        return parse_input(f.read(), path)


def check_required(inp, name):
    """Raises InputError ("<name>: missing ...") unless `inp` has chain lines,
    lattice, procs, muon_mass, lattice_spacing, out and cache."""
    missing = [key for key, given in (
        ("chain lines", inp.chains), ("lattice", inp.lattice), ("procs", inp.procs),
        ("muon_mass", inp.muon_mass), ("lattice_spacing", inp.lattice_spacing),
        ("out", inp.out), ("cache", inp.cache)) if not given]
    if missing:
        raise InputError(f"{name}: missing {', '.join(missing)}")


def footprint(steps, lattice):
    """What a chain touches: the wrapped source positions, as {position: window
    that solves it}, and the wrapped unordered pairs."""
    sources, pairs = {}, set()
    for step in steps:
        for src in step.sources:
            p = wrap(src.position, lattice)
            sources.setdefault(p, step.index)
            for q in src.partners:
                pairs.add(unordered(p, wrap(q, lattice)))
    return sources, pairs


MAX_OVERLAP_LINES = 20     # pairs of chains listed
MAX_SHARED_SOURCES = 10    # shared sources listed per pair of chains


def fmt_vec4(v):
    return "(" + ",".join(str(x) for x in v) + ")"


def overlap_report(footprints):
    """The overlap warning, from each chain's footprint(): the sources and pairs
    in more than one chain, and per pair of chains the first shared sources
    with the window of each chain that solves them. Empty if there is no
    overlap. Same lines as hlbl_hypercube."""
    source_count, pair_count = Counter(), Counter()
    for sources, pairs in footprints:
        source_count.update(sources.keys())
        pair_count.update(pairs)
    n_sources = sum(1 for n in source_count.values() if n > 1)
    n_pairs = sum(1 for n in pair_count.values() if n > 1)
    if not n_sources and not n_pairs:
        return []
    lines = [f"WARNING: {n_sources} source positions are in more than one chain "
             f"(solved once per chain) and {n_pairs} pairs are computed in more "
             f"than one chain"]
    shown = more = 0
    for (i, (si, pi)), (j, (sj, pj)) in itertools.combinations(enumerate(footprints), 2):
        shared = sorted(si.keys() & sj.keys())
        np_ = len(pi & pj)
        if not shared and not np_:
            continue
        if shown >= MAX_OVERLAP_LINES:
            more += 1
            continue
        shown += 1
        lines.append(f"chains {i} and {j} share {len(shared)} sources and {np_} pairs")
        for pos in shared[:MAX_SHARED_SOURCES]:
            lines.append(f"  source {fmt_vec4(pos)} in window {si[pos]} of chain {i} "
                         f"and window {sj[pos]} of chain {j}")
        if len(shared) > MAX_SHARED_SOURCES:
            lines.append(f"  ... and {len(shared) - MAX_SHARED_SOURCES} more sources")
    if more:
        lines.append(f"... and {more} more pairs of chains")
    return lines


# =====================================================================
# Placeholders for the expensive parts
# =====================================================================

@dataclass(frozen=True)
class Propagator:
    """Stand-in. A real one holds, per flavor, (12, volume, 12) complex:
    12 spin-color point sources x sink spinor-color at every site."""
    site: tuple


def invert_propagator(site):
    """Placeholder for 12 * N_FLAVORS solves from a point source at `site`
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


def run(steps, lattice, mode):
    pending = Counter()
    for older, newer in all_pairs(steps):
        pending[older] += 1
        pending[newer] += 1
    cache = PropagatorCache(pending)
    stats = RunStats(mode)

    def invert(position):
        cache.add(position, invert_propagator(wrap(position, lattice)))
        stats.n_propagators += 1
        stats.n_solves += N_SPIN_COLOR * N_FLAVORS

    def contract(older, newer):
        yv = sub(newer, older)  # (+-n, +-n, +-n, +-n)
        for flavor in range(N_FLAVORS):
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
        for src in step.sources:
            invert(src.position)
        for src in step.sources:
            for q in src.partners:
                contract(q, src.position)
        stats.per_step.append((k, step.n_new, step.n_pairs, len(cache.props)))

    assert not cache.props, "propagators left in the cache after the last pair"
    stats.peak_cache = cache.peak
    return stats


def combine_stats(parts):
    """The RunStats of chains run one after the other with one cache: the counts
    add up, the cache peak is the largest (the cache is empty between chains)."""
    if len(parts) == 1:
        return parts[0]
    total = RunStats(parts[0].mode)
    for st in parts:
        total.n_propagators += st.n_propagators
        total.n_solves += st.n_solves
        total.n_pairs += st.n_pairs
        total.n_kernel_sums += st.n_kernel_sums
    total.peak_cache = max(st.peak_cache for st in parts)
    return total


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


def print_orientations(pairs):
    """How often each of the 8 orientations +-g of y is sampled, for a list of
    (older, newer) pairs."""
    counts = Counter()
    for older, newer in pairs:
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


def memory_report(stats_by_mode, lattice, procs, precision, sparsen,
                  gpu_mem_gb, host_mem_gb, gpus_per_node, cache_limit,
                  nodes_scan, local_multiple, solver_gb=None, reserve_gb=0.0,
                  stack=(KERNEL_STACK_BYTES, GPU_SMS, GPU_THREADS_PER_SM),
                  solver_host_gb=SOLVER_HOST_GB, cache_mode="device"):
    """Prints the memory tables. Returns the need and the budget per rank, in
    bytes, of the run on `procs` with `cache_mode` (interleaved order), or None
    if QUDA was not measured at that local volume."""
    volume = math.prod(lattice)
    bytes_complex = 16 if precision == "double" else 8
    kept_sites = volume / sparsen
    per_flavor = N_SPIN_COLOR * N_SPIN_COLOR * bytes_complex * kept_sites
    per_position = N_FLAVORS * per_flavor
    stack_gb = math.prod(stack) / GB
    host_rest = solver_host_gb * GB

    def fit(cache, block, res):
        """GPU: cache + res on the device. host: cache in pinned host memory
        beside the solver's, the device holds res and three blocks (two staging
        slots and a solve target). -: neither."""
        if cache + res <= gpu_mem_gb * GB and host_rest <= host_mem_gb * GB:
            return "GPU"
        if block * 3 + res <= gpu_mem_gb * GB and cache + host_rest <= host_mem_gb * GB:
            return "host"
        return "-"

    print("\n" + "=" * 78)
    print("MEMORY  (peak positions x position size + QUDA solver + kernel stack)")
    print("=" * 78)
    print(f"lattice {tuple(lattice)} = {volume:,} sites, {precision} precision")
    if sparsen > 1:
        print(f"sparsened: 1 site in {sparsen:g} kept -> {kept_sites:,.0f} sites per propagator")
    print(f"one flavor propagator  12x12 x {bytes_complex} B x sites = {per_flavor / GB:10.2f} GB")
    print(f"one source position    x {N_FLAVORS} flavors            = {per_position / GB:10.2f} GB")
    print(f"budget per rank: {gpu_mem_gb:g} GB GPU, {host_mem_gb:g} GB host")
    if solver_gb is None:
        print("solver: QUDA peak measured with time_invert (4/8/16/32/64 nodes, 128x64^3)")
    else:
        print(f"solver: {solver_gb:g} GB (--solver-gb)")
    print(f"kernel stack: {stack[0]} B x {stack[1]} SMs x {stack[2]} threads = "
          f"{stack_gb:.2f} GB per GPU")
    print("  (--kernel-stack-bytes: STACK of ker_4pt_contraction in `cuobjdump "
          "--dump-resource-usage build/hlbl_hypercube`;")
    print("   --gpu-sms, --gpu-threads-per-sm: multiProcessorCount and "
          "maxThreadsPerMultiProcessor)")
    if reserve_gb:
        print(f"plus {reserve_gb:g} GB flat reserve (--reserve-gb)")
    print(f"host beside the cache: {solver_host_gb:g} GB per rank (--solver-host-gb; the "
          f"MaxRSS of a --dummy-run with cache = device)")
    print()
    print(f"  {'mode':<12} {'positions':>9} {'flavor-props':>12} {'total GB':>10} {'<= limit':>8}")
    for mode, st in stats_by_mode.items():
        print(f"  {mode:<12} {st.peak_cache:>9} {st.peak_cache * N_FLAVORS:>12} "
              f"{st.peak_cache * per_position / GB:>10.1f} "
              f"{'yes' if st.peak_cache <= cache_limit else 'no':>8}")
    print(f"  'positions' = source points held at once (all flavors each);")
    print(f"  '<= limit' compares positions with --cache-limit {cache_limit}.")

    print(f"\n  GB per rank by node count ({gpus_per_node} GPUs = ranks per node): "
          f"cache, and cache + solver + kernel stack = total")
    print(f"  fits: GPU = everything on the device, host = cache in pinned host memory "
          f"(device keeps solver + stack + 3 blocks), - = neither")
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
        res_b = (solver + stack_gb + reserve_gb) * GB
        line = f"  {nodes:>5} {ranks:>5}  {grid_s:<10} {solver:>7.2f}"
        for st in stats_by_mode.values():
            cache = st.peak_cache * per_position / ranks
            # no process grid -> the solver cannot run there, whatever the memory says
            verdict = "-" if grid is None else fit(cache, per_position / ranks, res_b)
            line += f"  {'':>9}  {cache / GB:>6.1f} {(cache + res_b) / GB:>6.1f} {verdict:>4}"
        print(line)
    print("  'none': no process grid of that size fits the lattice and the MG blocking.")
    print("  'no data': QUDA was not measured at this local volume.")

    # the run on `procs`, for the PRECHECK block
    ranks = math.prod(procs)
    solver = solver_gb if solver_gb is not None else quda_dev_gb(volume // ranks)
    if solver is None:
        return None
    cache = stats_by_mode["interleaved"].peak_cache * per_position / ranks
    res = (solver + stack_gb + reserve_gb) * GB
    return {
        "ranks": ranks,
        "gpu_need": (cache if cache_mode == "device" else 3 * per_position / ranks) + res,
        "gpu_have": gpu_mem_gb * GB,
        "host_need": (cache if cache_mode == "host" else 0) + host_rest,
        "host_have": host_mem_gb * GB,
    }


def cost_report(stats, n_chains, configs, procs, gpus_per_node,
                t_invert, t_kernel_sum, t_setup):
    """`stats`: the totals of the input file's `n_chains` chains."""
    ranks = math.prod(procs)
    nodes = ranks / gpus_per_node
    n_prop = stats.n_propagators
    n_ks = stats.n_kernel_sums

    print("\n" + "=" * 78)
    print("CALL COUNTS AND NODE HOURS")
    print("=" * 78)
    print(f"per config: {n_chains} chain(s) -> {n_prop} propagators ({stats.n_solves} solves), "
          f"{stats.n_pairs} pairs, {n_ks} kernel-sum calls")
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
        description="Plans an hlbl_hypercube run: schedules, memory and cost of its input file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("schedule")
    g.add_argument("--in", dest="input", metavar="FILE", required=True,
                   help="input file of hlbl_hypercube (scripts/input/hlbl.input.example)")
    g.add_argument("--mode", choices=MODES, default="interleaved",
                   help="execution order used for the per-step table")

    g = p.add_argument_group("machine")
    g.add_argument("--gpus-per-node", type=int, default=4)
    g.add_argument("--gpu-mem-gb", type=float, default=GPU_MEM_GB,
                   help="per GPU, as nvidia-smi memory.total (GH200: 97871 MiB)")
    g.add_argument("--host-mem-gb", type=float, default=120.0, help="per rank (GH200 Grace)")
    g.add_argument("--gpu-sms", type=int, default=GPU_SMS,
                   help="multiProcessorCount of the GPU (deviceQuery)")
    g.add_argument("--gpu-threads-per-sm", type=int, default=GPU_THREADS_PER_SM,
                   help="maxThreadsPerMultiProcessor of the GPU (deviceQuery)")
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
    g.add_argument("--kernel-stack-bytes", type=int, default=KERNEL_STACK_BYTES,
                   help="stack per thread of the 4-point kernel, reserved for every thread "
                        "slot of the GPU: the STACK column of ker_4pt_contraction in "
                        "`cuobjdump --dump-resource-usage build/hlbl_hypercube`")
    g.add_argument("--reserve-gb", type=float, default=0.0,
                   help="flat extra per rank (CUDA context, cuBLAS and MPI buffers)")
    g.add_argument("--solver-host-gb", type=float, default=SOLVER_HOST_GB,
                   help="host memory per rank beside the cache: the sacct MaxRSS of "
                        "`hlbl_hypercube --dummy-run` with cache = device")

    g = p.add_argument_group("cost (seconds, from time_hlbl on the same process grid)")
    g.add_argument("--t-invert", type=float, default=None,
                   help="per source position, all flavors (the invert-y timer)")
    g.add_argument("--t-kernel-sum", type=float, default=None,
                   help="per kernel-sum call, one flavor (kernel-sum + dzu-dzsu); "
                        "measure it with the same sparsening")
    g.add_argument("--t-setup", type=float, default=0.0,
                   help="per config, paid once however many chains run (gauge read, MG setup)")
    g.add_argument("--configs", type=int, default=1)

    g = p.add_argument_group("output")
    g.add_argument("--show-pairs", type=int, nargs="*", default=None, metavar="K",
                   help="print the pair table of these windows (default window 1)")
    g.add_argument("--show-steps", type=int, default=5, help="rows of the per-step table")
    args = p.parse_args()

    try:
        inp = parse_input_file(args.input)
        check_required(inp, args.input)
    except (OSError, InputError) as e:
        p.error(str(e))
    chains, lattice, procs, cache_mode = inp.chains, inp.lattice, inp.procs, inp.cache
    multi = len(chains) > 1
    if any(L % P for L, P in zip(lattice, procs)):
        p.error(f"procs {procs} does not divide lattice {lattice}")

    # ---- what the PRECHECK block at the end lists
    xunit = inp.muon_mass / HBARC_MEV_FM * inp.lattice_spacing
    print(f"muon mass {inp.muon_mass:.10g} MeV, lattice spacing "
          f"{inp.lattice_spacing:.10g} fm: xunit = m_mu a = {xunit:.10g}")
    checks, warnings, errors = [f"physics inputs: xunit = m_mu a = {xunit:.10g}"], [], []

    # ---- one schedule per chain
    schedules, footprints, n_steps_of = [], [], []
    for c, chain in enumerate(chains):
        h, stride, origin = chain.direction, chain.stride, chain.start
        tag = f"chain {c}: " if multi else ""
        max_steps = max_steps_without_wrap(h, stride, lattice, origin)
        n_steps = max_steps if chain.steps < 0 else chain.steps

        if multi:
            print(f"\nchain {c} of {len(chains)}")
        print(f"direction h = {h}, stride n = {stride}, |y| = 2n = {2 * stride}, "
              f"y = (+-{stride},+-{stride},+-{stride},+-{stride})")
        print(f"lattice {lattice}, origin {origin}, {n_steps} steps ({n_steps + 1} windows)")
        if 2 * stride >= min(lattice):
            print(f"WARNING: {tag}stride {stride} >= L/2 = {min(lattice) // 2}: the minimum-image "
                  f"separation is not (+-n,+-n,+-n,+-n).")
            warnings.append(f"{tag}stride {stride} >= L/2: the minimum-image separation is "
                            f"not (+-n,+-n,+-n,+-n)")
        if n_steps > max_steps:
            print(f"WARNING: {tag}beyond {max_steps} steps sources wrap onto sites already used.")
            warnings.append(f"{tag}{n_steps} steps wrap sources onto sites already used "
                            f"(at most {max_steps} do not)")

        steps = build_schedule(h, stride, n_steps, origin)
        check_complete(steps, stride)
        print("check: every |y| = 2n pair among the scheduled sources is scheduled exactly once")

        if args.show_pairs is not None:
            for k in (args.show_pairs or [min(1, n_steps)]):
                if 0 <= k <= n_steps:
                    print_pairs(steps, k, origin, stride)
        schedules.append(steps)
        n_steps_of.append(n_steps)
        footprints.append(footprint(steps, lattice))

    checks.append("schedules: every |y| = 2n pair among the scheduled sources once")
    overlap = overlap_report(footprints)
    if overlap:
        print()
        for line in overlap:
            print(line)
        warnings.append(overlap[0][len("WARNING: "):])
    print_orientations([pair for steps in schedules for pair in all_pairs(steps)])

    # ---- counts: chains run one after the other with one cache
    per_chain = [{mode: run(steps, lattice, mode) for mode in MODES}
                 for steps in schedules]
    for by_mode in per_chain:
        ref = by_mode["serial"]
        for st in by_mode.values():
            assert (st.n_propagators, st.n_kernel_sums) == (ref.n_propagators, ref.n_kernel_sums)
    stats_by_mode = {mode: combine_stats([by_mode[mode] for by_mode in per_chain])
                     for mode in MODES}
    reference = stats_by_mode["serial"]

    chosen = stats_by_mode[args.mode]
    if not multi:
        print(f"\nper-step table, mode {args.mode} (cache = positions held after the step):")
        print(f"  {'step':>5} {'new props':>9} {'pairs':>6} {'cache':>6}")
        rows = chosen.per_step
        shown = rows if len(rows) <= args.show_steps + 1 else rows[:args.show_steps] + [None] + rows[-1:]
        for row in shown:
            print("  " + ("   ..." if row is None else f"{row[0]:>5} {row[1]:>9} {row[2]:>6} {row[3]:>6}"))
        print(f"  total: {chosen.n_propagators} propagators, {chosen.n_pairs} pairs "
              f"(16 + 15*K and 15 + 65*K for K = {n_steps_of[0]})")
    else:
        print(f"\nper-chain table, mode {args.mode}:")
        print(f"  {'chain':>5} {'start':<18} {'direction':<15} {'stride':>6} {'steps':>5} "
              f"{'props':>6} {'pairs':>6} {'peak':>5}")
        for c, chain in enumerate(chains):
            st = per_chain[c][args.mode]
            print(f"  {c:>5} {str(chain.start):<18} {str(chain.direction):<15} {chain.stride:>6} "
                  f"{n_steps_of[c]:>5} {st.n_propagators:>6} {st.n_pairs:>6} {st.peak_cache:>5}")
        print(f"  total: {len(chains)} chains, {chosen.n_propagators} propagators, "
              f"{chosen.n_pairs} pairs, cache peak {chosen.peak_cache} (the largest of the chains)")

    chosen = memory_report(
        stats_by_mode, lattice, procs, args.precision, args.sparsen,
        args.gpu_mem_gb, args.host_mem_gb, args.gpus_per_node, args.cache_limit,
        args.nodes_scan, tuple(args.local_multiple), args.solver_gb, args.reserve_gb,
        (args.kernel_stack_bytes, args.gpu_sms, args.gpu_threads_per_sm),
        args.solver_host_gb, cache_mode)
    where = (f"{math.prod(procs) / args.gpus_per_node:g} nodes ({math.prod(procs)} ranks), "
             f"cache = {cache_mode}")
    if chosen is None:
        warnings.append(f"memory at {where}: QUDA was not measured at this local volume "
                        f"(give --solver-gb)")
    else:
        line = (f"memory at {where}: GPU {chosen['gpu_need'] / GB:.1f} of "
                f"{chosen['gpu_have'] / GB:.1f} GB, host {chosen['host_need'] / GB:.1f} of "
                f"{chosen['host_have'] / GB:.1f} GB per rank")
        if chosen["gpu_need"] <= chosen["gpu_have"] and chosen["host_need"] <= chosen["host_have"]:
            checks.append(line)
        else:
            errors.append(line + ": does not fit")
    cost_report(reference, len(chains), args.configs, procs, args.gpus_per_node,
                args.t_invert, args.t_kernel_sum, args.t_setup)

    print("\n" + "=" * 78)
    print("PRECHECK")
    print("=" * 78)
    for line in checks:
        print(f"  ok       {line}")
    for line in warnings:
        print(f"  WARNING  {line}")
    for line in errors:
        print(f"  ERROR    {line}")
    if errors:
        print(f"\n{len(errors)} error(s): fix them before submitting")
        sys.exit(1)
    print("\nno errors; next: hlbl_hypercube --dummy-run on the resources of the run")


if __name__ == "__main__":
    main()
