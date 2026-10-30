#!/usr/bin/env bash
# compare_schedule.sh: checks the C++ schedules (sanity_check/schedule_dump)
# against the Python reference (hlbl_mII_hypercube_schedule.py): several
# directions, strides and step counts; counts, totals and overlap report of
# multi-chain input files; the 2+2 plan and warnings of 2p2 lines; and the
# error messages for bad input files.
#
# Usage: scripts/compare_schedule.sh <path-to-schedule_dump>

set -euo pipefail

BIN="${1:?usage: compare_schedule.sh <path-to-schedule_dump>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3.11}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
fail=0

# the keys every input file needs besides lattice and chains
header='procs = 1,1,1,1
cache = device
out = result.h5
muon_mass = 105.6583745
lattice_spacing = 0.07957'

# The lines schedule_dump prints, from the Python reference: FILE [window]
py_dump() {
  "$PY" -c "
import sys
sys.path.insert(0, '$HERE')
import hlbl_mII_hypercube_schedule as hs

def vec(v):
    return '(' + ','.join(str(x) for x in v) + ')'

inp = hs.parse_input_file('$1')
hs.check_required(inp, '$1')
chains, lattice, window = inp.chains, inp.lattice, ${2:--1}
keys = ('n_propagators', 'n_solves', 'n_pairs', 'n_kernel_sums')
total = dict.fromkeys(keys, 0)
peak = 0
footprints = []
schedules = []
for c, ch in enumerate(chains):
    n = ch.steps if ch.steps >= 0 else hs.max_steps_without_wrap(ch.direction, ch.stride, lattice, ch.start)
    steps = hs.build_schedule(ch.direction, ch.stride, n, ch.start)
    hs.check_complete(steps, ch.stride)
    if len(chains) > 1:
        print(f'chain {c} of {len(chains)}')
    print(f'direction={vec(ch.direction)} stride={ch.stride} origin={vec(ch.start)} steps={n} windows={n + 1}')
    for mode in hs.MODES:
        s = hs.run(steps, lattice, mode)
        print(f'mode={mode} n_propagators={s.n_propagators} n_solves={s.n_solves} '
              f'n_pairs={s.n_pairs} n_kernel_sums={s.n_kernel_sums} peak_cache={s.peak_cache}')
        if mode == 'interleaved':
            for k in keys:
                total[k] += getattr(s, k)
            peak = max(peak, s.peak_cache)
    if 0 <= window < len(steps):
        def rel(p):
            return tuple((x - o) // ch.stride for x, o in zip(p, ch.start))
        for a, b in sorted(tuple(sorted((rel(src.position), rel(q))))
                           for src in steps[window].sources for q in src.partners):
            print('edge ' + vec(a) + ' ' + vec(b))
    footprints.append(hs.footprint(steps, lattice))
    schedules.append(steps)
if len(chains) > 1:
    print(f'total chains={len(chains)} ' + ' '.join(f'{k}={v}' for k, v in total.items())
          + f' peak_cache={peak}')
    for line in hs.overlap_report(footprints):
        print(line)
if inp.sources_2p2:
    sources = [g.source for g in inp.sources_2p2]
    plan = hs.plan_2p2(schedules, sources, lattice)
    n_sources = n_targets = 0
    for c, steps in enumerate(schedules):
        for step in steps:
            for src in step.sources:
                i = plan.at[c].get(src.position)
                if i is None:
                    continue
                g = inp.sources_2p2[i]
                print(f'2p2 chain={c} window={step.index} source={vec(g.source)} targets={len(g.targets)}')
                n_sources += 1
                n_targets += len(g.targets)
    print(f'2p2 total sources={n_sources} targets={n_targets}')
    for line in hs.report_2p2(plan, sources):
        print(line)
"
}

compare() {  # FILE WINDOW LABEL
  local py cpp
  py="$(py_dump "$1" "$2")"
  cpp="$("$BIN" "$1" "$2")"
  if [ "$py" = "$cpp" ]; then
    echo "PASS  $3"
  else
    echo "FAIL  $3"
    diff <(echo "$py") <(echo "$cpp") || true
    fail=1
  fi
}

# ---------------------------------------------------------------------------
# One chain: counts, cache peaks and the pairs of one window
# ---------------------------------------------------------------------------
# h0 h1 h2 h3  stride  steps  window
cases=(
  "1 1 1 1    1   3   0"
  "1 1 1 1    1   3   1"
  "1 1 1 1    4  30   1"
  "1 1 1 1    4  30   15"
  "1 -1 1 1   2   5   1"
  "-1 -1 -1 -1 3  4   2"
  "1 1 -1 -1  1   6   0"
  "-1 1 -1 1  2   6   3"
  "1 -1 -1 -1 4  30  15"
)
for c in "${cases[@]}"; do
  read -r h0 h1 h2 h3 stride steps window <<<"$c"
  printf '%s\nlattice = 128,64,64,64\nchain start=0,0,0,0 dir=%s,%s,%s,%s stride=%s steps=%s\n' \
    "$header" "$h0" "$h1" "$h2" "$h3" "$stride" "$steps" > "$tmp/one.input"
  compare "$tmp/one.input" "$window" \
    "direction=($h0,$h1,$h2,$h3) stride=$stride steps=$steps window=$window"
done

# ---------------------------------------------------------------------------
# Several chains: per-chain counts, totals, overlap report
# ---------------------------------------------------------------------------
# chains 0 and 3 overlap a lot; chain 2 has another stride and auto steps
cat > "$tmp/multi.input" <<EOF
$header
lattice = 64,32,32,32
chain start=0,0,0,0   dir=1,1,1,1   stride=2 steps=4
chain start=0,0,0,0   dir=1,-1,1,1  stride=2 steps=3
chain start=8,8,8,8   dir=-1,1,-1,1 stride=4
chain start=6,6,6,6   dir=1,1,1,1   stride=2 steps=2
EOF

# 8 equal chains: 28 overlapping pairs of chains, of which 20 are listed, each
# sharing 16 sources of which 10 are listed
{
  echo "$header"
  echo "lattice = 64,32,32,32"
  for i in 1 2 3 4 5 6 7 8; do
    echo "chain start=0,0,0,0 dir=1,1,1,1 stride=2 steps=1"
  done
} > "$tmp/many.input"

for f in multi many; do
  if ! "$BIN" "$tmp/$f.input" | grep -q '^  source '; then
    echo "FAIL  $f.input: no shared sources listed"
    fail=1
  else
    compare "$tmp/$f.input" -1 "input file with several chains ($f.input)"
  fi
done

# ---------------------------------------------------------------------------
# 2p2 lines: which solve computes each 2+2, and the warnings
# ---------------------------------------------------------------------------
# (6,6,6,6): window 3 of chain 0, again window 0 of chain 3 -> 2+2 in chain 0
# only. (10,10,10,10): first solved by chain 3. (1,1,1,1): no chain solves it
# (all sites are even). Most solved sites are no 2+2 source (list cut at 10).
{
  cat "$tmp/multi.input"
  echo "2p2 source=6,6,6,6 targets=(0,0,0,0),(2,2,2,2)"
  echo "2p2 source=0,0,0,0 targets=(6,6,6,6)"
  echo "2p2 source=10,10,10,10 targets=(6,6,6,6),(0,0,0,0),(4,4,4,4)"
  echo "2p2 source=1,1,1,1 targets=(0,0,0,0)"
} > "$tmp/multi2p2.input"

out="$("$BIN" "$tmp/multi2p2.input")"
if ! grep -q '^2p2 chain=0 window=3 source=(6,6,6,6) targets=2$' <<<"$out" ||
   grep -q '^2p2 chain=3 .*source=(6,6,6,6)' <<<"$out" ||
   ! grep -q '^2p2 chain=3 .*source=(10,10,10,10)' <<<"$out" ||
   ! grep -q '^  (1,1,1,1)$' <<<"$out" ||
   ! grep -q '^  \.\.\. and [0-9]* more$' <<<"$out"; then
  echo "FAIL  multi2p2.input: first solves, unsolved source or the list cut"
  echo "$out" | grep '2p2\|WARNING\|^  (1,1'
  fail=1
else
  compare "$tmp/multi2p2.input" -1 "2p2 lines on several chains (multi2p2.input)"
fi

# ---------------------------------------------------------------------------
# Bad input files: the same message from both parsers
# ---------------------------------------------------------------------------
c='chain start=0,0,0,0 dir=1,1,1,1 stride=4'
bad=(
  "chain start=0,0,0,0 dir=1,1,1,2 stride=4"
  "chain dir=1,1,1,1 stride=4"
  "chain start=0,0,0,0 dir=1,1,1,1"
  "chain start=0,0,0,0 dir=1,1,1,1 stride=0"
  "$c steps=-2"
  "$c steps"
  "$c stride=5"
  "$c colour=1"
  "$c windows=1:3"
  "chain start=0,0,0 dir=1,1,1,1 stride=4"
  "chain start=1_0,0,0,0 dir=1,1,1,1 stride=4"
  "chain start=0,0,0,x dir=1,1,1,1 stride=4"
  "cache = gpu"
  "random_fields = maybe"
  "force = yes"
  "flavors = 2"
  "muon_mass = 0"
  "muon_mass = 105.66 MeV"
  "lattice_spacing = -0.08"
  "lattice_spacing = 1e999"
  "lattice_spacing = inf"
  "lattice_spacing = 0x1p-3"
  "lattice = 6,6,6"
  "colour = red"
  $'# comment\n\ncache = host\n'"$c steps=a"
  "$c"
  "$header"$'\nlattice = 8,8,8,8'
  "2p2 targets=(1,1,1,1)"
  "2p2 source=1,1,1,1"
  "2p2 source=1,1,1,1 targets=(2,2,2,2) colour=2"
  "2p2 source=1,1,1,1 source=2,2,2,2 targets=(1,1,1,1)"
  "2p2 source=1,1,1,1 targets=2,2,2,2"
  "2p2 source=1,1,1,1 targets=(2,2,2,2)(3,3,3,3)"
  "2p2 source=1,1,1,1 targets=(2,2,2,2),"
  "2p2 source=1,1,1,1 targets=((2,2,2,2))"
  "2p2 source=1,1,1,1 targets=(2,2,2)"
  "2p2 source=1,1,1 targets=(2,2,2,2)"
  "2p2 source=1,1,1,1 targets=(2,2,2,2),(3,3,3,3),(2,2,2,2)"
  "2p2 source=1,1,1,1 targets=(2,2,2,2) extra"
  $'2p2 source=1,1,1,1 targets=(2,2,2,2)\n2p2 source=1,1,1,1 targets=(3,3,3,3)'
  $'lattice = 8,8,8,8\n2p2 source=8,0,0,0 targets=(1,1,1,1)'
  $'2p2 source=0,0,0,0 targets=(1,1,1,1),(1,1,1,-1)\nlattice = 8,8,8,8'
)
for text in "${bad[@]}"; do
  printf '%s\n' "$text" > "$tmp/bad.input"
  cpp_err="$("$BIN" "$tmp/bad.input" 2>&1 >/dev/null \
             | sed -n 's/^schedule_dump: error: //p' || true)"
  py_err="$("$PY" -c "
import sys
sys.path.insert(0, '$HERE')
import hlbl_mII_hypercube_schedule as hs
try:
    hs.check_required(hs.parse_input_file('$tmp/bad.input'), '$tmp/bad.input')
except hs.InputError as e:
    print(e)
" 2>&1 || true)"
  if [ -n "$cpp_err" ] && [ "$py_err" = "$cpp_err" ]; then
    echo "PASS  bad input: ${cpp_err#$tmp/}"
  else
    echo "FAIL  bad input: $(echo "$text" | tail -1)"
    echo "      C++:    $cpp_err"
    echo "      Python: $py_err"
    fail=1
  fi
done

exit $fail
