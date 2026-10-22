#!/usr/bin/env bash
# compare_schedule.sh: checks the C++ schedules (sanity_check/schedule_dump)
# against the Python reference (hlbl_mII_hypercube_schedule.py): several
# directions, strides and step counts; counts, totals and overlap report of
# multi-chain input files; and the error messages for bad input files.
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
if len(chains) > 1:
    print(f'total chains={len(chains)} ' + ' '.join(f'{k}={v}' for k, v in total.items())
          + f' peak_cache={peak}')
    for line in hs.overlap_report(footprints):
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
