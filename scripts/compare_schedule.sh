#!/usr/bin/env bash
# compare_schedule.sh: checks hlbl_hypercube --dry-run against the Python
# reference (hlbl_mII_hypercube_schedule.py): schedules for several
# directions, strides and step counts; counts, totals and overlap report of
# multi-chain input files; and the error messages for bad input files.
#
# Usage: scripts/compare_schedule.sh <path-to-hlbl_hypercube>

set -euo pipefail

BIN="${1:?usage: compare_schedule.sh <path-to-hlbl_hypercube>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3.11}"

py_dump() {
  local h0=$1 h1=$2 h2=$3 h4=$4 stride=$5 steps=$6 window=$7
  "$PY" -c "
import sys
sys.path.insert(0, '$HERE')
import hlbl_mII_hypercube_schedule as hs

h = ($h0, $h1, $h2, $h4)
origin = (0, 0, 0, 0)
steps = hs.build_schedule(h, $stride, $steps, origin)
hs.check_complete(steps, $stride)

for mode in ('serial', 'interleaved'):
    s = hs.run(steps, (128, 64, 64, 64), 2, mode)
    print(f'mode={mode} n_propagators={s.n_propagators} n_solves={s.n_solves} '
          f'n_pairs={s.n_pairs} n_kernel_sums={s.n_kernel_sums} peak_cache={s.peak_cache}')

w = $window
if 0 <= w < len(steps):
    def rel(p):
        return tuple((c - o) // $stride for c, o in zip(p, origin))
    def fmt(t):
        return '(' + ','.join(str(x) for x in t) + ')'
    edges = []
    for src in steps[w].sources:
        off = rel(src.position)
        for q in src.partners:
            poff = rel(q)
            edges.append((off, poff) if off < poff else (poff, off))
    edges.sort()
    for a, b in edges:
        print('edge ' + fmt(a) + ' ' + fmt(b))
" | sort
}

cpp_dump() {
  local h0=$1 h1=$2 h2=$3 h4=$4 stride=$5 steps=$6 window=$7
  "$BIN" --dry-run --direction "$h0" "$h1" "$h2" "$h4" \
         --stride "$stride" --steps "$steps" --show-window "$window" \
    | grep -E '^(mode=|edge )' | sort
}

# h0 h1 h2 h3  stride  steps  window-to-compare
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

fail=0
for c in "${cases[@]}"; do
  read -r h0 h1 h2 h3 stride steps window <<<"$c"
  py_out="$(py_dump "$h0" "$h1" "$h2" "$h3" "$stride" "$steps" "$window")"
  cpp_out="$(cpp_dump "$h0" "$h1" "$h2" "$h3" "$stride" "$steps" "$window")"
  label="direction=($h0,$h1,$h2,$h3) stride=$stride steps=$steps window=$window"
  if [ "$py_out" = "$cpp_out" ]; then
    echo "PASS  $label"
  else
    echo "FAIL  $label"
    diff <(echo "$py_out") <(echo "$cpp_out") || true
    fail=1
  fi
done

# ---------------------------------------------------------------------------
# Input files with several chains: per-chain counts, totals, overlap report
# ---------------------------------------------------------------------------
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

py_multi() {
  "$PY" -c "
import sys
sys.path.insert(0, '$HERE')
import hlbl_mII_hypercube_schedule as hs

def vec(v):
    return '(' + ','.join(str(x) for x in v) + ')'

inp = hs.parse_input_file('$1')
lattice = inp.lattice or (128, 64, 64, 64)
chains = inp.chains
footprints = []
keys = ('n_propagators', 'n_solves', 'n_pairs', 'n_kernel_sums')
total = dict.fromkeys(keys, 0)
peak = 0
for c, ch in enumerate(chains):
    n = ch.steps if ch.steps >= 0 else hs.max_steps_without_wrap(ch.direction, ch.stride, lattice, ch.start)
    steps = hs.build_schedule(ch.direction, ch.stride, n, ch.start)
    hs.check_complete(steps, ch.stride)
    print(f'chain {c} of {len(chains)}')
    print(f'direction={vec(ch.direction)} stride={ch.stride} origin={vec(ch.start)} steps={n} windows={n + 1}')
    for mode in ('serial', 'interleaved'):
        s = hs.run(steps, lattice, 2, mode)
        print(f'mode={mode} n_propagators={s.n_propagators} n_solves={s.n_solves} '
              f'n_pairs={s.n_pairs} n_kernel_sums={s.n_kernel_sums} peak_cache={s.peak_cache}')
        if mode == 'interleaved':
            for k in keys:
                total[k] += getattr(s, k)
            peak = max(peak, s.peak_cache)
    footprints.append(hs.footprint(steps, ch, lattice))
print(f'total chains={len(chains)} ' + ' '.join(f'{k}={v}' for k, v in total.items()) + f' peak_cache={peak}')
for line in hs.overlap_report(footprints):
    print(line)
" | sort
}

cpp_multi() {
  "$BIN" --dry-run --in "$1" \
    | grep -E '^(chain |chains |direction=|mode=|total |WARNING|\.\.\. |  source |  \.\.\. )' | sort
}

# chains 0 and 3 overlap a lot; chain 1 runs only windows 1 and 2; chain 2 has
# another stride and auto steps
cat > "$tmp/multi.input" <<'EOF'
lattice = 64,32,32,32
cache = host
muon_mass = 105.6583745
lattice_spacing 0.07957   # fm
chain start=0,0,0,0   dir=1,1,1,1   stride=2 steps=4
chain start=0,0,0,0   dir=1,-1,1,1  stride=2 steps=3 windows=1:3
chain start=8,8,8,8   dir=-1,1,-1,1 stride=4
chain start=6,6,6,6   dir=1,1,1,1   stride=2 steps=2
EOF

# 8 equal chains: 28 overlapping pairs of chains, of which 20 are listed, each
# sharing 16 sources of which 10 are listed
{
  echo "lattice = 64,32,32,32"
  for i in 1 2 3 4 5 6 7 8; do
    echo "chain start=0,0,0,0 dir=1,1,1,1 stride=2 steps=1"
  done
} > "$tmp/many.input"

for f in multi many; do
  py_out="$(py_multi "$tmp/$f.input")"
  cpp_out="$(cpp_multi "$tmp/$f.input")"
  if ! grep -q '^WARNING' <<<"$cpp_out" || ! grep -q '^  source ' <<<"$cpp_out"; then
    echo "FAIL  $f.input: no overlap warning or no shared sources listed"
    fail=1
  elif [ "$py_out" = "$cpp_out" ]; then
    echo "PASS  input file with several chains ($f.input)"
  else
    echo "FAIL  input file with several chains ($f.input)"
    diff <(echo "$py_out") <(echo "$cpp_out") || true
    fail=1
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
  "$c windows=3:3"
  "$c windows=5"
  "$c steps=2 windows=3:"
  "chain start=0,0,0 dir=1,1,1,1 stride=4"
  "chain start=1_0,0,0,0 dir=1,1,1,1 stride=4"
  "chain start=0,0,0,x dir=1,1,1,1 stride=4"
  "cache = gpu"
  "random_fields = maybe"
  "force = yes"
  "muon_mass = 0"
  "muon_mass = 105.66 MeV"
  "lattice_spacing = -0.08"
  "lattice_spacing = 1e999"
  "lattice_spacing = inf"
  "lattice_spacing = 0x1p-3"
  "flavors = 3"
  "lattice = 6,6,6"
  "colour = red"
  $'# comment\n\ncache = host\n'"$c steps=a"
)
for text in "${bad[@]}"; do
  printf '%s\n' "$text" > "$tmp/bad.input"
  cpp_err="$("$BIN" --dry-run --in "$tmp/bad.input" 2>&1 >/dev/null \
             | sed -n 's/^\[hlbl_hypercube\] error: //p' || true)"
  py_err="$("$PY" -c "
import sys
sys.path.insert(0, '$HERE')
import hlbl_mII_hypercube_schedule as hs
try:
    hs.parse_input_file('$tmp/bad.input')
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
