#!/usr/bin/env bash
# compare_schedule.sh — checks hlbl_hypercube --dry-run's schedule against
# the Python reference (hlbl_mII_hypercube_schedule.py) for several step
# directions, strides and step counts.
#
# The Python side calls the reference's own build_schedule/check_complete/run
# functions directly and prints the same canonical lines the C++ side prints
# (see hlbl_hypercube.cpp), rather than scraping its human-readable report --
# that keeps the comparison exact without changing the reference at all.
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

for mode in ('serial', 'pipelined'):
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

exit $fail
