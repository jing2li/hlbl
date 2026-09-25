#!/bin/bash
# gen_invert_input.sh -- write the tmLQCD input (invert.input) a run needs.
#
#   gen_invert_input.sh -o RUNDIR -T 128 -L 64 -p NT,NX,NY,NZ [options]
#
# tmLQCD_invert_init reads invert.input from the working directory; the
# lattice, process grid, operators and multigrid setup all come from it, so
# this is the only input file the programs need. The template is
# scripts/input/invert.input.in, copied from cvc's time_hlbl (production
# reference run, 3-level multigrid with coarse-level deflation).
#
# Constraints, all checked here:
#  - every process count divides its lattice extent;
#  - every local extent is a multiple of 2 * prod(MGBlockSizes) of the
#    template, or tmLQCD's multigrid aborts with "Blocking error";
#  - memory: the multigrid setup of this template needs a local volume of
#    about 64x32x32x32 or less per GPU (16 GPUs for 128x64^3). Larger local
#    volumes were seen to run out of device memory.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

outdir=""; T=128; L=64; procs=""; nconf=500; threads=16
gauge="/capstor/store/cscs/userlab/lp142/cB211a.072.64/conf"

while getopts "o:T:L:p:n:j:g:h" opt; do
  case $opt in
    o) outdir=$OPTARG ;; T) T=$OPTARG ;; L) L=$OPTARG ;; p) procs=$OPTARG ;;
    n) nconf=$OPTARG ;; j) threads=$OPTARG ;; g) gauge=$OPTARG ;;
    h|*) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
         echo "options: -o DIR -T N -L N -p NT,NX,NY,NZ [-n conf] [-j omp threads] [-g gauge prefix]"
         exit 0 ;;
  esac
done
die() { echo "gen_invert_input.sh: $*" >&2; exit 1; }
[ -n "$outdir" ] || die "-o is required"
[ -n "$procs" ]  || die "-p is required"

IFS=, read -r NPT NPX NPY NPZ <<< "$procs"
for d in "$T:$NPT:T" "$L:$NPX:LX" "$L:$NPY:LY" "$L:$NPZ:LZ"; do
  IFS=: read -r ext np name <<< "$d"
  [ $(( ext % np )) -eq 0 ] || die "$name = $ext is not divisible by its process count $np"
done

tpl="$here/input/invert.input.in"
read -r DT DX DY DZ <<< "$(python3 - "$tpl" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
out = []
for d in "TXYZ":
    m = re.search(r"^\s*MGBlockSizes%s\s*=\s*([0-9,\s]+)$" % d, text, re.M)
    blocks = [int(b) for b in m.group(1).replace(" ", "").split(",") if b] if m else []
    p = 1
    for b in blocks:
        p *= b
    out.append(str(2 * p if blocks else 1))
print(" ".join(out))
PY
)"
for d in "$(( T / NPT )):$DT:T" "$(( L / NPX )):$DX:LX" "$(( L / NPY )):$DY:LY" "$(( L / NPZ )):$DZ:LZ"; do
  IFS=: read -r ext div name <<< "$d"
  [ $(( ext % div )) -eq 0 ] || die "local $name = $ext is not a multiple of $div (multigrid blocking)"
done

mkdir -p "$outdir"
sed -e "s|@T@|$T|g" -e "s|@L@|$L|g" \
    -e "s|@NPX@|$NPX|g" -e "s|@NPY@|$NPY|g" -e "s|@NPZ@|$NPZ|g" \
    -e "s|@NCONF@|$nconf|g" -e "s|@THREADS@|$threads|g" -e "s|@GAUGE@|$gauge|g" \
    "$tpl" > "$outdir/invert.input"
echo "gen_invert_input.sh: $outdir  ${T}x${L}^3  grid $procs" \
     "($(( NPT * NPX * NPY * NPZ )) ranks, local $(( T / NPT ))x$(( L / NPX ))x$(( L / NPY ))x$(( L / NPZ )))"
