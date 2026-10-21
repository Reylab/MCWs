#!/bin/bash
H=/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline/python_vs_wave_clus
OUT="/mnt/data0/MCWs Sessions/matlab validation/single channel/analysis_python_vs_wave_clus"
PY=/home/ajb/Documents/GitHub/MicroWireSorter/.venv/bin/python
export MPLCONFIGDIR="$OUT/logs/mpl"
m() { matlab -batch "addpath('$H'); pvw_matlab('$1')" > "$OUT/logs/matlab_$1.log" 2>&1; echo "matlab $1 EXIT $? errors $(grep -ac Error "$OUT/logs/matlab_$1.log") passed $(grep -ac 'ALL CHECKS PASSED' "$OUT/logs/matlab_$1.log")"; }
m gt_vs_python
m wave_clus
m diagnostics
m tables
cd /home/ajb/Documents/GitHub/MicroWireSorter
for c in 333 259 299 300 328 368; do $PY $H/build_pack.py $c 2>&1 | grep -v -i warn | tail -1; done
$PY $H/build_overview.py 2>&1 | grep -v -i warn | tail -2
$PY $H/where.py; echo "missing: $(grep -c MISSING "$OUT/where_everything_is.txt")"
