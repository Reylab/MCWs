#!/bin/bash
S=/tmp/claude-1017/-home-ajb-Documents-GitHub-MCWs-MATLAB-pipeline/1d8fd482-04b6-49a4-9526-6897f90343d6/scratchpad
H=/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline/python_vs_wave_clus
OUT="/mnt/data0/MCWs Sessions/matlab validation/single channel/analysis_python_vs_wave_clus"
r() { cd $S && matlab -batch "redo_runs('$S/old_runs.txt', '$1')" > $S/redo_$2.log 2>&1; echo "$2 EXIT $? redone $(grep -ac '^\[redo\]' $S/redo_$2.log) passed $(grep -ac 'ALL CHECKS PASSED' $S/redo_$2.log) errors $(grep -ac Error $S/redo_$2.log)"; }
r "ch 333" ch333 & r "ch 259" ch259 & r "ch 299" ch299 & r "ch 300" ch300 & wait
cd $S && matlab -batch "addpath('$H'); pvw_matlab('diagnostics'); pvw_matlab('tables')" > "$OUT/logs/matlab_diag_tables.log" 2>&1; echo "diagnostics+tables EXIT $? errors $(grep -ac Error "$OUT/logs/matlab_diag_tables.log")"
export MPLCONFIGDIR="$OUT/logs/mpl"; cd /home/ajb/Documents/GitHub/MicroWireSorter
for c in 333 259 299 300 328 368; do .venv/bin/python $H/build_pack.py $c 2>&1 | grep -v -i warn | tail -1; done
.venv/bin/python $H/build_overview.py 2>&1 | grep -v -i warn | tail -1
.venv/bin/python $H/where.py; echo "missing: $(grep -c MISSING "$OUT/where_everything_is.txt")"
