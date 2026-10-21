#!/bin/bash
# One job of run_batch.sh: (1) the python sorter on the channel's recording (skipped when its output folder exists or
# RECORDING is '-'), (2) mcws_validation in MATLAB on the channel folder. Usage (run_batch.sh calls it):
#   run_one.sh "<channel folder>|<recording folder or ->|<ch>|<gt_class, MATLAB>|<stims, MATLAB>"
# env: TAG, RUNS_ROOT, OVERRIDES, RESCUE_SD, LOGS (see README_batch.md)
V=$(cd "$(dirname "$0")/.." && pwd); MWS=${MCWS_ROOT:-/home/ajb/Documents/GitHub/MicroWireSorter}
IFS='|' read -r folder rec ch gtc stims <<< "$1"
name=$(basename "$folder" | tr ' ' '_'); log="$LOGS/${name}${TAG:+_$TAG}"; run="$RUNS_ROOT/$name"
if [ "$rec" != "-" ]; then
  if [ ! -d "$run" ]; then
    (cd "$MWS" && nice -n 10 .venv/bin/python "$V/run_mcws_with.py" $OVERRIDES -- "$rec" "$run" "$ch" > "$log.python.txt" 2>&1 < /dev/null)
    cp -n "$log.python.txt" "$run/batch.log" 2>/dev/null
  fi
  grep -q "Pipeline session ended (exit_code=0" "$run/batch.log" 2>/dev/null || { echo "$name: python run FAILED (see $log.python.txt)"; exit 1; }
  echo "$name: python run OK ($run)"; pr="'python_run', '$run', "
fi
nice -n 10 matlab -singleCompThread -batch "addpath('$V'); mcws_validation('folder', '$folder', 'ch', $ch, 'gt_class', $gtc, 'stims', $stims, ${pr}'tag', '$TAG', 'rescue_sd', ${RESCUE_SD:-1}); disp('JOB DONE')" < /dev/null > "$log.matlab.txt" 2>&1
grep -q "JOB DONE" "$log.matlab.txt" && echo "$name: $(grep -o '\[mcws_validation\] done.*' "$log.matlab.txt")" || echo "$name: MATLAB FAILED (see $log.matlab.txt): $(grep -m1 -A2 Error "$log.matlab.txt" | tr '\n' ' ')"
