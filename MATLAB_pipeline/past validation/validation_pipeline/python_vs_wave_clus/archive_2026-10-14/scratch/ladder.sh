#!/bin/bash
# per channel: replay + save width, mild, within (loose); sequential per channel, channels in parallel
S=/tmp/claude-1017/-home-ajb-Documents-GitHub-MCWs-MATLAB-pipeline/1d8fd482-04b6-49a4-9526-6897f90343d6/scratchpad
TR=/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline/python_vs_wave_clus/trial_replay.py
cd /home/ajb/Documents/GitHub/MicroWireSorter
for c in 259 299 300 328 333 334 337 368; do
  ( for t in width mild within; do MPLCONFIGDIR=$S/mpl .venv/bin/python $TR $c $t $t >> $S/ladder_$c.log 2>&1; done; echo "ch$c finished" >> $S/ladder_$c.log ) &
done
wait
grep -h "saved\|Traceback\|Error" $S/ladder_*.log
