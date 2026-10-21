#!/bin/bash
S=/tmp/claude-1017/-home-ajb-Documents-GitHub-MCWs-MATLAB-pipeline/1d8fd482-04b6-49a4-9526-6897f90343d6/scratchpad
V=/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline
cd $S && timeout 14000 matlab -batch "addpath('$V'); T = run_wave_clus_validations('$1', $2, 'redo_gt', false); disp(T(:,1:2))" > $S/wcv_logs/fix_$1.log 2>&1
echo "$1 EXIT $?"
