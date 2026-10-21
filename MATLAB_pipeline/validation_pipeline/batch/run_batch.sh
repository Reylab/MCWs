#!/bin/bash
# Run every job in a job file, N at a time (default 2). Usage:
#   TAG=isi3 RUNS_ROOT=/mnt/data0/recluster_runs/runs_isi3 OVERRIDES="second_pass:ISI_THR=3" \
#     setsid nohup ./run_batch.sh jobs.txt 4 > batch.out 2>&1 &
# Job file: one channel per line, '|'-separated, '#' lines ignored (see jobs_example.txt and README_batch.md).
jobs=$1; n=${2:-2}; B=$(cd "$(dirname "$0")" && pwd)
export TAG RUNS_ROOT OVERRIDES RESCUE_SD MCWS_ROOT; export LOGS=${LOGS:-$PWD/batch_logs}; mkdir -p "$LOGS"
[ -n "$RUNS_ROOT" ] && mkdir -p "$RUNS_ROOT"
grep -v '^\s*#' "$jobs" | grep -v '^\s*$' | xargs -d '\n' -P "$n" -n 1 "$B/run_one.sh"
echo "BATCH DONE $(date)"
