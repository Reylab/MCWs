#!/bin/bash
# mcws_validation on every channel in a job file, N at a time (default 2):
#   TAG=isi3 setsid nohup ./run_batch.sh jobs.txt 4 > batch.out 2>&1 &
jobs=$1; n=${2:-2}; B=$(cd "$(dirname "$0")" && pwd)
export TAG PYTHON DECK_PYTHON; export LOGS=${LOGS:-$PWD/batch_logs}; mkdir -p "$LOGS"
grep -v '^\s*#' "$jobs" | grep -v '^\s*$' | xargs -d '\n' -P "$n" -n 1 "$B/run_one.sh"
echo "BATCH DONE $(date)"
