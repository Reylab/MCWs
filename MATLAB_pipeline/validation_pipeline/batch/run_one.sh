#!/bin/bash
# One job of run_batch.sh: mcws_validation in MATLAB on one channel folder.
#   run_one.sh "<channel folder>|<python run folder to convert, or ->|<gt_class>|<stims>"
# env: TAG (run tag), PYTHON (set = also run mcws_validation's python section), DECK_PYTHON (python with
#      python-pptx for the deck, default PYTHON), LOGS (log folder)
V=$(cd "$(dirname "$0")/.." && pwd)
IFS='|' read -r folder run gtc stims <<< "$1"
log="$LOGS/$(basename "$folder" | tr ' ' '_')${TAG:+_$TAG}.txt"
opts="'folder', '$folder', 'tag', '$TAG'"
[ "$run" != "-" ] && opts="$opts, 'python_run', '$run'"
[ -n "$PYTHON" ] && opts="$opts, 'python', '$PYTHON'"
[ -n "$DECK_PYTHON" ] && opts="$opts, 'deck_python', '$DECK_PYTHON'"
nice -n 10 matlab -singleCompThread -batch "addpath('$V'); mcws_validation($gtc, $stims, $opts); disp('JOB DONE')" < /dev/null > "$log" 2>&1
if grep -q "JOB DONE" "$log"; then grep -o "\[mcws_validation\] GT c[0-9]*: [A-Z]*" "$log" | sed "s|^|$(basename "$folder"): |"
else echo "$(basename "$folder"): FAILED, see $log"; fi
