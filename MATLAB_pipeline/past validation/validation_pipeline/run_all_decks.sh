#!/bin/bash
# Flags + deck for every GT unit whose report passed its check (ppt_status.txt).
#   ppt_flags (MATLAB)  -> <report>/flags.csv
#   build_channel_deck.py -> <report>/<channel> wc_mcws_auto.pptx / .odp   (ch259 blueprint; never touches other files)
# Waits for run_all_ppt.sh to finish first. One unit at a time, single-threaded MATLAB (low CPU).
# Result: "/mnt/data0/MCWs Sessions/matlab validation/deck_status.txt"
V=/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline
PY=$HOME/.venvs/pptdeck/bin/python                 # python-pptx (python3 -m venv ~/.venvs/pptdeck; pip install python-pptx)
R="/mnt/data0/MCWs Sessions/matlab validation"; STATUS="$R/ppt_status.txt"; OUT="$R/deck_status.txt"
while ! grep -q "ALL DONE" "$STATUS" 2>/dev/null; do sleep 120; done
: > "$OUT"
grep -o "\[ppt_check\] [^:]*: PASS" "$STATUS" | sed 's/\[ppt_check\] //; s/: PASS//' > /tmp/deck_units_$$.txt
grep -q "^pt4_ch259 PASS" "$STATUS" && echo pt4_ch259 >> /tmp/deck_units_$$.txt
while read -r id; do
  rep=$(matlab -singleCompThread -batch "addpath('$V'); cfg = validation_sessions('$id'); r = fullfile(cfg.root, 'for ppt'); m = exist(fullfile(r, sprintf('GT c%d', cfg.gt.class)), 'dir') == 7; if m, r = fullfile(r, sprintf('GT c%d', cfg.gt.class)); end; ppt_flags('$id', m); fprintf('REPORT=%s\n', r);" < /dev/null 2>&1 | grep -o "REPORT=.*" | sed 's/REPORT=//')
  if [ -n "$rep" ] && "$PY" "$V/build_channel_deck.py" "$rep" >> "$OUT.log" 2>&1 < /dev/null; then
    echo "$id deck OK  ($(grep -c . "$rep/flags.csv" | awk '{print $1-1}') flags) -> $rep" >> "$OUT"
  else
    echo "$id deck FAILED (see $OUT.log)" >> "$OUT"
  fi
done < /tmp/deck_units_$$.txt
rm -f /tmp/deck_units_$$.txt
echo "ALL DONE $(date)" >> "$OUT"
