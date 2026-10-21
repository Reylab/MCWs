#!/bin/bash
# Rebuild every for ppt report with the corrected windows, check each, then tuck the no-notch runs away.
# Starts only if ch259's rebuild passed its check. 2 workers, single-threaded MATLAB (low CPU).
# Result: "/mnt/data0/MCWs Sessions/matlab validation/ppt_status.txt" (one line per GT unit: PASS / FAIL).
S=/tmp/claude-1017/-home-ajb-Documents-GitHub-MCWs-MATLAB-pipeline/1d8fd482-04b6-49a4-9526-6897f90343d6/scratchpad/ppt
V=/home/ajb/Documents/GitHub/MCWs/MATLAB_pipeline/validation_pipeline
SC="/mnt/data0/MCWs Sessions/matlab validation/single channel"; OUT="/mnt/data0/MCWs Sessions/matlab validation/ppt_status.txt"
while ps -eo args | grep -q "[p]pt_session('pt4_ch259'"; do sleep 60; done
if ! grep -q "\[ppt_check\] pt4_ch259: PASS" "$S/log_fix259.txt"; then echo "ch259 did not pass -- nothing else run. See $S/log_fix259.txt" > "$OUT"; exit 1; fi
echo "pt4_ch259 PASS" > "$OUT"
awk -F'\t' '{split($1,a,"_class"); print a[1] "\t" $0}' "$S/sessions.txt" | sort -s -k1,1 > "$S/grouped_f.txt"
cut -f1 "$S/grouped_f.txt" | uniq > "$S/channels_f.txt"
for w in 0 1; do
  ( awk -v w=$w 'NR%2==w' "$S/channels_f.txt" | while IFS= read -r c; do
      awk -F'\t' -v c="$c" '$1==c {print $2 "\t" $3}' "$S/grouped_f.txt" | while IFS=$'\t' read -r id multi; do
        matlab -singleCompThread -batch "addpath('$V'); cfg = validation_sessions('$id'); o = fullfile(cfg.root, 'for ppt'); if $multi, o = fullfile(o, sprintf('GT c%d', cfg.gt.class)); end; a = fullfile(cfg.root, 'archive_2026-10-17', 'for ppt_gt_windows'); if $multi, a = fullfile(a, sprintf('GT c%d', cfg.gt.class)); end; if exist(o, 'dir'), if ~exist(fileparts(a), 'dir'), mkdir(fileparts(a)); end; movefile(o, a); end; ppt_session('$id', logical($multi)); ppt_check_unit('$id', logical($multi));" > "$S/log_final_$id.txt" 2>&1
        r=$(grep -o "\[ppt_check\] .*" "$S/log_final_$id.txt" | head -1); [ -z "$r" ] && r="[ppt_check] $id: FAILED TO RUN ($(grep -m1 -A2 'Error' "$S/log_final_$id.txt" | tr '\n' ' '))"
        echo "$r" >> "$OUT"
      done
    done ) &
done
wait
cd "$SC"
for d in pt*; do
  [ -d "$d/wave_clus no notch" ] && { mkdir -p "$d/_no_notch"; mv "$d/wave_clus no notch" "$d/_no_notch/"; }
  find "$d/for ppt" -maxdepth 2 -type d -name "07_no_notch_vs_notch" 2>/dev/null | while read -r f; do p=$(dirname "$f"); mkdir -p "$p/_no_notch"; mv "$f" "$p/_no_notch/"; done
done
echo "ALL DONE $(date)" >> "$OUT"
