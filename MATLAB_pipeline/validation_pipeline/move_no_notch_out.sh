#!/bin/bash
# Keep no-notch material out of every for ppt folder: <channel>/wave_clus no notch and any
# 07_no_notch_vs_notch (anywhere under for ppt, incl. for ppt/_no_notch) -> <channel>/_no_notch/.
# Waits for run_all_ppt.sh to finish first.
SC="/mnt/data0/MCWs Sessions/matlab validation/single channel"; ST="/mnt/data0/MCWs Sessions/matlab validation/ppt_status.txt"
while ! grep -q "ALL DONE" "$ST" 2>/dev/null; do sleep 120; done
cd "$SC"
for d in pt*; do
  [ -d "$d/for ppt" ] || continue
  [ -d "$d/wave_clus no notch" ] && { mkdir -p "$d/_no_notch"; mv "$d/wave_clus no notch" "$d/_no_notch/"; }
  find "$d/for ppt" -type d -name "07_no_notch_vs_notch" -prune 2>/dev/null | while read -r f; do
    mkdir -p "$d/_no_notch"; dst="$d/_no_notch/07_no_notch_vs_notch"; [ -e "$dst" ] && dst="$dst"_$(basename "$(dirname "$f")" | tr ' ' '_')
    mv "$f" "$dst"; echo "moved $f -> $dst"
  done
  find "$d/for ppt" -type d -name "_no_notch" -empty -delete 2>/dev/null
done
echo "no-notch cleanup done $(date)"
