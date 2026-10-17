#!/bin/bash
# python vs wave_clus analysis, end to end (see README.md). Read-only on every
# sort; re-running skips validations that already finished.
#   ./run_analysis.sh            all steps
#   ./run_analysis.sh report     only rebuild the tables, appendices and overview
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/home/ajb/Documents/GitHub/MicroWireSorter/.venv/bin/python   # MicroWireSorter env (scipy, matplotlib, diptest)
export PVW_CONFIG=${PVW_CONFIG:-channels.json}   # PVW_CONFIG=channels_tuned.json ./run_analysis.sh for the tuned trial
OUT=$($PY -c "import json; print(json.load(open('$HERE/$PVW_CONFIG'))['out_dir'])")
mkdir -p "$OUT/logs"; export MPLCONFIGDIR="$OUT/logs/mpl"
CHS=$($PY -c "import json; print(' '.join(sorted({str(u['ch']) for u in json.load(open('$HERE/$PVW_CONFIG'))['units']})))")
m() { matlab -batch "addpath('$HERE'); pvw_matlab('$1')" > "$OUT/logs/matlab_$1.log" 2>&1; echo "  matlab $1 done"; }

if [ "$1" != "report" ]; then
  echo "1/5 GT + wave_clus export";         m gt_export; m wc_export
  echo "2/5 validations (GT vs python, wave_clus runs)"; m validations
  echo "3/5 diagnostic figures";             m diagnostics
  echo "4/5 log-det replay + covariance";
  cd /home/ajb/Documents/GitHub/MicroWireSorter
  for c in $CHS; do $PY "$HERE/logdet_replay.py" $c > "$OUT/logs/logdet_ch$c.log" 2>&1 && $PY "$HERE/logdet_fig.py" $c >> "$OUT/logs/logdet_ch$c.log" 2>&1 & done; wait
  $PY "$HERE/covariance_check.py" > "$OUT/logs/covariance.log" 2>&1
  for c in $CHS; do $PY "$HERE/wc_forced_fig.py" $c > "$OUT/logs/wc_forced_ch$c.log" 2>&1; $PY "$HERE/gt_fig.py" $c >> "$OUT/logs/wc_forced_ch$c.log" 2>&1; done
fi
echo "5/5 tables, appendices, overview"; m tables
cd /home/ajb/Documents/GitHub/MicroWireSorter
for c in $CHS; do $PY "$HERE/build_pack.py" $c 2>&1 | grep -v -i warn | tail -1; done
$PY "$HERE/build_overview.py" 2>&1 | grep -v -i warn | tail -1
$PY "$HERE/where.py"
