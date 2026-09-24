#!/bin/bash
# run_scan.sh <vis|inv> [n] — drive the ball-in-view screening to completion.
set -u
PHASE="${1:?usage: run_scan.sh <vis|inv> [n]}"; N="${2:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="/Library/Frameworks/Python.framework/Versions/3.14/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
mkdir -p "$HERE/_cache/logs"
for ((i=0;i<N;i++)); do
  ( while true; do
      nice -n 10 python3 "$HERE/fill_gaps.py" "scan_$PHASE" "$i" "$N" \
        >> "$HERE/_cache/logs/scan_${PHASE}_$i.log" 2>&1
      [ $? -eq 3 ] && break
    done ) &
done
wait
echo "== scan $PHASE complete =="
