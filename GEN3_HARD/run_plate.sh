#!/bin/bash
# run_plate.sh <vis|inv> [n_shards] — drive fill_gaps.py plate_* to completion.
set -u
PHASE="${1:?usage: run_plate.sh <vis|inv> [n]}"
N="${2:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="/Library/Frameworks/Python.framework/Versions/3.14/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
mkdir -p "$HERE/_cache/logs"
for ((i=0;i<N;i++)); do
  ( while true; do
      nice -n 10 python3 "$HERE/fill_gaps.py" "plate_$PHASE" "$i" "$N" \
        >> "$HERE/_cache/logs/plate_${PHASE}_$i.log" 2>&1
      [ $? -eq 3 ] && break
    done ) &
done
wait
echo "== plate $PHASE complete =="
