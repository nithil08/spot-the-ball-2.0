#!/bin/bash
# run_phase.sh <phase> [n_shards] [nice] — drive a resumable GEN4 engine phase.
# Same protocol as GEN3: exit 0 = did work, 3 = shard empty, anything else = died, retry.
set -u
PHASE="${1:?usage: run_phase.sh <phase> [n_shards] [nice]}"
N="${2:-3}"
NICE="${3:-10}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="/Library/Frameworks/Python.framework/Versions/3.14/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
LOGS="$HERE/_cache/logs"; mkdir -p "$LOGS"
for ((i=0;i<N;i++)); do
  (
    while true; do
      nice -n "$NICE" python3 "$HERE/gen4.py" "$PHASE" --shard "$i/$N" \
        >> "$LOGS/${PHASE}_$i.log" 2>&1
      code=$?
      [ $code -eq 3 ] && break
    done
    echo "shard $i done" >> "$LOGS/${PHASE}_$i.log"
  ) &
done
wait
echo "== $PHASE complete =="
