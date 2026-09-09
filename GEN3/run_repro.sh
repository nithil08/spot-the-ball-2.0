#!/bin/bash
# Drive verify_reproducible.py to completion. Same contract as run_phase.sh:
# exit 3 = shard empty, 0 = did work, anything else = the engine died, retry.
set -u
N="${1:-3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="$HERE/_cache/logs"; mkdir -p "$LOGS"
for ((i = 0; i < N; i++)); do
  (
    tries=0
    while true; do
      nice -n 15 python3 "$HERE/verify_reproducible.py" run --shard "$i/$N" \
        >> "$LOGS/repro_$i.log" 2>&1
      rc=$?
      [ $rc -eq 3 ] && break
      if [ $rc -ne 0 ]; then
        tries=$((tries + 1)); echo "shard $i: rc=$rc (retry $tries)" >> "$LOGS/repro_$i.log"
        [ $tries -ge 20 ] && break
      else tries=0; fi
    done
  ) &
done
wait
echo "== reproducible finished =="
