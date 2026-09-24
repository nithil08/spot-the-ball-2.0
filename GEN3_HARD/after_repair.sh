#!/bin/bash
# after_repair.sh [shards] — everything that follows a composed batch.
# Waits for finish_repair.sh, then rebuilds the evidence and the derived files.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
export PATH="/Library/Frameworks/Python.framework/Versions/3.14/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
N="${1:-5}"
LOG="$HERE/_cache/logs/after.log"
say() { echo "$(date '+%H:%M:%S')  $*" | tee -a "$LOG"; }
py() { python3 "$@" 2>&1 | grep -v "Gym has\|Please upgrade\|migration guide\|^See \|UNSUPPORTED"; return "${PIPESTATUS[0]}"; }

while pgrep -f finish_repair.sh > /dev/null; do sleep 30; done
if [ ! -f clips/ground_truth.csv ]; then say "compose never ran — stopping"; exit 1; fi
say "=== remapping the final-frame evidence the kept clips already have ==="
py remap_evidence.py | tee -a "$LOG"
say "=== measuring the new clips' final frames at full resolution ==="
for ((i=0;i<N;i++)); do
  ( while true; do nice -n 10 python3 verify_final_frame.py "$i" "$N" \
      >> "_cache/logs/finalframe_$i.log" 2>&1
    python3 -c "
import json,sys
from pathlib import Path
picks=json.loads(Path('_cache/picks.json').read_text())
sys.exit(0 if all((Path('_cache/finalframe')/f\"{p['clip']}.npz\").exists() for p in picks) else 1)" && break
  done ) &
done
wait
say "=== derived files ==="
py gen3_report.py classify | tee -a "$LOG"
py review_gen3_hard.py | tee -a "$LOG"
py review_rebuild.py | tee -a "$LOG"
say "=== verifier ==="
py verify_gen3.py | tee -a "$LOG" | tail -30
say "=== ALL DONE ==="
