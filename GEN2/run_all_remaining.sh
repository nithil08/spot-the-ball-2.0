#!/bin/bash
# run_all_remaining.sh — wait out the situations job, then run player-delta.
#
# Sequential on purpose: both stages are render-bound and this machine has 4
# performance cores and 16 GB. Overlapping them turns a render job into a swap storm.
set -u
cd "/Users/nithilbalamurugan/Desktop/Nithil Research/spot-the-ball-2.0/GEN2" || exit 1
LOG=_cache/all_remaining.log
exec >> "$LOG" 2>&1
echo ""
echo "======== waiting for situations v2 $(date '+%H:%M:%S') ========"
while pgrep -f "run_v2_finish.sh" > /dev/null; do sleep 30; done
echo "situations v2 released the machine at $(date '+%H:%M:%S')"
echo "======== player-delta starting $(date '+%H:%M:%S') ========"
SHARDS=4 COOL=0 /bin/bash run_tonight.sh
echo "======== player-delta finished $(date '+%H:%M:%S'), rc=$? ========"
