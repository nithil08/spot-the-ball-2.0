#!/bin/bash
# run_corners_v2.sh — sweep matches for the v2 corner rebuild.
#
# One match per process: the engine leaks and dies at roughly its 46th env, so a
# long-lived worker is guaranteed to die part-way. `sweep` exits 3 when its shard is
# empty, which is the ONLY reason to stop; anything else is the leak and is retried.
#
# Everything runs under `taskpolicy -b` (background QoS), which parks it on the
# efficiency cores. That is why SHARDS=4 is safe here despite the machine only having
# 4 performance cores — none of this touches them.
set -u
cd "/Users/nithilbalamurugan/Desktop/Nithil Research/spot-the-ball-2.0/GEN2" || exit 1

SHARDS=${SHARDS:-4}
PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
RUN="/usr/sbin/taskpolicy -b"

mkdir -p _cache/corners_v2
LOG=_cache/corners_v2/sweep.log
exec >> "$LOG" 2>&1
echo "==== corners_v2 sweep starting $(date '+%H:%M:%S'), $SHARDS shards ===="

for i in $(seq 0 $((SHARDS - 1))); do
  (
    fails=0
    while true; do
      $RUN "$PY" gen_corners_v2.py sweep --shard "$i/$SHARDS" \
          >> "_cache/corners_v2/shard_$i.log" 2>&1
      rc=$?
      if [ "$rc" -eq 3 ]; then
        echo "shard $i: complete" >> "_cache/corners_v2/shard_$i.log"
        break
      elif [ "$rc" -ne 0 ]; then
        fails=$((fails + 1))
        echo "shard $i: died rc=$rc, retry $fails" >> "_cache/corners_v2/shard_$i.log"
        [ "$fails" -ge 25 ] && { echo "shard $i: giving up"; break; }
      else
        fails=0
      fi
    done
  ) &
done
wait
echo "==== sweep finished $(date '+%H:%M:%S'): $(ls _cache/corners_v2/sweep/*.npz 2>/dev/null | wc -l | tr -d ' ') matches ===="
