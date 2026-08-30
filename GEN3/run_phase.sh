#!/bin/bash
# run_phase.sh <phase> [n_shards] — drive a resumable GEN3 engine phase to completion.
#
# Every engine phase does a bounded amount of work and exits, because the engine leaks:
# a process dies at roughly its 46th FootballEnv with no traceback. The loop below keeps
# re-invoking each shard until it reports EXIT_DONE(3), so a shard that DIES (any other
# non-zero code, including being killed) is simply retried with a fresh process.
#
#   exit 0 -> did some work, call again
#   exit 3 -> shard empty, stop
#   else   -> died, retry
#
# Usage:  bash run_phase.sh sweep 8
#         bash run_phase.sh probe 6
#
# NICE: these are CPU-bound AND each holds an OpenGL context, so a high shard count makes
# the whole desktop sluggish — GPU contention is not something nice(1) can fix, which is
# why the shard COUNT is the real throttle and the priority is only a second line of
# defence. On a 10-core machine, 3 shards leaves the interactive apps a clear majority of
# the cores. Throughput barely suffers, because at 6 shards the machine was thrashing:
# load averaged 8.1 and matches took two to three times their uncontended 7 minutes.
set -u
PHASE="${1:?usage: run_phase.sh <phase> [n_shards] [nice]}"
N="${2:-3}"
NICE="${3:-15}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="$HERE/_cache/logs"
mkdir -p "$LOGS"

echo "== $PHASE across $N shards (nice +$NICE) =="
for ((i = 0; i < N; i++)); do
  (
    tries=0
    while true; do
      nice -n "$NICE" python3 "$HERE/gen3.py" "$PHASE" --shard "$i/$N" >> "$LOGS/${PHASE}_$i.log" 2>&1
      rc=$?
      if [ $rc -eq 3 ]; then
        echo "shard $i: done" >> "$LOGS/${PHASE}_$i.log"
        break
      fi
      if [ $rc -ne 0 ]; then
        tries=$((tries + 1))
        echo "shard $i: rc=$rc (retry $tries)" >> "$LOGS/${PHASE}_$i.log"
        if [ $tries -ge 40 ]; then
          echo "shard $i: giving up after $tries failures" >> "$LOGS/${PHASE}_$i.log"
          break
        fi
      else
        tries=0
      fi
    done
  ) &
done
wait
echo "== $PHASE finished =="
