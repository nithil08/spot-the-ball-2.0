#!/bin/bash
# stop_phase.sh — actually stop a run_phase.sh phase.
#
# `pkill -f run_phase.sh` is NOT enough and quietly fails. run_phase.sh forks one SUBSHELL
# per shard, each looping `python3 gen3.py <phase>` until it exits 3. Killing the parent
# leaves those subshells alive, and within seconds each relaunches a worker — so a check
# straight after the kill reports zero engine processes and the phase carries right on.
#
# Kill the subshells FIRST (so nothing can respawn), then the workers, then verify after a
# grace period that nothing came back.
set -u
for sig in TERM KILL; do
  pkill -"$sig" -f "run_phase.sh" 2>/dev/null
  sleep 1
  pkill -"$sig" -f "gen3.py" 2>/dev/null
  sleep 1
done
sleep 4
d=$(pgrep -f "run_phase.sh" | wc -l | tr -d ' ')
w=$(pgrep -f "gen3.py" | wc -l | tr -d ' ')
echo "after stop: $d driver(s), $w worker(s)"
[ "$d" = "0" ] && [ "$w" = "0" ] && echo "STOPPED CLEANLY" || echo "STILL ALIVE — investigate"
