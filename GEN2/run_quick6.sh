#!/bin/bash
# run_quick6.sh — sweep, gate and build 2 clips for each of the three situation classes.
#
# SHARDS=3 rather than 4 on purpose. The 4-shard full-speed run on 2026-08-26 pinned every
# performance core and the machine rebooted seven minutes in, losing the whole job. A
# reboot costs far more time than the third of a core this gives back.
set -u
cd "/Users/nithilbalamurugan/Desktop/Nithil Research/spot-the-ball-2.0/GEN2" || exit 1

if [ -z "${Q6_CAFFEINATED:-}" ]; then
  export Q6_CAFFEINATED=1
  exec /usr/bin/caffeinate -ims /bin/bash "$0" "$@"
fi

PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
SHARDS=${SHARDS:-3}
LOG=_cache/quick6/run.log
mkdir -p _cache/quick6
exec >> "$LOG" 2>&1

echo ""
echo "================ run_quick6.sh starting $(date '+%Y-%m-%d %H:%M:%S %Z') ================"
step() { echo ""; echo "---- $* ---- $(date '+%H:%M:%S')"; }

step "sweep ($SHARDS shards, one match per process)"
for i in $(seq 0 $((SHARDS - 1))); do
  (
    fails=0
    while true; do
      "$PY" gen_quick6.py sweep --shard "$i/$SHARDS" >> "_cache/quick6/sweep_$i.log" 2>&1
      rc=$?
      if [ "$rc" -eq 3 ]; then
        echo "shard $i: complete" >> "_cache/quick6/sweep_$i.log"; break
      elif [ "$rc" -ne 0 ]; then
        # The engine leaks and dies at roughly its 46th env. A crash is retried with a
        # fresh process; only exit 3 (shard empty) ends the loop.
        fails=$((fails + 1))
        echo "shard $i: died rc=$rc, retry $fails" >> "_cache/quick6/sweep_$i.log"
        [ "$fails" -ge 25 ] && { echo "shard $i: giving up" >> "_cache/quick6/sweep_$i.log"; break; }
      else
        fails=0
      fi
    done
  ) &
done
wait
echo "sweep done: $(ls _cache/quick6/sweep 2>/dev/null | wc -l | tr -d ' ') matches"

step "pick (log gates + ball-coherence gate)"
"$PY" gen_quick6.py pick || { echo "PICK FAILED"; exit 1; }

for kind in header corner gk_throw; do
  step "gate $kind"
  "$PY" gen_quick6.py gate "$kind"
  step "build $kind"
  "$PY" gen_quick6.py build "$kind"
  echo "$kind build exit: $?"
done

step "done"
echo "======== run_quick6.sh finished $(date '+%H:%M:%S') ========"
