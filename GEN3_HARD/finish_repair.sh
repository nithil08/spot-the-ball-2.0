#!/bin/bash
# finish_repair.sh [shards] — drive the whole repair to a composed batch, unattended.
#
# The phases interlock: a solve can want counts it does not have, a count measurement can
# move a window to another level, and an audit can reject a window and send the solve back
# round. Driving them by hand costs a human round-trip at every one of those points, which
# is most of the wall clock. This loops them.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export PATH="/Library/Frameworks/Python.framework/Versions/3.14/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
N="${1:-5}"
LOG="$HERE/_cache/logs/finish.log"
say() { echo "$(date '+%H:%M:%S')  $*" | tee -a "$LOG"; }
py() { python3 "$@" 2>&1 | grep -v "Gym has\|Please upgrade\|migration guide\|^See \|UNSUPPORTED"; return "${PIPESTATUS[0]}"; }

# vis and inv are different bundles and so different processes; the inv pass consumes what
# the vis pass produces, so running them together is a pipeline rather than a race.
drive_pair() {   # drive_pair <phase-prefix> <script-cmd-vis> <script-cmd-inv>
  local tag="$1" vis="$2" inv="$3"
  for ((i=0;i<N;i++)); do
    ( while true; do nice -n 10 python3 fill_gaps.py "$vis" "$i" "$N" \
        >> "_cache/logs/${tag}_vis_$i.log" 2>&1; [ $? -eq 3 ] && break; done ) &
    ( while true; do nice -n 10 python3 fill_gaps.py "$inv" "$i" "$N" \
        >> "_cache/logs/${tag}_inv_$i.log" 2>&1
        # the inv side may simply be waiting on its vis pass, so it sleeps rather than
        # exiting the moment it finds nothing to do
        if [ $? -eq 3 ]; then sleep 20
          nice -n 10 python3 fill_gaps.py "$inv" "$i" "$N" >> "_cache/logs/${tag}_inv_$i.log" 2>&1
          [ $? -eq 3 ] && break
        fi
      done ) &
  done
  wait
}

say "=== screening candidates for ball-in-view ==="
drive_pair scan scan_vis scan_inv
say "screening done"

for round in 1 2 3 4 5; do
  say "=== round $round: solve ==="
  py fill_gaps.py solve | tee -a "$LOG" | tail -6
  left=$(python3 -c "import json;print(len(json.load(open('_cache/gaps/toverify.json'))))")
  if [ "$left" != "0" ]; then
    say "round $round: measuring $left windows at full resolution"
    for ((i=0;i<N;i++)); do
      ( while true; do nice -n 10 python3 fill_gaps.py verify "$i" "$N" \
          >> "_cache/logs/fillverify_$i.log" 2>&1; [ $? -eq 3 ] && break; done ) &
    done
    wait
    continue
  fi
  say "round $round: every chosen window is measured — emitting"
  py fill_gaps.py emit | tee -a "$LOG"
  cp _cache/gaps/picks_new.json _cache/picks.json
  say "round $round: rendering"
  bash run_phase.sh render_vis "$N" >> "$LOG" 2>&1
  bash run_phase.sh render_inv "$N" >> "$LOG" 2>&1
  say "round $round: audit"
  py fill_gaps.py audit | tee -a "$LOG"
  code=$?
  if [ $code -eq 5 ]; then
    drive_pair plate plate_vis plate_inv
    py fill_gaps.py audit | tee -a "$LOG"; code=$?
  fi
  if [ $code -eq 0 ]; then
    say "=== audit clean — composing ==="
    py gen3.py compose | tee -a "$LOG" | tail -5
    say "=== DONE ==="
    exit 0
  fi
  say "round $round: audit rejected windows, re-screening and solving again"
  py fill_gaps.py scanlist 8 | tee -a "$LOG"
  drive_pair scan scan_vis scan_inv
done
say "!!! did not converge in 5 rounds — inspect _cache/gaps/solution.json"
exit 1
