#!/bin/bash
# resume_at.sh HH:MM <phase> [shards] [nice] — sleep until a wall-clock time, then run a
# phase to completion. Detached and nohup'd so it survives the terminal.
#
# Every phase is resumable (it skips anything already cached), so the worst case for an
# interrupted run is redoing the matches that were in flight.
set -u
TARGET="${1:?usage: resume_at.sh HH:MM <phase> [shards] [nice]}"
PHASE="${2:?}"; SH="${3:-3}"; NI="${4:-15}"
HERE="$(cd "$(dirname "$0")" && pwd)"

now=$(date +%s)
when=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%F) $TARGET" +%s 2>/dev/null)
[ "$when" -le "$now" ] && when=$((when + 86400))     # already past today -> tomorrow
delay=$((when - now))
echo "$(date '+%a %H:%M:%S') waiting ${delay}s until $(date -r "$when" '+%a %H:%M') for phase '$PHASE'"
sleep "$delay"

echo "$(date '+%a %H:%M:%S') starting $PHASE on $SH shards (nice +$NI)"
bash "$HERE/run_phase.sh" "$PHASE" "$SH" "$NI"
echo "$(date '+%a %H:%M:%S') $PHASE finished"

# ballpix is the last engine stage before selection, and selection is cheap and pure CPU,
# so run it straight away: there is then a result waiting rather than another manual step.
if [ "$PHASE" = "ballpix" ]; then
  echo "$(date '+%a %H:%M:%S') running select"
  python3 "$HERE/gen3.py" select 2>&1 | grep -v "Gym\|gymnasium\|migration\|^See"
fi
