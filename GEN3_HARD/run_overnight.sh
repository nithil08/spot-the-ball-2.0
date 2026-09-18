#!/bin/bash
# run_overnight.sh — drive GEN3_HARD from wherever it is to finished clips, unattended.
#
# Every stage is resumable: it skips whatever is already cached, so this is safe to run
# again after an interruption and safe to run when some stages are already done.
#
# caffeinate keeps the machine awake for the duration — an unattended run that gets put
# to sleep half way through is the one failure mode that wastes a whole night.
set -u

# launchd starts jobs with a minimal PATH (/usr/bin:/bin:/usr/sbin:/sbin) that does NOT
# contain python3 — the engine lives under the Python framework. Without this the whole
# overnight run fails instantly with "command not found" and looks like nothing happened.
export PATH="/Library/Frameworks/Python.framework/Versions/3.14/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/_cache/overnight.log"
SH="${SHARDS:-3}"
NI="${NICENESS:-15}"

say() { echo "$(date '+%a %H:%M:%S')  $*" | tee -a "$LOG"; }

# `return ${PIPESTATUS[0]}` is load-bearing, not tidiness. Without it this function
# returns GREP's status, and grep exits 0 whenever it printed anything at all. That
# swallowed `audit`'s exit 4 on the first run: the audit correctly rejected three windows
# and deleted their renders, the script read 0, declared "audit clean", and walked into
# compose with six render files missing. Exit codes are how this pipeline reports, so
# never let a filter stand between a stage and its status.
py() {
  python3 "$HERE/gen3.py" "$@" 2>&1 |
    grep -v "Gym has been\|Please upgrade\|migration guide\|^See \|UNSUPPORTED"
  return "${PIPESTATUS[0]}"
}

exec 3>&1
say "=== GEN3_HARD overnight run starting (${SH} shards) ==="

# Keep the Mac awake while we work, and let it sleep again when we are done.
caffeinate -dimsu -w $$ &
CAF=$!
trap 'kill $CAF 2>/dev/null' EXIT

# ── engine stages ───────────────────────────────────────────────────────────────
say "probe: exact player count per frame, at each match's shipping camera offset"
bash "$HERE/run_phase.sh" probe "$SH" "$NI" >> "$LOG" 2>&1
say "probe done ($(ls "$HERE/_cache/counts" | wc -l | tr -d ' ')/89 matches)"

say "ballpix: exact ball pixel at each candidate end frame"
bash "$HERE/run_phase.sh" ballpix "$SH" "$NI" >> "$LOG" 2>&1
say "ballpix done"

# ── selection and the audit loop ────────────────────────────────────────────────
# `audit` exits 4 when it rejects windows (the ball ends up occluded in the shipped
# render, which no plate measurement can see). The fix is to select again with those
# windows blacklisted and re-render. GEN3 needed two rounds; allow four.
for round in 1 2 3 4; do
  say "--- selection round $round ---"
  py select | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" != "0" ]; then
    say "*** select exited $rc — no feasible 24 under the current blacklist, needs a human"
    exit "$rc"
  fi

  say "render: visible and invisible pair for each of the 24 picks"
  bash "$HERE/run_phase.sh" render_vis "$SH" "$NI" >> "$LOG" 2>&1
  bash "$HERE/run_phase.sh" render_inv "$SH" "$NI" >> "$LOG" 2>&1

  # Belt and braces: the render phases report per-shard, so a window that never rendered
  # is otherwise invisible until compose trips over the missing file.
  missing=$(python3 - "$HERE" <<'PY'
import json, sys, pathlib
h = pathlib.Path(sys.argv[1])
picks = json.loads((h / "_cache" / "picks.json").read_text())
print(sum(1 for p in picks for t in ("vis", "inv")
          if not (h / "_cache" / "renders" /
                  f"{p['match']}_{p['start']}_{p['end']}_{t}.npz").exists()))
PY
)
  if [ "$missing" != "0" ]; then
    say "*** $missing of 48 renders missing after both render phases — needs a human"
    exit 5
  fi

  say "audit: is the ball actually readable in the shipped render?"
  py audit | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" = "0" ]; then
    say "audit clean after $round round(s)"
    break
  fi
  if [ "$rc" != "4" ]; then
    say "*** audit exited $rc — stopping, this needs a human"
    exit "$rc"
  fi
  say "audit rejected windows — reselecting"
  if [ "$round" = "4" ]; then
    say "*** still rejecting after 4 rounds — stopping, this needs a human"
    exit 4
  fi
done

# ── output ──────────────────────────────────────────────────────────────────────
say "compose: cut the 48 clips and write ground truth"
py compose | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" != "0" ]; then
  say "*** compose exited $rc — no clips written, stopping"
  exit "$rc"
fi

say "verify: every spec check, measured from the rendered frames"
python3 "$HERE/verify_gen3.py" 2>&1 |
  grep -v "Gym has been\|Please upgrade\|migration guide\|^See \|UNSUPPORTED" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" != "0" ]; then
  say "*** verify exited $rc — the batch does NOT meet spec, see above"
  exit "$rc"
fi

say "scene graphs: per-frame game state, free from the sweep logs"
python3 "$HERE/scene_graph.py" >> "$LOG" 2>&1

say "=== GEN3_HARD overnight run finished ==="
say "clips in $HERE/clips, contact sheets in $HERE/_review"

# Fire once and only once: unload the scheduler so it does not run again tomorrow night.
launchctl unload "$HOME/Library/LaunchAgents/com.nithil.gen3hard.plist" 2>/dev/null
say "scheduler unloaded"
