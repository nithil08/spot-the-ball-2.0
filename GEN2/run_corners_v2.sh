#!/bin/bash
# run_corners_v2.sh — build the 5 rebuilt corner clips, unattended.
#
# sweep -> pick -> gate -> build. Every stage checkpoints, so re-running continues
# rather than redoing. 37 matches were already banked before this was scheduled.
#
# One match per process: the engine leaks and dies at roughly its 46th env, so a
# long-lived worker is guaranteed to die part-way. `sweep` exits 3 when its shard is
# empty, which is the ONLY reason to stop; anything else is the leak, and is retried.
#
# COOL=1 runs under `taskpolicy -b` (efficiency cores) — cool but ~5x slower, measured
# at 190 s/match against 36 s/match on the performance cores. This job is scheduled for
# 23:00 when nobody is using the machine, so it defaults to COOL=0 and takes the
# performance cores. Set COOL=1 to run it while you are working.
set -u
cd "/Users/nithilbalamurugan/Desktop/Nithil Research/spot-the-ball-2.0/GEN2" || exit 1

# Hold a sleep assertion for the run; pmset reports sleep=1 and neither launchd nor a
# backgrounded shell holds one. Display sleep is deliberately still allowed.
if [ -z "${CORNERS_CAFFEINATED:-}" ]; then
  export CORNERS_CAFFEINATED=1
  exec /usr/bin/caffeinate -ims /bin/bash "$0" "$@"
fi

SHARDS=${SHARDS:-4}
COOL=${COOL:-0}
PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
if [ "$COOL" = "1" ]; then RUN="/usr/sbin/taskpolicy -b"; else RUN=""; fi

mkdir -p _cache/corners_v2
exec >> _cache/corners_v2/run.log 2>&1
echo ""
echo "======== corners_v2 starting $(date '+%Y-%m-%d %H:%M:%S'), SHARDS=$SHARDS COOL=$COOL ========"

step() { echo ""; echo "---- $* ---- $(date '+%H:%M:%S')"; }

# ── 1. sweep ────────────────────────────────────────────────────────────────────
step "sweep"
for i in $(seq 0 $((SHARDS - 1))); do
  (
    fails=0
    while true; do
      $RUN "$PY" gen_corners_v2.py sweep --shard "$i/$SHARDS" \
          >> "_cache/corners_v2/shard_$i.log" 2>&1
      rc=$?
      if [ "$rc" -eq 3 ]; then
        echo "shard $i: complete" >> "_cache/corners_v2/shard_$i.log"; break
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
echo "sweep done: $(ls _cache/corners_v2/sweep/*.npz 2>/dev/null | wc -l | tr -d ' ') matches"

# ── 2. pick, gate, build ────────────────────────────────────────────────────────
step "pick (log gates)"
$RUN "$PY" gen_corners_v2.py pick || { echo "PICK FAILED"; exit 1; }

step "gate (render + measure on-screen occupancy)"
$RUN "$PY" gen_corners_v2.py gate || { echo "GATE FAILED"; exit 1; }

step "build (best 5 by worst frame)"
$RUN "$PY" gen_corners_v2.py build
BUILD_RC=$?
echo "build exit code: $BUILD_RC"

# ── 3. review sheet, so the result can be judged without opening 5 videos ───────
if [ "$BUILD_RC" -eq 0 ]; then
  step "review sheet"
  $RUN "$PY" - <<'PYEOF'
import sys, csv
sys.path.insert(0, '.')
import cv2, numpy as np
from pathlib import Path
OUT = Path("02_corner_kicks_v2")
rows = list(csv.DictReader(open(OUT / "ground_truth.csv")))
CW, CH = 420, 158
# 5 frames spread across the clip, so judder and dead frames both show up
picks = [0, 0.25, 0.5, 0.75, 0.99]
sheet = np.zeros((len(rows) * (CH + 22), 5 * CW, 3), np.uint8)
for r_i, r in enumerate(rows):
    cap = cv2.VideoCapture(str(OUT / "full_visibility" / f"{r['clip']}.mov"))
    fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    cap.release()
    y = r_i * (CH + 22)
    cv2.putText(sheet, f"{r['clip']}  {r['match']} f{r['start_frame']}  "
                f"{r['fps']}fps  min_onscreen={r['min_onscreen']}",
                (6, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 220, 255), 1)
    for c, p in enumerate(picks):
        idx = int(p * (len(fr) - 1))
        sheet[y + 22:y + 22 + CH, c * CW:(c + 1) * CW] = cv2.resize(fr[idx], (CW, CH))
        cv2.putText(sheet, f"f{idx}", (c * CW + 6, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
Path("_review").mkdir(exist_ok=True)
cv2.imwrite("_review/corners_v2.png", sheet)
print("wrote _review/corners_v2.png")
PYEOF
fi

step "done"
echo "======== corners_v2 finished $(date '+%H:%M:%S') ========"
