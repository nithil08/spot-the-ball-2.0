#!/bin/bash
# run_v2_finish.sh — take headers and goalkeeper throws to the corners-v2 bar.
#
# gate -> build for each class, then one contact sheet per class so the result can be
# judged without opening 20 videos. Both stages checkpoint (occupancy_<kind>.json), so
# re-running continues rather than redoing.
#
# Runs the classes SEQUENTIALLY on purpose. One 175-frame render pair is ~640 MB of
# frames held at once and this machine has 16 GB; two classes in parallel is how you
# turn a render job into a swap storm.
#
# NOT scheduled through launchd. The 23:00 launchd attempt on 2026-08-24 died instantly
# with "Operation not permitted" — launchd's bash has no TCC grant for ~/Desktop, so it
# cannot even read this script. Started from a normal shell it inherits the user's
# grants and works fine.
set -u
cd "/Users/nithilbalamurugan/Desktop/Nithil Research/spot-the-ball-2.0/GEN2" || exit 1

if [ -z "${V2_CAFFEINATED:-}" ]; then
  export V2_CAFFEINATED=1
  exec /usr/bin/caffeinate -ims /bin/bash "$0" "$@"
fi

PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
LOG=_cache/situations_v2
mkdir -p "$LOG"
exec >> "$LOG/run.log" 2>&1

# gfootball's gym shim and the macOS GL driver both natter on every process start.
quiet() { grep -vE "Gym has been unmaintained|Please upgrade to Gymnasium|migration guide|UNSUPPORTED \(log once\)"; }

echo ""
echo "======== situations v2 starting $(date '+%Y-%m-%d %H:%M:%S') ========"

# ── corners: re-gated because the gate now also requires the ball in shot ───────
# The first build died in detect_ball on an all-zero difference: sliding chases
# players, and on that window the camera settled on a crowd before the ball arrived.
# Occupancy is cached per window, so this is a fresh measurement, not a re-sweep.
echo ""
echo "---- corner: gate ---- $(date '+%H:%M:%S')"
$PY gen_corners_v2.py gate 2>&1 | quiet
echo "---- corner: build ---- $(date '+%H:%M:%S')"
$PY gen_corners_v2.py build 2>&1 | quiet
echo "corner build exit: $?"

for kind in header gk_throw; do
  echo ""
  echo "---- $kind: pick ---- $(date '+%H:%M:%S')"
  $PY gen_situations_v2.py pick "$kind" 2>&1 | quiet

  echo "---- $kind: gate ---- $(date '+%H:%M:%S')"
  # The gate stops as soon as 10 windows pass, so a generous budget costs nothing
  # when candidates are good and buys depth when they are not.
  $PY gen_situations_v2.py gate "$kind" 44 2>&1 | quiet

  echo "---- $kind: build ---- $(date '+%H:%M:%S')"
  $PY gen_situations_v2.py build "$kind" 2>&1 | quiet
  echo "$kind build exit: $?"
done

# ── contact sheets ─────────────────────────────────────────────────────────────
echo ""
echo "---- review sheets ---- $(date '+%H:%M:%S')"
$PY - <<'PYEOF' 2>&1 | quiet
import csv
from pathlib import Path
import cv2, numpy as np

CW, CH = 420, 158
PICKS = [0, 0.25, 0.5, 0.75, 0.99]          # spread, so a dead patch cannot hide
for out, sheet_name in (("01_headers_v2", "headers_v2"),
                        ("03_goalkeeper_throws_v2", "gk_throws_v2"),
                        ("02_corner_kicks_v2", "corners_v2")):
    gt = Path(out) / "ground_truth.csv"
    if not gt.exists():
        print(f"skip {out}: not built")
        continue
    rows = list(csv.DictReader(open(gt)))
    sheet = np.zeros((len(rows) * (CH + 22), len(PICKS) * CW, 3), np.uint8)
    for r_i, r in enumerate(rows):
        cap = cv2.VideoCapture(str(Path(out) / "full_visibility" / f"{r['clip']}.mov"))
        fr = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            fr.append(f)
        cap.release()
        y = r_i * (CH + 22)
        cv2.putText(sheet, f"{r['clip']}  {r['match']} f{r['start_frame']}  "
                    f"{r['fps']}fps  min_onscreen={r['min_onscreen']}  "
                    f"slid+{r['slid_frames']}",
                    (6, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 220, 255), 1)
        for c, p in enumerate(PICKS):
            idx = int(p * (len(fr) - 1))
            sheet[y + 22:y + 22 + CH, c * CW:(c + 1) * CW] = cv2.resize(fr[idx], (CW, CH))
            cv2.putText(sheet, f"f{idx}", (c * CW + 6, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    Path("_review").mkdir(exist_ok=True)
    cv2.imwrite(f"_review/{sheet_name}.png", sheet)
    print(f"wrote _review/{sheet_name}.png ({len(rows)} clips)")
PYEOF

echo ""
echo "======== situations v2 finished $(date '+%H:%M:%S') ========"
