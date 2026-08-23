#!/bin/bash
# run_tonight.sh — finish the GEN2 batch unattended.
#
# Scheduled via launchd for this evening (see com.nithil.gen2finish.plist). Everything it
# does is resumable, so a partial previous run just gets continued rather than redone.
#
# Steps: probe the remaining player-delta matches -> pick windows -> render vis + inv ->
# compose 40 clips -> verify the whole 100-clip batch -> build review sheets -> commit and
# push. Ground truth and review sheets are tracked; the .mov files are gitignored and stay
# on this machine.
#
# The probe is run ONE MATCH PER PROCESS on purpose. The engine leaks resources and a
# process dies at roughly its 46th FootballEnv, which is two matches' worth of passes —
# observed as four independent shards each completing exactly 2 matches and dying with no
# traceback. `probe_one` exits 1 when its shard is empty, so each while-loop ends by itself.

set -u
cd "/Users/nithilbalamurugan/Desktop/Nithil Research/spot-the-ball-2.0/GEN2" || exit 1

# Absolute interpreter. launchd does NOT inherit the interactive shell's PATH, and the
# bare name resolves differently there — /usr/bin/python3 is the system one and has no
# numpy at all. Pin the framework build that actually has numpy and gfootball.
PY=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3

# ── thermal throttle ────────────────────────────────────────────────────────────
# This machine has 4 performance cores. The probe used to launch 4 shards at default
# QoS, which pins every P-core at max clock and overheats the laptop. Two knobs:
#
#   SHARDS   how many probe processes run at once (default 2, was 4)
#   COOL     1 = run the engine at background QoS via taskpolicy -b, which parks it
#            on the 6 efficiency cores. Much slower per match, far less heat.
#            0 = full speed (use when on mains power with good airflow).
#
#   SHARDS=1 COOL=1 ./run_tonight.sh     # coolest
#   SHARDS=4 COOL=0 ./run_tonight.sh     # the old, hot behaviour
SHARDS=${SHARDS:-2}
COOL=${COOL:-1}
if [ "$COOL" = "1" ]; then RUN="/usr/sbin/taskpolicy -b"; else RUN=""; fi

LOG="_cache/run_tonight.log"
mkdir -p _cache
exec >> "$LOG" 2>&1
echo ""
echo "================ run_tonight.sh starting $(date '+%Y-%m-%d %H:%M:%S %Z') ================"

step() { echo ""; echo "---- $* ---- $(date '+%H:%M:%S')"; }

# ── 0. preflight: fail loudly NOW rather than after an hour of silence ───────────
step "preflight"
"$PY" - <<'PRE' || { echo "PREFLIGHT FAILED — aborting"; exit 1; }
import sys
sys.path.insert(0, '.')
import numpy                                   # noqa: F401
import gen2_lib as G
from lib import use_bundle
use_bundle(G.BUNDLE_VIS)
from scenario_factory import write_scenario
lvl = write_scenario(G.match_spec('g2_preflight'), force=True)
frames, _ = G.render_window(lvl, 11, 0, 3)     # can the engine actually render here?
assert len(frames) == 3 and frames[0].shape == (480, 1280, 3), frames[0].shape
print("preflight OK: engine renders", frames[0].shape)
PRE

# ── 1. probe the remaining player-delta matches, one match per process ───────────
step "probe (one match per process; $SHARDS shards, COOL=$COOL)"
# Exit 3 means "shard empty" and is the ONLY reason to stop. Any other non-zero means
# the process died mid-match (the ~46-env engine leak) and must simply be retried with a
# fresh one — an earlier version treated a crash as "empty" and quietly finished the probe
# with 13 of 39 maps. `fails` caps the retries so a genuinely broken shard cannot spin.
for i in $(seq 0 $((SHARDS - 1))); do
  (
    fails=0
    while true; do
      $RUN "$PY" gen_player_delta.py probe_one --shard "$i/$SHARDS" >> "_cache/pdprobe_$i.log" 2>&1
      rc=$?
      if [ "$rc" -eq 3 ]; then
        echo "shard $i: complete" >> "_cache/pdprobe_$i.log"
        break
      elif [ "$rc" -ne 0 ]; then
        fails=$((fails + 1))
        echo "shard $i: probe process died (rc=$rc), retry $fails" >> "_cache/pdprobe_$i.log"
        if [ "$fails" -ge 25 ]; then
          echo "shard $i: giving up after $fails crashes" >> "_cache/pdprobe_$i.log"
          break
        fi
      else
        fails=0
      fi
    done
  ) &
done
wait
echo "probe finished: $(ls _cache/player_delta/counts 2>/dev/null | wc -l | tr -d ' ') maps"

# ── 2. pick windows, then render and compose ─────────────────────────────────────
step "pick"
"$PY" gen_player_delta.py pick || { echo "PICK FAILED"; exit 1; }

step "render visible"
$RUN "$PY" gen_player_delta.py vis || { echo "VIS FAILED"; exit 1; }

step "render invisible"
$RUN "$PY" gen_player_delta.py inv || { echo "INV FAILED"; exit 1; }

step "compose"
"$PY" gen_player_delta.py compose || { echo "COMPOSE FAILED"; exit 1; }

# ── 3. verify the whole batch ────────────────────────────────────────────────────
step "verify"
"$PY" verify_batch.py
VERIFY_RC=$?
echo "verify exit code: $VERIFY_RC"

# ── 4. review sheets, so the result can be judged without opening 40 videos ──────
step "review sheets"
mkdir -p _review
"$PY" - <<'PYEOF'
import sys, json
sys.path.insert(0, '.')
import numpy as np
from PIL import Image, ImageDraw
from gen2_lib import compose_pair, CACHE

PD = CACHE / "player_delta"
picks = json.load(open(PD / "picks_natural.json"))
groups = {}
for i, p in enumerate(picks):
    groups.setdefault(p["end_count"], []).append(i)

CW, CH = 512, 192
for tag, sel in (("FINAL", -1), ("FIRST", 0)):
    rows = sorted(groups, reverse=True)
    sheet = Image.new("RGB", (5 * CW, len(rows) * (CH + 22)), (18, 18, 18))
    dr = ImageDraw.Draw(sheet)
    counters = {}
    for r, cnt in enumerate(rows):
        y = r * (CH + 22)
        dr.text((6, y + 5), f"END {cnt} PLAYERS", fill=(255, 220, 60))
        for c, idx in enumerate(groups[cnt][:5]):
            p = picks[idx]
            vis = np.load(PD / f"nd{idx:02d}_vis.npz")["frames"]
            inv = np.load(PD / f"nd{idx:02d}_inv.npz")["frames"]
            full, split, st, fi = compose_pair(vis, inv)
            sheet.paste(Image.fromarray(full[sel]).resize((CW, CH)), (c * CW, y + 22))
            n = counters.get(cnt, 0) + 1
            counters[cnt] = n
            lbl = f"end{cnt:02d}_{n:02d}  {p['near_last']} near ball"
            dr.rectangle([c * CW + 2, y + 24, c * CW + 250, y + 40], fill=(0, 0, 0))
            dr.text((c * CW + 6, y + 26), lbl, fill=(120, 255, 120))
    out = f"_review/player_delta_{tag}_frames.png"
    sheet.save(out)
    print("wrote", out)
PYEOF

# ── 5. commit and push ───────────────────────────────────────────────────────────
step "commit and push"
cd ..
git add -A
if git diff --cached --quiet; then
  echo "nothing to commit"
else
  git commit -q -m "GEN2: player-delta rebuilt without hiding — 40 clips

Count comes from selecting windows the camera naturally frames at 6/10/12/16,
with all 22 players rendered and nobody hidden. Generated unattended by
GEN2/run_tonight.sh; verify_batch.py exit code was ${VERIFY_RC}.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
  git push -q origin main && echo "pushed" || echo "PUSH FAILED"
fi

# ── 6. disarm: this is a one-shot, it must not fire again tomorrow ───────────────
step "disarm the launchd job"
PLIST="$HOME/Library/LaunchAgents/com.nithil.gen2finish.plist"
if [ -f "$PLIST" ]; then
  launchctl bootout "gui/$(id -u)/com.nithil.gen2finish" 2>/dev/null \
    || launchctl unload "$PLIST" 2>/dev/null
  mv "$PLIST" "$PLIST.ran-$(date +%Y%m%d-%H%M%S)"
  echo "unloaded and renamed $PLIST"
else
  echo "no plist to disarm (already removed, or started by hand)"
fi

step "done"
echo "clips on disk: $(find GEN2 -name '*.mov' | wc -l | tr -d ' ')"
echo "================ finished $(date '+%Y-%m-%d %H:%M:%S %Z') ================"
