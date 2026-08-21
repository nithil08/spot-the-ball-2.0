# GEN2 — where this stopped, and how to pick it up

Paused 2026-08-21 (fourth pause). All background engine jobs were stopped deliberately.
Every stage checkpoints, so resuming is re-running the commands below — completed work is
skipped.

## Where to look

```
~/Desktop/Nithil Research/spot-the-ball-2.0/GEN2/
```

**Not** under `code/`. The repo moved out of `code/` to the workspace root on 2026-08-21.
Desktop is also iCloud-synced, so a stale second copy exists under
`~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/` — don't work in that one.

| What | Where |
|---|---|
| The 60 finished clips | `01_headers/`, `02_corner_kicks/`, `03_goalkeeper_throws/`, each with `full_visibility/` and `split_1s_4s/` |
| Visual review, no video player needed | `_review/situations_{header,corner,gk_throw}.png` — frames 0 / 25 / 49 of every clip |
| Ground truth | `<class>/ground_truth.csv` and `SITUATIONS_GROUND_TRUTH.csv` |
| Spec check | `python3 verify_batch.py` |

Open a class at once: `open 01_headers/full_visibility/*.mov`

Clips are **not** in git — the repo has a blanket `*.mov` ignore rule that predates GEN2.
Ground truth, review sheets and all code are tracked.

## State at the pause

| Stage | State |
|---|---|
| Kit bundles (`gen2`, `gen2_ball_invisible`) | **done** — blue / red / yellow |
| Match sweep | 602 / 720 cached (no longer gates anything) |
| `01_headers` | **DONE — 20 clips** (10 situations x 2 variants) |
| `02_corner_kicks` | **DONE — 20 clips** |
| `03_goalkeeper_throws` | **DONE — 20 clips** |
| `04_player_delta` | **rebuilt, mid-probe** — 8 / 24 probe maps, 0 clips |

60 of the intended 100 clips exist.

## Player-delta: rejected, and what replaced it

All 20 of the original clips were rejected, for four reasons — the last of which kills the
mechanism rather than needing a tweak:

1. the ball sat stranded in empty grass;
2. the frames were too empty to read as a match;
3. windows opened on kickoff clumps, with no real action;
4. **hiding players is the wrong approach entirely.**

The count is now obtained by SELECTION, not subtraction: all 22 players render in every
frame, and windows are searched for ones the camera naturally frames at 6 / 10 / 12 / 16.
That also fixes (1) and (2) — the emptiness *was* the hiding. A window must additionally
have >= 3 players near the ball at BOTH ends, real ball travel, the on-screen players
grouped around the play, no set piece or restart inside, and must not start in the first
150 frames (the kickoff).

`gen_player_delta.py` was rewritten for this. The old hiding code, and the defects that
came with it (source reuse, one-sided teams), are gone with it — `verify_batch.py` still
checks source diversity, and its team-balance check now reads a set where nobody is
hidden, so it should pass trivially.

**Two cheaper ways to count players were tried and both failed. Do not retry them:**

* Fitting the camera frustum geometrically, so the 600 cached sweep matches could be
  scanned for free: 93.6% correct per player, but the per-frame COUNT was exactly right
  only **36%** of the time (sd 1.48). Geometry cannot know one player is behind another.
* Counting kit-coloured blobs in the rendered frame: bias **+5.8 players**, some frames
  off by 40+. The advertising hoardings are saturated blue and red, and one player's shirt
  and shorts split into separate blobs.

The count *is* the ground truth here, so the exact solo probe is used.

## THE PROBE DIES AFTER ~46 ENGINE INSTANCES — run it one match per process

Seen three times; the third pinned it down. Four independent shards each completed
**exactly 2 matches** and then died, no traceback, no error line. Each match creates 23
environments (a plate plus 22 solo passes), so every process died at roughly its 46th
`FootballEnv`. That is a resource leak in the engine — an env that is closed but does not
return its GL context or file descriptors — not a bug in this code, and a short test run
will never show it.

The fix is to let the OS reclaim everything between matches. `probe_one` probes a single
unprobed match and exits 0 if it did work, 1 when the shard is empty, so a shell `while`
loop terminates on its own:

```bash
for i in 0 1 2 3; do
  ( while python3 gen_player_delta.py probe_one --shard $i/4 > _cache/pdprobe_$i.log 2>&1; do :; done ) &
done
```

## Resume

```bash
cd GEN2

# 1. finish the player-delta probe (16 of 24 matches left, ~12 min over 4 shards)
for i in 0 1 2 3; do
  ( while python3 gen_player_delta.py probe_one --shard $i/4 >> _cache/pdprobe_$i.log 2>&1; do :; done ) &
done
# wait for all four loops to exit, then confirm:
ls _cache/player_delta/counts | wc -l          # want 24

# 2. pick windows and render the 40 clips
python3 gen_player_delta.py pick               # want 5 each of 6/10/12/16, 20 matches
python3 gen_player_delta.py vis
python3 gen_player_delta.py inv
python3 gen_player_delta.py compose

# 3. check the whole batch
python3 verify_batch.py
```

`verify_batch.py` currently expects 20 situations per class; it needs
`WANT_PER_CLASS = 10` for the three situation classes and 20 for player-delta, or it will
report the 10-per-class sets as short. Fix that before trusting its summary.

If `pick` cannot fill a count from 24 matches, add a fourth seed to `SEEDS` in
`gen_player_delta.py` and probe again rather than relaxing the football-quality tests —
those tests are what the rejection was about.

## Other things worth knowing

**Do not "fix" the corner window offsets.** Corner clips open at `award + 1`, not at the
award. Opening earlier drags in the referee's re-spot of the ball onto the corner arc — a
teleport small enough to slip under the continuity threshold, leaving the red start-circle
marking a spot the ball instantly leaves. Seen and fixed on `flankR_s001` (award f816,
delivery f819, bad window opened at f806).

**Header contest count is scored, not filtered.** GEN1 required 2+ bodies under the ball;
on GEN2's match shapes that rejects essentially every header (over 31 matches, 21 clear
the physical tests but only 1 has more than one player under it).

**Continuity is a property of the window, not the probe.** An early version rejected a
whole 450-frame probe if a goal happened anywhere in it, which threw away perfectly good
windows either side. It is now checked per 50-frame window.

## Not done, and not started

- The GEN1 base batch `30_1s_visible_4s_invisible/` still uses the **old cyan / dark-red
  kits**. It was not regenerated — that was not asked for, and it would invalidate the
  eval runs already recorded against it. If the whole benchmark should share the new
  blue/red/yellow palette, that is a separate regeneration.
