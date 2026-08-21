# GEN2 — where this stopped, and how to pick it up

Paused 2026-08-21 (third pause) for review. All background engine jobs were stopped
deliberately; nothing crashed and nothing is half-written. Every stage checkpoints, so
resuming is just re-running the commands below — completed work is skipped.

## Where to look

```
~/Desktop/Nithil Research/spot-the-ball-2.0/GEN2/
```

**Not** under `code/`. The repo moved out of `code/` to the workspace root on 2026-08-21.
Desktop is also iCloud-synced, so a stale second copy exists under
`~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/` — don't work in that one.

| What | Where |
|---|---|
| The 40 finished clips | `04_player_delta/full_visibility/` and `04_player_delta/split_1s_4s/` |
| Quick visual review (no video player needed) | `_review/player_delta_FINAL_frames.png`, `_review/player_delta_FIRST_frames.png` |
| Ground truth | `04_player_delta/ground_truth.csv` |
| Spec check | `python3 verify_batch.py` |

Open all 20 full-visibility clips at once:
`open 04_player_delta/full_visibility/*.mov`

Clips are **not** in git — the repo has a blanket `*.mov` ignore rule that predates GEN2.
Ground truth, review sheets and all code are tracked.

## State at the pause

| Stage | State |
|---|---|
| Kit bundles (`gen2`, `gen2_ball_invisible`) | **done** — blue / red / yellow, verified on screen |
| Match sweep | **602 / 720 matches cached** |
| Situation picks | headers **20/20**, corners **20/20**, gk throws **20/20** — all reached |
| **Player-delta clips** | **DONE — 40 clips, all spec checks pass** |
| Situation clips | **not rendered yet** — 120 clips outstanding |

All three situation classes hit 20/20 at 546 matches, so the sweep no longer gates
anything. The remaining 118 matches would only marginally improve which 20 get picked.

## Resume

Run from `GEN2/`. Two options:

```bash
# A. finish the sweep first (~8 min, 4 shards), then pick and render.
#    Slightly better picks, since the top-20 is drawn from a bigger pool.
for i in 0 1 2 3; do
  nohup python3 sweep_events.py 60 1100 --shard $i/4 > _cache/sweep_$i.log 2>&1 &
  sleep 4
done
# wait for those to exit, then:
python3 pick_situations.py
python3 gen_situations.py all

# B. or just render now off the 602 matches already cached — all three
#    situations are already at 20/20.
python3 pick_situations.py
python3 gen_situations.py all
```

Then check the whole batch:

```bash
python3 verify_batch.py     # expects 160 clips, exits non-zero on any problem
```

## Two defects found and fixed in the player-delta set — don't reintroduce them

**Source diversity.** `phase_pick` originally reserved matches per end-count, so one match
could supply all four counts. With windows offset by only 5 frames, `end16_01` and
`end12_01` shared 45 of their 50 frames — one passage of play shown four ways. Matches are
now consumed globally, one clip each, and the phase warns if forced to reuse one.

**Team balance.** `choose_hide_set` ranked the whole frame by dwell and cut the tail, which
wiped out a whole side: 4 of the 5 end-6 clips came out 0 blue / 6 red. The surplus is now
chosen per team. `verify_batch.py` checks both of these, so a regression will be caught.

## Other things worth knowing

**The probe process died on its own twice**, partway through, no traceback. It checkpoints
`_cache/player_delta/onscreen.json` per match and skips keys already present, so re-running
continues. Check the map count rather than assuming it finished.

**Do not "fix" the corner window offsets.** Corner clips open at `award + 1`, not at the
award. Opening earlier drags in the referee's re-spot of the ball onto the corner arc — a
teleport small enough to slip under the continuity threshold, leaving the red start-circle
marking a spot the ball instantly leaves. Seen and fixed on `flankR_s001` (award f816,
delivery f819, bad window opened at f806).

**Header contest count is scored, not filtered.** GEN1 required 2+ bodies under the ball;
on GEN2's match shapes that rejects essentially every header (over 31 matches, 21 clear
the physical tests but only 1 has more than one player under it).

## Not done, and not started

- The GEN1 base batch `30_1s_visible_4s_invisible/` still uses the **old cyan / dark-red
  kits**. It was not regenerated — that was not asked for, and it would invalidate the
  eval runs already recorded against it. If the whole benchmark should share the new
  blue/red/yellow palette, that is a separate regeneration.
