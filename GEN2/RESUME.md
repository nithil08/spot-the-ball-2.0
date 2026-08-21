# GEN2 — where this stopped, and how to pick it up

Paused 2026-08-21. All background engine jobs were stopped deliberately; nothing crashed
and nothing is half-written. Every stage checkpoints, so resuming is just re-running the
same commands — completed work is skipped.

## State at the pause

| Stage | State |
|---|---|
| Kit bundles (`gen2`, `gen2_ball_invisible`) | **done** — blue / red / yellow, verified on screen |
| Path fixes after the repo move | **done** |
| Match sweep | **116 / 720 matches cached** |
| Player-delta probe | 19 matches probed, 16 usable |
| Player-delta window picks | **done — 20/20** (5 each ending with 6, 10, 12, 16) |
| Situation picks (from 116 matches) | headers **20/20**, corners 5/20, gk throws 8/20 |
| Clips rendered | **none yet** — this is the only step with no output on disk |

Headers are already fully picked. Corners and goalkeeper throws are the only things
gated on more sweep: they run at roughly 0.043 and 0.069 per match, so the remaining
604 matches should yield ~26 more corners and ~40 more throws — comfortably past 20 each.

## Resume, in order

Run from `GEN2/`. The sweep is the long pole and the only slow step; shard it across
cores (4 shards took the estimate from ~178 min to ~47 min).

```bash
# 1. finish the sweep (resumable; skips the 116 already cached)
for i in 0 1 2 3; do
  nohup python3 sweep_events.py 60 1100 --shard $i/4 > _cache/sweep_$i.log 2>&1 &
  sleep 4
done

# 2. re-pick once the sweep is done — expect 20/20 for all three situations
python3 pick_situations.py

# 3. render the 120 situation clips (vis -> inv -> compose)
python3 gen_situations.py all

# 4. render the 40 player-delta clips.
#    The picks are already made, so the probe/pick phases can be skipped:
python3 gen_player_delta.py vis
python3 gen_player_delta.py inv
python3 gen_player_delta.py compose
```

If corners or throws still fall short of 20 after the full sweep, extend it rather than
loosening the filters — the filters are what make a clip a genuine Law 17 corner or a
genuine by-hand throw. `python3 sweep_events.py 100 1100 --shard i/4` adds 40 more seeds
per shape and reuses everything already cached.

## Two things worth knowing before touching this again

**The probe process died once on its own** partway through (16 of 30 matches, no
traceback). It checkpoints `_cache/player_delta/onscreen.json` after every match and skips
keys already present, so re-running it just continues. It is not on the critical path
anyway — the 16 usable maps were already enough to fill all 20 player-delta windows.

**Do not "fix" the corner window offsets.** Corner clips open at `award + 1`, not at the
award. Opening earlier drags in the referee's re-spot of the ball onto the corner arc,
which is a teleport small enough to slip under the continuity threshold and leaves the red
start-circle marking a spot the ball instantly leaves. This was seen and fixed on
`flankR_s001` (award f816, delivery f819, bad window opened at f806).

## Not done, and not started

- The GEN1 base batch `30_1s_visible_4s_invisible/` still uses the **old cyan / dark-red
  kits**. It was not regenerated — that was not asked for, and it would invalidate the
  eval runs already recorded against it. If the whole benchmark should share the new
  blue/red/yellow palette, that is a separate regeneration.
