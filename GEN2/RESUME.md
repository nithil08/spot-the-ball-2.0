# GEN2 — where this stopped, and how to pick it up

Paused 2026-08-21 (second pause). All background engine jobs were stopped deliberately;
nothing crashed and nothing is half-written. Every stage checkpoints, so resuming is just
re-running the commands below — completed work is skipped.

## Where the files are

`~/Desktop/Nithil Research/spot-the-ball-2.0/GEN2/`

**Not** under `code/`. The repo moved out of `code/` to the workspace root on 2026-08-21.
Desktop is also iCloud-synced, so a stale second copy exists under
`~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/` — don't work in that one.

## State at the pause

| Stage | State |
|---|---|
| Kit bundles (`gen2`, `gen2_ball_invisible`) | **done** — blue / red / yellow, verified on screen |
| Path fixes after the repo move | **done** |
| Match sweep | **260 / 720 matches cached** |
| Situation picks (from 260 matches) | headers **20/20**, gk throws **20/20**, corners **14/20** |
| Player-delta probe | 20 matches probed, 17 usable |
| Player-delta clips rendered | **40 clips on disk — but they need re-rendering, see below** |
| Situation clips rendered | **none yet** |

Only corners are still short. They run at ~0.054 per match, so the remaining 460 matches
should yield ~25 more — comfortably past 20.

## The one thing that must be redone

The 40 clips in `04_player_delta/` have **correct end counts but bad diversity**: all 20
windows came from just 5 matches, and the four counts from one match were offset by only
5 frames — `end16_01` and `end12_01` share 45 of their 50 frames. That is one passage of
play shown four ways, not four clips.

`phase_pick` has been **fixed** to consume matches globally (one clip per match), so 20
clips will be 20 different matches. The fix is committed but has **not been re-run**. It
needs ≥20 usable probe maps and there are currently 17, so finish the probe first.

## Resume, in order

Run from `GEN2/`. The sweep is the long pole; shard it across cores.

```bash
# 1. finish the sweep (resumable; skips the 260 already cached)
for i in 0 1 2 3; do
  nohup python3 sweep_events.py 60 1100 --shard $i/4 > _cache/sweep_$i.log 2>&1 &
  sleep 4
done

# 2. finish the player-delta probe (resumable; skips the 20 already mapped)
nohup python3 gen_player_delta.py probe >> _cache/probe.log 2>&1 &

# 3. once the probe is done — re-pick, and CHECK the printed match count is 20/20.
#    If it warns that a match is reused, add a 4th seed to SEEDS in gen_player_delta.py
#    and re-probe rather than accepting duplicates.
python3 gen_player_delta.py pick
rm -rf 04_player_delta                     # discard the low-diversity clips
python3 gen_player_delta.py vis
python3 gen_player_delta.py inv
python3 gen_player_delta.py compose

# 4. once the sweep is done — re-pick; expect 20/20 for all three situations
python3 pick_situations.py

# 5. render the 120 situation clips (vis -> inv -> compose)
python3 gen_situations.py all
```

If corners still fall short of 20 after the full sweep, extend the sweep rather than
loosening the filters — the filters are what make a clip a genuine Law 17 corner or a
genuine by-hand throw. `python3 sweep_events.py 100 1100 --shard i/4` adds 40 more seeds
per shape and reuses everything already cached.

## Things worth knowing before touching this again

**The probe process has died on its own twice**, partway through, with no traceback. It
checkpoints `_cache/player_delta/onscreen.json` after every match and skips keys already
present, so re-running just continues. Check the map count rather than assuming it
finished.

**Do not "fix" the corner window offsets.** Corner clips open at `award + 1`, not at the
award. Opening earlier drags in the referee's re-spot of the ball onto the corner arc,
which is a teleport small enough to slip under the continuity threshold and leaves the red
start-circle marking a spot the ball instantly leaves. Seen and fixed on `flankR_s001`
(award f816, delivery f819, bad window opened at f806).

**Header contest count is scored, not filtered.** GEN1 required 2+ bodies under the ball;
on GEN2's match shapes that rejects essentially every header (over 31 matches, 21 clear
the physical tests but only 1 is contested by more than one player).

## Not done, and not started

- The GEN1 base batch `30_1s_visible_4s_invisible/` still uses the **old cyan / dark-red
  kits**. It was not regenerated — that was not asked for, and it would invalidate the
  eval runs already recorded against it. If the whole benchmark should share the new
  blue/red/yellow palette, that is a separate regeneration.
