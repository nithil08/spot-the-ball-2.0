# GOALS — clips that contain a goal

None of the shipped batches has one. GEN3, GEN3_HARD, GEN2, DRILLS and MATCH_SITUATIONS
all run the same `continuous()` gate over a candidate window and reject it if the ball
crosses a goal line between the posts (`|x| > 1.0` with `|y| < 0.05`) or teleports by
>= 0.35 in a frame. A goal does both — the engine respots the ball on the centre circle
the frame after the score ticks — so a goal can never survive the one-unbroken-passage-of-
play rule the spot-the-ball task needs. Scene graphs confirm it: score is constant inside
all 24 GEN3 and all 24 GEN3_HARD clips (nonzero in a few, because the goal happened
before the window opened).

The goals still exist in the sweep the batches were picked from. `GEN3/_cache/sweep/`
holds 700 cached 110-second matches, and their per-frame `score` arrays contain
**621 goals across 409 matches** (`goals.csv`). Matches replay bit-exactly, so any of
them can be re-cut without a new sweep.

## Clips

| clip | match | goal at | scorer | what happens |
|------|-------|---------|--------|--------------|
| `clips/goal_mid_push_s319_f146.mov` | mid_push, seed 319 | f146 (5.8 s) | A (blue) 1–0 | **pick** — blue carries out of midfield and strikes from x = 0.57, ~23 m out; the shot beats the keeper high, crossing at z = 2.0 m. Longest-range goal in the sweep, and both banks of players are in shot |
| `clips/goal_countR_s317_f362.mov` | countR, seed 317 | f362 (14.5 s) | B (red) 0–1 | red breaks on the counter and finishes high across the keeper (z = 2.3 m) from ~17 m; few bodies in shot |
| `clips/goal_wideR_s343_f332.mov` | wideR, seed 343 | f332 (13.3 s) | B (red) 0–1 | red attacks down the left of frame and finishes from ~21 m, crossing at z = 0.9 m |

Each is 123 frames at 25 fps (4.9 s): 115 frames of build-up, the goal, then 8 frames.
The clip ends right after the ball crosses the line on purpose — one frame later the
engine teleports the ball to the centre spot, which looks like a cut. Locked render
format otherwise (`gen3` bundle, noname, HUD cropped); no grid overlay and no
ball-invisible variant, since these are for watching, not for the benchmark.

## Re-cutting another one

```
python3 render_goal.py list                      # all 621, same data as goals.csv
python3 render_goal.py render mid_push 319 146   # shape, seed, score frame
```

Takes ~20 s per clip; nothing is swept, the replay is deterministic.
`_review/` holds tiled PNG strips of the goal moment for eyeballing.
