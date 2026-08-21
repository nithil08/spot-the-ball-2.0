# spread_30_5s_grid_16x6 — 30 clips with the ball spread across the grid

Same recipe as `natural_30_5s_grid_16x6` (natural 11v11 play, 5.0 s @ 10 fps, no
goals, 16x6 yellow grid, `noname` bundle, tracking camera) with one change: each
clip's camera is **framed off the ball by a fixed per-clip offset**, so the answer
is no longer always in the middle of the frame.

## Why

The stock engine camera centres on the ball, so the ball renders mid-frame in
every clip regardless of where play happens. In the original 30-clip batch that
collapsed the answer space:

| | before | after |
|---|---|---|
| cells used (of 96) | 15 | **22** |
| rows used | 3/6 — B, C, D only (20/30 in row C) | **6/6** |
| columns used | 8/16 — 5–12 only | **11/16** |
| busiest cell | 6/30 (20%) | **3/30 (10%)** |
| spread of ball pixel | x sd 141px, y sd 41px | **x sd 219px, y sd 85px** |

The vertical axis was the badly broken one — y sd 41px is half a grid cell, i.e.
every answer in the same row band. It roughly doubles here, and all six rows are
now in play. Numbers are regenerated into `distribution_stats.json`.

This is camera geometry, not scenario design — moving play around the pitch does
not help, because the camera follows it. Drill variety cannot fix it.

## How

`match.cpp / UpdateIngameCamera()` reads `GFOOTBALL_CAM_OFFSET_X` and
`GFOOTBALL_CAM_OFFSET_Y` (world units) and shifts the camera's framing target by
that much. The camera still pans and tracks exactly as before — the shot is just
framed off-centre, the way a real operator frames play. With both unset the
camera is byte-identical to stock, so every existing pipeline is unaffected —
verified by re-rendering seed 42 on the unset path and recovering the original
benchmark's clip_01 ball pixel `(394.9, 242.8)` exactly.

Setting the variables to `0` is **not** the same as leaving them unset: `0` still
applies the branch's pitch bound, which is slightly wider than the stock follow
clamp. The generator sets them in every phase (to 0 during the probe) so the
baseline is measured through the same camera bounds it is later applied to.

Because the offset is a constant added to the target every frame and the camera's
smoothing is linear, the ball's screen position shifts by exactly `K * offset`:

    px: -26.58 px per world unit of offset X   (linear across the usable range)
    py: ~+13.6 px per unit of offset Y         (monotone, mildly nonlinear)

so each clip's target cell is solved directly from its measured natural position.

The offset target is clamped to the **playing surface** (`pitchHalfW * 0.95`,
`pitchHalfH * 0.80`). That bound is what keeps the clips usable: without it the
camera aims past the goal line or over the touchline and fills a third of the
shot with empty seating.

That clamp is also why the pipeline has a `world` phase. The linear calibration
only holds while the framing target is *inside* the clamp; once it bites, the
camera stops panning and the engine's yaw term couples the two axes, so the ball
lands nowhere near the solve (observed: 333 px out on a clip whose ball sat near
a goal line, versus 13–16 px for interior clips). `world` replays each window
**without rendering** — ~50x faster, and the camera does not affect the
simulation — to get the ball's world position, so the solve can cap the offset at
what the clamp will actually honour. Clips near a pitch end therefore get a
smaller offset rather than a wrong one, which costs some spread at the extreme
columns and buys a trustworthy model everywhere.

## Known limits

**Row A is not targeted.** Putting the ball in the top row requires aiming the
camera from beyond the near touchline, which puts hoardings and stands across the
bottom of the frame. The framing stops looking like football before the ball gets
that high, so targets span rows B–F. (One clip does land in A: its ball is in
flight, and the camera barely tracks ball height.)

**Columns 1–3 and 14–16 are thin.** Reaching them needs a large horizontal offset,
which is exactly what the pitch clamp refuses when play is already near a goal
line. Those clips get a smaller offset rather than a wrong one. Widening the
`framedW` bound would buy more columns at the cost of framing stands behind the
goal — that is the dial to turn if the extremes matter more than the look.

**17/30 clips land in their exact target cell**; the rest land a cell or two off,
mostly where the clamp limited the offset. This does not affect correctness —
ground truth is measured, not predicted — it just means the achieved distribution
is a softened version of the requested one, which is closer to what you want than
hitting every cell exactly.

## Files

    clip_NN.mov              the 30 clips (grid burned in, ball visible throughout)
    ground_truth.csv/.json   final-frame ball cell + exact px/py, and the offset used
    plan.json                per-clip natural position, target cell, solved offset
    world.json               ball world position at each final frame (+ obs z)
    framing_audit.json       pitch-fill fraction per clip (see below)
    distribution_stats.json  coverage stats, before vs after
    ball_heatmap_comparison.png   final-frame distribution, before vs after
    _debug/clip_NN_gt.png    final frame with a crosshair on the detected ball
    _final/, _probe/         cached render frames (regenerate-able, safe to delete)

## Ground truth

Unchanged method: the exact ball pixel is the diff between the visible and
invisible-ball renders of the same deterministic state, measured on the **actual
reframed render** — never predicted from the calibration. A calibration miss just
means the ball lands a cell over from its target; the recorded cell is still
exact. `ground_truth.csv` carries both `ball_cell` (truth) and `target_cell`
(what was aimed for) so the two can be compared.

The detector reports failure rather than a guess when the diff carries too little
signal. The old fallback branch returned the centroid of pure noise as if it were
a ball, which silently produced fake ground truth for any clip whose ball was
framed out of shot.

## Regenerating

    python3 ../code/gen_30_spread_code.py all

Phases (`probe`, `probe_inv`, `world`, `plan`, `vis`, `inv`, `gt`, `audit`) can be
run individually; each engine phase is its own process because the asset bundle
must be chosen before `gfootball` is imported. `audit` flags any clip whose
framing drifted off the pitch — it caught both bad frames in the first batch and
is cheap, so run it every time.

Only `plan` onwards needs re-running to reshape the distribution; `probe`,
`probe_inv` and `world` depend on the seeds and window selection alone.

`world` also reports which clips have an **airborne** ball at the final frame.
The camera barely tracks ball height (`ballPos.coords[2] *= 0.1f`), so a ball in
flight renders far higher in frame than its ground position implies and can be
framed out of shot entirely. Those clips are the likeliest to lose their ball,
and `gt` reports them as lost rather than inventing a position.
