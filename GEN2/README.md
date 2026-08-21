# GEN2 — the classification batch

Second-generation spot-the-ball stimuli, organised by **what kind of situation the clip
is**, so results can be broken down per class instead of averaged over one undifferentiated
pile of clips.

Everything here is cut from real deterministic 11 v 11 matches with the engine's own
referee running. Nothing is staged: a corner is in this set because the referee awarded
one under Law 17, not because a ball was dropped near a corner flag.

## Format (locked, identical for every clip)

| | |
|---|---|
| Length | **exactly 5.0 s** — 50 frames at 10 fps |
| Frame | 1280x480 (the 1280x720 render with the score strip and radar panel cropped) |
| Grid | 16 columns (1–16) x 6 rows (A–F), translucent yellow |
| Start marker | red circle around the ball on the first 3 frames |
| Kits | **team A blue, team B red, goalkeepers yellow**; officials rendered invisible |

## The two visibility variants

Every situation ships twice, in sibling folders:

- `full_visibility/` — ball visible for all 50 frames
- `split_1s_4s/` — ball visible for 10 frames (1 s), then invisible for 40 frames (4 s)

Both are cut from the **same pair of renders** of the same deterministic state — one with
the ball visible, one with it invisible. They are the same play frame for frame, and their
first 10 frames are pixel-identical; only the ball's visibility differs after that. That is
what makes the pair a controlled comparison rather than two similar clips.

It is also where ground truth comes from: the pixels that differ between the visible and
invisible renders **are** the ball, so the start and final cells are measured, never
hand-labelled.

## Contents — 160 clips

| Folder | Situations | Clips |
|---|---|---|
| `01_headers/` | 20 | 40 |
| `02_corner_kicks/` | 20 | 40 |
| `03_goalkeeper_throws/` | 20 | 40 |
| `04_player_delta/` | 20 | 40 |

`04_player_delta/` is 5 clips each ending with **6, 10, 12 and 16** players in frame.

### What counts as each situation

- **header** — a ball above chest height that *changes direction* at a player. The
  direction change is what separates a header from a ball bouncing past someone.
- **corner kick** — Law 17, awarded by the engine referee because a defender put the ball
  behind. The clip opens just after the award and on the *delivery*, so all 5 s are live
  football rather than 2 s of a ball sitting on the arc while the box is arranged.
- **goalkeeper throw** — Law 12 distribution *by hand*. While the keeper holds the ball it
  parks at ~1.35 m (his hands); the release then plays the overarm throw animation. A
  keeper with the ball at ~0.11 m is dribbling and about to punt it — not a throw.
- **player delta** — see below.

## Player delta: only the END count is controlled

All 22 players are in the simulation for all 50 frames of every clip. The number on screen
is a property of the **camera**, not the squad — a broadcast never shows all 22 at once.

Per the GEN2 brief, the starting count is deliberately *not* constrained; only the count on
the final frame is. Each clip is a single continuous render with a single fixed hide set,
so nobody pops in or out mid-clip: the count changes because players ran and the camera
tracked the ball.

"In frame" is measured, not estimated. For each base match the pipeline renders a plate
with all 22 hidden and then 22 solo passes with exactly one player shown, and diffs them
frame by frame — an exact per-frame, per-player on-screen map. Only the surplus players in
frame on the final frame are hidden; everyone else stays visible, which is what lets the
count fall naturally during the clip.

## Ground truth

- `01_headers/ground_truth.csv`, etc. — per situation
- `SITUATIONS_GROUND_TRUTH.csv` — all three situation classes combined
- `04_player_delta/ground_truth.csv` — includes `players_in_frame_first/last` and the
  hidden-slot count

Columns cover the source match and window, the ball's start cell/pixel and final
cell/pixel. The **final** cell is the answer for the spot-the-ball task.

## Rebuilding

Each engine phase is its own process because the asset bundle must be selected before
`gfootball` is imported.

```bash
python3 ../experiments/apply_gen2_kits.py   # once: build the blue/red/yellow bundles
python3 sweep_events.py 60 1100             # play 12 shapes x 60 seeds, cache the logs
python3 pick_situations.py                  # score the cache -> _cache/picks.json
python3 gen_situations.py all               # vis -> inv -> compose  (120 clips)
python3 gen_player_delta.py all             # probe -> pick -> vis -> inv -> compose (40)
```

`sweep_events.py` is resumable — already-cached matches are skipped, so it can be killed
and restarted, and re-run with more seeds if a situation comes up short.

## Why seeds, not scenario shapes

GEN1 set `deterministic=True`, which makes the engine ignore `game_engine_random_seed`:
every seed replays one identical match, so variety had to come from hand-written scenario
shapes. That capped the yield at 3 corners in 48 matches — nowhere near 20.

Measured here: with `deterministic=False` the same seed still replays **bit-identically**
(max ball-track difference 0.0 over 120 steps) while a different seed gives a genuinely
different match (0.95). Identical replay is the only property the visible/invisible diff
requires, so GEN2 keeps a small set of shapes and sweeps seeds across them.

## Relationship to the rest of the repo

`30_1s_visible_4s_invisible/` is the GEN1 base batch and still uses the old cyan/dark-red
kits. GEN2 is a separate, self-contained set; nothing here overwrites it.
