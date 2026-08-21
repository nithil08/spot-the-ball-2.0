# framerate_test — pick a final frame rate

3 normal **11v11** natural clips (the OG style, NOT the small-sided drills), each rendered
at **3 frame rates** so we can settle on one rate for the benchmark.

- 11_vs_11_stochastic, goal-free 5 s window, moderate ball movement, ball visible
  throughout, **16×6 yellow grid (no player names)**, noname bundle, tracking camera.
- Each clip is rendered ONCE at the 20 fps master (`physics_steps_per_frame = 100/fps = 5`),
  then 10 fps and 5 fps are made by dropping frames. So the play is **identical** across the
  three rates — the only difference is motion smoothness. All three are 5.0 s long.

| file | fps | frames | how |
|------|-----|--------|-----|
| `clipN_20fps.mov` | 20 | 100 | master (real engine temporal resolution) |
| `clipN_10fps.mov` | 10 | 50 | every 2nd master frame (= current default rate) |
| `clipN_05fps.mov` | 5  | 25 | every 4th master frame |

`clips_index.csv` lists the seed, ball path length, and frame counts per clip.

## How to judge
Watch the three rates of the same clip back to back:
- **5 fps** — choppiest, smallest files; ball can "jump" between frames.
- **10 fps** — current benchmark rate.
- **20 fps** — smoothest; larger files / more frames to send a VLM.

Pick the lowest rate where the ball still moves smoothly enough to track. Then set that as
the benchmark's `fps` (and the small-sided drills' `FPS`) going forward.

## Regenerate
```
cd DRILLS/framerate_test
python3 gen_framerate_clips.py
```
Edit `RATES` / `MASTER_FPS` at the top to try a different bracket (e.g. 10/20/30 — set
`MASTER_FPS=30`, but note 30 fps = psf 3 → ~33 fps, since psf must be an integer).
