# 30_1s_visible_4s_invisible — clips, ground truth, and how to run the eval

## Clips (30)
`clipNN_1s_visible_then_4s_invisible.mov` — 5 s, 16×6 grid (rows A–F, cols 1–16),
1280×480, 10 fps. Ball visible 1 s (10 frames) then invisible 4 s (40 frames).
**Red circle marks the ball's start on the VERY FIRST FRAME ONLY.**

## Ground truth (final frame)
- `ground_truth.csv` / `.json` / `.md` — the cell the ball is in on the FINAL frame,
  from a visible-vs-invisible pixel diff of the cached renders. Columns:
  `clip, ball_cell, grid_row, grid_col, px, py, changed_pixels`.
- `_gt_debug/clipNN.png` — final frame with a crosshair on the detected ball (sanity check).
- Regenerate: `python3 ../../gen_final_gt.py`  (reads `_frames/*.npz`).

## People in frame (per-frame headcount)
How many players are on screen in each of the 1,500 frames. Median **16** of the 22 on
the pitch, middle half 15–18, full range 7–20; the headcount is flat across the clip
(per-second means 16.2 / 16.4 / 16.1 / 16.3 / 16.3), so the ball going invisible at 1 s
does not change how crowded the frame is.

- `people_in_frame.png` — histogram + time course + per-clip heatmap.
- `people_per_frame.csv` — `clip, frame, t_sec, people, team_blue, team_red, goalkeepers`.
- `people_per_second.csv` — per clip, per second: mean / min / max.
- `people_in_frame_summary.json` — the aggregates behind the figure.
- `people_in_frame_check.png` — four frames with every counted player boxed (sanity check).
- `_people_counts/vis_seed<seed>.npz` — raw per (clip, frame, slot) body/shadow pixel
  counts and bounding boxes.

Method — the ball ground-truth diff trick, applied to players. Each seed's window is
replayed once per player slot with that one player rendered at 2% scale
(`GFOOTBALL_HIDE_SLOTS`) while he stays fully in play; diffing against the normal render
isolates exactly his pixels, so "is player N on screen" is a pixel fact rather than a
colour heuristic (colour segmentation fails on the night-lit clips and counts the crowd
in the clips that show the stands). A hidden player also loses his shadow, so only
pixels whose **hue** moves count as body; a player counts as in frame at ≥25 body pixels
(the mean moves only 16.6 → 15.8 across thresholds 1 → 100, so the cut is not load-bearing).
Referees are engine-invisible, so 22 is the ceiling.

```
python3 ../experiments/nithil_work/count_people_render.py 42 --verify   # frames reproduce bit for bit
for s in 42 7 11 23 3; do python3 ../experiments/nithil_work/count_people_render.py $s; done
python3 ../experiments/nithil_work/count_people_report.py               # CSVs + summary + figure
python3 ../experiments/nithil_work/count_people_check.py                # sanity image
```
Needs the patched engine tree (`GFOOTBALL_SRC`, default `~/gfootball_src`); ~3 min per seed.

## Run the model eval
Runner: `../../run_spot_ball_adc.py`. Sends each clip as inline video + the exact prompt,
parses `Ball position:`, scores against `ground_truth.csv`. Results land in `eval_runs/`.

Auth — Vertex AI via ADC (preferred):
```
bash <(curl -sSL https://storage.googleapis.com/cloud-samples-data/adc/setup_adc.sh)
export GOOGLE_CLOUD_PROJECT=<your-project-id>      # optional: GOOGLE_CLOUD_LOCATION
python3 ../../run_spot_ball_adc.py                  # all 30 clips
python3 ../../run_spot_ball_adc.py --n 3            # smoke test
```
Falls back to `GEMINI_API_KEY` (AI Studio) if `GOOGLE_CLOUD_PROJECT` is unset.

## Exact prompt
> You are watching a short clip from a soccer match. The ball has been digitally removed
> from every frame. Infer where the ball is located in the final frame of the clip.
> Respond in exactly this format:
> Reasoning: <one or two sentences>
> Ball position: <grid row #, grid colum#>
