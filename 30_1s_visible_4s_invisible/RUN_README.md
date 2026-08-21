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
