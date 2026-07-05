# data/

Organized clip library for the Spot-the-Ball project.

```
data/
  videos/          — all video batches, organized by batch → scenario
  code/            — generation script for each batch (named <batch>_code.py)
```

## videos/

Each batch folder contains the clips for that experiment. Every batch (and every scenario within a batch) has a `visibility_variants/` subfolder with a README explaining which clip corresponds to which visibility condition.

| Batch | Clips | Ball | Duration | Notes |
|-------|-------|------|----------|-------|
| `first_batch/` | 3 per scenario × 6 scenarios | visible + hidden | ~5s | Original deliverable clips |
| `ball_hidden_4s/` | 2 still-frames | hidden | freeze at 4s | Early probing; superseded |
| `ball_hidden_grid/` | 3 clips | hidden | 4s | Grid overlay; early probing |
| `5s_ball_visible_no_names/` | 4 clips | visible | 5s | Base clips for 5s splits |
| `5s_visibility_splits/` | 5 clips | split | 5s | 0–4s visible then hidden |
| `10s_ball_visible_no_names/` | 30 clips | visible | 10s | Main benchmark set (6 scenarios × 5 seeds) |
| `10s_visibility_splits/` | 90 clips (30×3) | split | 10s | 3s / 5s / 8s visible then hidden |

> **Note:** Actual video files (`.mov`, `.mp4`) are gitignored due to size.
> They live locally under `experiments/nithil_work/results/` and the top-level `data/` folder.
> Only folder structure, `visibility_variants/` READMEs, and code files are tracked in git.

## code/

Each file here is a copy of the generation script that produced the corresponding batch.

| File | Generates | Original script |
|------|-----------|-----------------|
| `first_batch_code.py` | `videos/first_batch/` | `experiments/nithil_work/gen_updated_deliverable.py` |
| `ball_hidden_4s_code.py` | `videos/ball_hidden_4s/` | `experiments/nithil_work/gen_ball_hidden_4s.py` |
| `ball_hidden_grid_code.py` | `videos/ball_hidden_grid/` | `experiments/nithil_work/gen_ball_hidden_grid_video.py` |
| `5s_ball_visible_no_names_code.py` | `videos/5s_ball_visible_no_names/` | `experiments/nithil_work/gen_5s_noname_clips.py` |
| `5s_visibility_splits_code.py` | `videos/5s_visibility_splits/` | `experiments/nithil_work/gen_visibility_variants.py` |
| `10s_ball_visible_no_names_code.py` | `videos/10s_ball_visible_no_names/` | `experiments/nithil_work/gen_10s_noname_clips.py` |
| `10s_visibility_splits_code.py` | `videos/10s_visibility_splits/` | `experiments/nithil_work/gen_10s_visibility_splits.py` |
