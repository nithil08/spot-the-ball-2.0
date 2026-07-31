natural_30_5s_grid — 30 natural 5-second gameplay clips (32x12 grid, ball visible)
=================================================================================
Generated: 2026-07-08
Script:    experiments/nithil_work/gen_natural_30_clips.py

WHAT IS IN THIS FOLDER
----------------------
30 clips of Google Research Football gameplay that follow the user's "game rules":
continuous natural match play (NOT academy drills like the previous 30-clip batch).

  clip_01.mov ... clip_30.mov   the 30 clips
  ground_truth.md / .csv / .json  ball location on the FINAL frame of each clip
  manifest.json                  seed / window / movement for each clip
  _final_frames/                 visible + invisible final frame PNGs (diff inputs)
  _debug/                        final grid frame with a red crosshair on the ball

CLIP SPECS
----------
  Scenario:    11_vs_11_stochastic (a real match, ball-tracking camera)
  Duration:    5.0 s each (50 steps @ 10 fps)
  Resolution:  1280 x 480 (HUD cropped: 60px top, 180px bottom)
  Ball:        VISIBLE, normal size, throughout
  Names:       none (noname bundle)
  Grid:        32 columns (1-32, left->right) x 12 rows (A-L, top->bottom), yellow
  Variants:    NONE — one clip per window, no visibility/hidden-ball variants

HOW THE 30 WINDOWS WERE CHOSEN
------------------------------
5 goal-free seeds (42, 7, 11, 23, 3) of one continuous match each, sliced into
back-to-back non-overlapping 50-frame windows. A window is kept only if it has
NO goal and MODERATE ball movement (path length 0.33-0.95) — excluding static
and frantic end-to-end windows. Up to 6 windows per seed.

GROUND TRUTH METHOD (exact, not a guess)
----------------------------------------
For each clip the final frame is rendered twice from the identical simulation
state: once with the ball visible (noname bundle) and once with the ball made
invisible (noname_ball_invisible bundle). The pixel-difference isolates the ball,
so its pixel position — and therefore its grid cell — is known exactly.

HOW TO REGENERATE
-----------------
  cd experiments/nithil_work
  python3 gen_natural_30_clips.py visible      # 30 clips + manifest + visible frames
  python3 gen_natural_30_clips.py invisible    # invisible final frames
  python3 gen_natural_30_clips.py groundtruth  # diff -> ground_truth.{json,csv,md}
