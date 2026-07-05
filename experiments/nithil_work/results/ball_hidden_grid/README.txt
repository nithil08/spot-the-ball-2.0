ball_hidden_grid — CLIP SET README
===================================
Generated: 2026-06-30 (earlier session)
Script:    experiments/nithil_work/gen_ball_hidden_grid_video.py

WHAT IS IN THIS FOLDER
-----------------------
3 sub-folders (each named with a random ID), each containing a clip where the
ball has been hidden.  These were used for early "locate the hidden ball" probing.

NOTE: Player names are NOT removed in this batch (predates the font fix).
      Use 5s_ball_visible_no_names/ or 10s_ball_visible_no_names/ for clean clips.

CLIP SPECS
----------
  Duration shown to model:  4 seconds (40 steps visible)
  Full render duration:     5 seconds (50 steps)
  Resolution:               1280 × 480 pixels (HUD cropped)
  Ball in clip.mov:         HIDDEN — shrunk to 5% size (ball_tiny bundle)
  Ball in debug_full.mov:   VISIBLE with red marker (NOT for model, debug only)
  Grid overlay:             32 × 12 yellow alphanumeric grid on every frame
  Player names:             STILL VISIBLE (not fixed in this batch)

FILES INSIDE EACH SUB-FOLDER
-----------------------------
  clip.mov          The clip shown to the model (4s, ball hidden)
  debug_full.mov    Full 5s clip with ball visible + red dot marker (for reference)
  meta.json         Machine-readable metadata including:
                      - scenario, seed
                      - ball grid cell at freeze frame (e.g. "H28")
                      - ball grid cell at 1 second after clip ends (ground truth)
                      - the exact questions to ask the model

SCENARIOS
---------
  08fcceebae/   academy_3_vs_1_with_keeper     seed: 23
  9066812d07/   academy_counterattack_easy     seed: 331
  666bb5de48/   academy_counterattack_hard     seed: 3

HOW THE BALL IS HIDDEN
-----------------------
Uses the "ball_tiny" asset bundle: ball mesh vertices scaled to 5% of original size.
The ball still physically exists in the simulation (affects game logic) but appears
as ~3-4 pixels — very hard to see at normal scale.

HOW TO REGENERATE
-----------------
  cd experiments/nithil_work
  python3 gen_ball_hidden_grid_video.py
