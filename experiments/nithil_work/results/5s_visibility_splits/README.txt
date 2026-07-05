5s_visibility_splits — CLIP SET README
======================================
Generated: 2026-07-03
Script:    experiments/nithil_work/gen_visibility_variants.py

WHAT IS IN THIS FOLDER
-----------------------
5 video clips each showing the same 5 seconds of gameplay but with the ball
visible for a different amount of time before being hidden.

These clips are designed for the "Spot The Ball" task: the viewer watches the
ball for some seconds, then the ball disappears and they must guess where it is.

CLIP SPECS
----------
  Duration:     5 seconds each (50 frames at 10 FPS)
  Resolution:   1280 × 480 pixels (HUD cropped)
  Grid overlay: 32 × 12 yellow alphanumeric grid on every frame
  Player names: REMOVED (same font trick as 5s_ball_visible_no_names)
  Title bar:    Shows visible/hidden split e.g. "VISIBLE: 2s → HIDDEN: 3s"

SOURCE SCENARIO
---------------
  All 5 clips come from the same scenario rendered twice with the same seed:
    Scenario:  academy_3_vs_1_with_keeper
    Seed:      23

  Render 1 (noname bundle):              ball at full normal size  → "visible" frames
  Render 2 (noname_ball_invisible bundle): ball shrunk to 5% size → "hidden" frames

  Because both renders use identical seed and scenario, the gameplay physics
  are frame-perfectly identical — only the ball size differs.  The two renders
  are then spliced at the right frame boundary to make each variant.

THE 5 CLIPS
-----------
  0s_vis_5s_hid.mov   Ball hidden the ENTIRE 5 seconds  (hardest task)
  1s_vis_4s_hid.mov   Ball visible 1s, then hidden 4s
  2s_vis_3s_hid.mov   Ball visible 2s, then hidden 3s
  3s_vis_2s_hid.mov   Ball visible 3s, then hidden 2s
  4s_vis_1s_hid.mov   Ball visible 4s, then hidden 1s  (easiest task)

WHAT "HIDDEN" MEANS
-------------------
The ball is NOT removed from the simulation.  It still physically moves and
affects gameplay.  It is made invisible by loading an asset bundle where the
ball's 3D mesh vertices have been scaled to 5% of their original size.  At
this size the ball is roughly 3-4 pixels wide and very hard to see.

Ball asset location:
  experiments/asset_bundles/noname_ball_invisible/data/media/objects/balls/generic.ase

HOW NAMES ARE REMOVED
----------------------
Same method as 5s_ball_visible_no_names (see README there).

HOW TO REGENERATE
-----------------
  cd experiments/nithil_work
  python3 gen_visibility_variants.py

Each bundle renders in a fresh subprocess to avoid GFOOTBALL_DATA_DIR being
cached after the first env creation.  The worker script is render_worker_exact.py.
