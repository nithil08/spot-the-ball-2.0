10s_visibility_splits — CLIP SET README
=========================================
Generated: 2026-07-03
Script:    experiments/nithil_work/gen_10s_visibility_splits.py

WHAT IS IN THIS FOLDER
-----------------------
90 video clips: 3 visibility-split variants for each of the 30 × 10s base clips.
Each sub-folder (clip_01/ through clip_30/) holds the 3 variants for that clip.

PURPOSE
-------
Test whether giving a model more time watching the ball before it disappears
helps it predict where the ball went.  The 3 variants per clip form a
difficulty gradient from hard (3s visible) to easy (8s visible).

VARIANTS PER CLIP
-----------------
  3s_vis_7s_hid.mov   Ball visible 3s, then hidden 7s  (hard — short look)
  5s_vis_5s_hid.mov   Ball visible 5s, then hidden 5s  (medium)
  8s_vis_2s_hid.mov   Ball visible 8s, then hidden 2s  (easy — long look)

CLIP SPECS
----------
  Total duration:  10 seconds per clip
  Resolution:      1280 × 480 px (HUD cropped)
  Grid:            32 × 12 yellow alphanumeric overlay, every frame
  Player names:    REMOVED (blank font via GFOOTBALL_FONT env var)
  Title bar:       Shows clip number, split, scenario, seed

WHAT 'HIDDEN' MEANS
--------------------
The ball is NOT removed — it still affects gameplay physics.
It is made nearly invisible by loading an asset bundle where the ball's
3D mesh is scaled to 5% of its original size (~3-4 pixels wide).
Bundle: experiments/asset_bundles/noname_ball_invisible/

SCENARIOS AND SEEDS
-------------------
  Scenarios: academy_3_vs_1_with_keeper, academy_counterattack_easy,
             academy_counterattack_hard, academy_run_pass_and_shoot_with_keeper,
             academy_pass_and_shoot_with_keeper, academy_run_to_score_with_keeper
  Seeds: 3, 7, 11, 23, 42
  Same as experiments/nithil_work/results/10s_ball_visible_no_names/

CLIP INDEX
----------
  Clip   Scenario                                    Seed
  01     academy_3_vs_1_with_keeper                   3
  02     academy_3_vs_1_with_keeper                   7
  03     academy_3_vs_1_with_keeper                   11
  04     academy_3_vs_1_with_keeper                   23
  05     academy_3_vs_1_with_keeper                   42
  06     academy_counterattack_easy                   3
  07     academy_counterattack_easy                   7
  08     academy_counterattack_easy                   11
  09     academy_counterattack_easy                   23
  10     academy_counterattack_easy                   42
  11     academy_counterattack_hard                   3
  12     academy_counterattack_hard                   7
  13     academy_counterattack_hard                   11
  14     academy_counterattack_hard                   23
  15     academy_counterattack_hard                   42
  16     academy_run_pass_and_shoot_with_keeper       3
  17     academy_run_pass_and_shoot_with_keeper       7
  18     academy_run_pass_and_shoot_with_keeper       11
  19     academy_run_pass_and_shoot_with_keeper       23
  20     academy_run_pass_and_shoot_with_keeper       42
  21     academy_pass_and_shoot_with_keeper           3
  22     academy_pass_and_shoot_with_keeper           7
  23     academy_pass_and_shoot_with_keeper           11
  24     academy_pass_and_shoot_with_keeper           23
  25     academy_pass_and_shoot_with_keeper           42
  26     academy_run_to_score_with_keeper             3
  27     academy_run_to_score_with_keeper             7
  28     academy_run_to_score_with_keeper             11
  29     academy_run_to_score_with_keeper             23
  30     academy_run_to_score_with_keeper             42

HOW TO REGENERATE
-----------------
  cd experiments/nithil_work
  python3 gen_10s_visibility_splits.py

Requires: noname and noname_ball_invisible asset bundles,
          ffmpeg, Pillow, numpy, gfootball.
Build bundles first: python3 experiments/build_noname_bundles.py
