10s_ball_visible_no_names — CLIP SET README
============================================
Generated: 2026-07-03
Script:    experiments/nithil_work/gen_10s_noname_clips.py

WHAT IS IN THIS FOLDER
-----------------------
30 video clips of Google Research Football gameplay.
6 scenarios × 5 random seeds = 30 clips total.

CLIP SPECS
----------
  Duration:        10 seconds each (100 simulation steps at 10 FPS)
  Resolution:      1280 × 480 pixels (60px top HUD + 180px bottom HUD cropped out)
  Ball:            VISIBLE at full normal size
  Player names:    REMOVED (see 'How names are removed' below)
  Grid overlay:    YES — 32 columns × 12 rows yellow alphanumeric grid
                   burned into every frame. Cols 1–32 left→right, rows A–L top→bottom.
  Title bar:       Black bar at top: clip number, scenario, seed, duration, config

SCENARIOS USED
--------------
  academy_3_vs_1_with_keeper
  academy_counterattack_easy
  academy_counterattack_hard
  academy_run_pass_and_shoot_with_keeper
  academy_pass_and_shoot_with_keeper
  academy_run_to_score_with_keeper

SEEDS USED
----------
  3, 7, 11, 23, 42

CLIP INDEX
----------
  Clip#  Scenario                                    Seed   Size
  01     academy_3_vs_1_with_keeper                   3      1.0 MB
  02     academy_3_vs_1_with_keeper                   7      1.0 MB
  03     academy_3_vs_1_with_keeper                   11     1.2 MB
  04     academy_3_vs_1_with_keeper                   23     0.9 MB
  05     academy_3_vs_1_with_keeper                   42     0.9 MB
  06     academy_counterattack_easy                   3      1.3 MB
  07     academy_counterattack_easy                   7      1.3 MB
  08     academy_counterattack_easy                   11     1.4 MB
  09     academy_counterattack_easy                   23     1.2 MB
  10     academy_counterattack_easy                   42     1.2 MB
  11     academy_counterattack_hard                   3      1.3 MB
  12     academy_counterattack_hard                   7      1.3 MB
  13     academy_counterattack_hard                   11     1.3 MB
  14     academy_counterattack_hard                   23     1.2 MB
  15     academy_counterattack_hard                   42     1.2 MB
  16     academy_run_pass_and_shoot_with_keeper       3      1.0 MB
  17     academy_run_pass_and_shoot_with_keeper       7      1.0 MB
  18     academy_run_pass_and_shoot_with_keeper       11     1.0 MB
  19     academy_run_pass_and_shoot_with_keeper       23     1.0 MB
  20     academy_run_pass_and_shoot_with_keeper       42     1.1 MB
  21     academy_pass_and_shoot_with_keeper           3      1.1 MB
  22     academy_pass_and_shoot_with_keeper           7      1.0 MB
  23     academy_pass_and_shoot_with_keeper           11     1.2 MB
  24     academy_pass_and_shoot_with_keeper           23     1.2 MB
  25     academy_pass_and_shoot_with_keeper           42     1.1 MB
  26     academy_run_to_score_with_keeper             3      1.3 MB
  27     academy_run_to_score_with_keeper             7      1.2 MB
  28     academy_run_to_score_with_keeper             11     1.2 MB
  29     academy_run_to_score_with_keeper             23     1.1 MB
  30     academy_run_to_score_with_keeper             42     1.2 MB

HOW PLAYER NAMES ARE REMOVED
-----------------------------
The gfootball C++ engine shows the name of whoever has the ball using a TTF
font from GFOOTBALL_FONT env var.  By default gfootball's __init__.py sets
this to the original font in the source tree.

Our fix:
  1. Built 'noname' asset bundle with a modified font (fontTools) where every
     character maps to the space glyph — space has no visible ink.
  2. Set GFOOTBALL_FONT to this blank font BEFORE importing gfootball, so
     __init__.py's 'if not already set' check leaves it alone.
  3. GFOOTBALL_DATA_DIR → noname bundle (normal ball size, all other assets default).

Blank font: experiments/asset_bundles/noname/data/media/fonts/alegreya/AlegreyaSansSC-ExtraBold.ttf

HOW TO REGENERATE
-----------------
  cd experiments/nithil_work
  python3 gen_10s_noname_clips.py

Prerequisites: gfootball at /Users/.../gfootball_src, ffmpeg, Pillow, numpy.
The noname asset bundle must exist (run experiments/build_noname_bundles.py first).
