5s_ball_visible_no_names — CLIP SET README
==========================================
Generated: 2026-07-03
Script:    experiments/nithil_work/gen_5s_noname_clips.py

WHAT IS IN THIS FOLDER
-----------------------
4 video clips of Google Research Football gameplay.

CLIP SPECS
----------
  Duration:        5 seconds each (50 simulation steps at 10 FPS)
  Resolution:      1280 × 480 pixels  (the 60px top HUD and 180px bottom HUD
                   are cropped out — only the playing field is shown)
  Ball:            VISIBLE at full normal size
  Player names:    REMOVED (see "How names are removed" below)
  Grid overlay:    YES — 32 columns × 12 rows yellow alphanumeric grid burned
                   into every frame. Columns labelled 1–32 left to right,
                   rows labelled A–L top to bottom.
  Title bar:       Yes — black bar at top of each frame with clip info

CLIPS IN THIS FOLDER
--------------------
  clip_1.mov   scenario: academy_3_vs_1_with_keeper        seed: 23
  clip_2.mov   scenario: academy_counterattack_easy        seed: 331
  clip_3.mov   scenario: academy_counterattack_hard        seed:  3
  clip_4.mov   scenario: academy_run_pass_and_shoot_with_keeper  seed: 3

HOW PLAYER NAMES ARE REMOVED
-----------------------------
The gfootball C++ engine shows the name of whoever has the ball using a TTF
font loaded from the GFOOTBALL_FONT environment variable.  By default that
variable is set (by gfootball_engine/__init__.py) to the original font in the
gfootball source tree.

Our fix:
  1. We built an asset bundle ("noname") where the font file has been replaced
     with a modified copy using fontTools.  In the modified font every character
     is remapped in the cmap table to the "space" glyph, which has no visible
     ink.  All text the engine renders becomes invisible.
  2. Before importing gfootball in the generation script, we explicitly set
     GFOOTBALL_FONT to the blank font path.  This prevents the gfootball
     __init__.py from overriding it with the original.
  3. GFOOTBALL_DATA_DIR is also pointed at the noname bundle so all other
     assets (player textures, grass, etc.) remain normal.

Blank font location:
  experiments/asset_bundles/noname/data/media/fonts/alegreya/AlegreyaSansSC-ExtraBold.ttf

HOW THE GRID IS ADDED
----------------------
After each frame is captured from the engine, a Python PIL script composites a
semi-transparent yellow grid overlay (RGBA, alpha=130) onto the frame.
Grid cells are 40×40 pixels.  Labels are drawn in opaque yellow.
No grid information is passed to the engine — it is purely a post-processing step.

HOW TO REGENERATE
-----------------
  cd experiments/nithil_work
  python3 gen_5s_noname_clips.py

Prerequisites: gfootball installed at /Users/.../gfootball_src, ffmpeg, PIL, numpy.
The noname asset bundle must exist (run experiments/build_noname_bundles.py first).
