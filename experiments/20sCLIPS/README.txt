20sCLIPS — EARLY CLIP SET (largely superseded)
===============================================
Generated: 2026-06-30 (earlier session)

WHAT IS IN THIS FOLDER
-----------------------
4 video clips of Google Research Football gameplay, 20 seconds each.

NOTE: These clips do NOT have the 32×12 grid overlay and player names are
      STILL VISIBLE.  They were generated early in the project before the
      grid-burn and name-suppression methods were developed.

      For clips with no names and a grid, use:
        experiments/nithil_work/results/5s_ball_visible_no_names/
        experiments/nithil_work/results/10s_ball_visible_no_names/

CLIP SPECS
----------
  Duration:      20 seconds each (200 steps at 10 FPS)
  Resolution:    1280 × 720 pixels (full render including HUD bars)
  Ball:          VISIBLE at normal size
  Player names:  VISIBLE (not removed)
  Grid overlay:  NONE
  Asset bundle:  default (unmodified gfootball assets)

CLIPS IN THIS FOLDER
--------------------
  clip_1.mov — clip_4.mov
  (Exact scenarios and seeds not recorded for this batch — see RESEARCH_LOG.md)

HOW TO REGENERATE (if needed)
------------------------------
The original script for these clips is no longer current.
Use gen_plain_grid_clips.py (generates 20s clips WITH grid) or
gen_10s_noname_clips.py (10s clips, no names, with grid) instead.
