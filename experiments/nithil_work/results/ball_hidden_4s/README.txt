ball_hidden_4s — EARLY EXPERIMENT (largely superseded)
=======================================================
Generated: 2026-06-30 (earliest session)
Script:    experiments/nithil_work/gen_ball_hidden_4s.py

WHAT IS IN THIS FOLDER
-----------------------
Single still frames (PNG) extracted at the 4-second mark of a gameplay clip.
Ball is hidden.  This was the very first exploration of the "hidden ball" concept.

THIS FOLDER IS LARGELY SUPERSEDED BY ball_hidden_grid/ AND 10s_ball_visible_no_names/
It is kept for reference.

WHAT'S IN EACH SUB-FOLDER
--------------------------
  frame_4s.png               Still frame at t=4s, ball hidden (no grid drawn on image)
  frame_4s_grid_annotated.png  Same frame with a grid manually described in the prompt
                               (NOT burned visually — an early approach that was abandoned)
  meta.json                  Scenario, seed, and probe questions

NOTE: In this early batch, the grid was described to the model in text rather than
      burned into the image.  The visual-grid approach (burning it onto frames) was
      adopted starting from ball_hidden_grid/ and all subsequent batches.

NOTE: Player names are visible in these frames (no name-suppression at this stage).
