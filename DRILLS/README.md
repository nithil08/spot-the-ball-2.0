# DRILLS — spot-the-invisible-ball drill clips

Separate workspace of small-sided football **drills** rendered for the spot-the-ball
benchmark. **8 drills** (× 3 variants × 2 versions = **up to 48 clips**; set-piece
variants with no continuous seed are skipped rather than emitted as goal/reset clips).

**Continuous play only:** every clip is one unbroken passage of play — no goals, no
kickoff resets, and no "ball dropped into the centre" restarts. A goal makes the engine
teleport the ball to the centre spot; the renderer detects that (and any goal-line
crossing or teleport) and keeps only the seeds where the ball keeps circulating.

## What each clip is
- 5.0 s, 1280×480, 10 fps, **16×6 yellow grid** (rows A–F, cols 1–16).
- **Red circle around the ball on the VERY FIRST FRAME** (marks the start location).
- Two versions per variant:
  - `*_visible.mov` — ball visible the whole 5 s.
  - `*_split.mov`   — ball visible 1 s (10 frames), then invisible 4 s (40 frames);
                      the model must infer the final-frame cell.
- Referees are already invisible / engine-side (inherited from the noname bundles).

Ground truth (ball start cell + final cell) is computed by pixel-diffing the identical
deterministic play rendered with the ball visible vs. invisible — same method as the
main 30-clip benchmark. See `ALL_GROUND_TRUTH.csv` and each drill's `ground_truth.csv`.

## The 8 drills (ball-movement character)
| # | Folder | Drill | Ball dynamics |
|---|--------|-------|---------------|
| 1 | `01_4v4_small_sided` | 4v4 small-sided | balanced central open play |
| 2 | `02_3v1_rondo` | 3v1 rondo (keep-away) | tight, fast handoffs (path ≈0.3–0.4) |
| 3 | `03_5v2_rondo` | 5v2 rondo | wider circulation + splitting passes |
| 4 | `04_fast_break_3v2` | fast break 3v2 to goal | directional forward travel |
| 5 | `05_wing_cross` | wing cross across the third | long sideways travel, pulled off the goalmouth (no goals) |
| 6 | `06_switch_of_play` | switch of play | max lateral displacement (path ≈1.0) |
| 7 | `07_free_kick` | free kick (set piece) | dead ball outside the box worked into open play |
| 8 | `08_corner_kick` | corner kick (set piece) | corner-flag delivery circulated in the box |

## Regenerate / tweak
```
cd DRILLS
python3 generate_drills.py all        # vis -> inv -> compose, all 36 clips
# narrow for iteration:
DRILL_ONLY=5 python3 generate_drills.py all
```
- `drills.py` — the 6 drill layouts + per-variant jitter (edit positions/roles here).
- `generate_drills.py` — 3-phase renderer (engine caches the asset bundle at import,
  so visible and invisible must render in separate processes).

## Status
- [x] First 6 drills, 36 clips rendered + ground truth. (2026-07-22)
- [ ] Added 2 mainstream set pieces (free kick, corner kick), fixed wing_cross to
      circulate off the goalmouth, enforced continuous-play-only (no goal/reset clips).
      Re-run `python3 generate_drills.py all` to render the 8-drill set. (2026-08-07)
