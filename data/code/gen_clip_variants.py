"""gen_clip_variants.py — variants of clips 29 & 30 (seed 3, windows 210 & 260).

Per clip, three 5s versions (all 1280x480, 16x6 grid where gridded):
  clipNN_nogrid.mov       ball visible, NO grid
  clipNN_grid.mov         ball visible, 16x6 grid
  clipNN_noball_grid.mov  ball invisible (spot-the-ball stimulus), 16x6 grid

Modes (bundle chosen before gfootball import):
  visible    -> noname bundle              -> _nogrid + _grid
  invisible  -> noname_ball_invisible      -> _noball_grid

Run with GRID_COLS=16 GRID_ROWS=6 so the shared build_grid() draws a 16x6 grid.
"""
import sys
from pathlib import Path

# reuse the generator's engine/grid helpers (reads GRID_COLS/GRID_ROWS at import)
import gen_natural_30_clips as G

CLIPS = {29: ("_work_c29", 210), 30: ("_work_c30", 260)}   # clip -> (workdir, window_start)
SEED = 3
OUT = Path(__file__).parent / "results" / "clip_variants"


def mode_visible():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lib import use_bundle, frames_to_mov
    use_bundle("noname")
    OUT.mkdir(parents=True, exist_ok=True)
    overlay = G.build_grid()
    starts = {s for _, s in CLIPS.values()}
    wins = G.run_seed(SEED, OUT / "_work", want_starts=starts)
    for cid, (_, start) in CLIPS.items():
        frames = wins[start][0]
        frames_to_mov(frames, OUT / f"clip{cid}_nogrid.mov", fps=G.FPS, crop_hud=False)
        gframes = [G.burn_grid(f, overlay) for f in frames]
        frames_to_mov(gframes, OUT / f"clip{cid}_grid.mov", fps=G.FPS, crop_hud=False)
        print(f"  clip{cid}: wrote _nogrid + _grid  ({len(frames)} frames)")


def mode_invisible():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lib import use_bundle, frames_to_mov
    use_bundle("noname_ball_invisible")
    OUT.mkdir(parents=True, exist_ok=True)
    overlay = G.build_grid()
    starts = {s for _, s in CLIPS.values()}
    wins = G.run_seed(SEED, OUT / "_work_inv", want_starts=starts)
    for cid, (_, start) in CLIPS.items():
        frames = wins[start][0]
        gframes = [G.burn_grid(f, overlay) for f in frames]
        frames_to_mov(gframes, OUT / f"clip{cid}_noball_grid.mov", fps=G.FPS, crop_hud=False)
        print(f"  clip{cid}: wrote _noball_grid  ({len(frames)} frames)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "visible"
    {"visible": mode_visible, "invisible": mode_invisible}[mode]()
