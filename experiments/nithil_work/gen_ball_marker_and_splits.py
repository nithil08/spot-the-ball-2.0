"""gen_ball_marker_and_splits.py — clips 29 & 30 (seed 3, windows 210 & 260).

Two deliverables, all 5s / 1280x480 / 16x6 grid:

  1) clipNN_start_marked.mov
       ball visible throughout; a small RED circle around the ball's STARTING
       location, drawn only on the first few frames.

  2) clipNN_visible{1,2,3,4}s_then_hidden.mov   (the 1-4 visibility split)
       ball visible for the first N seconds, then invisible for the rest.

Method: render each window twice — ball visible (noname) and ball invisible
(noname_ball_invisible) — from the identical deterministic state, then compose:
  * start location comes from the frame-0 visible-vs-invisible diff (exact pixel).
  * a split = first N visible frames + remaining invisible frames.

Run with GRID_COLS=16 GRID_ROWS=6 so the shared grid is 16x6.

Modes: render_visible | render_invisible | compose
"""
import sys
from pathlib import Path
import numpy as np

import gen_natural_30_clips as G   # reads GRID_COLS/GRID_ROWS at import

CLIPS = {29: 210, 30: 260}         # clip -> window_start (both seed 3)
SEED = 3
MARK_FRAMES = 3                    # circle shown on the first N frames
CIRCLE_R = 18                      # small circle radius (px)

OUT = Path(__file__).parent / "results" / "ball_marker_and_splits"
FR = OUT / "_frames"


def _render(bundle, tag):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lib import use_bundle
    use_bundle(bundle)
    OUT.mkdir(parents=True, exist_ok=True)
    FR.mkdir(exist_ok=True)
    wins = G.run_seed(SEED, OUT / f"_work_{tag}", want_starts=set(CLIPS.values()))
    for cid, start in CLIPS.items():
        frames = np.array(wins[start][0])          # (50, 480, 1280, 3)
        np.savez_compressed(FR / f"clip{cid}_{tag}.npz", frames=frames)
        print(f"  clip{cid} {tag}: {frames.shape}")


def mode_render_visible():
    _render("noname", "vis")


def mode_render_invisible():
    _render("noname_ball_invisible", "inv")


def mode_compose():
    from PIL import Image, ImageDraw
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lib import frames_to_mov
    overlay = G.build_grid()

    for cid in CLIPS:
        vis = np.load(FR / f"clip{cid}_vis.npz")["frames"]
        inv = np.load(FR / f"clip{cid}_inv.npz")["frames"]
        px, py, _ = G.detect_ball(vis[0], inv[0])   # ball start, exact pixel

        # 1) start-marked clip (ball visible throughout)
        marked = []
        for i, f in enumerate(vis):
            g = G.burn_grid(f, overlay)
            if i < MARK_FRAMES:
                img = Image.fromarray(g)
                d = ImageDraw.Draw(img)
                d.ellipse([px - CIRCLE_R, py - CIRCLE_R, px + CIRCLE_R, py + CIRCLE_R],
                          outline=(255, 0, 0), width=3)
                g = np.array(img)
            marked.append(g)
        frames_to_mov(marked, OUT / f"clip{cid}_start_marked.mov", fps=G.FPS, crop_hud=False)
        print(f"  clip{cid}: start circle @ ({px:.0f},{py:.0f}) on first {MARK_FRAMES} frames")

        # 2) 1-4s visibility splits (visible N s, then hidden)
        for secs in (1, 2, 3, 4):
            n = secs * G.FPS
            seq = list(vis[:n]) + list(inv[n:])
            g = [G.burn_grid(f, overlay) for f in seq]
            frames_to_mov(g, OUT / f"clip{cid}_visible{secs}s_then_hidden.mov",
                          fps=G.FPS, crop_hud=False)
            print(f"  clip{cid}: visible {secs}s then hidden ({n}/{len(seq)} frames)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "compose"
    {"render_visible": mode_render_visible,
     "render_invisible": mode_render_invisible,
     "compose": mode_compose}[mode]()
