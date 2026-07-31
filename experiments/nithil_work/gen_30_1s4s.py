"""gen_30_1s4s.py — all 30 clips as "1s visible then 4s invisible" + red start circle.

For every clip in natural_30_5s_grid/manifest.json, produce a 5s / 16x6-grid clip where
the ball is visible for the first 1 s, then invisible for the remaining 4 s, with a small
red circle around the ball's STARTING location on the first few frames.

Method mirrors gen_ball_marker_and_splits: render each window twice (ball visible via
noname, ball invisible via noname_ball_invisible) from the identical deterministic state,
then compose (first 10 visible frames + last 40 invisible frames). Referee is already
invisible engine-side.

Run with GRID_COLS=16 GRID_ROWS=6.  Modes: render_visible | render_invisible | compose
"""
import sys
import json
from pathlib import Path
import numpy as np

import gen_natural_30_clips as G

SRC_MANIFEST = Path(__file__).parent / "results" / "natural_30_5s_grid" / "manifest.json"
OUT = Path(__file__).parent / "results" / "30_1s_visible_4s_invisible"
FR = OUT / "_frames"
MARK_FRAMES = 1          # red circle on the VERY FIRST FRAME ONLY
CIRCLE_R = 18


def _manifest():
    return json.loads(SRC_MANIFEST.read_text())


def _render(bundle, tag):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lib import use_bundle
    use_bundle(bundle)
    OUT.mkdir(parents=True, exist_ok=True)
    FR.mkdir(exist_ok=True)
    by_seed = {}
    for m in _manifest():
        by_seed.setdefault(m["seed"], []).append(m)
    for seed, items in by_seed.items():
        starts = {m["window_start"] for m in items}
        wins = G.run_seed(seed, OUT / f"_work_{tag}", want_starts=starts)
        for m in items:
            frames = np.array(wins[m["window_start"]][0])
            np.savez_compressed(FR / f"clip{m['clip']:02d}_{tag}.npz", frames=frames)
        print(f"  seed {seed} [{tag}]: {len(items)} clips")


def mode_render_visible():
    _render("noname", "vis")


def mode_render_invisible():
    _render("noname_ball_invisible", "inv")


def mode_compose():
    from PIL import Image, ImageDraw
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lib import frames_to_mov
    overlay = G.build_grid()
    rows = []
    for m in _manifest():
        cid = m["clip"]
        vis = np.load(FR / f"clip{cid:02d}_vis.npz")["frames"]
        inv = np.load(FR / f"clip{cid:02d}_inv.npz")["frames"]
        px, py, _ = G.detect_ball(vis[0], inv[0])
        cell, r, c = G.cell_of(px, py)
        seq = list(vis[:10]) + list(inv[10:])          # 1s visible, 4s invisible
        out = []
        for i, f in enumerate(seq):
            g = G.burn_grid(f, overlay)
            if i < MARK_FRAMES:
                img = Image.fromarray(g)
                ImageDraw.Draw(img).ellipse(
                    [px - CIRCLE_R, py - CIRCLE_R, px + CIRCLE_R, py + CIRCLE_R],
                    outline=(255, 0, 0), width=3)
                g = np.array(img)
            out.append(g)
        name = f"clip{cid:02d}_1s_visible_then_4s_invisible.mov"
        frames_to_mov(out, OUT / name, fps=G.FPS, crop_hud=False)
        rows.append((cid, m["seed"], cell, round(px, 1), round(py, 1)))
        print(f"  clip{cid:02d} seed={m['seed']} start_circle={cell} ({px:.0f},{py:.0f})")

    csv = ["clip,seed,ball_start_cell,start_px,start_py"]
    csv += [f"{cid},{sd},{cell},{px},{py}" for cid, sd, cell, px, py in rows]
    (OUT / "ball_start_positions.csv").write_text("\n".join(csv) + "\n")
    print(f"\nwrote {len(rows)} clips + ball_start_positions.csv -> {OUT}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "compose"
    {"render_visible": mode_render_visible,
     "render_invisible": mode_render_invisible,
     "compose": mode_compose}[mode]()
