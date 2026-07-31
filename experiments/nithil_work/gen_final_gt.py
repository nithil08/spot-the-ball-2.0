"""gen_final_gt.py — final-frame ground truth for the 30_1s_visible_4s_invisible batch.

For every clip we already cached the visible and invisible renders of the identical
deterministic play (_frames/clipNN_vis.npz, clipNN_inv.npz). The ball is the ONLY thing
that differs between them, so diffing the LAST frame of each pair pinpoints the ball's
exact pixel in the final frame — the ground truth the model must infer.

Grid: 16x6 (rows A-F, cols 1-16), frame 1280x480 — the locked benchmark format.

Writes into results/30_1s_visible_4s_invisible/:
    ground_truth.csv   clip,ball_cell,grid_row,grid_col,px,py
    ground_truth.json  same, as records
    ground_truth.md    readable table
    _gt_debug/clipNN.png   final frame + crosshair on the detected ball (sanity check)
"""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "30_1s_visible_4s_invisible"
FR = OUT / "_frames"
DBG = OUT / "_gt_debug"

W, H = 1280, 480
COLS, ROWS = 16, 6
CELL_W, CELL_H = W / COLS, H / ROWS
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"


def cell_of(px, py):
    c = min(max(int(px // CELL_W), 0), COLS - 1)
    r = min(max(int(py // CELL_H), 0), ROWS - 1)
    return f"{chr(ord('A') + r)}{c + 1}", chr(ord('A') + r), c + 1


def detect_ball(vis, inv):
    """Ball (px,py) via visible-vs-invisible pixel diff (canonical GT method)."""
    v = vis.astype(np.int16)
    i = inv.astype(np.int16)
    d = np.abs(v - i).sum(axis=2)
    thr = max(40, d.max() * 0.35)
    mask = d > thr
    if mask.sum() == 0:
        mask = d > (d.max() * 0.5)
    bright = vis.astype(np.int32).sum(axis=2)
    ballmask = mask & (bright > 600)
    if ballmask.sum() < 3:
        ballmask = mask
    ys, xs = np.nonzero(ballmask)
    w = d[ys, xs].astype(float)
    return float(np.average(xs, weights=w)), float(np.average(ys, weights=w)), int(mask.sum())


def main():
    DBG.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(FONT_PATH, 18)
    clips = sorted(FR.glob("clip*_vis.npz"), key=lambda p: int(p.name[4:6]))
    rows = []
    for vp in clips:
        cid = int(vp.name[4:6])
        vis = np.load(vp)["frames"]
        inv = np.load(FR / f"clip{cid:02d}_inv.npz")["frames"]
        px, py, npix = detect_ball(vis[-1], inv[-1])       # FINAL frame
        cell, r, c = cell_of(px, py)
        rows.append({"clip": cid, "ball_cell": cell, "grid_row": r, "grid_col": c,
                     "px": round(px, 1), "py": round(py, 1), "changed_pixels": npix})
        # debug: final visible frame + crosshair at detected ball
        dbg = Image.fromarray(vis[-1]).convert("RGB")
        dr = ImageDraw.Draw(dbg)
        dr.line([(px - 14, py), (px + 14, py)], fill=(255, 0, 0), width=2)
        dr.line([(px, py - 14), (px, py + 14)], fill=(255, 0, 0), width=2)
        dr.text((px + 10, py + 8), cell, fill=(255, 0, 0), font=font)
        dbg.save(DBG / f"clip{cid:02d}.png")
        print(f"  clip{cid:02d}: final ball -> {cell}  (px={px:.0f},py={py:.0f}, dpix={npix})")

    (OUT / "ground_truth.json").write_text(json.dumps(rows, indent=2))
    with (OUT / "ground_truth.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["clip", "ball_cell", "grid_row", "grid_col",
                                           "px", "py", "changed_pixels"])
        wr.writeheader()
        wr.writerows(rows)
    md = ["# Final-frame ground truth — 30_1s_visible_4s_invisible (16x6 grid)", "",
          "Rows A-F (top->bottom), cols 1-16 (left->right). Frame 1280x480.",
          "Ball pixel = visible-vs-invisible diff on the FINAL frame.", "",
          "| Clip | Ball cell | Row | Col | px | py |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['clip']:02d} | **{r['ball_cell']}** | {r['grid_row']} | "
                  f"{r['grid_col']} | {r['px']} | {r['py']} |")
    (OUT / "ground_truth.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote ground_truth.(csv|json|md) + _gt_debug/ -> {OUT}")


if __name__ == "__main__":
    main()
