"""make_16x6_gt_heatmap.py — re-bin the 30 exact ball pixel positions onto a 16x6
grid, then write ground_truth.{json,csv,md} and ball_heatmap.png.

Ball pixel positions are grid-independent, so we reuse the exact px/py measured for
the 32x12 batch (visible-vs-invisible diff) — no re-rendering needed.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).parent / "results"
SRC = BASE / "natural_30_5s_grid" / "ground_truth.json"     # exact px/py
OUT = BASE / "natural_30_5s_grid_16x6"
W, H, COLS, ROWS = 1280, 480, 16, 6
CELL_W, CELL_H = W / COLS, H / ROWS                          # 80 x 80
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(sz):
    try:
        return ImageFont.truetype(FONT_PATH, sz)
    except Exception:
        return ImageFont.load_default()


def cell_of(px, py):
    c = min(max(int(px // CELL_W), 0), COLS - 1)
    r = min(max(int(py // CELL_H), 0), ROWS - 1)
    return f"{chr(ord('A') + r)}{c + 1}", r, c


def ramp(t):
    stops = [(0.0, (24, 28, 34)), (0.01, (40, 60, 110)), (0.35, (30, 140, 200)),
             (0.6, (60, 200, 160)), (0.8, (240, 220, 60)), (1.0, (220, 50, 40))]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if t <= b:
            f = 0 if b == a else (t - a) / (b - a)
            return tuple(int(ca[i] + (cb[i] - ca[i]) * f) for i in range(3))
    return stops[-1][1]


def main():
    src = json.load(open(SRC))
    rows = []
    counts = np.zeros((ROWS, COLS), int)
    for r in src:
        cell, ri, ci = cell_of(r["px"], r["py"])
        counts[ri, ci] += 1
        rows.append({"clip": r["clip"], "file": r["file"], "seed": r["seed"],
                     "ball_cell": cell, "grid_row": chr(ord('A') + ri),
                     "grid_col": ci + 1, "px": r["px"], "py": r["py"]})

    # ground truth files
    (OUT / "ground_truth.json").write_text(json.dumps(rows, indent=2))
    csv = ["clip,file,seed,ball_cell,grid_row,grid_col,px,py"]
    for r in rows:
        csv.append(f"{r['clip']},{r['file']},{r['seed']},{r['ball_cell']},"
                   f"{r['grid_row']},{r['grid_col']},{r['px']},{r['py']}")
    (OUT / "ground_truth.csv").write_text("\n".join(csv) + "\n")
    md = ["# Ground truth — ball location on final frame (16x6 grid)", "",
          "Grid: columns 1-16 left->right, rows A-F top->bottom. Frame 1280x480 "
          "(cells 80x80).", "",
          "| Clip | Seed | Ball cell | Row | Col | px | py |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['clip']:02d} | {r['seed']} | **{r['ball_cell']}** | "
                  f"{r['grid_row']} | {r['grid_col']} | {r['px']} | {r['py']} |")
    (OUT / "ground_truth.md").write_text("\n".join(md) + "\n")

    # heatmap
    cmax = counts.max()
    img = Image.new("RGB", (W, H + 70), (16, 18, 22))
    d = ImageDraw.Draw(img)
    for rr in range(ROWS):
        for cc in range(COLS):
            n = counts[rr, cc]
            t = 0 if cmax == 0 else n / cmax
            d.rectangle([cc * CELL_W, rr * CELL_H, (cc + 1) * CELL_W, (rr + 1) * CELL_H],
                        fill=ramp(t))
            if n:
                d.text((cc * CELL_W + CELL_W / 2 - 5, rr * CELL_H + CELL_H / 2 - 9),
                       str(n), fill=(255, 255, 255), font=font(20))
    for c in range(COLS + 1):
        d.line([(int(c * CELL_W), 0), (int(c * CELL_W), H)], fill=(120, 120, 40), width=1)
    for r in range(ROWS + 1):
        d.line([(0, int(r * CELL_H)), (W, int(r * CELL_H))], fill=(120, 120, 40), width=1)
    lf = font(13)
    for c in range(COLS):
        d.text((int(c * CELL_W) + 3, 2), str(c + 1), fill=(230, 230, 120), font=lf)
    for r in range(ROWS):
        d.text((2, int(r * CELL_H) + CELL_H - 16), chr(ord("A") + r),
               fill=(230, 230, 120), font=lf)
    for r in rows:
        px, py = r["px"], r["py"]
        d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(255, 255, 255), outline=(0, 0, 0))
    d.text((6, H + 8), "Final-frame ball positions — 30 clips on 16x6 grid  "
           "(cell number = count, white dot = exact spot)",
           fill=(230, 230, 230), font=font(14))
    lx, ly, lw, lh = 6, H + 40, 300, 16
    for i in range(lw):
        d.line([(lx + i, ly), (lx + i, ly + lh)], fill=ramp(i / (lw - 1)))
    d.rectangle([lx, ly, lx + lw, ly + lh], outline=(180, 180, 180))
    d.text((lx, ly + lh + 2), "0", fill=(200, 200, 200), font=font(11))
    d.text((lx + lw - 8, ly + lh + 2), str(cmax), fill=(200, 200, 200), font=font(11))
    d.text((lx + lw + 12, ly + 1), "balls per cell", fill=(200, 200, 200), font=font(12))
    img.save(OUT / "ball_heatmap.png")
    print(f"wrote GT + heatmap -> {OUT} | max per cell = {cmax} | total = {int(counts.sum())}")


if __name__ == "__main__":
    main()
