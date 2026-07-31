"""make_heatmap.py — heatmap of final-frame ball positions on an empty 32x12 grid.

Reads results/natural_30_5s_grid/ground_truth.json and renders:
  ball_heatmap.png  — each grid cell shaded by how many of the 30 balls landed in
                      it, with the exact ball positions dotted on top + a legend.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "results" / "natural_30_5s_grid"
W, H, CELL = 1280, 480, 40
COLS, ROWS = W // CELL, H // CELL          # 32 x 12
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(sz):
    try:
        return ImageFont.truetype(FONT_PATH, sz)
    except Exception:
        return ImageFont.load_default()


def ramp(t):
    """t in [0,1] -> color. dark slate (empty) -> blue -> cyan -> yellow -> red."""
    stops = [(0.0, (24, 28, 34)), (0.01, (40, 60, 110)), (0.35, (30, 140, 200)),
             (0.6, (60, 200, 160)), (0.8, (240, 220, 60)), (1.0, (220, 50, 40))]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if t <= b:
            f = 0 if b == a else (t - a) / (b - a)
            return tuple(int(ca[i] + (cb[i] - ca[i]) * f) for i in range(3))
    return stops[-1][1]


def main():
    rows = json.load(open(OUT / "ground_truth.json"))
    counts = np.zeros((ROWS, COLS), int)
    for r in rows:
        counts[ord(r["grid_row"]) - ord("A"), r["grid_col"] - 1] += 1
    cmax = counts.max()

    img = Image.new("RGB", (W, H + 70), (16, 18, 22))
    d = ImageDraw.Draw(img)

    # shaded cells
    for rr in range(ROWS):
        for cc in range(COLS):
            n = counts[rr, cc]
            t = 0 if cmax == 0 else n / cmax
            d.rectangle([cc * CELL, rr * CELL, (cc + 1) * CELL, (rr + 1) * CELL],
                        fill=ramp(t))
            if n:
                d.text((cc * CELL + CELL / 2 - 3, rr * CELL + CELL / 2 - 6),
                       str(n), fill=(255, 255, 255), font=font(13))

    # grid lines
    line = (255, 255, 0, 90)
    for c in range(COLS + 1):
        d.line([(c * CELL, 0), (c * CELL, H)], fill=(120, 120, 40), width=1)
    for r in range(ROWS + 1):
        d.line([(0, r * CELL), (W, r * CELL)], fill=(120, 120, 40), width=1)

    # labels
    lf = font(11)
    for c in range(COLS):
        d.text((c * CELL + 2, 1), str(c + 1), fill=(230, 230, 120), font=lf)
    for r in range(ROWS):
        d.text((1, r * CELL + CELL - 13), chr(ord("A") + r),
               fill=(230, 230, 120), font=lf)

    # exact ball dots
    for r in rows:
        px, py = r["px"], r["py"]
        d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(255, 255, 255),
                  outline=(0, 0, 0))

    # legend / colorbar
    d.text((6, H + 8), f"Final-frame ball positions — 30 clips  (cell number = count, "
           f"white dot = exact spot)", fill=(230, 230, 230), font=font(14))
    lx, ly, lw, lh = 6, H + 40, 300, 16
    for i in range(lw):
        d.line([(lx + i, ly), (lx + i, ly + lh)], fill=ramp(i / (lw - 1)))
    d.rectangle([lx, ly, lx + lw, ly + lh], outline=(180, 180, 180))
    d.text((lx, ly + lh + 2), "0", fill=(200, 200, 200), font=font(11))
    d.text((lx + lw - 8, ly + lh + 2), str(cmax), fill=(200, 200, 200), font=font(11))
    d.text((lx + lw + 12, ly + 1), "balls per cell", fill=(200, 200, 200), font=font(12))

    p = OUT / "ball_heatmap.png"
    img.save(p)
    print("wrote", p, "| max per cell =", cmax, "| total balls =", int(counts.sum()))


if __name__ == "__main__":
    main()
