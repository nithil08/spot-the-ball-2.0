"""make_spread_heatmap.py — final-frame ball distribution on the 16x6 grid.

Renders the camera-offset batch beside the original centre-biased batch so the
change in coverage is directly readable. Encoding is sequential magnitude (balls
per cell), so it uses ONE hue light->dark — not the rainbow ramp the older
make_16x6_gt_heatmap.py used, which made mid counts look like a different kind of
thing rather than simply more.

    python3 make_spread_heatmap.py
"""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
VID = HERE.parent / "videos"
NEW = VID / "spread_30_5s_grid_16x6"
OLD = VID / "natural_30_5s_grid_16x6"

W, H, COLS, ROWS = 1280, 480, 16, 6
CELL_W, CELL_H = W / COLS, H / ROWS
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"

# sequential blue ramp, light -> dark (steps 100..700)
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE = (250, 250, 249)
INK = (28, 28, 26)
INK_MUTED = (110, 110, 105)
GRIDLINE = (214, 214, 210)
EMPTY = (241, 241, 239)          # zero reads as surface, not as a colour


def _hex(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def font(sz, bold=False):
    p = FONT_PATH.replace("Arial.ttf", "Arial Bold.ttf") if bold else FONT_PATH
    try:
        return ImageFont.truetype(p, sz)
    except Exception:
        return ImageFont.load_default()


def ramp(t):
    """t in [0,1] -> sequential blue. t==0 handled by the caller (EMPTY)."""
    i = int(round(t * (len(RAMP) - 1)))
    return _hex(RAMP[min(max(i, 0), len(RAMP) - 1)])


def load(path):
    """(counts 6x16, [(px,py)...]) from a ground_truth.csv."""
    counts = np.zeros((ROWS, COLS), int)
    pts = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            px, py = float(r["px"]), float(r["py"])
            c = min(max(int(px // CELL_W), 0), COLS - 1)
            rr = min(max(int(py // CELL_H), 0), ROWS - 1)
            counts[rr, c] += 1
            pts.append((px, py))
    return counts, pts


def stats(counts, pts):
    n = int(counts.sum())
    occ = int((counts > 0).sum())
    top = int(counts.max())
    rows_used = int((counts.sum(axis=1) > 0).sum())
    cols_used = int((counts.sum(axis=0) > 0).sum())
    pxs = np.array([p[0] for p in pts])
    pys = np.array([p[1] for p in pts])
    return {"n": n, "occupied": occ, "top": top, "top_pct": 100.0 * top / max(n, 1),
            "rows_used": rows_used, "cols_used": cols_used,
            "px_sd": float(pxs.std()), "py_sd": float(pys.std())}


def panel(draw, ox, oy, counts, pts, title, subtitle, cmax):
    f_cell = font(19, bold=True)
    f_axis = font(12)
    f_title = font(20, bold=True)
    f_sub = font(13)

    # title/subtitle clear of the column ruler at oy-16
    draw.text((ox, oy - 64), title, fill=INK, font=f_title)
    draw.text((ox, oy - 40), subtitle, fill=INK_MUTED, font=f_sub)

    for rr in range(ROWS):
        for cc in range(COLS):
            n = int(counts[rr, cc])
            x0, y0 = ox + cc * CELL_W, oy + rr * CELL_H
            # 2px surface gap between cells so adjacent fills stay separable
            draw.rectangle([x0 + 1, y0 + 1, x0 + CELL_W - 1, y0 + CELL_H - 1],
                           fill=(EMPTY if n == 0 else ramp(n / cmax)))
            if n:
                # label ink flips on dark fills to hold contrast
                col = (255, 255, 255) if n / cmax > 0.45 else INK
                tw = draw.textlength(str(n), font=f_cell)
                draw.text((x0 + CELL_W / 2 - tw / 2, y0 + CELL_H / 2 - 12),
                          str(n), fill=col, font=f_cell)

    for c in range(COLS + 1):
        x = ox + int(c * CELL_W)
        draw.line([(x, oy), (x, oy + H)], fill=GRIDLINE, width=1)
    for r in range(ROWS + 1):
        y = oy + int(r * CELL_H)
        draw.line([(ox, y), (ox + W, y)], fill=GRIDLINE, width=1)

    for c in range(COLS):
        t = str(c + 1)
        tw = draw.textlength(t, font=f_axis)
        draw.text((ox + c * CELL_W + CELL_W / 2 - tw / 2, oy - 16), t,
                  fill=INK_MUTED, font=f_axis)
    for r in range(ROWS):
        draw.text((ox - 16, oy + r * CELL_H + CELL_H / 2 - 8), chr(ord("A") + r),
                  fill=INK_MUTED, font=f_axis)

    # exact ball positions, ringed so they stay visible on any fill
    for px, py in pts:
        x, y = ox + px, oy + py
        draw.ellipse([x - 3.5, y - 3.5, x + 3.5, y + 3.5],
                     fill=(255, 255, 255), outline=(40, 40, 40))


def main():
    new_counts, new_pts = load(NEW / "ground_truth.csv")
    old_counts, old_pts = load(OLD / "ground_truth.csv")
    ns, os_ = stats(new_counts, new_pts), stats(old_counts, old_pts)
    cmax = max(new_counts.max(), old_counts.max())

    PAD, GAP, TOP = 60, 112, 130
    img = Image.new("RGB", (W + 2 * PAD, TOP + H * 2 + GAP + 150), SURFACE)
    d = ImageDraw.Draw(img)

    d.text((PAD, 30), "Where the ball ends up — final frame, 16x6 grid",
           fill=INK, font=font(26, bold=True))

    panel(d, PAD, TOP + 20, old_counts, old_pts,
          "Before — stock ball-tracking camera",
          f"{os_['occupied']}/96 cells · {os_['rows_used']}/6 rows · "
          f"{os_['cols_used']}/16 cols · busiest cell {os_['top']}/{os_['n']} "
          f"({os_['top_pct']:.0f}%)", cmax)

    y2 = TOP + 20 + H + GAP
    panel(d, PAD, y2, new_counts, new_pts,
          "After — per-clip camera framing offset",
          f"{ns['occupied']}/96 cells · {ns['rows_used']}/6 rows · "
          f"{ns['cols_used']}/16 cols · busiest cell {ns['top']}/{ns['n']} "
          f"({ns['top_pct']:.0f}%)", cmax)

    # legend
    ly = y2 + H + 34
    lx, lw, lh = PAD, 260, 14
    for i in range(lw):
        d.line([(lx + i, ly), (lx + i, ly + lh)], fill=ramp(i / (lw - 1)))
    d.rectangle([lx, ly, lx + lw, ly + lh], outline=GRIDLINE)
    d.text((lx, ly + lh + 5), "1", fill=INK_MUTED, font=font(11))
    tw = d.textlength(str(cmax), font=font(11))
    d.text((lx + lw - tw, ly + lh + 5), str(cmax), fill=INK_MUTED, font=font(11))
    d.text((lx + lw + 14, ly + 1), "clips per cell   ·   white dot = exact ball pixel",
           fill=INK_MUTED, font=font(12))

    d.text((PAD, ly + 44),
           f"Spread of final ball pixel:  before  x sd {os_['px_sd']:.0f}px / "
           f"y sd {os_['py_sd']:.0f}px      after  x sd {ns['px_sd']:.0f}px / "
           f"y sd {ns['py_sd']:.0f}px",
           fill=INK, font=font(13))

    out = NEW / "ball_heatmap_comparison.png"
    img.save(out)
    (NEW / "distribution_stats.json").write_text(
        json.dumps({"before": os_, "after": ns}, indent=2))
    print(f"wrote {out}")
    print(f"  before: {os_}")
    print(f"  after : {ns}")


if __name__ == "__main__":
    main()
