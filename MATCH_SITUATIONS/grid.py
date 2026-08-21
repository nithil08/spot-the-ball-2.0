"""grid.py — the locked benchmark overlay: 16x6 yellow grid, red start circle, ball GT.

Same format as the DRILLS / 30-clip benchmark, kept in one place so the situation clips
and the frame-visibility clips can't drift apart:

  * 1280x480 frame (the engine's 1280x720 render with the score strip and the radar /
    sticky-action panel cropped off).
  * 16 columns numbered 1..16, 6 rows lettered A..F, drawn in translucent yellow.
  * A red circle around the ball on the VERY FIRST frame, marking where it started.
  * Ball ground truth comes from pixel-diffing the identical deterministic play rendered
    with the ball visible (bundle `noname`) against the ball invisible
    (`noname_ball_invisible`). The only thing that changes between the two renders is the
    ball, so the difference IS the ball — no engine instrumentation needed.
"""
from pathlib import Path

W, H = 1280, 480
COLS, ROWS = 16, 6
CELL_W, CELL_H = W / COLS, H / ROWS
FPS = 10
CIRCLE_R = 18
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def build_grid():
    from PIL import Image, ImageDraw
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    line = (255, 255, 0, 130)
    for c in range(COLS + 1):
        x = int(round(c * CELL_W))
        dr.line([(x, 0), (x, H)], fill=line, width=1)
    for r in range(ROWS + 1):
        y = int(round(r * CELL_H))
        dr.line([(0, y), (W, y)], fill=line, width=1)
    font = _font(13)
    for c in range(COLS):
        dr.text((int(round(c * CELL_W)) + 2, 0), str(c + 1),
                fill=(255, 255, 0, 255), font=font)
    for r in range(ROWS):
        dr.text((1, int(round(r * CELL_H)) + 1), chr(ord("A") + r),
                fill=(255, 255, 0, 255), font=font)
    return ov


def burn_grid(frame_rgb, overlay):
    from PIL import Image
    import numpy as np
    base = Image.fromarray(frame_rgb).convert("RGBA")
    return np.array(Image.alpha_composite(base, overlay).convert("RGB"))


def cell_of(px, py):
    c = min(max(int(px // CELL_W), 0), COLS - 1)
    r = min(max(int(py // CELL_H), 0), ROWS - 1)
    return f"{chr(ord('A') + r)}{c + 1}"


def circle_ball(frame_rgb, px, py):
    from PIL import Image, ImageDraw
    import numpy as np
    img = Image.fromarray(frame_rgb)
    ImageDraw.Draw(img).ellipse(
        [px - CIRCLE_R, py - CIRCLE_R, px + CIRCLE_R, py + CIRCLE_R],
        outline=(255, 0, 0), width=3)
    return np.array(img)


def detect_ball(vis, inv):
    """Ball centre in pixels, from the visible-vs-invisible pixel difference."""
    import numpy as np
    d = np.abs(vis.astype(np.int16) - inv.astype(np.int16)).sum(axis=2)
    thr = max(40, d.max() * 0.35)
    mask = d > thr
    if mask.sum() == 0:
        mask = d > (d.max() * 0.5)
    bright = vis.astype(np.int32).sum(axis=2)          # the ball is near-white
    ballmask = mask & (bright > 600)
    if ballmask.sum() < 3:
        ballmask = mask
    ys, xs = np.nonzero(ballmask)
    w = d[ys, xs].astype(float)
    return float(np.average(xs, weights=w)), float(np.average(ys, weights=w))
