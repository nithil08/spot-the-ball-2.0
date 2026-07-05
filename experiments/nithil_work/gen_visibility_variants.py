"""Generate 5 visibility-split variants from clip_1 (academy_3_vs_1_with_keeper, seed=23).

Each variant is exactly 5 seconds.  The first N seconds show the BALL (default bundle);
the remaining (5-N) seconds hide the BALL (ball_tiny bundle).  Same seed = identical
physics, so the two renders are frame-perfectly aligned and can be spliced.

Variants produced:
  0s_vis_5s_hid.mov  — ball hidden the entire clip  (baseline: fully occluded)
  1s_vis_4s_hid.mov  — 1 s visible, then 4 s hidden
  2s_vis_3s_hid.mov  — 2 s visible, then 3 s hidden
  3s_vis_2s_hid.mov  — 3 s visible, then 2 s hidden
  4s_vis_1s_hid.mov  — 4 s visible, then 1 s hidden

Every frame gets:
  - HUD cropped (top 60px + bottom 180px removed)
  - 32x12 yellow alphanumeric grid burned in
  - Title bar at top showing split and clip info
  - No player name labels (zero controllable player slots)

Output: results/visibility_variants/
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EXPERIMENTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS))
from lib import (frames_to_mov,
                 HUD_TOP_PX, HUD_BOTTOM_PX, FRAME_W, FRAME_H, FPS)

WORKER = Path(__file__).parent / "render_worker_exact.py"

OUT = Path(__file__).parent / "results" / "visibility_variants"

# Source clip: Clip 1 from gen_5s_noname_clips.py
LEVEL = "academy_3_vs_1_with_keeper"
SEED  = 23
STEPS = 50   # 5 seconds @ 10 fps

# Variants: (visible_seconds, hidden_seconds)
VARIANTS = [(0, 5), (1, 4), (2, 3), (3, 2), (4, 1)]

CELL = 40
CROP_W = FRAME_W
CROP_H = FRAME_H - HUD_TOP_PX - HUD_BOTTOM_PX   # 480
COLS = CROP_W // CELL   # 32
ROWS = CROP_H // CELL   # 12

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"


def _font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def build_grid_overlay() -> Image.Image:
    overlay = Image.new("RGBA", (CROP_W, CROP_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line = (255, 255, 0, 130)
    for c in range(COLS + 1):
        draw.line([(c * CELL, 0), (c * CELL, CROP_H)], fill=line, width=1)
    for r in range(ROWS + 1):
        draw.line([(0, r * CELL), (CROP_W, r * CELL)], fill=line, width=1)
    font = _font(11)
    for c in range(COLS):
        draw.text((c * CELL + 2, 0), str(c + 1), fill=(255, 255, 0, 255), font=font)
    for r in range(ROWS):
        draw.text((1, r * CELL + 1), chr(ord("A") + r), fill=(255, 255, 0, 255), font=font)
    return overlay


GRID_OVERLAY = build_grid_overlay()


def crop_frame(frame: np.ndarray) -> np.ndarray:
    return frame[HUD_TOP_PX: FRAME_H - HUD_BOTTOM_PX, :]


def burn_grid(frame_rgb: np.ndarray) -> np.ndarray:
    base = Image.fromarray(frame_rgb).convert("RGBA")
    out = Image.alpha_composite(base, GRID_OVERLAY)
    return np.array(out.convert("RGB"))


def burn_title(frame_rgb: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    """Burn a two-line title bar onto the top of a frame."""
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    bar_h = 42 if subtitle else 24
    draw.rectangle([0, 0, img.width, bar_h], fill=(0, 0, 0))
    draw.text((6, 4), title, fill=(255, 255, 255), font=_font(14))
    if subtitle:
        draw.text((6, 24), subtitle, fill=(180, 180, 180), font=_font(11))
    return np.array(img)


def render_bundle(bundle: str, work_label: str) -> list:
    """Render STEPS frames in a fresh subprocess to avoid GFOOTBALL_DATA_DIR caching.

    The C++ engine locks in the data dir on first env creation per process.
    Running each bundle in its own subprocess guarantees the correct assets load.
    """
    work_dir = OUT / f"_work_{work_label}"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_npy = work_dir / "frames.npy"

    result = subprocess.run(
        [sys.executable, str(WORKER), LEVEL, str(SEED), bundle, str(STEPS), str(out_npy)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  Worker stderr:\n{result.stderr[-800:]}")
        raise RuntimeError(f"Worker failed (exit {result.returncode})")

    frames_arr = np.load(str(out_npy))
    frames = [frames_arr[i] for i in range(len(frames_arr))]
    print(f"  rendered {bundle}: {len(frames)} frames")
    return frames


def make_variant(vis_s: int, hid_s: int,
                 frames_vis: list, frames_hid: list) -> Path:
    vis_n = vis_s * FPS   # frames from visible render
    hid_n = hid_s * FPS   # frames from hidden render

    spliced = frames_vis[:vis_n] + frames_hid[vis_n: vis_n + hid_n]

    title = f"VISIBLE: {vis_s}s  →  HIDDEN: {hid_s}s   (total: 5s)"
    subtitle = f"academy_3_vs_1_with_keeper | seed=23 | 32x12 grid | no player names"

    processed = []
    for i, f in enumerate(spliced):
        frame = burn_grid(crop_frame(f))
        frame = burn_title(frame, title, subtitle)
        processed.append(frame)

    filename = f"{vis_s}s_vis_{hid_s}s_hid.mov"
    out_path = OUT / filename
    frames_to_mov(processed, out_path, crop_hud=False)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"  {filename}  ({vis_s}s visible + {hid_s}s hidden)  {size_mb:.1f} MB")
    return out_path, size_mb


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"\nRendering with NONAME bundle (ball visible, no names)...")
    frames_vis = render_bundle("noname", "visible")

    print(f"\nRendering with NONAME_BALL_INVISIBLE bundle (ball invisible, no names)...")
    frames_hid = render_bundle("noname_ball_invisible", "hidden")

    print(f"\nGenerating {len(VARIANTS)} variants...")
    results = []
    for vis_s, hid_s in VARIANTS:
        out_path, size_mb = make_variant(vis_s, hid_s, frames_vis, frames_hid)
        results.append((vis_s, hid_s, size_mb))

    print(f"\ndone -> {OUT}")

    sys.path.insert(0, str(Path(__file__).parent))
    from log_entry import log
    bullets = [
        f"{vs}s_vis_{hs}s_hid.mov: {vs}s ball visible then {hs}s ball hidden, {sm:.1f} MB"
        for vs, hs, sm in results
    ]
    log(
        "Generated visibility-split variants (gen_visibility_variants.py)",
        f"Source: academy_3_vs_1_with_keeper, seed=23, 5s total, 32x12 grid, no player names",
        *bullets,
        "Output: experiments/nithil_work/results/visibility_variants/",
        "Visible = noname bundle (normal ball, no names) | Hidden = noname_ball_invisible (5% ball, no names)",
    )


if __name__ == "__main__":
    main()
