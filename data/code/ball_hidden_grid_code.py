"""Ball-hidden occlusion task, upgraded to match Spot The Ball's own methodology.

The earlier nithil_work/gen_ball_hidden_4s.py sent models a single frozen
frame with a 6x4 grid *described only in the text prompt* -- never drawn.
That doesn't match our own reference paper (Balamurugan et al., "Spot The
Ball: A Benchmark for Visual Social Inference", arXiv:2511.00261), which
burns an alphanumeric grid directly onto the image shown to both humans and
models, and evaluates on real video-like context rather than a lone frame.

This script fixes both gaps:
  - renders an actual moving clip of each play with the ball invisible
    (ball_tiny bundle) instead of one freeze-frame
  - burns a fine SQUARE grid onto every frame of the video itself: 32 cols
    x 12 rows = 384 cells, 40x40px each, tiling the 1280x480 playing-field
    crop exactly (the paper uses a coarser 6x10 = 60-cell grid; this is
    deliberately much finer per instruction, and square rather than
    rectangular)
  - keeps the "locate now" / "predict +1s" question pair from the earlier
    single-frame version, now graded against grid cells instead of raw xy

Ground truth cropping/labeling always goes through the SAME pixel
projection (`pitch_to_pixel`) used to draw the debug ball marker, so the
graded cell and the visual grid can never disagree with each other --
independent of how well-calibrated that projection is against the engine's
true camera (lib.py itself flags it as an approximation).

Writes per item, under results/ball_hidden_grid/<item_id>/:
  clip.mov         grid burned in, ball invisible -- what a model/human sees
  debug_full.mov   grid burned in, ball VISIBLE + red ground-truth marker --
                    our own QA only, never shown to a model
  meta.json        ground truth cells + question text
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lib
from lib import (use_bundle, make_env, run_clip_capture, frames_to_mov, latest_dump,
                  HUD_TOP_PX, HUD_BOTTOM_PX, FRAME_W, FRAME_H, pitch_to_pixel, FPS)

OUT = Path(__file__).parent / "results" / "ball_hidden_grid"

CELL = 40
CROP_W = FRAME_W                                   # 1280
CROP_H = FRAME_H - HUD_TOP_PX - HUD_BOTTOM_PX       # 480
COLS = CROP_W // CELL                              # 32
ROWS = CROP_H // CELL                              # 12
assert CROP_W % CELL == 0 and CROP_H % CELL == 0, "grid must tile the frame exactly in squares"

FREEZE_STEP = 40    # 4s @ 10fps -- visible portion shown to the model
PREDICT_STEP = 50   # 5s @ 10fps -- ground truth for "predict 1s further"

TARGET_LEVELS = [
    "academy_3_vs_1_with_keeper",
    "academy_counterattack_easy",
    "academy_counterattack_hard",
    "academy_pass_and_shoot_with_keeper",
    "academy_run_pass_and_shoot_with_keeper",
    "academy_3_vs_1_with_keeper",
]
# Many episodes end early (goal / out of bounds) before reaching PREDICT_STEP.
# Try seeds in order per level and keep the first that survives long enough --
# see probe: academy_3_vs_1_with_keeper alone ranges from 12 to 51 surviving
# steps depending on seed.
SEED_POOL = [3, 11, 23, 47, 89, 113, 191, 257, 331, 401, 461, 523, 601, 677]


def cell_label(col: int, row: int) -> str:
    return f"{chr(ord('A') + row)}{col + 1}"


def build_grid_overlay() -> Image.Image:
    overlay = Image.new("RGBA", (CROP_W, CROP_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line = (255, 255, 0, 130)
    for c in range(COLS + 1):
        x = c * CELL
        draw.line([(x, 0), (x, CROP_H)], fill=line, width=1)
    for r in range(ROWS + 1):
        y = r * CELL
        draw.line([(0, y), (CROP_W, y)], fill=line, width=1)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
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


def pixel_bin(x: float, y: float) -> tuple:
    """Grid cell for a pitch coord, via the SAME pixel projection used to draw
    the debug marker -- so ground truth and the visual grid always agree."""
    px, py = pitch_to_pixel(x, y)
    py_c = py - HUD_TOP_PX
    col = max(0, min(COLS - 1, px // CELL))
    row = max(0, min(ROWS - 1, py_c // CELL))
    return int(col), int(row)


def render_clip(level: str, seed: int, bundle: str, work_dir: Path):
    use_bundle(bundle)
    env = make_env(level, seed, work_dir, write_video=False, hud=False)
    info = run_clip_capture(env, steps=PREDICT_STEP + 1, dump_name=f"work_{bundle}")
    env.close()
    return info["frames"], latest_dump(work_dir)


def make_item(level: str, used_seeds: set):
    item_id = frames_hidden = dump_hidden = seed = None
    for candidate in SEED_POOL:
        if candidate in used_seeds:
            continue
        item_id = hashlib.md5(f"ballhidgrid|{level}|{candidate}".encode()).hexdigest()[:10]
        item_dir = OUT / item_id
        item_dir.mkdir(parents=True, exist_ok=True)
        frames_hidden, dump_hidden = render_clip(level, candidate, "ball_tiny", item_dir / "_work_hidden")
        if len(frames_hidden) > PREDICT_STEP:
            seed = candidate
            break
        print(f"  (skip) {level} seed={candidate}: episode ended at {len(frames_hidden)} steps, need >{PREDICT_STEP}")
        shutil.rmtree(item_dir, ignore_errors=True)
    if seed is None:
        raise RuntimeError(f"{level}: no seed in pool survived past step {PREDICT_STEP}")
    used_seeds.add(seed)
    visible = [burn_grid(crop_frame(f)) for f in frames_hidden[:FREEZE_STEP + 1]]
    clip_mov = item_dir / "clip.mov"
    frames_to_mov(visible, clip_mov, crop_hud=False)  # already cropped in numpy above

    ball_at_freeze = ball_at_predict = None
    for idx, step in lib.read_dump(dump_hidden):
        obs = step["observation"]
        if idx == FREEZE_STEP:
            ball_at_freeze = obs["ball"][:2].tolist()
        if idx == PREDICT_STEP:
            ball_at_predict = obs["ball"][:2].tolist()
    if ball_at_freeze is None or ball_at_predict is None:
        raise RuntimeError(f"{item_id}: dump ended before step {PREDICT_STEP}")

    bin_freeze = pixel_bin(*ball_at_freeze)
    bin_predict = pixel_bin(*ball_at_predict)

    # 2. Debug render: same seed -> identical physics, ball VISIBLE this time,
    # grid + a red ground-truth marker. For our own QA only -- never a model input.
    frames_debug, dump_debug = render_clip(level, seed, "default", item_dir / "_work_debug")
    debug_frames = []
    for (idx, step), f in zip(lib.read_dump(dump_debug), frames_debug[:PREDICT_STEP + 1]):
        img = burn_grid(crop_frame(f))
        obs = step["observation"]
        px, py = pitch_to_pixel(*obs["ball"][:2].tolist())
        py_c = py - HUD_TOP_PX
        img_pil = Image.fromarray(img)
        d = ImageDraw.Draw(img_pil)
        d.ellipse([px - 8, py_c - 8, px + 8, py_c + 8], outline=(255, 0, 0), width=3)
        debug_frames.append(np.array(img_pil))
    debug_mov = item_dir / "debug_full.mov"
    frames_to_mov(debug_frames, debug_mov, crop_hud=False)

    meta = {
        "id": item_id,
        "level": level,
        "seed": seed,
        "grid": {
            "cols": COLS, "rows": ROWS, "cell_px": CELL, "total_cells": COLS * ROWS,
            "note": f"{COLS}x{ROWS} squares ({COLS*ROWS} total), {CELL}x{CELL}px, burned into "
                    f"every frame. Columns 1-{COLS} left->right, rows A-{chr(ord('A')+ROWS-1)} "
                    f"top->bottom. Spot The Ball (Balamurugan et al., arXiv:2511.00261) uses a "
                    f"6x10 (60-cell) alphanumeric grid burned onto the image; this is the same "
                    f"convention at much finer, square resolution.",
        },
        "freeze_step": FREEZE_STEP,
        "predict_step": PREDICT_STEP,
        "clip_visible_s": FREEZE_STEP / FPS,
        "ground_truth": {
            "ball_xy_at_freeze": ball_at_freeze,
            "ball_cell_at_freeze": cell_label(*bin_freeze),
            "ball_bin_at_freeze": list(bin_freeze),
            "ball_xy_at_predict": ball_at_predict,
            "ball_cell_at_predict": cell_label(*bin_predict),
            "ball_bin_at_predict": list(bin_predict),
        },
        "questions": {
            "locate_now": f"The ball has been made invisible. This clip shows "
                          f"{FREEZE_STEP/FPS:.0f}s of play. A {COLS}x{ROWS} grid is overlaid on "
                          f"the video (columns 1-{COLS} left to right, rows A-"
                          f"{chr(ord('A')+ROWS-1)} top to bottom). Based on player positions, "
                          f"body orientation, and motion, which grid cell is the ball in RIGHT "
                          f"NOW, at the final frame of the clip? Answer with a cell like 'F14'.",
            "predict_next": "Now predict one second further: which grid cell will the ball be "
                            "in at 1 second after this clip ends? Answer in the same format.",
        },
        "stimulus": "clip.mov",
        "debug_stimulus_NOT_FOR_MODEL": "debug_full.mov",
    }
    (item_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{item_id}  {level:38s} seed={seed:<4d} freeze_cell={meta['ground_truth']['ball_cell_at_freeze']:>5s}  "
          f"predict_cell={meta['ground_truth']['ball_cell_at_predict']:>5s}")
    return item_id, meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    used_seeds = set()
    for level in TARGET_LEVELS:
        item_id, meta = make_item(level, used_seeds)
        summary.append({"id": item_id, "level": level, "seed": meta["seed"]})
    (OUT / "_index.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(summary)} ball_hidden_grid items -> {OUT}")


if __name__ == "__main__":
    main()
