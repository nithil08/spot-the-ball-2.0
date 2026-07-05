"""Generate 4 plain 20-second gameplay clips: ball visible, grid overlaid.

Uses the default asset bundle so the ball is fully visible.
Burns the same 32x12 alphanumeric yellow grid used in gen_ball_hidden_grid_video.py.

Output: results/plain_grid_clips/clip_<N>.mov  (HUD-cropped, H.264)
"""

import sys
from pathlib import Path

GFOOTBALL_SRC = Path("/Users/nithilbalamurugan/gfootball_src")
for p in [str(GFOOTBALL_SRC), str(GFOOTBALL_SRC / "third_party")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import (use_bundle, make_env, run_clip_capture, frames_to_mov,
                 HUD_TOP_PX, HUD_BOTTOM_PX, FRAME_W, FRAME_H, FPS)

OUT = Path(__file__).parent / "results" / "plain_grid_clips"

STEPS = 200  # 20 seconds at 10 fps

CELL = 40
CROP_W = FRAME_W                              # 1280
CROP_H = FRAME_H - HUD_TOP_PX - HUD_BOTTOM_PX  # 480
COLS = CROP_W // CELL                         # 32
ROWS = CROP_H // CELL                         # 12

CLIPS = [
    ("academy_3_vs_1_with_keeper",          23),
    ("academy_counterattack_easy",         331),
    ("academy_counterattack_hard",           3),
    ("academy_run_pass_and_shoot_with_keeper", 3),
]


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
    composited = Image.alpha_composite(base, GRID_OVERLAY)
    return np.array(composited.convert("RGB"))


def capture_frames(env, total_steps: int) -> list:
    """Collect exactly total_steps frames, resetting if an episode ends early."""
    env.reset()
    frames = []
    done = False
    for _ in range(total_steps):
        if done:
            env.reset()
        _, _, done, _ = env.step([])
        frame = env.render("rgb_array")
        if frame is not None:
            frames.append(np.array(frame))
    return frames


def make_clip(idx: int, level: str, seed: int):
    work_dir = OUT / f"_work_clip_{idx}"
    env = make_env(level, seed, work_dir, write_video=False, hud=False)
    frames = capture_frames(env, STEPS)
    env.close()
    print(f"  captured {len(frames)} frames ({len(frames)/FPS:.1f}s)")
    processed = [burn_grid(crop_frame(f)) for f in frames]
    out_path = OUT / f"clip_{idx}.mov"
    frames_to_mov(processed, out_path, crop_hud=False)
    print(f"  -> {out_path}")
    return out_path


def main():
    use_bundle("default")
    OUT.mkdir(parents=True, exist_ok=True)
    generated = []
    for i, (level, seed) in enumerate(CLIPS, start=1):
        print(f"\nclip {i}/4: {level}  seed={seed}")
        out_path = make_clip(i, level, seed)
        size_mb = out_path.stat().st_size / 1_048_576
        generated.append(f"clip_{i}.mov: {level}, seed={seed}, {STEPS/FPS:.0f}s, ball visible, 32×12 grid, {size_mb:.1f} MB")
    print(f"\ndone -> {OUT}")

    from log_entry import log
    log(
        "Generated plain 20s grid clips (gen_plain_grid_clips.py)",
        *generated,
        f"Output: {OUT.relative_to(OUT.parent.parent.parent.parent)}",
        "Ball: visible (default bundle) | Grid: 32×12 yellow burned on every frame | HUD: off",
    )


if __name__ == "__main__":
    main()
