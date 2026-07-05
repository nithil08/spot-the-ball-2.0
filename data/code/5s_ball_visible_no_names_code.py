"""Generate 4 clips — 5 seconds long, NO player name labels, ball visible, 32x12 grid.

These are the basis clips for the visibility-split experiment (Task 2).

Details per clip:
  - Duration:  5 seconds (50 steps @ 10 fps)
  - Ball:      visible (default asset bundle)
  - Grid:      32 cols x 12 rows yellow alphanumeric grid, burned on every frame
  - Names:     OFF  (zero controllable player slots -> C++ AI drives all players)
  - HUD:       OFF
  - Title:     burned at top of every frame

Output: results/5s_noname_clips/clip_<N>.mov
"""

import sys
from pathlib import Path

GFOOTBALL_SRC = Path("/Users/nithilbalamurugan/gfootball_src")
for p in [str(GFOOTBALL_SRC), str(GFOOTBALL_SRC / "third_party")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EXPERIMENTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS))
from lib import (use_bundle, frames_to_mov,
                 HUD_TOP_PX, HUD_BOTTOM_PX, FRAME_W, FRAME_H, FPS)

OUT = Path(__file__).parent / "results" / "5s_noname_clips"

STEPS = 50   # 5 seconds at 10 fps

CELL = 40
CROP_W = FRAME_W
CROP_H = FRAME_H - HUD_TOP_PX - HUD_BOTTOM_PX   # 480
COLS = CROP_W // CELL   # 32
ROWS = CROP_H // CELL   # 12

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"

CLIPS = [
    ("academy_3_vs_1_with_keeper",             23),
    ("academy_counterattack_easy",            331),
    ("academy_counterattack_hard",              3),
    ("academy_run_pass_and_shoot_with_keeper",  3),
]


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


def burn_title(frame_rgb: np.ndarray, title: str) -> np.ndarray:
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 24], fill=(0, 0, 0))
    draw.text((6, 5), title, fill=(255, 255, 255), font=_font(13))
    return np.array(img)


def make_env_noname(level: str, seed: int, out_dir: Path):
    """Env with zero externally-controllable player slots -> no name labels."""
    import os
    # gfootball_engine/__init__.py overrides GFOOTBALL_FONT with the original
    # source-tree font if the variable isn't already set.  Set it to our blank
    # font first so the engine renders all text as invisible.
    bundle = os.environ.get("GFOOTBALL_DATA_DIR", "")
    blank_font = Path(bundle) / "media/fonts/alegreya/AlegreyaSansSC-ExtraBold.ttf"
    if blank_font.exists():
        os.environ["GFOOTBALL_FONT"] = str(blank_font)

    from gfootball.env import config as cfg
    from gfootball.env import football_env
    values = {
        "level": level,
        "players": [],          # no external slots = no player name overlays
        "action_set": "full",
        "write_video": False,
        "dump_full_episodes": True,
        "dump_scores": False,
        "tracesdir": str(out_dir),
        "real_time": False,
        "game_engine_random_seed": seed,
        "video_quality_level": 2,
        "display_game_stats": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    env = football_env.FootballEnv(cfg.Config(values))
    env.render("rgb_array")
    return env


def capture_frames(env, total_steps: int) -> list:
    """Capture frames, auto-resetting if an episode ends early."""
    env.reset()
    frames, done = [], False
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
    env = make_env_noname(level, seed, work_dir)
    raw = capture_frames(env, STEPS)
    env.close()

    scenario_short = level.replace("academy_", "").replace("_", " ")
    title = f"Clip {idx}: {scenario_short}  |  5s  |  ball visible  |  grid 32x12  |  no player names"

    processed = [burn_title(burn_grid(crop_frame(f)), title) for f in raw]
    out_path = OUT / f"clip_{idx}.mov"
    frames_to_mov(processed, out_path, crop_hud=False)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"  clip_{idx}.mov  {len(raw)/FPS:.1f}s  {size_mb:.1f} MB  -> {out_path.name}")
    return out_path, size_mb


def main():
    use_bundle("noname")
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for i, (level, seed) in enumerate(CLIPS, start=1):
        print(f"\nclip {i}/4: {level}  seed={seed}")
        out_path, size_mb = make_clip(i, level, seed)
        results.append((level, seed, size_mb))

    print(f"\ndone -> {OUT}")

    sys.path.insert(0, str(Path(__file__).parent))
    from log_entry import log
    bullets = [
        f"clip_{i}: {lv}, seed={sd}, 5s, ball visible, 32x12 grid, no player names, {sm:.1f} MB"
        for i, (lv, sd, sm) in enumerate(results, start=1)
    ]
    log("Generated 5s no-name grid clips (gen_5s_noname_clips.py)", *bullets,
        "Output: experiments/nithil_work/results/5s_noname_clips/",
        "Bundle: default | Steps: 50 | FPS: 10 | Grid: 32x12 | Names: OFF | HUD: OFF")


if __name__ == "__main__":
    main()
