"""Generate 30 clips — 10 seconds long, NO player name labels, ball visible, 32x12 grid.

6 scenarios × 5 seeds = 30 clips.  Same rendering approach as gen_5s_noname_clips.py:
  - Asset bundle:  noname  (blank font so names are invisible, normal ball size)
  - GFOOTBALL_FONT set to blank font BEFORE importing gfootball so the engine
    does not override it with the original source-tree font.
  - 32×12 yellow alphanumeric grid burned on every frame via PIL post-processing.
  - HUD cropped: top 60px and bottom 180px removed → 1280×480 output.
  - Title bar burned at top of every frame.

Output: results/10s_ball_visible_no_names/
"""

import os
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

OUT = Path(__file__).parent / "results" / "10s_ball_visible_no_names"

STEPS = 100   # 10 seconds at 10 FPS

CELL = 40
CROP_W = FRAME_W
CROP_H = FRAME_H - HUD_TOP_PX - HUD_BOTTOM_PX   # 480
COLS = CROP_W // CELL   # 32
ROWS = CROP_H // CELL   # 12

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"

# 6 scenarios × 5 seeds = 30 clips
SCENARIOS = [
    "academy_3_vs_1_with_keeper",
    "academy_counterattack_easy",
    "academy_counterattack_hard",
    "academy_run_pass_and_shoot_with_keeper",
    "academy_pass_and_shoot_with_keeper",
    "academy_run_to_score_with_keeper",
]
SEEDS = [3, 7, 11, 23, 42]

CLIPS = [
    (scenario, seed)
    for scenario in SCENARIOS
    for seed in SEEDS
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


def make_env(level: str, seed: int, out_dir: Path):
    # Set GFOOTBALL_FONT to our blank font before importing gfootball.
    # gfootball_engine/__init__.py sets GFOOTBALL_FONT to the original source-tree
    # font only if the variable is not already set — so we must set it first.
    bundle_path = os.environ.get("GFOOTBALL_DATA_DIR", "")
    blank_font = Path(bundle_path) / "media/fonts/alegreya/AlegreyaSansSC-ExtraBold.ttf"
    if blank_font.exists():
        os.environ["GFOOTBALL_FONT"] = str(blank_font)

    from gfootball.env import config as cfg
    from gfootball.env import football_env

    values = {
        "level": level,
        "players": [],
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
    work_dir = OUT / f"_work_clip_{idx:02d}"
    env = make_env(level, seed, work_dir)
    raw = capture_frames(env, STEPS)
    env.close()

    scenario_short = level.replace("academy_", "").replace("_", " ")
    title = (f"Clip {idx:02d}: {scenario_short}  |  seed={seed}  |  "
             f"10s  |  ball visible  |  no names  |  grid 32x12")

    processed = [burn_title(burn_grid(crop_frame(f)), title) for f in raw]
    out_path = OUT / f"clip_{idx:02d}.mov"
    frames_to_mov(processed, out_path, crop_hud=False)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"  clip_{idx:02d}.mov  {level}  seed={seed}  "
          f"{len(raw)/FPS:.0f}s  {size_mb:.1f} MB")
    return size_mb


def main():
    use_bundle("noname")
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {len(CLIPS)} clips → {OUT}\n")
    results = []
    for i, (level, seed) in enumerate(CLIPS, start=1):
        size_mb = make_clip(i, level, seed)
        results.append((i, level, seed, size_mb))

    # Write a README alongside the clips
    readme = OUT / "README.txt"
    lines = [
        "10s_ball_visible_no_names — CLIP SET README",
        "============================================",
        f"Generated: 2026-07-03",
        "Script:    experiments/nithil_work/gen_10s_noname_clips.py",
        "",
        "WHAT IS IN THIS FOLDER",
        "-----------------------",
        "30 video clips of Google Research Football gameplay.",
        "6 scenarios × 5 random seeds = 30 clips total.",
        "",
        "CLIP SPECS",
        "----------",
        "  Duration:        10 seconds each (100 simulation steps at 10 FPS)",
        "  Resolution:      1280 × 480 pixels (60px top HUD + 180px bottom HUD cropped out)",
        "  Ball:            VISIBLE at full normal size",
        "  Player names:    REMOVED (see 'How names are removed' below)",
        "  Grid overlay:    YES — 32 columns × 12 rows yellow alphanumeric grid",
        "                   burned into every frame. Cols 1–32 left→right, rows A–L top→bottom.",
        "  Title bar:       Black bar at top: clip number, scenario, seed, duration, config",
        "",
        "SCENARIOS USED",
        "--------------",
        "  academy_3_vs_1_with_keeper",
        "  academy_counterattack_easy",
        "  academy_counterattack_hard",
        "  academy_run_pass_and_shoot_with_keeper",
        "  academy_pass_and_shoot_with_keeper",
        "  academy_run_to_score_with_keeper",
        "",
        "SEEDS USED",
        "----------",
        "  3, 7, 11, 23, 42",
        "",
        "CLIP INDEX",
        "----------",
        "  Clip#  Scenario                                    Seed   Size",
    ]
    for idx, level, seed, size_mb in results:
        lines.append(f"  {idx:02d}     {level:<44} {seed:<6} {size_mb:.1f} MB")

    lines += [
        "",
        "HOW PLAYER NAMES ARE REMOVED",
        "-----------------------------",
        "The gfootball C++ engine shows the name of whoever has the ball using a TTF",
        "font from GFOOTBALL_FONT env var.  By default gfootball's __init__.py sets",
        "this to the original font in the source tree.",
        "",
        "Our fix:",
        "  1. Built 'noname' asset bundle with a modified font (fontTools) where every",
        "     character maps to the space glyph — space has no visible ink.",
        "  2. Set GFOOTBALL_FONT to this blank font BEFORE importing gfootball, so",
        "     __init__.py's 'if not already set' check leaves it alone.",
        "  3. GFOOTBALL_DATA_DIR → noname bundle (normal ball size, all other assets default).",
        "",
        "Blank font: experiments/asset_bundles/noname/data/media/fonts/alegreya/AlegreyaSansSC-ExtraBold.ttf",
        "",
        "HOW TO REGENERATE",
        "-----------------",
        "  cd experiments/nithil_work",
        "  python3 gen_10s_noname_clips.py",
        "",
        "Prerequisites: gfootball at /Users/.../gfootball_src, ffmpeg, Pillow, numpy.",
        "The noname asset bundle must exist (run experiments/build_noname_bundles.py first).",
    ]

    readme.write_text("\n".join(lines) + "\n")
    print(f"\nREADME written -> {readme}")

    print(f"\ndone -> {OUT}")

    sys.path.insert(0, str(Path(__file__).parent))
    from log_entry import log
    bullets = [
        f"clip_{idx:02d}: {lv}, seed={sd}, 10s, ball visible, no names, 32x12 grid, {sm:.1f} MB"
        for idx, lv, sd, sm in results
    ]
    log(
        "Generated 30 × 10s no-name grid clips (gen_10s_noname_clips.py)",
        f"6 scenarios × 5 seeds = 30 clips total",
        *bullets,
        "Output: experiments/nithil_work/results/10s_ball_visible_no_names/",
        "Bundle: noname | GFOOTBALL_FONT: blank font (all chars -> space glyph) | Steps: 100 | FPS: 10",
    )


if __name__ == "__main__":
    main()
