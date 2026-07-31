"""gen_match_clips.py — 5-second 11v11 match, 3 camera views + 29×9 grid variants.

Generates 6 clips in results/match_clips/ (ALL 1280×480, no resizing):

  og_5s.mov              standard render, camera tracks ball, first 5 s of match
  full_field_5s.mov      same style but 10–15 s into match (ball in play, full pitch)
  half_field_5s.mov      right-half crop of full_field frames, 2× uniform scale → 1280×480
  og_5s_grid.mov         og_5s  + 29 cols × 9 rows yellow grid
  full_field_grid.mov    full_field + grid
  half_field_grid.mov    half_field + grid

Scenario: 11_vs_11_stochastic  (proper match, not an academy drill)
Seed: 42  |  FPS: 10
"""

import os
import sys
from pathlib import Path

GFOOTBALL_SRC = Path("/Users/nithilbalamurugan/gfootball_src")
for _p in [str(GFOOTBALL_SRC), str(GFOOTBALL_SRC / "third_party")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EXPERIMENTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS))
from lib import use_bundle, frames_to_mov, HUD_TOP_PX, HUD_BOTTOM_PX, FRAME_H, FPS

OUT    = Path(__file__).parent / "results" / "match_clips"
SEED   = 42
LEVEL  = "11_vs_11_stochastic"
W, H   = 1280, 480      # every clip is exactly this — never changes

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"

# 29 columns × 9 rows on 1280×480
COLS, ROWS = 29, 9
CELL_W = W / COLS   # 44.138 px
CELL_H = H / ROWS   # 53.333 px


def _font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


# ── Environment ────────────────────────────────────────────────────────────────

def make_env(work_dir: Path):
    bundle = os.environ.get("GFOOTBALL_DATA_DIR", "")
    blank = Path(bundle) / "media/fonts/alegreya/AlegreyaSansSC-ExtraBold.ttf"
    if blank.exists():
        os.environ["GFOOTBALL_FONT"] = str(blank)

    from gfootball.env import config as cfg, football_env
    values = {
        "level": LEVEL, "players": [], "action_set": "full",
        "write_video": False, "dump_full_episodes": False,
        "dump_scores": False, "tracesdir": str(work_dir),
        "real_time": False, "game_engine_random_seed": SEED,
        "video_quality_level": 2, "display_game_stats": False,
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    env = football_env.FootballEnv(cfg.Config(values))
    env.render("rgb_array")
    return env


def capture(env, warmup: int, steps: int):
    """Warm up for `warmup` steps, then capture `steps` HUD-cropped frames."""
    env.reset()
    done = False

    # Warm-up: advance match without capturing
    for _ in range(warmup):
        if done:
            env.reset()
        _, _, done, _ = env.step([])
        env.render("rgb_array")   # keep renderer in sync

    # Capture
    raw_frames = []
    for _ in range(steps):
        if done:
            env.reset()
        _, _, done, _ = env.step([])
        frame = env.render("rgb_array")
        if frame is not None:
            raw_frames.append(np.array(frame))

    return [f[HUD_TOP_PX: FRAME_H - HUD_BOTTOM_PX, :] for f in raw_frames]


# ── Half-field: crop right half → 2× uniform scale → center-crop to H ─────────

def make_half_field(frames):
    """
    Take the right half of each frame (640×480), scale 2× uniformly (1280×960),
    then take the centre 480 rows → 1280×480.  No horizontal stretching.
    """
    out = []
    for f in frames:
        right = f[:, W // 2:, :]                            # 640×480
        big   = np.array(Image.fromarray(right)
                         .resize((W, H * 2), Image.LANCZOS))  # 1280×960
        crop  = big[H // 2: H // 2 + H, :, :]              # centre 480 rows
        out.append(crop)
    return out


# ── 29×9 grid overlay ──────────────────────────────────────────────────────────

def _build_grid() -> Image.Image:
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    font    = _font(10)
    line    = (255, 255, 0, 140)

    for c in range(COLS + 1):
        x = int(round(c * CELL_W))
        draw.line([(x, 0), (x, H)], fill=line, width=1)
    for r in range(ROWS + 1):
        y = int(round(r * CELL_H))
        draw.line([(0, y), (W, y)], fill=line, width=1)

    for c in range(COLS):
        draw.text((int(round(c * CELL_W)) + 2, 1),
                  str(c + 1), fill=(255, 255, 0, 255), font=font)
    for r in range(ROWS):
        draw.text((1, int(round(r * CELL_H)) + 1),
                  chr(ord('A') + r), fill=(255, 255, 0, 255), font=font)
    return overlay


GRID = _build_grid()


def burn_grid(frame_rgb: np.ndarray) -> np.ndarray:
    base = Image.fromarray(frame_rgb).convert("RGBA")
    return np.array(Image.alpha_composite(base, GRID).convert("RGB"))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    use_bundle("noname")
    OUT.mkdir(parents=True, exist_ok=True)

    env = make_env(OUT / "_work")

    # 1 — OG: first 5 s from kick-off (0 warm-up steps)
    print("Capturing og_5s (steps 1–50, kick-off)…")
    og_frames = capture(env, warmup=0, steps=50)
    print(f"  {len(og_frames)} frames")

    # 2 — Full-field: steps 101–150 (ball in play, natural wide view)
    print("Capturing full_field_5s (steps 101–150)…")
    ff_frames = capture(env, warmup=100, steps=50)
    print(f"  {len(ff_frames)} frames")

    env.close()

    # 3 — Half-field: crop right half of full-field, uniform 2× scale, centre-crop height
    print("Building half_field_5s…")
    hf_frames = make_half_field(ff_frames)

    # Write base clips
    base = [
        ("og_5s.mov",         og_frames),
        ("full_field_5s.mov", ff_frames),
        ("half_field_5s.mov", hf_frames),
    ]
    # Write grid variants
    grid_clips = [
        ("og_5s_grid.mov",        [burn_grid(f) for f in og_frames]),
        ("full_field_grid.mov",   [burn_grid(f) for f in ff_frames]),
        ("half_field_grid.mov",   [burn_grid(f) for f in hf_frames]),
    ]

    for i, (name, frames) in enumerate(base + grid_clips, start=1):
        p = OUT / name
        frames_to_mov(frames, p, crop_hud=False)
        h, w = frames[0].shape[:2]
        print(f"{i}/6  {name}  {p.stat().st_size / 1e6:.1f} MB  ({w}×{h})")

    print(f"\ndone → {OUT}")


if __name__ == "__main__":
    main()
