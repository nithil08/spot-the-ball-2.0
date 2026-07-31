"""gen_natural_passing_clips.py — 2 natural 5-second passing clips, 3 grid variants each.

Brief (from user):
  * Continuous play — NO goal scoring. Just passing, players moving, some chasing
    a moving ball (not everyone dribbling).
  * Natural / normal framing — do NOT try to keep all 22 players on frame.
    Uses the engine's standard ball-tracking camera (shows part of the pitch).
  * Ball visible throughout (noname bundle = default ball, no player-name captions).
  * No player names.
  * For EACH clip, produce 3 grid variants, each labelled:
        1) 32 x 12   2) 16 x 6   3) 28 x 8

Match: 11_vs_11_stochastic, seed 42 (verified goal-free across 450 steps, score 0-0).
Two moderate-movement windows are captured from ONE continuous playthrough so the
clips are natural mid-match passing, not frantic counterattacks and not static.

Every output is exactly 1280x480, 50 frames, 10 fps -> 5.0 s.  Output:
  results/natural_passing_clips/
    clip1_base.mov  clip1_grid_32x12.mov  clip1_grid_16x6.mov  clip1_grid_28x8.mov
    clip2_base.mov  clip2_grid_32x12.mov  clip2_grid_16x6.mov  clip2_grid_28x8.mov
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

OUT   = Path(__file__).parent / "results" / "natural_passing_clips"
SEED  = 42
LEVEL = "11_vs_11_stochastic"
W, H  = 1280, 480          # every clip is exactly this
STEPS = 50                 # 50 frames @ 10 fps = 5.0 s

# Two moderate-movement, goal-free windows picked from the probe scan.
# (warmup steps before each 50-frame capture, measured from a single reset)
CLIP_WINDOWS = [
    ("clip1", 30),         # settled midfield possession / passing
    ("clip2", 110),        # ball travels laterally, some chasing
]

# Grid variants: (label, cols, rows)
GRID_VARIANTS = [
    ("32x12", 32, 12),
    ("16x6",  16,  6),
    ("28x8",  28,  8),
    ("8x3",    8,  3),
]

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"


def _font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


# ── Environment ──────────────────────────────────────────────────────────────

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


def _crop_hud(frame):
    return frame[HUD_TOP_PX: FRAME_H - HUD_BOTTOM_PX, :]


def capture_windows(env):
    """Play ONE continuous match and grab both windows in order.

    Guarantees no goal occurs (seed 42 stays 0-0) and no reset mid-window, so the
    captured play is natural and matches the probe windows exactly.
    """
    env.reset()
    clips = {}
    step = 0
    for name, warmup in CLIP_WINDOWS:
        # advance to the window start (no capture)
        while step < warmup:
            obs, reward, done, _ = env.step([])
            env.render("rgb_array")
            assert not done and float(np.sum(np.asarray(reward))) == 0.0, \
                f"unexpected goal/reset at step {step}"
            step += 1
        # capture STEPS frames
        frames = []
        for _ in range(STEPS):
            obs, reward, done, _ = env.step([])
            frame = env.render("rgb_array")
            assert not done and float(np.sum(np.asarray(reward))) == 0.0, \
                f"unexpected goal/reset at step {step}"
            if frame is not None:
                frames.append(_crop_hud(np.array(frame)))
            step += 1
        clips[name] = frames
        print(f"  {name}: captured {len(frames)} frames (steps {warmup}-{warmup+STEPS})")
    return clips


# ── Grid overlays ────────────────────────────────────────────────────────────

def build_grid(cols: int, rows: int, label: str) -> Image.Image:
    """Uniform cols x rows yellow alphanumeric grid + a corner variant label."""
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line = (255, 255, 0, 140)
    cell_w = W / cols
    cell_h = H / rows

    for c in range(cols + 1):
        x = int(round(c * cell_w))
        draw.line([(x, 0), (x, H)], fill=line, width=1)
    for r in range(rows + 1):
        y = int(round(r * cell_h))
        draw.line([(0, y), (W, y)], fill=line, width=1)

    cell_font = _font(10)
    for c in range(cols):
        draw.text((int(round(c * cell_w)) + 2, 1),
                  str(c + 1), fill=(255, 255, 0, 255), font=cell_font)
    for r in range(rows):
        # letters wrap past Z (AA, AB, ...) for grids with >26 rows (none here, but safe)
        letter = chr(ord('A') + r) if r < 26 else "A" + chr(ord('A') + r - 26)
        draw.text((1, int(round(r * cell_h)) + 1),
                  letter, fill=(255, 255, 0, 255), font=cell_font)

    # Variant label, bottom-right, on a translucent plate so it stays readable.
    tag = f"grid {label}"
    tag_font = _font(20)
    tb = draw.textbbox((0, 0), tag, font=tag_font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    pad = 6
    x0, y0 = W - tw - 2 * pad - 4, H - th - 2 * pad - 4
    draw.rectangle([x0, y0, x0 + tw + 2 * pad, y0 + th + 2 * pad],
                   fill=(0, 0, 0, 170))
    draw.text((x0 + pad, y0 + pad - tb[1]), tag,
              fill=(255, 255, 0, 255), font=tag_font)
    return overlay


def burn(frame_rgb: np.ndarray, overlay: Image.Image) -> np.ndarray:
    base = Image.fromarray(frame_rgb).convert("RGBA")
    return np.array(Image.alpha_composite(base, overlay).convert("RGB"))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    use_bundle("noname")          # default ball (visible) + no player-name captions
    OUT.mkdir(parents=True, exist_ok=True)

    env = make_env(OUT / "_work")
    print("Capturing two natural passing windows from one goal-free match…")
    clips = capture_windows(env)
    env.close()

    overlays = {label: build_grid(cols, rows, label)
                for label, cols, rows in GRID_VARIANTS}

    written = []
    for name, _ in CLIP_WINDOWS:
        frames = clips[name]

        # base (no grid)
        p = OUT / f"{name}_base.mov"
        frames_to_mov(frames, p, crop_hud=False)
        written.append(p)

        # 3 grid variants
        for label, _cols, _rows in GRID_VARIANTS:
            ov = overlays[label]
            gframes = [burn(f, ov) for f in frames]
            p = OUT / f"{name}_grid_{label}.mov"
            frames_to_mov(gframes, p, crop_hud=False)
            written.append(p)

    print(f"\nWrote {len(written)} clips -> {OUT}")
    for p in written:
        h, w = H, W
        print(f"  {p.name:28s} {p.stat().st_size/1e6:5.2f} MB  ({w}x{h}, {STEPS/FPS:.1f}s)")


if __name__ == "__main__":
    main()
