"""Generate 3 visibility-split variants for each of the 30 × 10s clips.

For each clip the ball is visible for the first N seconds, then shrunk to 5%
(near-invisible) for the remaining (10-N) seconds.  Variants per clip:
  3s_vis_7s_hid.mov  — ball visible 3s, hidden 7s  (hard)
  5s_vis_5s_hid.mov  — ball visible 5s, hidden 5s  (medium)
  8s_vis_2s_hid.mov  — ball visible 8s, hidden 2s  (easy)

Purpose: test whether giving the model more time watching the ball before it
disappears helps it predict the ball's location.

Total output: 30 clips × 3 variants = 90 .mov files

Method:
  - Two renders per clip with identical seed → frame-perfect physics alignment
  - Visible render: noname bundle (normal ball, no names)
  - Hidden render:  noname_ball_invisible bundle (ball at 5%, no names)
  - Visible renders run in the current process (one bundle throughout)
  - Hidden renders run in ONE subprocess batch (different bundle, isolated to
    avoid GFOOTBALL_DATA_DIR caching in the same process)
  - Frames spliced in Python then encoded via ffmpeg

Output: results/10s_visibility_splits/clip_<NN>/<Nv>s_vis_<Nh>s_hid.mov
"""

import json
import os
import subprocess
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

OUT = Path(__file__).parent / "results" / "10s_visibility_splits"
WORK = OUT / "_work"
WORKER = Path(__file__).parent / "render_batch_worker.py"

STEPS = 100  # 10 seconds at 10 FPS

CELL = 40
CROP_W = FRAME_W
CROP_H = FRAME_H - HUD_TOP_PX - HUD_BOTTOM_PX  # 480
COLS = CROP_W // CELL   # 32
ROWS = CROP_H // CELL   # 12

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"

# 3 visibility splits — (visible_seconds, hidden_seconds)
VARIANTS = [(3, 7), (5, 5), (8, 2)]

# Same 30 clips as gen_10s_noname_clips.py
SCENARIOS = [
    "academy_3_vs_1_with_keeper",
    "academy_counterattack_easy",
    "academy_counterattack_hard",
    "academy_run_pass_and_shoot_with_keeper",
    "academy_pass_and_shoot_with_keeper",
    "academy_run_to_score_with_keeper",
]
SEEDS = [3, 7, 11, 23, 42]
CLIPS = [(i + 1, sc, sd) for i, (sc, sd) in enumerate(
    (sc, sd) for sc in SCENARIOS for sd in SEEDS
)]


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
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    bar_h = 40 if subtitle else 24
    draw.rectangle([0, 0, img.width, bar_h], fill=(0, 0, 0))
    draw.text((6, 4), title, fill=(255, 255, 255), font=_font(13))
    if subtitle:
        draw.text((6, 22), subtitle, fill=(180, 180, 180), font=_font(10))
    return np.array(img)


def render_visible_batch() -> dict:
    """Render all 30 clips with noname (ball visible) in current process.
    Returns {clip_idx: np.ndarray of shape (STEPS, H, W, 3)}.
    """
    use_bundle("noname")

    bundle_path = os.environ.get("GFOOTBALL_DATA_DIR", "")
    blank_font = Path(bundle_path) / "media/fonts/alegreya/AlegreyaSansSC-ExtraBold.ttf"
    if blank_font.exists():
        os.environ["GFOOTBALL_FONT"] = str(blank_font)

    from gfootball.env import config as cfg
    from gfootball.env import football_env

    results = {}
    for clip_idx, level, seed in CLIPS:
        work_dir = WORK / f"vis_{clip_idx:02d}"
        work_dir.mkdir(parents=True, exist_ok=True)
        values = {
            "level": level, "players": [], "action_set": "full",
            "write_video": False, "dump_full_episodes": True, "dump_scores": False,
            "tracesdir": str(work_dir), "real_time": False,
            "game_engine_random_seed": seed, "video_quality_level": 2,
            "display_game_stats": False,
        }
        env = football_env.FootballEnv(cfg.Config(values))
        env.render("rgb_array")
        env.reset()
        frames, done = [], False
        for _ in range(STEPS):
            if done:
                env.reset()
            _, _, done, _ = env.step([])
            frame = env.render("rgb_array")
            if frame is not None:
                frames.append(np.array(frame))
        env.close()
        results[clip_idx] = np.stack(frames)
        print(f"  [vis] clip_{clip_idx:02d}  {level}  seed={seed}  {len(frames)} frames",
              flush=True)
    return results


def render_hidden_batch() -> dict:
    """Render all 30 clips with noname_ball_invisible in a fresh subprocess batch.
    Returns {clip_idx: np.ndarray of shape (STEPS, H, W, 3)}.
    """
    clips_spec = []
    for clip_idx, level, seed in CLIPS:
        out_npy = WORK / f"hid_{clip_idx:02d}" / "frames.npy"
        clips_spec.append({
            "level": level, "seed": seed, "steps": STEPS,
            "out_npy": str(out_npy)
        })
    clips_json = WORK / "hidden_clips.json"
    clips_json.write_text(json.dumps(clips_spec, indent=2))

    print(f"\n  Launching hidden-render subprocess for all 30 clips...", flush=True)
    result = subprocess.run(
        [sys.executable, str(WORKER), "noname_ball_invisible", str(clips_json)],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if line.startswith("OK:"):
            parts = line.split(":")
            idx_in_list = int(parts[1])
            clip_idx = CLIPS[idx_in_list][0]
            print(f"  [hid] clip_{clip_idx:02d} done ({parts[2]} frames)", flush=True)
    if result.returncode != 0:
        print(f"  Worker stderr:\n{result.stderr[-1000:]}")
        raise RuntimeError(f"Hidden render worker failed (exit {result.returncode})")

    frames_dict = {}
    for clip_idx, level, seed in CLIPS:
        out_npy = WORK / f"hid_{clip_idx:02d}" / "frames.npy"
        frames_dict[clip_idx] = np.load(str(out_npy))
    return frames_dict


def make_variants(clip_idx: int, level: str, seed: int,
                  frames_vis: np.ndarray, frames_hid: np.ndarray) -> list:
    """Splice and encode 3 variants for one clip. Returns list of (vis_s, hid_s, size_mb)."""
    clip_dir = OUT / f"clip_{clip_idx:02d}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    scenario_short = level.replace("academy_", "").replace("_", " ")
    results = []
    for vis_s, hid_s in VARIANTS:
        vis_n = vis_s * FPS
        hid_n = hid_s * FPS
        raw = np.concatenate([frames_vis[:vis_n], frames_hid[vis_n: vis_n + hid_n]], axis=0)

        title = f"Clip {clip_idx:02d}: VISIBLE {vis_s}s → HIDDEN {hid_s}s  (10s total)"
        subtitle = f"{scenario_short}  |  seed={seed}  |  32×12 grid  |  no names"

        processed = [
            burn_title(burn_grid(crop_frame(raw[i])), title, subtitle)
            for i in range(len(raw))
        ]
        out_path = clip_dir / f"{vis_s}s_vis_{hid_s}s_hid.mov"
        frames_to_mov(processed, out_path, crop_hud=False)
        size_mb = out_path.stat().st_size / 1_048_576
        results.append((vis_s, hid_s, size_mb))
    return results


def write_readme(all_results: list):
    readme = OUT / "README.txt"
    lines = [
        "10s_visibility_splits — CLIP SET README",
        "=========================================",
        "Generated: 2026-07-03",
        "Script:    experiments/nithil_work/gen_10s_visibility_splits.py",
        "",
        "WHAT IS IN THIS FOLDER",
        "-----------------------",
        "90 video clips: 3 visibility-split variants for each of the 30 × 10s base clips.",
        "Each sub-folder (clip_01/ through clip_30/) holds the 3 variants for that clip.",
        "",
        "PURPOSE",
        "-------",
        "Test whether giving a model more time watching the ball before it disappears",
        "helps it predict where the ball went.  The 3 variants per clip form a",
        "difficulty gradient from hard (3s visible) to easy (8s visible).",
        "",
        "VARIANTS PER CLIP",
        "-----------------",
        "  3s_vis_7s_hid.mov   Ball visible 3s, then hidden 7s  (hard — short look)",
        "  5s_vis_5s_hid.mov   Ball visible 5s, then hidden 5s  (medium)",
        "  8s_vis_2s_hid.mov   Ball visible 8s, then hidden 2s  (easy — long look)",
        "",
        "CLIP SPECS",
        "----------",
        "  Total duration:  10 seconds per clip",
        "  Resolution:      1280 × 480 px (HUD cropped)",
        "  Grid:            32 × 12 yellow alphanumeric overlay, every frame",
        "  Player names:    REMOVED (blank font via GFOOTBALL_FONT env var)",
        "  Title bar:       Shows clip number, split, scenario, seed",
        "",
        "WHAT 'HIDDEN' MEANS",
        "--------------------",
        "The ball is NOT removed — it still affects gameplay physics.",
        "It is made nearly invisible by loading an asset bundle where the ball's",
        "3D mesh is scaled to 5% of its original size (~3-4 pixels wide).",
        "Bundle: experiments/asset_bundles/noname_ball_invisible/",
        "",
        "SCENARIOS AND SEEDS",
        "-------------------",
        "  Scenarios: academy_3_vs_1_with_keeper, academy_counterattack_easy,",
        "             academy_counterattack_hard, academy_run_pass_and_shoot_with_keeper,",
        "             academy_pass_and_shoot_with_keeper, academy_run_to_score_with_keeper",
        "  Seeds: 3, 7, 11, 23, 42",
        "  Same as experiments/nithil_work/results/10s_ball_visible_no_names/",
        "",
        "CLIP INDEX",
        "----------",
        "  Clip   Scenario                                    Seed",
    ]
    for clip_idx, level, seed, variants in all_results:
        lines.append(f"  {clip_idx:02d}     {level:<44} {seed}")
    lines += [
        "",
        "HOW TO REGENERATE",
        "-----------------",
        "  cd experiments/nithil_work",
        "  python3 gen_10s_visibility_splits.py",
        "",
        "Requires: noname and noname_ball_invisible asset bundles,",
        "          ffmpeg, Pillow, numpy, gfootball.",
        "Build bundles first: python3 experiments/build_noname_bundles.py",
    ]
    readme.write_text("\n".join(lines) + "\n")
    print(f"\nREADME written -> {readme}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Step 1/3: Rendering visible frames (noname bundle, 30 clips) ===")
    vis_frames = render_visible_batch()

    print(f"\n=== Step 2/3: Rendering hidden frames (noname_ball_invisible, 30 clips) ===")
    hid_frames = render_hidden_batch()

    print(f"\n=== Step 3/3: Splicing and encoding {len(CLIPS) * len(VARIANTS)} variants ===")
    all_results = []
    for clip_idx, level, seed in CLIPS:
        variant_results = make_variants(
            clip_idx, level, seed, vis_frames[clip_idx], hid_frames[clip_idx]
        )
        for vis_s, hid_s, size_mb in variant_results:
            print(f"  clip_{clip_idx:02d}/{vis_s}s_vis_{hid_s}s_hid.mov  {size_mb:.1f} MB")
        all_results.append((clip_idx, level, seed, variant_results))

    write_readme(all_results)
    print(f"\ndone -> {OUT}")

    sys.path.insert(0, str(Path(__file__).parent))
    from log_entry import log
    log(
        "Generated 90 × 10s visibility-split clips (gen_10s_visibility_splits.py)",
        "30 base clips × 3 variants each (3s/5s/8s visible, rest hidden)",
        "Variants: 3s_vis_7s_hid, 5s_vis_5s_hid, 8s_vis_2s_hid per clip",
        "Purpose: test whether more time watching ball improves model predictions",
        "Output: experiments/nithil_work/results/10s_visibility_splits/clip_01/ ... clip_30/",
        "Bundles: noname (visible) + noname_ball_invisible (hidden, 5% ball scale)",
    )


if __name__ == "__main__":
    main()
