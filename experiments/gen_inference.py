"""Generator: INFERENCE category.

Three sub-tasks, each produced from the same underlying clip so they share
visual context:

  ball_hidden:     rendered with the `ball_tiny` bundle so the ball is
                    invisible. Task: click where the ball is on the freeze
                    frame, or pick a 6x4 grid bin.
  player_hidden:   one player is occluded with a green disc in post (engine
                    has no per-player hide), using the GT position. Task:
                    where is the missing player?
  region_occluded: black rectangle covers a strip of the pitch. Task: how
                    many players are inside the occluded region?

Stimuli per item:
  stimuli/inference/<id>/ball_hidden.mov
  stimuli/inference/<id>/player_hidden.mov
  stimuli/inference/<id>/region_occluded.mov
  stimuli/inference/<id>/meta.json
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

import lib
from lib import (use_bundle, make_env, run_clip_capture, frames_to_mov, latest_dump,
                 REPO, CLIP_STEPS, FPS, pitch_to_pixel, HUD_TOP_PX, HUD_BOTTOM_PX,
                 FRAME_W, FRAME_H)

OUT = REPO / "experiments/stimuli/inference"

LEVELS = ["academy_3_vs_1_with_keeper", "academy_counterattack_easy", "5_vs_5"]
SEEDS = [3, 11, 23, 47, 89, 113]
# Which sub-task variants to produce per (level,seed).
SUBTASKS = ["ball_hidden", "player_hidden", "region_occluded"]


def render_clip(level: str, seed: int, bundle: str, work_dir: Path):
    """Render a 10 s clip using the chosen bundle; returns (frames_list, dump_path)."""
    use_bundle(bundle)
    env = make_env(level, seed, work_dir, write_video=False, hud=False)
    info = run_clip_capture(env, steps=CLIP_STEPS, dump_name=f"work_{bundle}")
    env.close()
    return info["frames"], latest_dump(work_dir)


def post_occlude_player(frames: list, dump: Path, out_mov: Path,
                        hide_team: str, hide_idx: int, radius: int = 28):
    """Paint a green disc over the chosen player in every frame, write H.264 mov."""
    GRASS = (0, 180, 0)   # RGB grass-green
    processed = []
    for (idx, step), frame in zip(lib.read_dump(dump), frames):
        img = frame.copy()
        obs = step["observation"]
        team = obs[f"{hide_team}_team"]
        if hide_idx < team.shape[0]:
            px, py = pitch_to_pixel(float(team[hide_idx][0]), float(team[hide_idx][1]))
            cv2.circle(img, (px, py), radius, GRASS, -1)
        processed.append(img)
    frames_to_mov(processed, out_mov)


def post_occlude_region(frames: list, dump: Path, out_mov: Path,
                        band_y0: int, band_y1: int):
    """Cover a horizontal strip with black; return per-frame player-in-band counts."""
    in_band_per_frame = []
    processed = []
    for (idx, step), frame in zip(lib.read_dump(dump), frames):
        img = frame.copy()
        h, w = img.shape[:2]
        img[band_y0:band_y1, :] = 0   # black band (RGB, so this is black)
        obs = step["observation"]
        cnt = sum(
            1 for team_key in ("left_team", "right_team")
            for p in obs[team_key]
            if band_y0 <= pitch_to_pixel(float(p[0]), float(p[1]))[1] <= band_y1
        )
        in_band_per_frame.append(cnt)
        processed.append(img)
    frames_to_mov(processed, out_mov)
    return in_band_per_frame


def make_item(level: str, seed: int):
    item_id = hashlib.md5(f"inf|{level}|{seed}".encode()).hexdigest()[:10]
    item_dir = OUT / item_id
    item_dir.mkdir(parents=True, exist_ok=True)

    # 1. ball_hidden — render with the ball_tiny asset bundle.
    work_ball = item_dir / "_work_ball"
    frames_ball, dump_ball = render_clip(level, seed, "ball_tiny", work_ball)
    ball_mov = item_dir / "ball_hidden.mov"
    frames_to_mov(frames_ball, ball_mov)

    # 2. player_hidden + region_occluded — both built from a default render.
    work_def = item_dir / "_work_default"
    frames_def, dump_def = render_clip(level, seed, "default", work_def)

    # Pick the most "interesting" player to hide: the right-team's first
    # non-GK player on academy scenarios (the lone defender), else left CM.
    hide_team, hide_idx = ("right", 1) if "academy" in level else ("left", 1)

    ph_mov = item_dir / "player_hidden.mov"
    post_occlude_player(frames_def, dump_def, ph_mov, hide_team, hide_idx)

    # Region: middle horizontal band of the frame.
    band_y0 = int(FRAME_H * 0.35)
    band_y1 = int(FRAME_H * 0.65)
    ro_mov = item_dir / "region_occluded.mov"
    region_counts = post_occlude_region(frames_def, dump_def, ro_mov, band_y0, band_y1)

    # Ground truth from the final frame.
    last_obs = None
    for idx, step in lib.read_dump(dump_def):
        last_obs = step["observation"]
    ball_xy_final = last_obs["ball"][:2].tolist() if last_obs is not None else None
    hidden_player_xy = (
        last_obs[f"{hide_team}_team"][hide_idx].tolist()
        if last_obs is not None and hide_idx < last_obs[f"{hide_team}_team"].shape[0]
        else None
    )

    meta = {
        "category": "inference",
        "id": item_id,
        "level": level,
        "seed": seed,
        "subtasks": SUBTASKS,
        "ground_truth": {
            "ball_xy_final": ball_xy_final,
            "hidden_player": {
                "team": hide_team, "idx": hide_idx, "xy": hidden_player_xy,
            },
            "region_band_pixels": [band_y0, band_y1],
            "region_player_count_final": region_counts[-1] if region_counts else None,
            "region_player_count_per_frame": region_counts,
        },
        "questions": {
            "ball_hidden": "The ball is rendered invisible. Click where it is now.",
            "player_hidden":
                "One player has been hidden behind grass. Click where they are.",
            "region_occluded":
                "How many players are inside the black band right now?",
        },
        "answer_format": "click coord OR MC bin / count",
        "stimuli": {
            "ball_hidden": "ball_hidden.mov",
            "player_hidden": "player_hidden.mov",
            "region_occluded": "region_occluded.mov",
        },
    }
    (item_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return item_id, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items = [(lvl, s) for lvl in LEVELS for s in SEEDS][: args.n]
    summary = []
    for lvl, s in items:
        iid, meta = make_item(lvl, s)
        summary.append({"id": iid, "level": lvl, "seed": s})
        gt = meta["ground_truth"]
        print(f"  {iid}  {lvl}  seed={s}  ball={gt['ball_xy_final']}  "
              f"hidden={gt['hidden_player']}  band_count={gt['region_player_count_final']}")
    (OUT / "_index.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(summary)} inference items → {OUT}")


if __name__ == "__main__":
    main()
