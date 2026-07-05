"""Generator: PREDICTION category.

10 s clip is rendered; we cut the visible portion early (5 s or 7 s) and
ask the model/human to predict what happens in the remaining 3-5 s.

Two prediction sub-tasks per item:
  (a) where will the ball be at the end of the clip (grid bin)?
  (b) will either team score within the full 10 s window?

Stimuli per item:
  stimuli/prediction/<id>/clip.mov     # visible portion (5 s or 7 s)
  stimuli/prediction/<id>/full.mov     # full 10 s, for our debugging only
  stimuli/prediction/<id>/meta.json    # ground-truth ball xy at cutoff + endpoint + scored
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import lib
from lib import use_bundle, make_env, run_clip_capture, frames_to_mov, latest_dump, REPO, CLIP_STEPS, FPS

OUT = REPO / "experiments/stimuli/prediction"

# Scenarios where action is dense and outcomes (goal / no-goal) are uncertain.
LEVELS = [
    "academy_3_vs_1_with_keeper",
    "academy_counterattack_easy",
    "academy_counterattack_hard",
    "academy_pass_and_shoot_with_keeper",
    "academy_run_pass_and_shoot_with_keeper",
]
SEEDS = [3, 11, 23, 47, 89, 113, 191, 257]
CUTOFFS_S = [5, 7]  # show first 5 s or first 7 s, predict the remaining 3-5 s


def bin_xy(x: float, y: float, nx: int = 6, ny: int = 4) -> tuple:
    """Discretize pitch coords into a 6x4 grid for MC answers."""
    col = max(0, min(nx - 1, int((x + 1.0) / (2.0 / nx))))
    row = max(0, min(ny - 1, int((y + 0.42) / (0.84 / ny))))
    return col, row


def trim_clip(src: Path, dst: Path, duration_s: float):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(src), "-t", f"{duration_s}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(dst)],
        check=True,
    )


def make_item(level: str, seed: int, cutoff_s: int):
    item_id = hashlib.md5(f"pred|{level}|{seed}|{cutoff_s}".encode()).hexdigest()[:10]
    item_dir = OUT / item_id
    item_dir.mkdir(parents=True, exist_ok=True)
    work = item_dir / "_work"
    env = make_env(level, seed, work, write_video=False, hud=False)
    info = run_clip_capture(env, steps=CLIP_STEPS, dump_name=f"p_{item_id}")
    env.close()

    dump = latest_dump(work)
    frames = info["frames"]
    # Full transcoded version (for debugging — not shown to subjects).
    full_mov = item_dir / "full.mov"
    frames_to_mov(frames, full_mov)
    # Visible portion: first cutoff_s seconds.
    clip_mov = item_dir / "clip.mov"
    trim_clip(full_mov, clip_mov, duration_s=cutoff_s)

    # Ground truth: ball at cutoff frame and at final frame, did anyone score.
    # Episodes may end early (end_on_score etc) — fall back to last available.
    cutoff_frame = cutoff_s * FPS
    final_frame = CLIP_STEPS - 1
    ball_at_cutoff = None
    ball_at_end = None
    ever_scored = (0, 0)
    last_idx = 0
    for idx, step in lib.read_dump(dump):
        obs = step["observation"]
        if idx <= cutoff_frame:
            ball_at_cutoff = obs["ball"].tolist()
        ball_at_end = obs["ball"].tolist()    # tracks the latest seen
        ever_scored = tuple(obs["score"])
        last_idx = idx
    actual_end_frame = last_idx

    meta = {
        "category": "prediction",
        "id": item_id,
        "level": level,
        "seed": seed,
        "cutoff_s": cutoff_s,
        "clip_s": CLIP_STEPS / FPS,
        "ground_truth": {
            "ball_at_cutoff_xy": ball_at_cutoff,
            "ball_at_end_xy": ball_at_end,
            "ball_end_bin_xy": bin_xy(*ball_at_end[:2]) if ball_at_end else None,
            "left_scored": int(ever_scored[0]) > 0,
            "right_scored": int(ever_scored[1]) > 0,
            "actual_end_frame": actual_end_frame,
            "ended_early": actual_end_frame < final_frame,
        },
        "question_variants": [
            f"Where will the ball be {CLIP_STEPS / FPS - cutoff_s:.0f}s after the clip ends?",
            "Will the team in possession score within the next 5 seconds?",
            "Which team will be in possession when the play ends?",
        ],
        "answer_format": "ball: 6x4 grid bin (24 options) or click. score: yes/no.",
        "stimulus": "clip.mov",
    }
    (item_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return item_id, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--bundle", default="default")
    args = ap.parse_args()

    use_bundle(args.bundle)
    OUT.mkdir(parents=True, exist_ok=True)
    items = [(lvl, s, c) for lvl in LEVELS for s in SEEDS for c in CUTOFFS_S]
    items = items[: args.n]

    summary = []
    for lvl, s, c in items:
        iid, meta = make_item(lvl, s, c)
        gt = meta["ground_truth"]
        summary.append({"id": iid, "level": lvl, "seed": s, "cutoff": c})
        print(f"  {iid}  {lvl}  seed={s}  cut={c}s  "
              f"end_bin={gt['ball_end_bin_xy']}  L={gt['left_scored']} R={gt['right_scored']}")

    (OUT / "_index.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(summary)} prediction items → {OUT}")


if __name__ == "__main__":
    main()
