"""Generator: DESCRIPTION category.

Static frames at varied team densities. Task: count the players on each
team (or total). Calibration-tier — humans and VLMs should both nail this
when it works. If a model can't count players, every downstream claim is
suspect.

Produces, per item:
  stimuli/description/<id>/frame.png       # static stimulus
  stimuli/description/<id>/meta.json       # ground truth + parameters
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import lib
from lib import use_bundle, make_env, run_clip_capture, frame_to_png, REPO

OUT = REPO / "experiments/stimuli/description"

# Diverse density levels available out of the box.
LEVELS = [
    ("custom_1v1_base", 1, 1),                      # very sparse
    ("academy_3_vs_1_with_keeper", 3, 1),           # sparse, asymmetric
    ("academy_counterattack_easy", 4, 4),           # medium
    ("5_vs_5", 5, 5),                               # medium
    ("11_vs_11_easy_stochastic", 11, 11),           # dense
]

# Multiple seeds per level give us varied player layouts.
SEEDS = [11, 23, 47, 89, 113]
# Sample a few frames per clip to vary the stimulus across time.
SAMPLE_FRAMES = [10, 30, 50, 70]


def make_item(level: str, seed: int, sample_frame: int):
    item_id_input = f"desc|{level}|{seed}|{sample_frame}".encode()
    item_id = hashlib.md5(item_id_input).hexdigest()[:10]
    item_dir = OUT / item_id
    item_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run a short clip just long enough to reach the sample frame.
    work = item_dir / "_work"
    env = make_env(level, seed, work, write_video=False, hud=False)
    info = run_clip_capture(env, steps=max(sample_frame + 5, 30), dump_name=f"d_{item_id}")
    env.close()

    # 2. Extract the stimulus frame from the captured buffer.
    captured = info["frames"]
    target = min(sample_frame, len(captured) - 1)
    frame_png = item_dir / "frame.png"
    frame_to_png(captured[target], frame_png)

    # 3. Record ground truth.
    # Read the dump to count actually-rendered players at the chosen frame.
    dump = sorted(work.glob("*.dump"))[-1]
    obs = None
    for idx, step in lib.read_dump(dump):
        if idx == sample_frame:
            obs = step["observation"]
            break
    left_n = int(obs["left_team"].shape[0]) if obs is not None else None
    right_n = int(obs["right_team"].shape[0]) if obs is not None else None
    total = (left_n + right_n) if (left_n and right_n) else None

    meta = {
        "category": "description",
        "id": item_id,
        "level": level,
        "seed": seed,
        "sample_frame": sample_frame,
        "ground_truth": {
            "n_left": left_n,
            "n_right": right_n,
            "n_total": total,
        },
        "question_variants": [
            "How many players are on the left team (red)?",
            "How many players are on the right team (yellow)?",
            "How many players are in this scene in total?",
        ],
        "answer_format": "MC: 1 / 2 / 3 / 5 / 7 / 11",
        "stimulus": "frame.png",
    }
    (item_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return item_id, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="how many items to generate (test mode = 4)")
    ap.add_argument("--bundle", default="default")
    args = ap.parse_args()

    use_bundle(args.bundle)
    OUT.mkdir(parents=True, exist_ok=True)

    items = []
    for level, _, _ in LEVELS:
        for seed in SEEDS:
            for frame in SAMPLE_FRAMES:
                items.append((level, seed, frame))
    items = items[: args.n]

    summary = []
    for level, seed, frame in items:
        item_id, meta = make_item(level, seed, frame)
        summary.append({"id": item_id, "level": level, "seed": seed})
        print(f"  {item_id}  {level}  seed={seed}  frame={frame}  "
              f"L={meta['ground_truth']['n_left']} R={meta['ground_truth']['n_right']}")

    (OUT / "_index.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(summary)} description items → {OUT}")


if __name__ == "__main__":
    main()
