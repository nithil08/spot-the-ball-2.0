"""One-off stimulus: ball invisible, freeze at 4s, predict 5th-second position.

Renders a clip with the ball scaled to invisible (`ball_tiny` bundle),
plays it for 5 simulated seconds (50 steps @ 10fps), and:
  - saves a freeze-frame PNG at the 4s mark (t=40) — this is what the model sees
  - records ground truth ball xy at t=40 (4s) and t=50 (5s) straight from the
    simulator's own observation dump (i.e. we "watch the clip ourselves" to
    get ground truth, no rendering/vision involved)
  - discretizes both into a 6x4 pitch grid (cols 0-5 left->right, rows 0-3
    top->bottom), same convention as gen_prediction.py's bin_xy()

Usage: python gen_ball_hidden_4s.py
Writes: results/ball_hidden_4s/<item_id>/frame_4s.png + meta.json
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lib
from lib import use_bundle, make_env, run_clip_capture, latest_dump, frame_to_png, FPS

LEVEL = "academy_counterattack_easy"
SEED = 23
FREEZE_STEP = 40   # 4s @ 10fps
PREDICT_STEP = 50  # 5s @ 10fps

OUT = Path(__file__).parent / "results" / "ball_hidden_4s"


def bin_xy(x: float, y: float, nx: int = 6, ny: int = 4) -> list:
    col = max(0, min(nx - 1, int((x + 1.0) / (2.0 / nx))))
    row = max(0, min(ny - 1, int((y + 0.42) / (0.84 / ny))))
    return [col, row]


def main():
    item_id = hashlib.md5(f"ballhidden4s|{LEVEL}|{SEED}".encode()).hexdigest()[:10]
    item_dir = OUT / item_id
    item_dir.mkdir(parents=True, exist_ok=True)
    work = item_dir / "_work"

    use_bundle("ball_tiny")
    env = make_env(LEVEL, SEED, work, write_video=False, hud=False)
    info = run_clip_capture(env, steps=PREDICT_STEP + 1, dump_name="work")
    env.close()

    frames = info["frames"]
    if len(frames) <= FREEZE_STEP:
        raise RuntimeError(f"episode ended early at step {len(frames)}, before freeze point {FREEZE_STEP}")

    freeze_png = item_dir / "frame_4s.png"
    frame_to_png(frames[FREEZE_STEP], freeze_png)

    dump = latest_dump(work)
    ball_at_4s = ball_at_5s = None
    for idx, step in lib.read_dump(dump):
        obs = step["observation"]
        if idx == FREEZE_STEP:
            ball_at_4s = obs["ball"][:2].tolist()
        if idx == PREDICT_STEP:
            ball_at_5s = obs["ball"][:2].tolist()

    if ball_at_4s is None or ball_at_5s is None:
        raise RuntimeError(f"dump ended before step {PREDICT_STEP} (episode likely terminated early)")

    meta = {
        "id": item_id,
        "level": LEVEL,
        "seed": SEED,
        "freeze_step": FREEZE_STEP,
        "predict_step": PREDICT_STEP,
        "grid": "6 cols (0-5, left->right) x 4 rows (0-3, top->bottom)",
        "ground_truth": {
            "ball_xy_at_4s": ball_at_4s,
            "ball_bin_at_4s": bin_xy(*ball_at_4s),
            "ball_xy_at_5s": ball_at_5s,
            "ball_bin_at_5s": bin_xy(*ball_at_5s),
        },
        "stimulus": "frame_4s.png",
        "questions": {
            "locate_now": "The ball has been made invisible in this frame, which is frozen "
                          "4 seconds into the play. Based on player positions, body orientation, "
                          "and motion, where is the ball RIGHT NOW? Answer with a grid cell: "
                          "col=<0-5> row=<0-3>, where col 0 is the far left of the pitch and "
                          "col 5 is the far right, row 0 is the top of the frame and row 3 is "
                          "the bottom.",
            "predict_next": "Now predict one second further: where will the ball be at the "
                            "5-second mark (1 second after this frame)? Answer with the same "
                            "grid format: col=<0-5> row=<0-3>.",
        },
    }
    (item_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"item_id={item_id}")
    print(f"frame -> {freeze_png}")
    print(f"ground truth @4s: xy={ball_at_4s} bin={meta['ground_truth']['ball_bin_at_4s']}")
    print(f"ground truth @5s: xy={ball_at_5s} bin={meta['ground_truth']['ball_bin_at_5s']}")
    return item_dir


if __name__ == "__main__":
    main()
