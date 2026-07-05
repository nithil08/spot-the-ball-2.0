"""Render multiple clips with ONE asset bundle in a single process.

Called as a subprocess by gen_10s_visibility_splits.py so that GFOOTBALL_DATA_DIR
and GFOOTBALL_FONT are locked to the desired bundle for the entire batch, without
interfering with a different bundle already loaded in the parent process.

Usage:
  python render_batch_worker.py <bundle> <clips_json_path>

clips_json_path must contain a JSON array of objects:
  [{"level": "...", "seed": N, "steps": N, "out_npy": "/abs/path/frames.npy"}, ...]

Each clip is rendered sequentially.  Frames are saved as float32 npy arrays of
shape (steps, H, W, 3).  Prints "OK:<clip_index>:<n_frames>" after each clip.
"""

import json
import os
import sys
from pathlib import Path

GFOOTBALL_SRC = Path("/Users/nithilbalamurugan/gfootball_src")
for p in [str(GFOOTBALL_SRC), str(GFOOTBALL_SRC / "third_party")]:
    if p not in sys.path:
        sys.path.insert(0, p)

EXPERIMENTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS))
from lib import use_bundle  # noqa: E402

import numpy as np  # noqa: E402


def main():
    bundle, clips_json_path = sys.argv[1], sys.argv[2]
    clips = json.loads(Path(clips_json_path).read_text())

    use_bundle(bundle)

    # Set GFOOTBALL_FONT to blank font before importing gfootball so
    # gfootball_engine/__init__.py does not override it with the original.
    bundle_data = Path(os.environ.get("GFOOTBALL_DATA_DIR", ""))
    blank_font = bundle_data / "media/fonts/alegreya/AlegreyaSansSC-ExtraBold.ttf"
    if blank_font.exists():
        os.environ["GFOOTBALL_FONT"] = str(blank_font)

    from gfootball.env import config as cfg
    from gfootball.env import football_env

    for i, clip in enumerate(clips):
        level = clip["level"]
        seed = int(clip["seed"])
        steps = int(clip["steps"])
        out_npy = Path(clip["out_npy"])
        out_npy.parent.mkdir(parents=True, exist_ok=True)

        values = {
            "level": level,
            "players": [],
            "action_set": "full",
            "write_video": False,
            "dump_full_episodes": True,
            "dump_scores": False,
            "tracesdir": str(out_npy.parent),
            "real_time": False,
            "game_engine_random_seed": seed,
            "video_quality_level": 2,
            "display_game_stats": False,
        }

        env = football_env.FootballEnv(cfg.Config(values))
        env.render("rgb_array")
        env.reset()

        frames = []
        done = False
        for _ in range(steps):
            if done:
                env.reset()
            _, _, done, _ = env.step([])
            frame = env.render("rgb_array")
            if frame is not None:
                frames.append(np.array(frame))
        env.close()

        np.save(str(out_npy), np.stack(frames))
        print(f"OK:{i}:{len(frames)}", flush=True)


if __name__ == "__main__":
    main()
