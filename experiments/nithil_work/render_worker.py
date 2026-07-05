"""Render exactly one (level, seed, bundle) clip in its own fresh process.

The gfootball C++ engine reads GFOOTBALL_DATA_DIR once, the first time an
env is constructed in a process, and ignores later changes -- so rendering
two different asset bundles (e.g. "default" then "ball_tiny") in the same
Python process silently reuses whichever bundle loaded first. Isolating
each bundle in its own process (invoked via subprocess) sidesteps that.

No fps/step-count math: `real_time=True` makes the engine self-throttle to
actual wall-clock speed, and we just let it run for a genuine <duration_s>
seconds, recording whatever frames come out at whatever native rate that
is (measured: ~8.4fps for a 5s window, ~42 frames -- it varies slightly
run to run, which is expected).

Usage: python render_worker.py <level> <seed> <bundle> <duration_s> <out_npy>
Writes a .npy array of shape (n_frames, H, W, 3) uint8, plus a sibling
<out_npy>.fps.txt containing the native fps (n_frames / actual_elapsed_s)
to encode at.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import use_bundle


def make_env_real_time(level: str, seed: int, out_dir: Path):
    from gfootball.env import config as cfg
    from gfootball.env import football_env

    values = {
        "level": level,
        "players": [
            "bot:left_players=1,right_players=0",
            "bot:left_players=0,right_players=1",
        ],
        "action_set": "full",
        "write_video": False,
        "dump_full_episodes": True,
        "dump_scores": False,
        "tracesdir": str(out_dir),
        "real_time": True,
        "game_engine_random_seed": seed,
        "video_quality_level": 2,
        "display_game_stats": False,
    }
    c = cfg.Config(values)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = football_env.FootballEnv(c)
    env.render("rgb_array")
    return env


def main():
    level, seed, bundle, duration_s, out_path = sys.argv[1:6]
    seed = int(seed)
    duration_s = float(duration_s)
    out_path = Path(out_path)

    use_bundle(bundle)
    work_dir = out_path.parent / f"_work_{out_path.stem}"
    env = make_env_real_time(level, seed, work_dir)
    env.reset()
    frames = []
    done = False
    start = time.time()
    while time.time() - start < duration_s:
        if done:
            break
        _, _, done, _ = env.step([])
        frame = env.render("rgb_array")
        if frame is not None:
            frames.append(np.array(frame))
    elapsed = time.time() - start
    env.close()

    if len(frames) < 2:
        print(f"SHORT:{len(frames)}")
        sys.exit(1)
    np.save(out_path, np.stack(frames))
    native_fps = len(frames) / elapsed
    out_path.with_suffix(".fps.txt").write_text(str(native_fps))
    print(f"OK:{len(frames)}:{elapsed:.3f}:{native_fps:.4f}")


if __name__ == "__main__":
    main()
