"""Render exactly N steps with a given asset bundle in an isolated process.

The gfootball C++ engine caches GFOOTBALL_DATA_DIR after the first env is
created in a process.  Rendering two different bundles in the same process
silently reuses whichever bundle loaded first.  Running each bundle in its own
subprocess via this worker avoids that problem.

Usage (called by gen_visibility_variants.py):
  python render_worker_exact.py <level> <seed> <bundle> <steps> <out.npy>

Output: numpy array of shape (steps, H, W, 3) saved to out.npy
"""

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
    level, seed_s, bundle, steps_s, out_path_s = sys.argv[1:6]
    seed = int(seed_s)
    steps = int(steps_s)
    out_path = Path(out_path_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_bundle(bundle)

    # gfootball_engine/__init__.py sets GFOOTBALL_FONT to the original font in
    # the source tree (ignoring GFOOTBALL_DATA_DIR) if the env var is not already
    # set.  We set it here to our blank font BEFORE importing so __init__.py
    # leaves it alone.
    import os
    blank_font = (
        Path(__file__).resolve().parent.parent
        / "asset_bundles" / bundle / "data"
        / "media/fonts/alegreya/AlegreyaSansSC-ExtraBold.ttf"
    )
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
        "tracesdir": str(out_path.parent),
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
    np.save(str(out_path), np.stack(frames))
    print(f"OK:{len(frames)}", flush=True)


if __name__ == "__main__":
    main()
