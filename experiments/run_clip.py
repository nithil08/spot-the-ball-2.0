"""Render a short clip with a chosen asset bundle.

Usage:
  GFOOTBALL_DATA_DIR=<bundle>/data python run_clip.py \
      --bundle <name> --level <scenario> --steps <N> --out <dir>
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENGINE_BUILD = REPO / "football/build/lib.macosx-26.0-arm64-cpython-314"
sys.path.insert(0, str(ENGINE_BUILD))

from absl import app, flags, logging  # noqa: E402

flags.DEFINE_string("bundle", "default", "Asset bundle name under experiments/asset_bundles")
flags.DEFINE_string("level", "academy_3_vs_1_with_keeper", "Scenario name")
flags.DEFINE_integer("steps", 100, "env steps to record (~10s @ 10fps)")
flags.DEFINE_string("out", str(REPO / "experiments/clips"), "output dir")
flags.DEFINE_integer("seed", 42, "engine seed for repeatability across bundles")
FLAGS = flags.FLAGS


def main(_):
    bundle_data = REPO / "experiments/asset_bundles" / FLAGS.bundle / "data"
    if not bundle_data.exists():
        sys.exit(f"bundle not found: {bundle_data}")
    os.environ["GFOOTBALL_DATA_DIR"] = str(bundle_data)

    out_dir = Path(FLAGS.out) / FLAGS.bundle
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import AFTER setting GFOOTBALL_DATA_DIR.
    from gfootball.env import config as cfg
    from gfootball.env import football_env

    # Bot-vs-bot so we have action without writing a policy.
    from gfootball.env import scenario_builder
    scn = scenario_builder.Scenario(cfg.Config({"level": FLAGS.level}))
    left_n = scn.config().controllable_left_players
    right_n = scn.config().controllable_right_players
    players = (["bot:left_players=1,right_players=0"] * left_n
               + ["bot:left_players=0,right_players=1"] * right_n)

    c = cfg.Config({
        "level": FLAGS.level,
        "players": players,
        "action_set": "full",
        "write_video": True,
        "dump_full_episodes": True,
        "dump_scores": False,
        "tracesdir": str(out_dir),
        "real_time": False,
        "game_engine_random_seed": FLAGS.seed,
        "video_quality_level": 2,
    })

    env = football_env.FootballEnv(c)
    env.render("rgb_array")
    env.reset()
    done = False
    dump_prefix = None
    for i in range(FLAGS.steps):
        if done:
            logging.info("episode done at step %d (dump auto-written)", i)
            break
        _, _, done, _ = env.step([])
    if not done:
        dump_prefix = env.write_dump(f"clip_{FLAGS.bundle}")
    env.close()
    print("OK", FLAGS.bundle, "wrote files in", out_dir)


if __name__ == "__main__":
    app.run(main)
