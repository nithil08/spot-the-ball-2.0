# coding=utf-8

"""Record two short clips (10 actions each) for before/after comparison.

Variant A: baseline scenario with idle actions.
Variant B: shifted player scenario with an initial directional action sequence
to "rotate/turn" movement direction.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from gfootball.env import config as cfg
from gfootball.env import football_env

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "output_dir",
    None,
    "Directory for output .avi/.dump files.")
flags.DEFINE_integer(
    "steps",
    10,
    "Number of actions/steps to execute for each clip.")
flags.DEFINE_bool(
    "render",
    True,
    "Enable full engine rendering (human players) in saved video.")


def _build_env(level, output_dir):
  values = {
      "level": level,
      "players": [
          # One externally controlled left player.
          "agent:left_players=1,right_players=0",
          # One bot on the right side.
          "bot:left_players=0,right_players=1",
      ],
      "action_set": "full",
      "write_video": True,
      "dump_full_episodes": False,
      "dump_scores": False,
      "tracesdir": output_dir,
      "real_time": False,
  }
  return football_env.FootballEnv(cfg.Config(values))


def _run_clip(level, actions, output_dir, dump_name):
  env = _build_env(level, output_dir)
  try:
    if FLAGS.render:
      # Must be called before reset() to enable frame capture from the engine.
      env.render()
    env.reset()
    done = False
    for i in range(len(actions)):
      if done:
        logging.warning("Episode ended early at step %d for %s", i, dump_name)
        break
      _, _, done, _ = env.step(actions[i])
    dump_path = env.write_dump(dump_name)
    logging.info("Wrote %s prefix: %s", dump_name, dump_path)
    return dump_path
  finally:
    env.close()


def main(_):
  if not FLAGS.output_dir:
    raise app.UsageError("--output_dir is required")

  # full action set indices:
  # 0 idle, 3 top, 14 release_direction
  actions_base = [0] * FLAGS.steps

  actions_shifted = [0] * FLAGS.steps
  if FLAGS.steps >= 1:
    actions_shifted[0] = 3   # top (sticky directional action)
  if FLAGS.steps >= 2:
    actions_shifted[1] = 14  # release_direction

  base_prefix = _run_clip(
      "custom_1v1_base", actions_base, FLAGS.output_dir, "custom_base_10a")
  shifted_prefix = _run_clip(
      "custom_1v1_shifted", actions_shifted, FLAGS.output_dir, "custom_shifted_10a")

  logging.info("Done.")
  logging.info("Base prefix: %s", base_prefix)
  logging.info("Shifted prefix: %s", shifted_prefix)


if __name__ == "__main__":
  app.run(main)
