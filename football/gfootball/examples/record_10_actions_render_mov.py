# coding=utf-8

"""Record rendered GRF clips (10 actions) directly to .mov via ffmpeg.

This bypasses gfootball's built-in cv2 video writer path.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import sys
import time
import types

from absl import app
from absl import flags
from absl import logging

# Avoid importing OpenCV (cv2) at gfootball import time, which can clash with
# SDL used by the rendering engine in some local setups.
if "cv2" not in sys.modules:
  sys.modules["cv2"] = types.SimpleNamespace(FONT_HERSHEY_SIMPLEX=0)

from gfootball.env import config as cfg  # pylint: disable=wrong-import-position
from gfootball.env import football_env  # pylint: disable=wrong-import-position

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", None, "Directory for output .mov files.")
flags.DEFINE_integer("steps", 10, "Number of actions/steps to execute.")


def _first_obs(obs):
  return obs[0] if isinstance(obs, list) else obs


def _encode_mov_from_frames(frames, out_path, fps=10):
  if not frames:
    raise ValueError("No frames to encode")
  h, w, _ = frames[0].shape
  cmd = [
      "ffmpeg",
      "-y",
      "-f",
      "rawvideo",
      "-pix_fmt",
      "rgb24",
      "-s",
      f"{w}x{h}",
      "-r",
      str(fps),
      "-i",
      "-",
      "-c:v",
      "libx264",
      "-pix_fmt",
      "yuv420p",
      "-movflags",
      "+faststart",
      out_path,
  ]
  proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
  try:
    for frame in frames:
      proc.stdin.write(frame.tobytes())
  finally:
    proc.stdin.close()
  rc = proc.wait()
  if rc != 0:
    raise RuntimeError(f"ffmpeg failed with code {rc}")


def _run(level, actions, output_dir, basename):
  values = {
      "level": level,
      "players": [
          "agent:left_players=1,right_players=0",
          "bot:left_players=0,right_players=1",
      ],
      "action_set": "full",
      "write_video": False,
      "dump_full_episodes": False,
      "dump_scores": False,
      "tracesdir": output_dir,
      "real_time": False,
  }
  env = football_env.FootballEnv(cfg.Config(values))
  frames = []
  try:
    env.render()
    obs = env.reset()
    obs0 = _first_obs(obs)
    if "frame" in obs0:
      frames.append(obs0["frame"])
    done = False
    for i, action in enumerate(actions):
      if done:
        logging.warning("Episode ended early at step %d for %s", i, basename)
        break
      obs, _, done, _ = env.step(action)
      o = _first_obs(obs)
      if "frame" in o:
        frames.append(o["frame"])
  finally:
    env.close()

  ts = time.strftime("%Y%m%d-%H%M%S")
  out = os.path.join(output_dir, f"{basename}_{ts}.mov")
  _encode_mov_from_frames(frames, out, fps=10)
  return out


def main(_):
  if not FLAGS.output_dir:
    raise app.UsageError("--output_dir is required")
  os.makedirs(FLAGS.output_dir, exist_ok=True)

  actions_base = [0] * FLAGS.steps
  actions_shifted = [0] * FLAGS.steps
  if FLAGS.steps >= 1:
    actions_shifted[0] = 3   # top
  if FLAGS.steps >= 2:
    actions_shifted[1] = 14  # release_direction

  p1 = _run("custom_1v1_base", actions_base, FLAGS.output_dir, "custom_base_10a_render")
  p2 = _run(
      "custom_1v1_shifted",
      actions_shifted,
      FLAGS.output_dir,
      "custom_shifted_10a_render",
  )
  logging.info("Saved: %s", p1)
  logging.info("Saved: %s", p2)


if __name__ == "__main__":
  app.run(main)
