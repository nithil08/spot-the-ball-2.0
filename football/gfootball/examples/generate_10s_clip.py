# coding=utf-8
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Record ~10s of simulated match video (first segment after kickoff).

With default settings, each env step produces one video frame at
PHYSICS_STEPS_PER_SECOND / physics_steps_per_frame == 10 fps, so 100 steps
is about 10 seconds of footage.

Example:
  python3 -m gfootball.examples.generate_10s_clip \\
      --output_dir=/tmp/gfootball_clips --level=11_vs_11_easy_stochastic
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from gfootball.env import config as cfg
from gfootball.env import constants as const
from gfootball.env import football_env

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'output_dir', None,
    'Directory for .avi and .dump (must be writable).')
flags.DEFINE_string(
    'level', '11_vs_11_easy_stochastic',
    'Scenario name (see gfootball/scenarios).')
flags.DEFINE_integer(
    'clip_steps', 100,
    'Number of env steps to record (100 @ default fps ≈ 10 s).')
flags.DEFINE_string(
    'dump_name', 'clip_10s',
    'Base name passed to write_dump (output files get a timestamp suffix).')


def _scenario_team_sizes(level_name):
  """Returns (left_controllable, right_controllable) for common levels."""
  scenario_cfg = cfg.Config({'level': level_name}).ScenarioConfig()
  return (scenario_cfg.controllable_left_players,
          scenario_cfg.controllable_right_players)


def main(_):
  if not FLAGS.output_dir:
    raise app.UsageError('--output_dir is required')

  left_n, right_n = _scenario_team_sizes(FLAGS.level)
  if left_n < 1 or right_n < 1:
    logging.warning(
        'Scenario %s reports controllable left=%d right=%d; '
        'both sides should be >= 1 for bot-vs-bot. Trying anyway.',
        FLAGS.level, left_n, right_n)

  # Each bot instance controls exactly one player.
  players = (['bot:left_players=1,right_players=0'] * left_n +
             ['bot:left_players=0,right_players=1'] * right_n)
  values = {
      'level': FLAGS.level,
      'players': players,
      'action_set': 'full',
      'write_video': True,
      'dump_full_episodes': False,
      'dump_scores': False,
      'tracesdir': FLAGS.output_dir,
      'real_time': False,
  }
  c = cfg.Config(values)
  fps = const.PHYSICS_STEPS_PER_SECOND / c['physics_steps_per_frame']
  logging.info('Recording %d steps (~%.1f s of video at %.1f fps).',
               FLAGS.clip_steps, FLAGS.clip_steps / fps, fps)

  env = football_env.FootballEnv(c)
  env.render('rgb_array')  # activate 3D rendering engine before reset
  env.reset()
  done = False
  for i in range(FLAGS.clip_steps):
    if done:
      logging.warning(
          'Episode finished after %d steps; trace only contains frames '
          'since last reset, so the clip may be shorter than requested.',
          i)
      break
    _, _, done, _ = env.step([])

  dump_path = env.write_dump(FLAGS.dump_name)
  env.close()
  if dump_path:
    logging.info('Wrote dump prefix: %s (.avi / .dump next to it)', dump_path)
  else:
    logging.error('write_dump returned nothing; check logs above.')


if __name__ == '__main__':
  app.run(main)
