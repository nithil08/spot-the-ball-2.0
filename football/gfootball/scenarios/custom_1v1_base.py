# coding=utf-8

"""Simple 1v1 scenario for short recording experiments."""

from . import *


def build_scenario(builder):
  # Keep deterministic to make before/after clips comparable.
  builder.config().game_duration = 300
  builder.config().deterministic = True
  builder.config().offsides = False
  builder.config().end_episode_on_score = False
  builder.config().end_episode_on_out_of_play = False
  builder.config().end_episode_on_possession_change = False

  # Ball starts close to the left outfield player.
  builder.SetBallPosition(-0.60, 0.00)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(-0.62, 0.00, e_PlayerRole_CB)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(0.10, 0.00, e_PlayerRole_CB)
