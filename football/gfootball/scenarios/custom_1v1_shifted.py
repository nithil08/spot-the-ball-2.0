# coding=utf-8

"""Same as custom_1v1_base, but one left player is moved."""

from . import *


def build_scenario(builder):
  builder.config().game_duration = 300
  builder.config().deterministic = True
  builder.config().offsides = False
  builder.config().end_episode_on_score = False
  builder.config().end_episode_on_out_of_play = False
  builder.config().end_episode_on_possession_change = False

  # Keep ball in the same region as the shifted player.
  builder.SetBallPosition(-0.35, 0.20)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  # Shifted from (-0.62, 0.00) to demonstrate position editing.
  builder.AddPlayer(-0.35, 0.20, e_PlayerRole_CB)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(0.10, 0.00, e_PlayerRole_CB)
