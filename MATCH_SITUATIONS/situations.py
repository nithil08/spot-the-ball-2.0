"""situations.py — the 11v11 match scenarios + real-football event detectors.

Everything in MATCH_SITUATIONS is built on ONE idea: do NOT fake a set piece by
dropping the ball near a corner flag and hoping. Instead run a real deterministic
11 v 11 match and wait for the ENGINE'S OWN REFEREE to award the set piece, then cut
the clip around it. That is what makes the corner / free kick behave like the real
thing, because the engine already implements the real laws:

  * Corner (Law 17): referee.cpp awards e_GameMode_Corner when the ball fully crosses
    the goal line last touched by a DEFENDER. Ball is re-spotted on the corner arc,
    play stops ~2 s, teamAIcontroller.cpp::PrepareSetPiece(e_GameMode_Corner) walks the
    attackers into the box and the defenders onto their markers, then the whistle goes
    and the taker delivers it. Goals can be scored directly; there is no offside.
  * Free kick (Law 13): referee.cpp::CheckFoul awards e_GameMode_FreeKick at the foul
    spot (or a penalty inside the box). PrepareSetPiece(e_GameMode_FreeKick) pulls the
    defending side back into a wall in front of the ball while the attacking side pushes
    up, then the taker strikes it. Offside (offsides=True) also produces a real
    indirect free kick, exactly as in the laws.
  * Throw-in (Law 15) uses the engine's genuine two-handed over-the-head throw anim
    (media/animations/{pass,highpass}/*/special/*_throw.anim).
  * Goalkeeper distribution (Law 12): when the keeper CATCHES the ball he enters the
    engine's "holdball" retain state (deflect/*_holdball.anim -> outgoing_retain_state
    = right_elbow). Every pass he then plays is rendered with the throw animation
    (highpass/idle/special/000_throw.anim requires incoming_retain_state right_elbow),
    i.e. a real overarm throw-out with the HANDS, not a kick.
  * Header: a high ball (ball z above head height) that changes direction at a player
    is a header — the engine has e_FunctionType_Header and picks the head-contact anims.

gfootball convention: BOTH teams are given in the left team's coordinate frame; the
engine mirrors the right team internally.
"""

import sys
from pathlib import Path

SRC = "/Users/nithilbalamurugan/gfootball_src"
EXP = "/Users/nithilbalamurugan/Desktop/Nithil Research/code/spot-the-ball-2.0/experiments"
for _p in (SRC, SRC + "/third_party", EXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scenario_factory import ScenarioSpec, PlayerSpec  # noqa: E402

# game_mode enum (engine src/defines.hpp)
GM_NORMAL, GM_KICKOFF, GM_GOALKICK, GM_FREEKICK, GM_CORNER, GM_THROWIN, GM_PENALTY = range(7)
GM_NAME = {GM_NORMAL: "normal", GM_KICKOFF: "kickoff", GM_GOALKICK: "goalkick",
           GM_FREEKICK: "freekick", GM_CORNER: "corner", GM_THROWIN: "throwin",
           GM_PENALTY: "penalty"}


def P(x, y, role="CM"):
    return PlayerSpec(x, y, role)


# ── the 11 v 11 shape (4-3-3-ish, same skeleton the stock 11_vs_11 scenario uses) ──
# Index order matters: it is the AddPlayer order, which is what GFOOTBALL_HIDE_SLOTS
# addresses as L0..L10 / R0..R10.
def eleven(push_up=0.0):
    return [
        P(-1.000, 0.000, "GK"),      # 0
        P(-0.010 + push_up, 0.020, "RM"),      # 1
        P(-0.010 + push_up, -0.020, "CF"),     # 2
        P(-0.422 + push_up, -0.196, "LB"),     # 3
        P(-0.500 + push_up, -0.064, "CB"),     # 4
        P(-0.500 + push_up, 0.064, "CB"),      # 5
        P(-0.422 + push_up, 0.196, "RB"),      # 6
        P(-0.184 + push_up, -0.106, "CM"),     # 7
        P(-0.268 + push_up, 0.000, "CM"),      # 8
        P(-0.184 + push_up, 0.106, "CM"),      # 9
        P(-0.010 + push_up, -0.216, "LM"),     # 10
    ]


def match_spec(name, ball=(0.0, 0.0), offsides=True, difficulty=(0.75, 0.75),
               push_left=0.0, push_right=0.0, duration=3000):
    """A full, honest 11 v 11 match: 22 players, real referee, nothing ends the episode
    early so the referee's set pieces actually get played out on camera."""
    return ScenarioSpec(
        name=name, ball=ball,
        left=eleven(push_left), right=eleven(push_right),
        game_duration=duration, deterministic=True, offsides=offsides,
        end_on_score=False, end_on_out=False, end_on_possession_change=False,
        left_difficulty=difficulty[0], right_difficulty=difficulty[1],
    )


# ── the match shapes we sweep ────────────────────────────────────────────────────
# `deterministic=True` makes game_engine_random_seed irrelevant — every seed replays the
# identical match — so variety has to come from the SCENARIO SHAPE, not from seeds.

def variants_general():
    """Balanced matches: the everyday source of headers and midfield play."""
    i = 0
    for bx, by in [(0.0, 0.0), (0.35, 0.10), (0.55, -0.15), (-0.30, 0.25),
                   (0.70, 0.05), (0.20, -0.30)]:
        for pl, pr in [(0.0, 0.0), (0.35, 0.15), (0.55, 0.30), (0.20, 0.40)]:
            for diff in [(0.8, 0.8), (1.0, 0.6)]:
                i += 1
                name = f"ms_v{i:02d}"
                yield name, match_spec(name, ball=(bx, by), offsides=True,
                                       difficulty=diff, push_left=pl, push_right=pr)


def variants_pressure():
    """A strong side camped on a weak side's box. Outmatched defenders concede corners
    (deflections behind) and commit fouls, which is where set pieces come from.
    Offsides are OFF here so the referee's free kicks are fouls, not offside awards."""
    i = 0
    for bx, by in [(0.55, 0.0), (0.70, 0.12), (0.62, -0.18), (0.80, 0.05),
                   (0.45, 0.20), (0.75, -0.05)]:
        for pl in [0.45, 0.60, 0.75]:
            for diff in [(1.0, 0.2), (1.0, 0.05), (0.9, 0.35)]:
                i += 1
                name = f"pr_v{i:02d}"
                yield name, match_spec(name, ball=(bx, by), offsides=False,
                                       difficulty=diff, push_left=pl, push_right=0.35)


def variants_wide():
    """Ball out on the flank in the final third: the situation that actually produces
    corners (a cross or a shot deflected behind by a defender) and crosses to head."""
    i = 0
    for bx in [0.62, 0.75, 0.85]:
        for by in [0.34, -0.34, 0.26, -0.26]:
            for pl, pr in [(0.55, 0.30), (0.70, 0.45), (0.40, 0.20)]:
                for diff in [(1.0, 0.4), (0.85, 0.85)]:
                    i += 1
                    name = f"wd_v{i:02d}"
                    yield name, match_spec(name, ball=(bx, by), offsides=False,
                                           difficulty=diff, push_left=pl, push_right=pr)


def all_variants():
    yield from variants_general()
    yield from variants_pressure()
    yield from variants_wide()


# ── detectors, run over the per-step observation log of one episode ──────────────

def find_mode_events(log, mode):
    """Frame indices where game_mode first becomes `mode` (i.e. the award)."""
    out = []
    prev = GM_NORMAL
    for i, s in enumerate(log):
        if s["game_mode"] == mode and prev != mode:
            out.append(i)
        prev = s["game_mode"]
    return out


def find_gk_throws(log, hold_min=3, travel_min=0.25):
    """Goalkeeper catches the ball in his hands, holds it, then releases it a long way.

    Returns (catch_frame, release_frame). `hold_min` frames of uninterrupted keeper
    possession is what puts him in the engine's holdball retain state; the release is
    then rendered as the overarm throw animation.
    """
    import numpy as np
    out = []
    i = 0
    n = len(log)
    while i < n:
        s = log[i]
        if s["ball_owned_player"] == 0 and s["ball_owned_team"] in (0, 1):
            team = s["ball_owned_team"]
            j = i
            while (j < n and log[j]["ball_owned_player"] == 0
                   and log[j]["ball_owned_team"] == team):
                j += 1
            held = j - i
            if held >= hold_min and j < n:
                # how far does the ball travel in the 12 frames after release?
                k = min(n - 1, j + 12)
                travel = float(np.linalg.norm(
                    np.array(log[k]["ball"][:2]) - np.array(log[j - 1]["ball"][:2])))
                if travel >= travel_min:
                    out.append((i, j - 1, held, travel))
            i = j
        else:
            i += 1
    return out


def find_headers(log, z_min=1.25, turn_max=0.35, near=0.035):
    """A high ball that changes direction next to a player = a header.

    z_min ~1.25 m keeps it above chest height; turn_max is the cosine of the
    incoming/outgoing direction, so a genuine redirection (not a bounce-through);
    `near` requires a player within ~3.5% of pitch length of the contact point.
    """
    import numpy as np
    out = []
    for i in range(2, len(log) - 2):
        z = log[i]["ball"][2]
        if z < z_min:
            continue
        p0 = np.array(log[i - 2]["ball"][:2])
        p1 = np.array(log[i]["ball"][:2])
        p2 = np.array(log[i + 2]["ball"][:2])
        v_in, v_out = p1 - p0, p2 - p1
        n_in, n_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
        if n_in < 0.02 or n_out < 0.02:
            continue
        cos = float(np.dot(v_in, v_out) / (n_in * n_out))
        if cos > turn_max:
            continue
        players = np.vstack([log[i]["left_team"], log[i]["right_team"]])
        dist = float(np.min(np.linalg.norm(players - p1, axis=1)))
        if dist > near:
            continue
        out.append((i, z, cos, dist))
    return out
