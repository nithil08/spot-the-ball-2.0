"""drills.py — the 8 small-sided drill definitions (3 variants each = 24 scenarios).

Each drill is a small-sided football setup chosen to produce a DIFFERENT ball-movement
character, which is what makes the "spot the invisible ball" inference easier/harder:

  1  4v4 small-sided     — balanced central open play, natural transitions.
  2  3v1 rondo           — tight keep-away; ball snaps between clustered players (hard).
  3  5v2 rondo           — wider keep-away with occasional splitting passes through middle.
  4  fast break 3v2      — directional counter driving at goal; long FORWARD ball travel.
  5  wing cross          — ball worked wide then delivered across the attacking third;
                           long SIDEWAYS travel WITHOUT driving into the goalmouth (so it
                           circulates instead of scoring — no goals, no kickoff reset).
  6  switch of play      — deliberate cross-pitch switch; maximum lateral displacement.
  7  free kick           — mainstream set piece: dead ball just outside the box worked
                           into open play (authentic delivery; scoring seeds are dropped).
  8  corner kick         — mainstream set piece: ball at the corner flag delivered into
                           the box (authentic delivery; scoring seeds are dropped).

NOTE ON CONTINUOUS PLAY: every clip must be one unbroken passage of play. A goal makes
the engine teleport the ball to the centre spot for a kickoff (the "ball dropped into the
centre" reset). generate_drills.py rejects any seed that scores / resets, so drills whose
geometry sits near goal (4, 5, 7, 8) simply keep only the seeds where the ball circulates.

gfootball convention: BOTH teams are given in the left team's coordinate frame; the
engine mirrors the right team internally (so a right GK at x=-1.0 renders at the far goal).

Each variant applies a small deterministic position jitter so its 3 clips differ.
"""

import sys
from pathlib import Path

# make scenario_factory importable (engine + experiments on path)
SRC = "/Users/nithilbalamurugan/gfootball_src"
# Repo root = the nearest ancestor holding experiments/; keeps working if the
# checkout is moved (it was, from code/spot-the-ball-2.0 to the Desktop root).
EXP = str(next(p for p in Path(__file__).resolve().parents
               if (p / "experiments").is_dir()) / "experiments")
for _p in (SRC, SRC + "/third_party", EXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scenario_factory import ScenarioSpec, PlayerSpec  # noqa: E402


def P(x, y, role="CM"):
    return PlayerSpec(x, y, role)


# ── the 6 base drills (variant 0 layout); returns (ball, left, right) ────────────
def _drill_base(idx):
    if idx == 1:  # 4v4 small-sided — central open play
        ball = (0.0, 0.0)
        left = [P(-1.0, 0.0, "GK"), P(-0.15, 0.10, "CM"), P(-0.15, -0.10, "CM"),
                P(-0.35, 0.0, "CB"), P(0.05, 0.0, "CF")]
        right = [P(-1.0, 0.0, "GK"), P(-0.15, 0.10, "CM"), P(-0.15, -0.10, "CM"),
                 P(-0.35, 0.0, "CB"), P(0.05, 0.0, "CF")]
    elif idx == 2:  # 3v1 rondo — tight keep-away, defender in the middle
        ball = (0.10, 0.0)
        left = [P(-1.0, 0.0, "GK"), P(0.0, 0.16, "CM"), P(0.0, -0.16, "CM"),
                P(0.22, 0.0, "CM")]
        right = [P(-1.0, 0.0, "GK"), P(0.08, 0.0, "CB")]
    elif idx == 3:  # 5v2 rondo — wider ring, two defenders
        ball = (0.14, 0.02)
        left = [P(-1.0, 0.0, "GK"), P(0.0, 0.24, "CM"), P(0.24, 0.14, "CM"),
                P(0.24, -0.14, "CM"), P(0.0, -0.24, "CM"), P(-0.10, 0.0, "CM")]
        right = [P(-1.0, 0.0, "GK"), P(0.06, 0.06, "CB"), P(0.06, -0.06, "CB")]
    elif idx == 4:  # fast break 3v2 to goal — directional counter
        ball = (0.45, 0.0)
        left = [P(-1.0, 0.0, "GK"), P(0.42, 0.0, "CF"), P(0.40, 0.22, "CM"),
                P(0.40, -0.22, "CM")]
        right = [P(-1.0, 0.0, "GK"), P(-0.28, 0.10, "CB"), P(-0.28, -0.10, "CB")]
    elif idx == 5:  # wing cross — ball wide on the flank, delivered ACROSS the
                    # attacking third (targets pulled back off the goalmouth so the
                    # ball circulates sideways instead of being shot into the net).
        ball = (0.28, 0.34)
        left = [P(-1.0, 0.0, "GK"), P(0.30, 0.34, "RM"), P(0.50, 0.08, "CM"),
                P(0.50, -0.12, "CF")]
        right = [P(-1.0, 0.0, "GK"), P(0.60, 0.05, "CB"), P(0.60, -0.10, "CB")]
    elif idx == 6:  # switch of play — spread full width, ball starts far side
        ball = (-0.10, -0.35)
        left = [P(-1.0, 0.0, "GK"), P(-0.10, -0.35, "LM"), P(-0.10, -0.10, "CM"),
                P(-0.10, 0.10, "CM"), P(-0.10, 0.35, "RM"), P(0.18, 0.0, "CF")]
        right = [P(-1.0, 0.0, "GK"), P(-0.05, -0.20, "CM"), P(-0.05, 0.20, "CM"),
                 P(0.0, 0.0, "CB")]
    elif idx == 7:  # free kick — dead ball just outside the box, taker + support,
                    # a defensive wall in front of goal. Authentic attacking free kick.
        ball = (0.55, 0.06)
        left = [P(-1.0, 0.0, "GK"), P(0.53, 0.06, "CF"), P(0.40, 0.22, "CM"),
                P(0.40, -0.20, "CM"), P(0.62, -0.08, "AM")]
        right = [P(-1.0, 0.0, "GK"), P(0.72, 0.10, "CB"), P(0.72, 0.02, "CB"),
                 P(0.72, -0.06, "CB")]
    elif idx == 8:  # corner kick — ball at the corner flag, attackers in the box,
                    # defenders marking. Authentic delivery into the six-yard area.
        ball = (0.98, 0.38)
        left = [P(-1.0, 0.0, "GK"), P(0.97, 0.37, "RM"), P(0.80, 0.06, "CF"),
                P(0.80, -0.08, "CF"), P(0.83, 0.16, "CM")]
        right = [P(-1.0, 0.0, "GK"), P(0.85, 0.05, "CB"), P(0.85, -0.08, "CB"),
                 P(0.88, 0.14, "CB")]
    else:
        raise ValueError(idx)
    return ball, left, right


DRILL_NAMES = {
    1: "4v4_small_sided", 2: "3v1_rondo", 3: "5v2_rondo",
    4: "fast_break_3v2", 5: "wing_cross", 6: "switch_of_play",
    7: "free_kick", 8: "corner_kick",
}

# per-variant jitter (dx, dy) nudges applied to every outfield player + the ball,
# so each of a drill's 3 variants renders a distinct clip.
_JITTER = {0: (0.0, 0.0), 1: (0.06, -0.05), 2: (-0.05, 0.06)}


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def variant_spec(drill_idx, variant_idx):
    """Return a ScenarioSpec for (drill, variant). Deterministic — identical across phases."""
    ball, left, right = _drill_base(drill_idx)
    dx, dy = _JITTER[variant_idx]

    def nudge(players):
        out = []
        for i, p in enumerate(players):
            if p.role == "GK":            # keepers stay on their line
                out.append(PlayerSpec(p.x, p.y, p.role))
            else:
                out.append(PlayerSpec(_clamp(p.x + dx, -0.99, 0.99),
                                      _clamp(p.y + dy, -0.41, 0.41), p.role))
        return out

    bx = _clamp(ball[0] + dx, -0.99, 0.99)
    by = _clamp(ball[1] + dy, -0.41, 0.41)
    name = f"drill{drill_idx}_{DRILL_NAMES[drill_idx]}_v{variant_idx + 1}"
    return ScenarioSpec(
        name=name, ball=(bx, by), left=nudge(left), right=nudge(right),
        game_duration=600, deterministic=True, offsides=False,
        end_on_score=False, end_on_out=False, end_on_possession_change=False,
        left_difficulty=0.6, right_difficulty=0.6,
    )


# iteration order shared by all phases: drill 1..8, variant 1..3
def all_variants():
    for d in range(1, 9):
        for v in range(3):
            yield d, v
