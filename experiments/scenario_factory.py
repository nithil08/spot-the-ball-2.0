"""Programmatically emit gfootball scenario modules.

gfootball loads scenarios by Python module name from gfootball/scenarios/,
so the factory writes a .py file there and returns the level name to use
with the env. This lets us generate hundreds of paired/perturbed scenarios
for stimulus generation without hand-writing each one.
"""

import importlib.util
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


def _resolve_scenarios_dir() -> Path:
    """Find the gfootball.scenarios package directory.

    On Neha's Mac it's in the vendored build under football/build/lib.*/.
    On Windows / fresh clones it's wherever pip dropped gfootball.
    Either way, we need the *actual* import path so the engine can find what
    we write.
    """
    # Delegate vendored-build discovery to lib so we share the "must have a
    # real engine binary" rule.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lib import ENGINE_BUILD  # noqa: E402
    if ENGINE_BUILD is not None:
        d = ENGINE_BUILD / "gfootball/scenarios"
        if d.exists():
            if str(ENGINE_BUILD) not in sys.path:
                sys.path.insert(0, str(ENGINE_BUILD))
            return d
    # Fall back to wherever gfootball.scenarios resolves via the normal path.
    spec = importlib.util.find_spec("gfootball.scenarios")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "Cannot locate gfootball.scenarios. Install via `pip install "
            "gfootball` or vendor a build under football/build/lib.*/."
        )
    return Path(list(spec.submodule_search_locations)[0])


SCENARIOS_DIR = _resolve_scenarios_dir()

# Mirror the role constants — these are the names the gen'd files must use.
ROLES = ("GK", "CB", "LB", "RB", "DM", "CM", "LM", "RM", "AM", "CF")


@dataclass
class PlayerSpec:
    x: float
    y: float
    role: str = "CM"          # one of ROLES
    lazy: bool = False
    controllable: bool = True


@dataclass
class ScenarioSpec:
    name: str                                  # output .py module name
    ball: Tuple[float, float] = (0.0, 0.0)
    left: List[PlayerSpec] = field(default_factory=list)
    right: List[PlayerSpec] = field(default_factory=list)
    game_duration: int = 200
    deterministic: bool = True
    offsides: bool = False
    end_on_score: bool = True
    end_on_out: bool = True
    end_on_possession_change: bool = False
    left_difficulty: float = 0.6
    right_difficulty: float = 0.6


def _validate(spec: ScenarioSpec):
    if not spec.name.replace("_", "").isalnum():
        raise ValueError(f"bad scenario name: {spec.name}")
    for p in spec.left + spec.right:
        if p.role not in ROLES:
            raise ValueError(f"bad role: {p.role}")
        if not (-1.0 <= p.x <= 1.0) or not (-0.42 <= p.y <= 0.42):
            raise ValueError(f"player out of pitch: ({p.x},{p.y})")
    if not (-1.0 <= spec.ball[0] <= 1.0):
        raise ValueError(f"ball out of pitch: {spec.ball}")


def _emit(spec: ScenarioSpec) -> str:
    """Render the scenario as a Python source string."""
    def pl(p: PlayerSpec) -> str:
        flags = []
        if p.lazy:
            flags.append("lazy=True")
        if not p.controllable:
            flags.append("controllable=False")
        flagstr = (", " + ", ".join(flags)) if flags else ""
        return f"  builder.AddPlayer({p.x:.4f}, {p.y:.4f}, e_PlayerRole_{p.role}{flagstr})"

    left_lines = "\n".join(pl(p) for p in spec.left)
    right_lines = "\n".join(pl(p) for p in spec.right)
    lines = [
        "# AUTO-GENERATED. Do not edit by hand.",
        "from . import *",
        "",
        "",
        "def build_scenario(builder):",
        f"  builder.config().game_duration = {spec.game_duration}",
        f"  builder.config().deterministic = {spec.deterministic}",
        f"  builder.config().offsides = {spec.offsides}",
        f"  builder.config().end_episode_on_score = {spec.end_on_score}",
        f"  builder.config().end_episode_on_out_of_play = {spec.end_on_out}",
        f"  builder.config().end_episode_on_possession_change = {spec.end_on_possession_change}",
        f"  builder.config().left_team_difficulty = {spec.left_difficulty}",
        f"  builder.config().right_team_difficulty = {spec.right_difficulty}",
        f"  builder.SetBallPosition({spec.ball[0]:.4f}, {spec.ball[1]:.4f})",
        "",
        "  builder.SetTeam(Team.e_Left)",
        left_lines,
        "",
        "  builder.SetTeam(Team.e_Right)",
        right_lines,
        "",
    ]
    return "\n".join(lines)


def write_scenario(spec: ScenarioSpec, force: bool = False) -> str:
    """Write a scenario module to disk; return its importable level name."""
    _validate(spec)
    out = SCENARIOS_DIR / f"{spec.name}.py"
    if out.exists() and not force:
        # Same name implies same intent (caller should suffix with a hash).
        pass
    out.write_text(_emit(spec))
    return spec.name


# ---------- convenience builders ----------

def base_3v1_attack() -> ScenarioSpec:
    """3 attackers vs 1 defender + GK, ball with the attackers, right-side approach."""
    return ScenarioSpec(
        name="x_base_3v1_attack",
        ball=(0.62, 0.0),
        left=[
            PlayerSpec(-1.0, 0.0, "GK"),
            PlayerSpec(0.60, 0.0, "CM"),
            PlayerSpec(0.70, 0.20, "CM"),
            PlayerSpec(0.70, -0.20, "CM"),
        ],
        right=[
            PlayerSpec(-1.0, 0.0, "GK"),
            PlayerSpec(-0.75, 0.0, "CB"),
        ],
        end_on_score=True,
        end_on_out=True,
        game_duration=200,
    )


def remove_left_player(spec: ScenarioSpec, idx: int, new_name: str) -> ScenarioSpec:
    """Return a copy of spec with the i-th left outfield player removed."""
    s = ScenarioSpec(**{**spec.__dict__, "name": new_name})
    s.left = [p for j, p in enumerate(spec.left) if j != idx]
    return s


def perturb_player(spec: ScenarioSpec, team: str, idx: int,
                   dx: float, dy: float, new_name: str) -> ScenarioSpec:
    """Move a player by (dx, dy) and return a new spec."""
    s = ScenarioSpec(**{**spec.__dict__, "name": new_name})
    src = list(spec.left if team == "left" else spec.right)
    p = src[idx]
    src[idx] = PlayerSpec(p.x + dx, p.y + dy, p.role, p.lazy, p.controllable)
    if team == "left":
        s.left = src
    else:
        s.right = src
    return s


def laze_player(spec: ScenarioSpec, team: str, idx: int, new_name: str) -> ScenarioSpec:
    s = ScenarioSpec(**{**spec.__dict__, "name": new_name})
    src = list(spec.left if team == "left" else spec.right)
    p = src[idx]
    src[idx] = PlayerSpec(p.x, p.y, p.role, lazy=True, controllable=p.controllable)
    if team == "left":
        s.left = src
    else:
        s.right = src
    return s
