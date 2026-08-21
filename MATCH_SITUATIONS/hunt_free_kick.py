"""hunt_free_kick.py — find a real ATTACKING free kick (the iconic Law 13 situation).

The general sweep only ever produced free kicks that get hoofed clear, for a reason worth
recording: with offsides on, most of the engine's free kicks are OFFSIDE awards, and an
offside free kick is by definition given to the DEFENDING side deep in their own half —
so it always gets launched upfield. The dangerous free kick (ball 18-35 yds out, defenders
dropped in front of it, taker striking at goal) only comes from a FOUL committed by a
defender on an attacker in the final third.

So this sweep is tuned for exactly that:
  * offsides OFF, so every free kick in the log is a foul, not an offside;
  * a strong attacking side against a weak defending side, so play camps in the final
    third and the outmatched defenders keep having to tackle;
  * both lines pushed up so the contact happens near the box.

Run:  python3 hunt_free_kick.py    ->  _events/free_kicks.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = "/Users/nithilbalamurugan/gfootball_src"
# Repo root = the nearest ancestor holding experiments/; keeps working if the
# checkout is moved (it was, from code/spot-the-ball-2.0 to the Desktop root).
EXP = str(next(p for p in Path(__file__).resolve().parents
               if (p / "experiments").is_dir()) / "experiments")
for _p in (SRC, SRC + "/third_party", EXP, str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from situations import match_spec, find_mode_events, GM_FREEKICK  # noqa: E402
from scan_events import run_log, SEED  # noqa: E402
from pick_situations import has_reset, near_count  # noqa: E402

STEPS = 900


def fk_variants():
    i = 0
    for bx, by in [(0.55, 0.0), (0.70, 0.12), (0.62, -0.18), (0.80, 0.05),
                   (0.45, 0.20), (0.75, -0.05)]:
        for pl in [0.45, 0.60, 0.75]:
            for diff in [(1.0, 0.2), (1.0, 0.05), (0.9, 0.35)]:
                i += 1
                name = f"fk_v{i:02d}"
                # offsides OFF => every free kick below is a foul, never an offside
                yield name, match_spec(name, ball=(bx, by), offsides=False,
                                       difficulty=diff, push_left=pl, push_right=0.35)


def classify(log, f, window=80):
    """Attacking free kick = delivered towards the goal nearest the ball, ending in/near
    the box, from 18-35 yds out — and with play continuing (no goal/kickoff reset)."""
    import numpy as np
    if has_reset(log, f + 1, f + window):
        return None
    spot = np.array(log[f]["ball"][:2])
    end = min(len(log) - 1, f + 35)
    seg = log[f:end]
    apex = max(s["ball"][2] for s in seg)
    xs = [s["ball"][0] for s in seg]
    goal_side = 1.0 if spot[0] >= 0 else -1.0
    towards = (max(xs) - spot[0]) if goal_side > 0 else (spot[0] - min(xs))
    reach = max(abs(x) for x in xs)
    dist_to_goal = 1.0 - abs(spot[0])
    attacking = towards > 0.04 and reach > 0.88
    in_range = 0.06 <= dist_to_goal <= 0.34
    central = abs(spot[1]) < 0.22
    contest = max((near_count(log[i]) for i in range(f + 8, min(end, f + 30))), default=0)
    return {"frame": f, "spot": [round(float(v), 3) for v in spot],
            "dist_to_goal": round(dist_to_goal, 3), "apex": round(apex, 2),
            "attacking": bool(attacking), "in_range": bool(in_range),
            "central": bool(central), "contested_by": contest,
            "score": (80 if attacking else 0) + (30 if in_range else 0)
                     + (15 if central else 0) + contest * 5 + apex}


def main():
    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle("noname")
    found = []
    for name, spec in fk_variants():
        write_scenario(spec, force=True)
        log = run_log(name, SEED, STEPS)
        fks = find_mode_events(log, GM_FREEKICK)
        for f in fks:
            r = classify(log, f)
            if r:
                found.append({"level": name, **r})
        print(f"  {name}: {len(fks)} free kicks", flush=True)
    found.sort(key=lambda d: -d["score"])
    (HERE / "_events" / "free_kicks.json").write_text(json.dumps(found, indent=2))
    print(f"\n=== top free kicks ({len(found)} total) ===")
    for r in found[:12]:
        print("   ", r)


if __name__ == "__main__":
    main()
