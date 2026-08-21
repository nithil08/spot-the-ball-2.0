"""hunt_corners.py — find enough clean Law 17 corners to make 5 clips.

The balanced match shapes only produced two corners that survive the no-goal rule, so this
sweeps the shapes that actually generate them: ball out on the flank in the final third
with a strong attack against a weaker defence, which is where a defender ends up putting
the ball behind. Stops as soon as it has enough, and CHECKPOINTS after every match so a
kill doesn't throw the work away.

A corner is kept only if the 80 frames after the award contain no goal and no restart:
  * no teleport (any single-frame ball jump >= 0.35 = the engine moved the ball, i.e. a
    goal sending it to the centre spot, or a mid-clip respot);
  * the ball never crosses a goal line BETWEEN THE POSTS (|x| > 1.0 and |y| < 0.05).
    Note |x| > 1.0 alone is NOT a goal — a corner is taken from x = 1.011 by definition.

Run:  python3 hunt_corners.py [n_wanted]   ->  _events/corners.json
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

from situations import (variants_wide, variants_pressure, find_mode_events,  # noqa: E402
                        GM_CORNER)
from scan_events import run_log, SEED  # noqa: E402
from pick_situations import near_count  # noqa: E402

STEPS = 620
WINDOW = 80
OUT = HERE / "_events" / "corners.json"


def clean(log, f):
    import numpy as np
    end = min(len(log), f + WINDOW)
    balls = np.array([s["ball"][:2] for s in log[f:end]])
    if len(balls) < WINDOW - 5:
        return False
    steps = np.linalg.norm(np.diff(balls, axis=0), axis=1)
    if steps.max() >= 0.35:
        return False
    in_goal = (np.abs(balls[:, 0]) > 1.0) & (np.abs(balls[:, 1]) < 0.05)
    return not bool(in_goal.any())


def main():
    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle("noname")
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    found = []
    for name, spec in list(variants_wide()) + list(variants_pressure()):
        write_scenario(spec, force=True)
        log = run_log(name, SEED, STEPS)
        for f in find_mode_events(log, GM_CORNER):
            if not clean(log, f):
                print(f"    {name}@{f}: rejected (goal or restart in window)")
                continue
            end = min(len(log) - 1, f + WINDOW)
            apex = max(s["ball"][2] for s in log[f:end])
            drop = [i for i in range(f + 8, end)
                    if log[i]["ball"][2] < 1.5 and abs(log[i]["ball"][0]) > 0.82]
            contest = max((near_count(log[i]) for i in drop[:6]), default=0)
            found.append({"level": name, "frame": f, "apex": round(apex, 2),
                          "contested_by": contest, "score": contest * 10 + apex})
            print(f"    KEEP {name}@{f} apex={apex:.1f} contested_by={contest}")
        OUT.write_text(json.dumps(sorted(found, key=lambda d: -d["score"]),
                                  indent=2, default=float))
        print(f"  {name} ({len(found)}/{want})", flush=True)
        if len(found) >= want:
            break
    print(f"\nfound {len(found)} clean corners -> {OUT}")


if __name__ == "__main__":
    main()
