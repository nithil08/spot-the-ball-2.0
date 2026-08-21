"""pick_situations.py — score every candidate window and pick the best clip of each type.

scan_events.py only says "a corner happened at frame N". This decides WHICH corner (and
which free kick, header, keeper throw) actually looks like the real thing on camera, using
rules taken from the laws and from how these situations look in match footage:

  corner       Law 17 delivery into the box that is CONTESTED (several bodies under the
               ball when it drops) and does not end in a goal — a goal teleports the ball
               to the centre spot for the kickoff, which breaks continuous play.
  free kick    Law 13. The iconic one is an ATTACKING free kick: the ball is delivered
               towards the goal the taker is attacking and arrives in/near the box.
               A free kick deep in a team's own half that gets hoofed clear is a real
               free kick but a boring one, so attacking ones score higher.
  gk throw     Law 12 distribution BY HAND. The engine tell is unambiguous: while the
               keeper holds the ball it sits parked at ~1.35 m (his hands), then the
               release arcs away. A keeper who keeps the ball at ~0.11 m is dribbling and
               about to PUNT it — that is not a throw, so height is what separates them.
  header       a ball above chest height, redirected, with more than one player under it.

Run:  python3 pick_situations.py      ->  _events/picks.json  (+ printed ranking)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = "/Users/nithilbalamurugan/gfootball_src"
EXP = "/Users/nithilbalamurugan/Desktop/Nithil Research/code/spot-the-ball-2.0/experiments"
for _p in (SRC, SRC + "/third_party", EXP, str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from situations import find_mode_events, GM_CORNER, GM_FREEKICK  # noqa: E402
from situations import all_variants  # noqa: E402
from scan_events import run_log, SEED  # noqa: E402

WINDOW = 80          # frames of clip we intend to cut after the award
STEPS = 900


def _np():
    import numpy as np
    return np


def has_reset(log, a, b):
    """True if play is broken in [a,b): a goal/kickoff teleports the ball to the centre
    spot, and any teleport shows up as an impossibly large single-frame ball step."""
    np = _np()
    b = min(b, len(log))
    balls = np.array([s["ball"][:2] for s in log[a:b]])
    if len(balls) < 3:
        return True
    steps = np.linalg.norm(np.diff(balls, axis=0), axis=1)
    return bool(steps.max() >= 0.35)


def near_count(s, radius=0.06):
    np = _np()
    pl = np.vstack([s["left_team"], s["right_team"]])
    return int((np.linalg.norm(pl - np.array(s["ball"][:2]), axis=1) < radius).sum())


def score_corner(log, f):
    """Delivery quality: how high it was whipped and how many bodies met it."""
    np = _np()
    if has_reset(log, f + 1, f + WINDOW):
        return None
    end = min(len(log) - 1, f + WINDOW)
    seg = log[f:end]
    apex = max(s["ball"][2] for s in seg)
    # the contest is where the delivery comes back down inside the box
    drop = [i for i in range(f, end) if log[i]["ball"][2] < 1.5
            and abs(log[i]["ball"][0]) > 0.82 and i > f + 8]
    contest = max((near_count(log[i]) for i in drop[:6]), default=0)
    return {"frame": f, "apex": round(apex, 2), "contested_by": contest,
            "score": contest * 10 + apex}


def score_freekick(log, f):
    """Attacking free kicks (delivered towards the goal being attacked) score highest."""
    np = _np()
    if has_reset(log, f + 1, f + WINDOW):
        return None
    spot = np.array(log[f]["ball"][:2])
    end = min(len(log) - 1, f + 35)
    seg = log[f:end]
    apex = max(s["ball"][2] for s in seg)
    # which way did the delivery go relative to the nearer goal?
    goal_side = 1.0 if spot[0] >= 0 else -1.0
    xs = [s["ball"][0] for s in seg]
    towards = (max(xs) - spot[0]) * goal_side if goal_side > 0 else (spot[0] - min(xs))
    into_box = max(abs(x) for x in xs) > 0.85
    dist_to_goal = 1.0 - abs(spot[0])
    attacking = towards > 0.05 and into_box
    # 0.10..0.30 from the goal line is roughly 18-35 yds: prime free-kick range
    in_range = 0.08 <= dist_to_goal <= 0.32
    return {"frame": f, "spot": [round(v, 3) for v in spot], "apex": round(apex, 2),
            "attacking": bool(attacking), "in_range": bool(in_range),
            "score": (60 if attacking else 0) + (25 if in_range else 0) + apex}


def score_gk_throws(log):
    """A hold at hand height (~1.35 m) followed by a release = a throw, not a punt."""
    np = _np()
    out = []
    n = len(log)
    i = 0
    while i < n:
        s = log[i]
        if s["ball_owned_player"] == 0 and s["ball_owned_team"] in (0, 1):
            team = s["ball_owned_team"]
            j = i
            while (j < n and log[j]["ball_owned_player"] == 0
                   and log[j]["ball_owned_team"] == team):
                j += 1
            hand = [k for k in range(i, j) if 1.15 <= log[k]["ball"][2] <= 1.75]
            if len(hand) >= 4 and j < n:
                rel = j - 1
                end = min(n - 1, rel + 30)
                apex = max(s2["ball"][2] for s2 in log[rel:end])
                travel = float(np.linalg.norm(np.array(log[end]["ball"][:2])
                                              - np.array(log[rel]["ball"][:2])))
                if not has_reset(log, i, end):
                    out.append({"catch": i, "hand_frames": len(hand), "release": rel,
                                "apex": round(apex, 2), "travel": round(travel, 3),
                                "score": travel * 100 + len(hand)})
            i = j
        else:
            i += 1
    return out


def score_headers(log):
    """Contested aerial headers only — 2+ bodies under a ball above chest height."""
    from situations import find_headers
    out = []
    for f, z, cos, dist in find_headers(log):
        if not (1.35 <= z <= 2.9):
            continue
        n = near_count(log[f])
        if n < 2:
            continue
        if has_reset(log, max(0, f - 25), f + 45):
            continue
        apex_in = max(s["ball"][2] for s in log[max(0, f - 18):f + 1])
        out.append({"frame": f, "z": round(z, 2), "contested_by": n,
                    "apex_in": round(apex_in, 2), "turn_cos": round(cos, 2),
                    "score": n * 10 + apex_in})
    return out


def main():
    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle("noname")
    picks = {"corner": [], "freekick": [], "gk_throw": [], "header": []}
    for name, spec in all_variants():
        write_scenario(spec, force=True)
        log = run_log(name, SEED, STEPS)
        for f in find_mode_events(log, GM_CORNER):
            r = score_corner(log, f)
            if r:
                picks["corner"].append({"level": name, **r})
        for f in find_mode_events(log, GM_FREEKICK):
            r = score_freekick(log, f)
            if r:
                picks["freekick"].append({"level": name, **r})
        for r in score_gk_throws(log):
            picks["gk_throw"].append({"level": name, **r})
        for r in score_headers(log):
            picks["header"].append({"level": name, **r})
        print(f"  scanned {name}", flush=True)

    # One real event spans several frames (a header shows up on both frames of contact,
    # a corner is only ever one award) — collapse anything from the same level within
    # 40 frames so "top 5" means five different situations, not five views of one.
    for k in picks:
        picks[k].sort(key=lambda d: -d["score"])
        kept, seen = [], []
        for r in picks[k]:
            f = r.get("frame", r.get("catch"))
            if any(lv == r["level"] and abs(f - g) < 40 for lv, g in seen):
                continue
            seen.append((r["level"], f))
            kept.append(r)
        picks[k] = kept
    (HERE / "_events" / "picks.json").write_text(
        json.dumps(picks, indent=2, default=float))
    for k, v in picks.items():
        print(f"\n=== {k} (top 6 of {len(v)}) ===")
        for r in v[:6]:
            print("   ", r)


if __name__ == "__main__":
    main()
