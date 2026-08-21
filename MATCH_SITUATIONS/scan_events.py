"""scan_events.py — phase 1: find real match situations in deterministic 11v11 play.

Runs full 11 v 11 matches (22 players, engine referee live) and logs the observation
stream ONLY — no frames are kept, so this is cheap enough to sweep many match shapes.
It then reports where the engine's OWN referee awarded a corner / free kick / throw-in,
where the keeper caught-and-threw, and where a header happened.

Two things learned the hard way and encoded here:

  * `deterministic=True` makes game_engine_random_seed irrelevant — every seed replays
    the identical match. Variety therefore comes from the SCENARIO SHAPE (where the ball
    starts, how far up the pitch each line is pushed, team difficulty), not from seeds.
  * The engine fast-forwards dead-ball time (match.cpp skips simulation while the game
    is on hold), so a set piece occupies only ~5 env steps between the award and the
    delivery. Clips are therefore cut to START at the award frame, which also hides the
    instantaneous player reposition that PrepareSetPiece does.

Run:  python3 scan_events.py            # sweep every variant
      python3 scan_events.py 6 900      # first 6 variants, 900 steps each
Out:  _events/events.json
"""
import json
import sys
import time
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

from situations import (match_spec, find_mode_events, find_gk_throws, find_headers,  # noqa: E402
                        GM_CORNER, GM_FREEKICK, GM_THROWIN, GM_NAME)

OUT = HERE / "_events"
SEED = 42          # deterministic => the seed is only a formality


def variants():
    """Match shapes to sweep. Pushing a line up the pitch puts the two teams in contact
    sooner, which is what generates fouls (free kicks), deflections (corners) and shots
    the keeper has to catch."""
    i = 0
    for bx, by in [(0.0, 0.0), (0.35, 0.10), (0.55, -0.15), (-0.30, 0.25),
                   (0.70, 0.05), (0.20, -0.30)]:
        for pl, pr in [(0.0, 0.0), (0.35, 0.15), (0.55, 0.30), (0.20, 0.40)]:
            for diff in [(0.8, 0.8), (1.0, 0.6)]:
                i += 1
                name = f"ms_v{i:02d}"
                yield name, match_spec(name, ball=(bx, by), offsides=True,
                                       difficulty=diff, push_left=pl, push_right=pr)


def run_log(level, seed, steps):
    """Play `level` for `steps` steps in render mode, returning the observation log."""
    import numpy as np
    from gfootball.env import config as cfg, football_env
    work = HERE / "_events" / "_work"
    work.mkdir(parents=True, exist_ok=True)
    values = {
        "level": level, "players": [], "action_set": "full",
        "write_video": False, "dump_full_episodes": False, "dump_scores": False,
        "tracesdir": str(work), "real_time": False,
        "game_engine_random_seed": seed, "video_quality_level": 2,
        "display_game_stats": False,
    }
    env = football_env.FootballEnv(cfg.Config(values))
    env.render("rgb_array")          # render mode ON => same timing as the render pass
    obs = env.reset()
    log = []
    for _ in range(steps):
        obs, r, done, _ = env.step([])
        o = obs[0] if isinstance(obs, list) else obs
        log.append({
            "ball": np.array(o["ball"], dtype=float).tolist(),
            "ball_owned_team": int(o["ball_owned_team"]),
            "ball_owned_player": int(o["ball_owned_player"]),
            "game_mode": int(o["game_mode"]),
            "left_team": np.array(o["left_team"], dtype=float).tolist(),
            "right_team": np.array(o["right_team"], dtype=float).tolist(),
            "score": list(map(int, o["score"])),
        })
        if done:
            break
    env.close()
    return log


def describe_setpiece(log, frame, header_window=40):
    """Where the restart is and what happens to the delivery — used to pick the
    most realistic-looking example of each set piece."""
    import numpy as np
    spot = log[frame]["ball"][:2]
    end = min(len(log) - 1, frame + header_window)
    seg = log[frame:end]
    apex = max((s["ball"][2] for s in seg), default=0.0)
    heads = [h for h in find_headers(log) if frame < h[0] <= end]
    travel = float(np.linalg.norm(np.array(log[end]["ball"][:2]) - np.array(spot)))
    return {"frame": frame, "spot": [round(v, 3) for v in spot],
            "apex_m": round(apex, 2), "headers_after": [h[0] for h in heads],
            "travel": round(travel, 3)}


def main():
    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle("noname")
    allv = list(variants())
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(allv)
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 900

    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, spec in allv[:n]:
        level = write_scenario(spec, force=True)
        t0 = time.time()
        log = run_log(level, SEED, steps)
        ev = {
            "n_steps": len(log),
            "corner": [describe_setpiece(log, f) for f in find_mode_events(log, GM_CORNER)],
            "freekick": [describe_setpiece(log, f) for f in find_mode_events(log, GM_FREEKICK)],
            "throwin": [describe_setpiece(log, f) for f in find_mode_events(log, GM_THROWIN)],
            "gk_throw": find_gk_throws(log),
            "header": [[h[0], round(h[1], 2), round(h[2], 2), round(h[3], 3)]
                       for h in find_headers(log)],
            "final_score": log[-1]["score"] if log else None,
            "modes": sorted({GM_NAME[s["game_mode"]] for s in log}),
        }
        results[name] = ev
        print(f"[{name}] {len(log)} steps {time.time() - t0:.0f}s | "
              f"corners={[c['frame'] for c in ev['corner']]} "
              f"fk={[(c['frame'], c['spot']) for c in ev['freekick']]} "
              f"throwin={[c['frame'] for c in ev['throwin']]} "
              f"gk={[(a, b) for a, b, _, _ in ev['gk_throw']]} "
              f"hdr={[h[0] for h in ev['header']]}", flush=True)
        (OUT / "events.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT / 'events.json'}")


if __name__ == "__main__":
    main()
