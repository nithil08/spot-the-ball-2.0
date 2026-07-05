"""Ground-truth rollout harness.

Given a scenario module name, run M independent rollouts and report
empirical P(left scores), P(right scores), and per-rollout final states.
This is the engine of the Hypothetical and Counterfactual categories:
every "what would happen if..." answer is grounded in many rollouts.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from lib import use_bundle, REPO


def score_rate(level: str, n_seeds: int = 20, max_steps: int = 200, bundle: str = "default"):
    """Run n_seeds rollouts; return summary dict.

    Output:
      {
        "level": "...",
        "n": 20,
        "left_score_rate": 0.35,
        "right_score_rate": 0.10,
        "ended_naturally": 0.95,
        "per_seed": [{"seed": 0, "left": 1, "right": 0, "steps": 78}, ...]
      }
    """
    use_bundle(bundle)
    from gfootball.env import config as cfg
    from gfootball.env import football_env

    per_seed = []
    for seed in range(n_seeds):
        values = {
            "level": level,
            "players": [
                "bot:left_players=1,right_players=0",
                "bot:left_players=0,right_players=1",
            ],
            "action_set": "full",
            "write_video": False,
            "dump_full_episodes": False,
            "dump_scores": False,
            "tracesdir": tempfile.gettempdir(),
            "real_time": False,
            "game_engine_random_seed": seed,
            "video_quality_level": 0,
            "display_game_stats": False,
        }
        env = football_env.FootballEnv(cfg.Config(values))
        env.reset()
        done = False
        steps = 0
        last = None
        while not done and steps < max_steps:
            obs, _, done, _ = env.step([])
            last = obs
            steps += 1
        score = last[0]["score"] if isinstance(last, (list, tuple)) else last["score"]
        per_seed.append({
            "seed": seed,
            "left": int(score[0]),
            "right": int(score[1]),
            "steps": steps,
            "natural_end": bool(done),
        })
        env.close()

    n = float(n_seeds)
    summary = {
        "level": level,
        "n": n_seeds,
        "left_score_rate": sum(1 for r in per_seed if r["left"] > 0) / n,
        "right_score_rate": sum(1 for r in per_seed if r["right"] > 0) / n,
        "natural_end_rate": sum(1 for r in per_seed if r["natural_end"]) / n,
        "per_seed": per_seed,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--bundle", default="default")
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args()

    s = score_rate(args.level, args.n, args.max_steps, args.bundle)
    out_path = Path(args.out) if args.out else REPO / "experiments" / f"rollout_{args.level}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(s, indent=2))
    print(f"left score: {s['left_score_rate']:.2f}  right: {s['right_score_rate']:.2f}  "
          f"(n={args.n})  → {out_path}")


if __name__ == "__main__":
    main()
