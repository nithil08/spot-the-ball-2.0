"""Generator: COUNTERFACTUAL (responsibility) category.

Each item is a 10 s clip ending in a goal. We compute per-player
"but-for" responsibility by re-running the same scenario with one player
removed or lazied, measuring how much the goal probability drops.
Higher ΔP(goal) when player X is ablated → X is more responsible.

Per item:
  stimuli/counterfactual/<id>/clip.mov         # the actual scoring play
  stimuli/counterfactual/<id>/meta.json        # responsibility rank from rollouts
"""

import argparse
import hashlib
import json
from pathlib import Path

import lib
from lib import use_bundle, make_env, run_clip_capture, frames_to_mov, REPO, CLIP_STEPS
import scenario_factory as sf
from rollout import score_rate

OUT = REPO / "experiments/stimuli/counterfactual"


def make_item(seed: int, n_seeds: int = 10):
    """Render the base scoring play + compute ablation responsibility."""
    item_id = hashlib.md5(f"cf|{seed}".encode()).hexdigest()[:10]
    item_dir = OUT / item_id
    item_dir.mkdir(parents=True, exist_ok=True)

    base = sf.base_3v1_attack()
    base.name = f"x_cf_{item_id}_base"
    sf.write_scenario(base, force=True)

    # 1. Render the base clip (the one shown to subjects).
    work = item_dir / "_work"
    env = make_env(base.name, seed, work, write_video=False, hud=False)
    info = run_clip_capture(env, steps=CLIP_STEPS, dump_name=f"cf_{item_id}")
    env.close()
    clip_mov = item_dir / "clip.mov"
    frames_to_mov(info["frames"], clip_mov)

    # 2. Baseline P(goal): run base scenario many times.
    base_summary = score_rate(base.name, n_seeds=n_seeds, max_steps=150)
    p_base = base_summary["left_score_rate"]

    # 3. Per-left-attacker: replace with lazy and re-measure.
    # left = [GK, CM, CM, CM]  — attackers are idx 1, 2, 3
    delta_attacker = {}
    for i, p in enumerate(base.left):
        if p.role == "GK":
            continue
        lazied = sf.laze_player(base, "left", i, f"x_cf_{item_id}_lazL{i}")
        sf.write_scenario(lazied, force=True)
        s = score_rate(lazied.name, n_seeds=n_seeds, max_steps=150)
        # ΔP — drop in attack success when player i is removed
        delta_attacker[f"left_{i}_{p.role}"] = p_base - s["left_score_rate"]

    # 4. Right-team defender (idx 1, the CB) — removing them should help the attack.
    delta_defender = {}
    for i, p in enumerate(base.right):
        if p.role == "GK":
            continue
        removed = sf.ScenarioSpec(**{**base.__dict__,
                                     "name": f"x_cf_{item_id}_remR{i}"})
        removed.right = [base.right[j] for j in range(len(base.right)) if j != i]
        sf.write_scenario(removed, force=True)
        s = score_rate(removed.name, n_seeds=n_seeds, max_steps=150)
        # ΔP — drop in goal probability if defender stayed (negative = they were helpful)
        delta_defender[f"right_{i}_{p.role}"] = s["left_score_rate"] - p_base

    # Rank attackers by responsibility (larger ΔP when removed = more responsible).
    attacker_rank = sorted(delta_attacker.items(), key=lambda kv: -kv[1])

    meta = {
        "category": "counterfactual",
        "id": item_id,
        "base_scenario": base.name,
        "seed": seed,
        "n_rollout_seeds": n_seeds,
        "ground_truth": {
            "p_score_base": p_base,
            "responsibility_attacker": delta_attacker,
            "responsibility_defender_helpful": delta_defender,
            "attacker_responsibility_rank": [k for k, _ in attacker_rank],
        },
        "questions": [
            "Which player was most responsible for the goal?",
            "How responsible was each player for the goal (1-7 Likert)?",
            "What single change would have prevented the goal?",
        ],
        "answer_format": "responsibility: per-player Likert; cause: MC over players",
        "stimulus": "clip.mov",
    }
    (item_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return item_id, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--rollout_seeds", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    use_bundle("default")

    seeds = [3, 13, 29, 41, 67, 91, 137, 211][: args.n]
    summary = []
    for s in seeds:
        iid, meta = make_item(s, n_seeds=args.rollout_seeds)
        gt = meta["ground_truth"]
        summary.append({"id": iid, "p_base": gt["p_score_base"]})
        print(f"  {iid}  P_base={gt['p_score_base']:.2f}  "
              f"rank={gt['attacker_responsibility_rank']}")
    (OUT / "_index.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(summary)} counterfactual items → {OUT}")


if __name__ == "__main__":
    main()
