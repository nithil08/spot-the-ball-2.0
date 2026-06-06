"""Generator: HYPOTHETICAL (intervention) category.

Each item is a goal-approaching tableau. We render the base scene as a
static frame and four candidate-intervention scenes side-by-side, each
shown as its own static frame. The model/human picks which intervention
most increases scoring chance. Ground truth is the empirical P(left
scores) under each intervention from many rollouts.

Per item:
  stimuli/hypothetical/<id>/frame_base.png
  stimuli/hypothetical/<id>/frame_a_add_attacker.png
  stimuli/hypothetical/<id>/frame_b_remove_defender.png
  stimuli/hypothetical/<id>/frame_c_ball_forward.png
  stimuli/hypothetical/<id>/frame_d_no_change.png
  stimuli/hypothetical/<id>/meta.json    # P(goal) per intervention
"""

import argparse
import hashlib
import json
from pathlib import Path

import lib
from lib import use_bundle, make_env, run_clip, latest_avi, extract_frame, REPO
import scenario_factory as sf
from rollout import score_rate

OUT = REPO / "experiments/stimuli/hypothetical"


def render_static(level: str, seed: int, frame: int, out_png: Path):
    """Render a static frame from a scenario at the given step."""
    work = out_png.parent / f"_work_{level}_{seed}"
    work.mkdir(parents=True, exist_ok=True)
    env = make_env(level, seed, work, write_video=True, hud=False)
    run_clip(env, steps=max(frame + 5, 20), dump_name=f"hyp_{level}_{seed}")
    env.close()
    avi = latest_avi(work)
    extract_frame(avi, frame, out_png)


def make_item(seed: int, frame: int = 15, n_seeds: int = 12):
    """Build one hypothetical item: 1 base + 4 intervention variants."""
    item_id = hashlib.md5(f"hyp|{seed}|{frame}".encode()).hexdigest()[:10]
    item_dir = OUT / item_id
    item_dir.mkdir(parents=True, exist_ok=True)

    base = sf.base_3v1_attack()
    base.name = f"x_hyp_{item_id}_base"
    sf.write_scenario(base, force=True)

    # Intervention A: add an attacker near the box.
    add_atk = sf.ScenarioSpec(**{**base.__dict__, "name": f"x_hyp_{item_id}_a"})
    add_atk.left = list(base.left) + [sf.PlayerSpec(0.80, 0.05, "CF")]
    sf.write_scenario(add_atk, force=True)

    # Intervention B: remove the defender.
    rem_def = sf.ScenarioSpec(**{**base.__dict__, "name": f"x_hyp_{item_id}_b"})
    rem_def.right = [base.right[0]]   # keep only GK
    sf.write_scenario(rem_def, force=True)

    # Intervention C: ball moved 0.15 forward (closer to goal).
    ball_fwd = sf.ScenarioSpec(**{**base.__dict__,
                                  "name": f"x_hyp_{item_id}_c",
                                  "ball": (base.ball[0] + 0.15, base.ball[1])})
    sf.write_scenario(ball_fwd, force=True)

    # Intervention D: no change (control).
    no_change = sf.ScenarioSpec(**{**base.__dict__, "name": f"x_hyp_{item_id}_d"})
    sf.write_scenario(no_change, force=True)

    variants = [
        ("base", base.name),
        ("a_add_attacker", add_atk.name),
        ("b_remove_defender", rem_def.name),
        ("c_ball_forward", ball_fwd.name),
        ("d_no_change", no_change.name),
    ]

    # Render static stimulus for each variant.
    for tag, lvl in variants:
        render_static(lvl, seed, frame, item_dir / f"frame_{tag}.png")

    # Ground truth: P(left scores) per intervention.
    p_score = {}
    for tag, lvl in variants:
        if tag == "base":
            continue       # base == d_no_change visually but evaluated separately
        summary = score_rate(lvl, n_seeds=n_seeds, max_steps=150)
        p_score[tag] = summary["left_score_rate"]

    # Rank interventions (excl. base) by P(goal).
    ranked = sorted([(p, t) for t, p in p_score.items()], reverse=True)
    best = ranked[0][1] if ranked else None

    meta = {
        "category": "hypothetical",
        "id": item_id,
        "base_scenario": base.name,
        "seed": seed,
        "frame": frame,
        "ground_truth": {
            "p_score_per_intervention": p_score,
            "best_intervention": best,
            "n_rollout_seeds": n_seeds,
        },
        "questions": [
            "The left (red) team is attacking. Which intervention most "
            "increases their chance of scoring?",
        ],
        "answer_format": "MC: A / B / C / D (see frame_*.png labels)",
        "stimuli": {tag: f"frame_{tag}.png" for tag, _ in variants},
    }
    (item_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return item_id, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--rollout_seeds", type=int, default=8,
                    help="rollouts per intervention for ground truth")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    use_bundle("default")

    seeds = [7, 13, 29, 41, 67, 91, 137, 211][: args.n]
    summary = []
    for s in seeds:
        iid, meta = make_item(s, frame=15, n_seeds=args.rollout_seeds)
        summary.append({"id": iid, "best": meta["ground_truth"]["best_intervention"]})
        print(f"  {iid}  best={meta['ground_truth']['best_intervention']}  "
              f"P={meta['ground_truth']['p_score_per_intervention']}")
    (OUT / "_index.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(summary)} hypothetical items → {OUT}")


if __name__ == "__main__":
    main()
