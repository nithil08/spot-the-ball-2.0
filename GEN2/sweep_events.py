"""sweep_events.py — play many matches and cache the observation logs.

Every (shape, seed) pair is a different match (gen2_lib explains why seeds work).
Each match is played once, headless, and its log is cached to
_cache/sweep/<shape>_s<seed>.npz. Nothing is scored here: scoring is cheap and gets
re-run and re-tuned many times, so it must not require replaying the engine.

Resumable by design — an already-cached pair is skipped, so this can be killed and
restarted freely, and the picker can start working off a partial sweep.

Run:  python3 sweep_events.py [n_seeds] [steps]
      python3 sweep_events.py 40 1100        # ~480 matches
"""
import sys
import time

import numpy as np

from gen2_lib import (SHAPES, SWEEP, run_log, shape_spec, sweep_jobs, tag,
                      GM_CORNER, GM_FREEKICK)

DEFAULT_SEEDS = 40
DEFAULT_STEPS = 1100          # ~110 s of match: long enough for several set pieces


def summarise(log):
    """A one-line count of what turned up, so sweep progress is readable."""
    gm = log["game_mode"]
    changed = np.flatnonzero(np.diff(gm) != 0) + 1
    corners = int(sum(1 for i in changed if gm[i] == GM_CORNER))
    fks = int(sum(1 for i in changed if gm[i] == GM_FREEKICK))
    gk = int(((log["owned_player"] == 0) & (log["owned_team"] >= 0)).sum())
    goals = int(np.abs(np.diff(log["score"], axis=0)).sum())
    return corners, fks, gk, goals


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEEDS
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_STEPS
    seeds = list(range(1, n_seeds + 1))

    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle("gen2")

    SWEEP.mkdir(parents=True, exist_ok=True)
    # Scenario modules depend only on the shape, so write each one once.
    levels = {name: write_scenario(shape_spec(name), force=True)
              for name, *_ in SHAPES}

    jobs = list(sweep_jobs(seeds))
    todo = [(sh, sd) for sh, sd in jobs if not (SWEEP / f"{tag(sh, sd)}.npz").exists()]
    print(f"{len(jobs)} matches ({len(SHAPES)} shapes x {n_seeds} seeds), "
          f"{len(jobs) - len(todo)} already cached, {len(todo)} to play", flush=True)

    t0 = time.time()
    for n, (shape, seed) in enumerate(todo, 1):
        log = run_log(levels[shape], seed, steps)
        np.savez_compressed(SWEEP / f"{tag(shape, seed)}.npz", **log)
        c, f, g, go = summarise(log)
        el = time.time() - t0
        eta = (el / n) * (len(todo) - n) / 60
        print(f"  [{n}/{len(todo)}] {tag(shape, seed):18s} "
              f"corners={c} freekicks={f} gk_hold={g:4d} goals={go}  "
              f"eta {eta:.0f}m", flush=True)
    print(f"sweep done in {(time.time() - t0)/60:.1f} min -> {SWEEP}")


if __name__ == "__main__":
    main()
