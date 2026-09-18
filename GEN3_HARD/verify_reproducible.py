"""verify_reproducible.py — prove every GEN3 clip replays bit-identically, and record a
fingerprint so any future rebuild can be checked against the batch that shipped.

WHY THIS EXISTS
  The clips are wanted for counterfactual work ("what would have happened without this
  player"), which is only meaningful if the FACTUAL is pinned down exactly. A clip is
  reproducible if and only if the play is a pure function of things we have written down:

      (scenario source, game_engine_random_seed, physics_steps_per_frame)  ->  the play

  Everything else the pipeline touches — camera offset, hidden players, which asset
  bundle — must be RENDER-ONLY, i.e. provably unable to change the football. That is not
  a comfortable assumption to carry into a counterfactual study, because if hiding a
  player changed the play even slightly, then "hide a player" would look like a
  counterfactual while actually being a different match. So it is measured here.

WHAT IT MEASURES, PER CLIP
  pass A   canonical: the clip's own camera offset, nobody hidden        -> reference log
  pass B   identical to A                                                -> determinism
  pass C   different camera offset AND two players hidden                -> render-only

  A vs B bit-identical  =>  replay is deterministic over the clip's FULL prefix (up to
                            2450 steps, not the 120 the scenario docstring measured)
  A vs C bit-identical  =>  offset and hide-slots cannot influence the play, so neither
                            is a counterfactual knob

  "Bit-identical" is exact equality on every logged array — ball xyz, all 22 player
  positions, possession, game mode and score — for every step, not a tolerance.

WHAT IT WRITES
  clips/reproducibility.csv   one row per clip: the full spec needed to reproduce it,
                              the sha256 of the scenario source, and the sha256 of the
                              play itself. Re-run `check` later and any drift in the
                              engine, the scenario table or the seed handling shows up as
                              a changed play_sha rather than as a silently different clip.

USAGE
  python3 verify_reproducible.py run --shard 0/3     # bounded work, resumable
  bash run_phase.sh reproducible 3                   # (via gen3.py) all shards
  python3 verify_reproducible.py report              # collate -> reproducibility.csv
  python3 verify_reproducible.py check               # re-verify against a written CSV

NOTE ON THE ASSET BUNDLE
  The bundle is deliberately NOT varied inside a process: use_bundle sets
  GFOOTBALL_DATA_DIR and must run before the engine loads, so a mid-process swap would
  not test what it appears to test. Bundle-invariance is already established elsewhere —
  the visible and invisible renders of a window differ only in ball pixels, which is the
  assumption `audit` and `ballpix` are built on and which they would fail loudly without.
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))

import gen3_lib as G  # noqa: E402

EXIT_DONE = 3
OUT = HERE / "_cache" / "reproducible"
GT = HERE / "clips" / "ground_truth.csv"
CSV_OUT = HERE / "clips" / "reproducibility.csv"

PER_PROCESS = 3          # 3 clips x 3 passes = 9 envs, well inside the ~46-env leak

# Pass C's perturbation. The offset is far from any shipping value and the hidden slots
# are one keeper and one outfielder, on opposite teams — if hiding could affect play at
# all, a hidden keeper is the most likely way to see it.
C_OFFSET = (-18.0, 6.0)
C_HIDE = "L0,R7"

LOG_KEYS = ("ball", "owned_team", "owned_player", "game_mode", "left", "right", "score")


def rows():
    with GT.open() as f:
        return list(csv.DictReader(f))


def run_log_ex(level, seed, steps, offset, hide_slots):
    """G.run_log, but with hide_slots exposed.

    Deliberately a copy of gen3_lib.run_log rather than a wrapper: run_log hardcodes an
    empty hide set, and the point of pass C is to drive the SAME code path with a
    non-empty one. Any divergence between this and run_log would invalidate the
    comparison, so keep them in step.
    """
    G._prep(hide_slots, offset)
    env = G._env(level, seed, G.CACHE / "_work_repro")
    env.render("rgb_array")
    env.reset()
    ball, ot, op, gm, lt, rt, sc = [], [], [], [], [], [], []
    for _ in range(steps):
        obs, _r, done, _i = env.step([])
        o = obs[0] if isinstance(obs, list) else obs
        ball.append(np.asarray(o["ball"], dtype=np.float32))
        ot.append(int(o["ball_owned_team"]))
        op.append(int(o["ball_owned_player"]))
        gm.append(int(o["game_mode"]))
        lt.append(np.asarray(o["left_team"], dtype=np.float32))
        rt.append(np.asarray(o["right_team"], dtype=np.float32))
        sc.append(list(map(int, o["score"])))
        if done:
            break
    env.close()
    return {"ball": np.array(ball), "owned_team": np.array(ot),
            "owned_player": np.array(op), "game_mode": np.array(gm),
            "left": np.array(lt), "right": np.array(rt), "score": np.array(sc)}


def log_sha(log):
    """sha256 over every logged array, in a fixed order, including dtype and shape.

    Hashing the raw bytes makes this exact: a float that differs in the last ulp changes
    the digest, which is the standard we want. Shape and dtype go in too, so a truncated
    or re-typed log can never collide with a full one.
    """
    h = hashlib.sha256()
    for k in LOG_KEYS:
        a = np.ascontiguousarray(log[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def identical(a, b):
    """Exact equality on every array. Returns (ok, first_differing_key, first_step)."""
    for k in LOG_KEYS:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            return False, k, -1
        if not np.array_equal(x, y):
            diff = np.argwhere(x != y)
            return False, k, int(diff[0][0])
    return True, None, None


def scenario_sha(shape):
    """sha256 of the emitted scenario module — the initial positions, ball, difficulty,
    offsides and duration that define the match setup.

    This is half the reproducibility contract. The seed alone is meaningless if the
    SHAPES table has been edited since, and that edit would otherwise be invisible.
    """
    from scenario_factory import SCENARIOS_DIR, write_scenario
    write_scenario(G.shape_spec(shape), force=False)
    src = (SCENARIOS_DIR / f"g3h_{shape}.py").read_bytes()
    return hashlib.sha256(src).hexdigest()


def cmd_run(argv):
    from lib import use_bundle
    from scenario_factory import write_scenario
    shard_i, shard_n = parse_shard(argv)
    use_bundle(G.BUNDLE_VIS)
    for name, *_ in G.SHAPES:
        write_scenario(G.shape_spec(name), force=False)
    OUT.mkdir(parents=True, exist_ok=True)

    todo = [r for i, r in enumerate(rows())
            if i % shard_n == shard_i
            and not (OUT / f"{r['clip']}.json").exists()]
    if not todo:
        print(f"shard {shard_i}/{shard_n}: nothing left")
        return EXIT_DONE
    print(f"shard {shard_i}/{shard_n}: {len(todo)} clips left", flush=True)

    for r in todo[:PER_PROCESS]:
        clip, shape, seed = r["clip"], r["shape"], int(r["seed"])
        steps = int(r["end_frame"])
        level = f"g3h_{shape}"
        off = (float(r["offset_x"]), float(r["offset_y"]))

        a = G.run_log(level, seed, steps, offset=off)
        b = G.run_log(level, seed, steps, offset=off)
        c = run_log_ex(level, seed, steps, C_OFFSET, C_HIDE)

        ab_ok, ab_key, ab_step = identical(a, b)
        ac_ok, ac_key, ac_step = identical(a, c)
        rec = {
            "clip": clip, "shape": shape, "seed": seed, "steps": steps,
            "logged_steps": int(len(a["ball"])),
            "start_frame": int(r["start_frame"]), "end_frame": int(r["end_frame"]),
            "offset_x": off[0], "offset_y": off[1],
            "scenario_sha": scenario_sha(shape),
            "play_sha": log_sha(a),
            "determinism_ok": bool(ab_ok),
            "determinism_first_diff": None if ab_ok else f"{ab_key}@{ab_step}",
            "render_invariant_ok": bool(ac_ok),
            "render_invariant_first_diff": None if ac_ok else f"{ac_key}@{ac_step}",
        }
        (OUT / f"{clip}.json").write_text(json.dumps(rec, indent=2))
        flag = "OK " if (ab_ok and ac_ok) else "FAIL"
        print(f"  [{flag}] {clip} {shape}_s{seed:03d} {steps} steps "
              f"play={rec['play_sha'][:12]}", flush=True)
    return 0


def parse_shard(argv):
    for i, a in enumerate(argv):
        if a.startswith("--shard"):
            spec = a.split("=", 1)[1] if "=" in a else argv[i + 1]
            n, d = spec.split("/")
            return int(n), int(d)
    return 0, 1


FIELDS = ["clip", "shape", "seed", "scenario_sha", "play_sha", "steps", "logged_steps",
          "start_frame", "end_frame", "offset_x", "offset_y",
          "determinism_ok", "render_invariant_ok",
          "determinism_first_diff", "render_invariant_first_diff"]


def cmd_report(argv):
    recs = [json.loads(p.read_text()) for p in sorted(OUT.glob("clip_*.json"))]
    if len(recs) != len(rows()):
        print(f"only {len(recs)} of {len(rows())} clips measured — run the shards first")
        return 1
    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k) for k in FIELDS})

    det = sum(r["determinism_ok"] for r in recs)
    inv = sum(r["render_invariant_ok"] for r in recs)
    uniq = len({r["play_sha"] for r in recs})
    print(f"clips                 {len(recs)}")
    print(f"deterministic         {det}/{len(recs)}")
    print(f"render-invariant      {inv}/{len(recs)}")
    print(f"distinct play_sha     {uniq}/{len(recs)}")
    print(f"total steps replayed  {sum(r['steps'] for r in recs) * 3}")
    print(f"-> {CSV_OUT}")
    bad = [r["clip"] for r in recs if not (r["determinism_ok"] and r["render_invariant_ok"])]
    if bad:
        print("FAILURES:", ", ".join(bad))
        return 1
    return 0


def cmd_check(argv):
    """Re-run the canonical pass and compare play_sha against the written CSV.

    This is the cheap regression gate: one replay per clip instead of three. Use it after
    any engine rebuild, scenario edit or dependency bump to confirm the batch still
    reproduces the football it shipped with.
    """
    from lib import use_bundle
    use_bundle(G.BUNDLE_VIS)
    if not CSV_OUT.exists():
        print(f"no {CSV_OUT} — run `run` then `report` first")
        return 1
    with CSV_OUT.open() as f:
        want = {r["clip"]: r for r in csv.DictReader(f)}
    only = [a for a in argv if not a.startswith("--")]
    bad = []
    for clip, r in want.items():
        if only and clip not in only:
            continue
        sc = scenario_sha(r["shape"])
        log = G.run_log(f"g3h_{r['shape']}", int(r["seed"]), int(r["steps"]),
                        offset=(float(r["offset_x"]), float(r["offset_y"])))
        ps = log_sha(log)
        ok = (ps == r["play_sha"]) and (sc == r["scenario_sha"])
        if not ok:
            bad.append(clip)
        why = "" if ok else (" scenario CHANGED" if sc != r["scenario_sha"] else " play CHANGED")
        print(f"  [{'OK ' if ok else 'FAIL'}] {clip} {ps[:12]}{why}", flush=True)
    print("all reproduce" if not bad else f"DRIFT: {', '.join(bad)}")
    return 0 if not bad else 1


CMDS = {"run": cmd_run, "report": cmd_report, "check": cmd_check}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print(__doc__)
        sys.exit(2)
    sys.exit(CMDS[sys.argv[1]](sys.argv[2:]))
