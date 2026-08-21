"""gen_situations.py — render the GEN2 match-situation clips.

Three situations x 20 windows x 2 visibility variants = 120 clips:

    01_headers/            full_visibility/  split_1s_4s/
    02_corner_kicks/       full_visibility/  split_1s_4s/
    03_goalkeeper_throws/  full_visibility/  split_1s_4s/

Every clip is exactly 50 frames (5.0 s at 10 fps) on the locked 16x6 grid, with a red
circle on the ball's start position for the first 3 frames.

Each window is rendered TWICE from the identical deterministic state — once with the ball
visible (bundle gen2) and once with it invisible (gen2_ball_invisible). Both visibility
variants are cut from that one pair, so full_visibility and split_1s_4s are the same play
frame for frame; only the ball differs. The pixels that differ between the two renders ARE
the ball, which is where the start and final ground-truth cells come from — no engine
instrumentation, no hand labelling.

The asset bundle has to be chosen before gfootball is imported, so the visible and
invisible passes are separate processes.

Run:  python3 gen_situations.py all
      (phases: vis -> inv -> compose)
"""
import json
import subprocess
import sys

import numpy as np

from gen2_lib import (BUNDLE_INV, BUNDLE_VIS, CACHE, CLIP_FRAMES, HERE, SHAPES,
                      compose_pair, continuous, render_window, shape_spec, write_clip)

PICKS = CACHE / "picks.json"
FR = CACHE / "situations"
PLAN = FR / "plan.json"
# 10 situations per class (-> 20 clips per class, 60 in total). The picker still ranks the
# full candidate pool and keeps 20, so this takes the BEST 10 of those rather than the
# first 10 found.
N_PER_KIND = 10

# kind key in picks.json -> output folder, clip-name stem
KINDS = {
    "header": ("01_headers", "header"),
    "corner": ("02_corner_kicks", "corner_kick"),
    "gk_throw": ("03_goalkeeper_throws", "goalkeeper_throw"),
}


def _levels():
    """Scenario modules must be on disk and identical to what the sweep played."""
    from scenario_factory import write_scenario
    return {name: write_scenario(shape_spec(name), force=True) for name, *_ in SHAPES}


def phase_vis():
    from lib import use_bundle
    use_bundle(BUNDLE_VIS)
    levels = _levels()
    picks = json.loads(PICKS.read_text())
    FR.mkdir(parents=True, exist_ok=True)
    kept = []
    for kind, (folder, stem) in KINDS.items():
        n = 0
        for p in picks.get(kind, []):
            if n >= N_PER_KIND:
                break
            frames, balls = render_window(levels[p["shape"]], p["seed"],
                                          p["start"], p["end"])
            # The render replays the same match the sweep logged, so this should always
            # agree with the pick-time check — it is here to catch a mismatch loudly
            # rather than ship a clip with a goal in it.
            if len(frames) != CLIP_FRAMES or not continuous(balls):
                print(f"  SKIP {p['match']}@{p['start']} ({kind}): "
                      f"{len(frames)} frames, play breaks", flush=True)
                continue
            idx = len(kept)
            np.savez_compressed(FR / f"sit{idx:03d}_vis.npz", frames=np.array(frames))
            kept.append({**p, "index": idx, "folder": folder, "stem": stem,
                         "n_in_kind": n + 1})
            n += 1
            print(f"  [vis] {kind:9s} {n:2d}/{N_PER_KIND} {p['match']} "
                  f"f{p['start']}-{p['end']}", flush=True)
        print(f"  {kind}: {n} clips", flush=True)
    PLAN.write_text(json.dumps(kept, indent=2, default=float))
    print(f"phase vis: {len(kept)} windows -> {PLAN}")


def phase_inv():
    from lib import use_bundle
    use_bundle(BUNDLE_INV)
    levels = _levels()
    plan = json.loads(PLAN.read_text())
    for p in plan:
        frames, _b = render_window(levels[p["shape"]], p["seed"], p["start"], p["end"])
        np.savez_compressed(FR / f"sit{p['index']:03d}_inv.npz", frames=np.array(frames))
        print(f"  [inv] {p['kind']:9s} {p['match']}", flush=True)
    print(f"phase inv: {len(plan)} renders")


def phase_compose():
    from grid import cell_of
    plan = json.loads(PLAN.read_text())
    header = ("clip,situation,shape,seed,match,start_frame,end_frame,n_frames,seconds,"
              "ball_start_cell,start_px,start_py,ball_final_cell,final_px,final_py,detail")
    per_folder, all_rows = {}, []
    for p in plan:
        vis = np.load(FR / f"sit{p['index']:03d}_vis.npz")["frames"]
        inv = np.load(FR / f"sit{p['index']:03d}_inv.npz")["frames"]
        full, split, (spx, spy), (fpx, fpy) = compose_pair(vis, inv)
        clip = f"{p['stem']}_{p['n_in_kind']:02d}"
        out = HERE / p["folder"]
        write_clip(full, out / "full_visibility" / f"{clip}.mov")
        write_clip(split, out / "split_1s_4s" / f"{clip}.mov")
        detail = {k: p[k] for k in p
                  if k not in ("match", "shape", "seed", "start", "end", "kind",
                               "index", "folder", "stem", "n_in_kind")}
        row = (f"{clip},{p['kind']},{p['shape']},{p['seed']},{p['match']},"
               f"{p['start']},{p['end']},{len(full)},{len(full)/10:.1f},"
               f"{cell_of(spx, spy)},{spx:.1f},{spy:.1f},"
               f"{cell_of(fpx, fpy)},{fpx:.1f},{fpy:.1f},"
               f"{json.dumps(detail).replace(',', ';')}")
        per_folder.setdefault(p["folder"], []).append(row)
        all_rows.append(row)
        print(f"  [compose] {clip}: ball {cell_of(spx, spy)} -> {cell_of(fpx, fpy)}")
    for folder, rows in per_folder.items():
        (HERE / folder / "ground_truth.csv").write_text(
            header + "\n" + "\n".join(rows) + "\n")
    (HERE / "SITUATIONS_GROUND_TRUTH.csv").write_text(
        header + "\n" + "\n".join(all_rows) + "\n")
    print(f"phase compose: {len(plan)} situations -> {len(plan) * 2} clips")


def phase_all():
    for mode in ("vis", "inv", "compose"):
        print(f"\n===== phase {mode} =====", flush=True)
        subprocess.run([sys.executable, __file__, mode], check=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    {"vis": phase_vis, "inv": phase_inv, "compose": phase_compose,
     "all": phase_all}[mode]()
