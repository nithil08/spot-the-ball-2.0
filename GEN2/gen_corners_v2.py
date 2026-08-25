"""gen_corners_v2.py — corner-kick clips, rebuilt to fix what GEN2 v1 got wrong.

WHAT WAS WRONG WITH v1, measured rather than guessed:

1. 10 fps. `ffprobe` on 02_corner_kicks: r_frame_rate=10/1. That is not a setting
   anyone chose — it is gfootball's native observation rate, PHYSICS_STEPS_PER_SECOND
   (100) / physics_steps_per_frame (10). Every clip was a 100 ms sample of continuous
   motion, i.e. stop-motion. v2 runs PSF=4 for 25 fps; measured per-frame ball
   displacement drops from 0.0231 to 0.0086 (p95 0.0541 -> 0.0146).

2. Dead frames. Frames 4-11 of corner_kick_01 are near-empty pitch and stand: the
   camera tracks the ball in flight and frames neither the taker nor the box. Diffing
   a normal render against one with all 22 players hidden gives the exact on-screen
   player count: that clip opens on 4-5 visible players against a median of 18. This
   is the same "too empty to read as a match" defect that got player-delta rejected,
   and corners never got the gate. v2 gates on MIN_ONSCREEN in EVERY frame.

3. Trailing unrelated play. A corner resolves in 2-3 s but the window always ran the
   full 5 s, so the back half was post-clearance play — which is where "someone
   kicking it from the back" comes from. v2 requires the ball to stay in the corner's
   own final third for the whole window.

WHY THE OCCUPANCY GATE MUST RENDER

Log-based occupancy does not work, and this was measured before being abandoned:
across all 36 v1 corner windows, the minimum count of players within 0.35 of the ball
is 0 or 1 for every single window, and corner_kick_01 — the known-bad one — scores in
the UPPER HALF on the mean. Emptiness is a property of camera framing, not of where
the players are standing. Only the rendered frame knows. (Same lesson as the grid
centre bias being camera geometry rather than scenario design.)

The player-diff trick is exact and needs no colour heuristics, unlike the two counting
methods that already failed here: frustum geometry (36% exact) and kit-colour blobs
(+5.8 bias, the hoardings are saturated blue and red).

A SWEEP CACHED AT PSF=10 CANNOT BE REUSED AT PSF=4. Lower PSF makes the built-in AI
act more often, so the match diverges — identical for ~10 s, then a discrete event
flips it (max ball-track difference 0.51 on pressL2 s7). v2 keeps its own sweep cache.

Run:
    python3 gen_corners_v2.py sweep [--shard i/n]   # one match per process
    python3 gen_corners_v2.py pick                  # log gates -> shortlist
    python3 gen_corners_v2.py build                 # render, gate on occupancy, write
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen2_lib as G                                    # noqa: E402

# PSF must be set before any env is constructed — see the pool-reuse trap in gen2_lib.
G.PSF = 4

import numpy as np                                      # noqa: E402
from gen2_lib import (ALL_SLOTS, GM_CORNER, continuous,  # noqa: E402
                      render_window, run_log, shape_spec, tag)

# ── format ─────────────────────────────────────────────────────────────────────
FPS = 25                    # 100 / PSF
CLIP_FRAMES = 125           # 5.0 s, the locked GEN2 duration
VIS_FRAMES = 25             # the "1 s visible" half of the split variant
MARK_FRAMES = 8             # red start-circle, ~0.3 s
LEAD = 10                   # delivery sits 0.4 s into the clip
SWEEP_STEPS = 2750          # 110 s of match at 25 fps

# ── gates ──────────────────────────────────────────────────────────────────────
MIN_ONSCREEN = 10           # visible players required in EVERY frame
MIN_ABS_X = 0.50            # ball stays in the corner's own final third
N_WANT = 5

OUT = HERE / "02_corner_kicks_v2"
CACHE = G.CACHE / "corners_v2"
SWEEP = CACHE / "sweep"
SHORTLIST = CACHE / "shortlist.json"

# Corners concentrate hard in the flank/press shapes: measured over the 602-match v1
# sweep, these four yield 0.06-0.10 usable corner windows per match while midpush
# manages 0.02. Sweeping only these is what makes 5 clips affordable.
SHAPES = ["pressR2", "flankL", "pressL", "boxL"]
SEEDS = list(range(60, 100))


def jobs():
    return [(sh, sd) for sd in SEEDS for sh in SHAPES]


def parse_shard(argv):
    for a in argv:
        if a.startswith("--shard"):
            i, n = argv[argv.index(a) + 1].split("/") if "=" not in a else a.split("=")[1].split("/")
            return int(i), int(n)
    return 0, 1


# ── 1. sweep ───────────────────────────────────────────────────────────────────
def cmd_sweep(argv):
    """Play ONE uncached match, then exit. Exit 3 means the shard is empty.

    One match per process because the engine leaks resources and dies at roughly its
    46th env; the caller loops. Same pattern as the player-delta probe.
    """
    shard_i, shard_n = parse_shard(argv)
    SWEEP.mkdir(parents=True, exist_ok=True)
    todo = [(sh, sd) for k, (sh, sd) in enumerate(jobs())
            if k % shard_n == shard_i and not (SWEEP / f"{tag(sh, sd)}.npz").exists()]
    if not todo:
        print(f"shard {shard_i}/{shard_n}: empty")
        return 3
    shape, seed = todo[0]
    from lib import use_bundle
    use_bundle(G.BUNDLE_VIS)
    from scenario_factory import write_scenario
    lvl = write_scenario(shape_spec(shape), force=(shard_i == 0))
    log = run_log(lvl, seed, SWEEP_STEPS)
    np.savez_compressed(SWEEP / f"{tag(shape, seed)}.npz", **log)
    print(f"swept {tag(shape, seed)}: {len(log['ball'])} steps, "
          f"{len(todo) - 1} left in shard")
    return 0


# ── 2. log gates ───────────────────────────────────────────────────────────────
def find_corners(log, quiet=0.012):
    """Law 17 awards, anchored on the DELIVERY — the first frame the ball moves again
    after the referee re-spots it onto the arc. Anchoring on the award instead spends
    the opening seconds on a static ball, and opening BEFORE the award shows the
    re-spot as a teleport small enough to slip under the continuity threshold.

    Windows are 2.5x denser than v1, so every frame constant here is scaled with PSF.
    """
    gm, ball = log["game_mode"], log["ball"]
    awards = [i for i in range(1, len(gm))
              if gm[i] == GM_CORNER and gm[i - 1] != GM_CORNER]
    out = []
    for f in awards:
        speed = np.linalg.norm(np.diff(ball[f:f + 225, :2], axis=0), axis=1)
        moving = np.flatnonzero(speed > quiet / 2.5)
        if len(moving) == 0:
            continue
        delivery = f + int(moving[0])
        end = min(len(ball) - 1, delivery + 112)
        out.append({"anchor": delivery, "award": f, "min_start": f + 1,
                    "apex": round(float(ball[delivery:end, 2].max()), 2)})
    return out


def no_setpiece_inside(log, s, e):
    modes = (1, 2, 3, 4, 5, 6)
    gm = log["game_mode"][s:e]
    return not any(gm[i] in modes and gm[i] != gm[i - 1] for i in range(1, len(gm)))


def cmd_pick(argv):
    """Cheap log-only gates. Everything that survives gets rendered and gated again."""
    import glob
    rows = []
    for p in sorted(glob.glob(str(SWEEP / "*.npz"))):
        t = Path(p).stem
        shape, seed = t.rsplit("_s", 1)
        d = np.load(p)
        log = {k: d[k] for k in d.files}
        n = len(log["ball"])
        for r in find_corners(log):
            s = max(r["anchor"] - LEAD, r["min_start"])
            e = s + CLIP_FRAMES
            if e > n:
                continue
            if not continuous(log["ball"][s:e]):
                continue
            if not no_setpiece_inside(log, s, e):
                continue
            bx = log["ball"][s:e, 0]
            if np.abs(bx).min() < MIN_ABS_X:
                continue                       # play left the corner's final third
            if not (np.sign(bx) == np.sign(bx[0])).all():
                continue                       # play switched ends
            rows.append({"shape": shape, "seed": int(seed), "match": t,
                         "start": int(s), "end": int(e), "apex": r["apex"]})
    rows.sort(key=lambda r: -r["apex"])
    CACHE.mkdir(parents=True, exist_ok=True)
    SHORTLIST.write_text(json.dumps(rows, indent=1))
    by_shape = {}
    for r in rows:
        by_shape[r["shape"]] = by_shape.get(r["shape"], 0) + 1
    print(f"shortlist: {len(rows)} corner windows survive the log gates {by_shape}")
    print(f"need {N_WANT}; render-gating keeps only those with >= {MIN_ONSCREEN} "
          f"players on screen in every frame")
    return 0


# ── 3. render, gate on what is actually on screen, write ───────────────────────
def onscreen_counts(normal, empty):
    """Exact per-frame visible-player count from the render diff.

    `empty` is the identical deterministic state with all 22 players shrunk away by
    GFOOTBALL_HIDE_SLOTS, so the pixels that differ ARE the players — no colour
    thresholds, and the saturated blue/red hoardings cannot contaminate it.
    """
    import cv2
    counts, areas = [], []
    for a, b in zip(normal, empty):
        d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2) > 28
        d = cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        _n, _lab, stats, _c = cv2.connectedComponentsWithStats(d, 8)
        counts.append(sum(1 for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 60))
        areas.append(int(d.sum()))
    return np.array(counts), np.array(areas)


def compose(vis, inv):
    """The two visibility variants from one render pair, grid burned in and the red
    start-circle on the first MARK_FRAMES frames. Both variants share an identical
    opening second, so a model sees the same lead-in either way."""
    from grid import build_grid, burn_grid, circle_ball, detect_ball
    overlay = build_grid()
    spx, spy = detect_ball(vis[0], inv[0])
    fpx, fpy = detect_ball(vis[-1], inv[-1])

    def finish(seq):
        out = []
        for i, f in enumerate(seq):
            g = burn_grid(np.asarray(f), overlay)
            if i < MARK_FRAMES:
                g = circle_ball(g, spx, spy)
            out.append(g)
        return out
    return (finish(list(vis)),
            finish(list(vis[:VIS_FRAMES]) + list(inv[VIS_FRAMES:])),
            (spx, spy), (fpx, fpy))


OCCUPANCY = CACHE / "occupancy.json"


def cmd_gate(argv):
    """Measure on-screen occupancy for every shortlisted window.

    Frames are measured and DISCARDED, not kept: one 125-frame render is 230 MB
    (125 x 480 x 1280 x 3), so holding a dozen candidates would be ~3 GB. The five
    winners are re-rendered in `build`. One extra pass for the winners is much
    cheaper than carrying every candidate in memory, especially with sweep shards
    still running.

    Resumable — already-measured windows are skipped.
    """
    from lib import use_bundle
    from scenario_factory import write_scenario
    rows = json.loads(SHORTLIST.read_text())
    done = json.loads(OCCUPANCY.read_text()) if OCCUPANCY.exists() else {}
    seen = set()
    for r in rows:
        if r["match"] in seen:
            continue                       # one corner per match, for source diversity
        seen.add(r["match"])
        key = f"{r['match']}@{r['start']}"
        if key in done:
            continue
        use_bundle(G.BUNDLE_VIS)
        lvl = write_scenario(shape_spec(r["shape"]), force=False)
        vis, _b = render_window(lvl, r["seed"], r["start"], r["end"])
        empty, _b = render_window(lvl, r["seed"], r["start"], r["end"],
                                  hide_slots=",".join(ALL_SLOTS))
        counts, areas = onscreen_counts(vis, empty)
        del vis, empty
        done[key] = {"min": int(counts.min()), "med": int(np.median(counts)),
                     "p10": int(np.percentile(counts, 10)),
                     "area_med": int(np.median(areas))}
        OCCUPANCY.write_text(json.dumps(done, indent=1))
        verdict = "PASS" if counts.min() >= MIN_ONSCREEN else "fail"
        print(f"  {verdict} {key}: min on screen {counts.min():2d}, "
              f"median {int(np.median(counts)):2d}")
    ok = sum(1 for v in done.values() if v["min"] >= MIN_ONSCREEN)
    print(f"\n{len(done)} measured, {ok} pass the >= {MIN_ONSCREEN} floor "
          f"(need {N_WANT})")
    return 0


def cmd_build(argv):
    from lib import use_bundle, frames_to_mov
    from scenario_factory import write_scenario
    rows = json.loads(SHORTLIST.read_text())
    occ = json.loads(OCCUPANCY.read_text())

    # Rank by the WORST frame, not the average: the v1 defect was a good clip with a
    # dead second in it, which an average hides completely.
    cand = []
    seen = set()
    for r in rows:
        key = f"{r['match']}@{r['start']}"
        if key not in occ or r["match"] in seen:
            continue
        seen.add(r["match"])
        r.update(min_onscreen=occ[key]["min"], med_onscreen=occ[key]["med"])
        cand.append(r)
    cand.sort(key=lambda r: (-r["min_onscreen"], -r["med_onscreen"]))
    kept = [r for r in cand if r["min_onscreen"] >= MIN_ONSCREEN][:N_WANT]

    if len(kept) < N_WANT:
        print(f"only {len(kept)} of {len(cand)} windows clear the >= {MIN_ONSCREEN} "
              f"floor; need {N_WANT}. Best available:")
        for r in cand[:8]:
            print(f"   {r['match']} f{r['start']}: min {r['min_onscreen']}, "
                  f"med {r['med_onscreen']}")
        print("Sweep more matches rather than lowering the floor — the floor is the fix.")
        return 1

    for r in kept:
        use_bundle(G.BUNDLE_VIS)
        lvl = write_scenario(shape_spec(r["shape"]), force=False)
        r["vis"], _b = render_window(lvl, r["seed"], r["start"], r["end"])
        use_bundle(G.BUNDLE_INV)
        r["inv"], _b = render_window(lvl, r["seed"], r["start"], r["end"])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "full_visibility").mkdir(exist_ok=True)
    (OUT / "split_1s_4s").mkdir(exist_ok=True)
    import csv
    with open(OUT / "ground_truth.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "shape", "seed", "match", "start_frame", "end_frame",
                    "n_frames", "fps", "seconds", "start_px", "start_py",
                    "final_px", "final_py", "min_onscreen", "med_onscreen", "apex"])
        for i, r in enumerate(kept, 1):
            name = f"corner_kick_{i:02d}"
            full, split, (spx, spy), (fpx, fpy) = compose(r["vis"], r["inv"])
            frames_to_mov(full, OUT / "full_visibility" / f"{name}.mov", fps=FPS, crop_hud=False)
            frames_to_mov(split, OUT / "split_1s_4s" / f"{name}.mov", fps=FPS, crop_hud=False)
            w.writerow([name, r["shape"], r["seed"], r["match"], r["start"], r["end"],
                        CLIP_FRAMES, FPS, round(CLIP_FRAMES / FPS, 2),
                        round(spx, 1), round(spy, 1), round(fpx, 1), round(fpy, 1),
                        r["min_onscreen"], r["med_onscreen"], r["apex"]])
            print(f"wrote {name}: {CLIP_FRAMES} frames @ {FPS} fps, "
                  f"min on screen {r['min_onscreen']}")
    print(f"\n{len(kept)} clips written to {OUT}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pick"
    sys.exit({"sweep": cmd_sweep, "pick": cmd_pick, "gate": cmd_gate,
              "build": cmd_build}[cmd](sys.argv[2:]))
