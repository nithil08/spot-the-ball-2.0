"""gen_situations_v2.py — headers and goalkeeper throws, rebuilt to the corners-v2 bar.

The v1 clips in `01_headers/` and `03_goalkeeper_throws/` are real football and were
reviewed as good, but they carry two of the three defects that forced the corner
rebuild (see gen_corners_v2.py):

1. 10 fps. `ffprobe` says r_frame_rate=10/1 on every v1 clip — gfootball's native
   observation rate, not a choice. Each frame is a 100 ms sample of continuous motion,
   which reads as stop-motion. This matters MORE for headers than for corners: a header
   is a fast direction change, and at 100 ms sampling the redirection is smeared across
   one or two frames. v2 runs PSF=4 for 25 fps.

2. No occupancy gate. Corners were rebuilt because clips opened on near-empty pitch;
   headers and keeper throws were never checked for the same thing at all. v2 puts both
   through the identical render-diff gate.

Defect 3 (trailing unrelated play) was corner-specific — it came from the ball being
cleared out of the box — so there is no positional gate here. A header or a throw is
allowed to lead into open play; that IS the situation.

WHAT IS REUSED, AND WHY THAT IS SAFE

The 160 PSF=4 matches swept for corners are full match logs, so they already contain
every other kind of event: measured over that cache, 343 headers in 88 matches and 17
keeper throws in 17. Headers therefore need no new sweeping at all. Keeper throws are
scarce enough that the seed range usually has to be extended — `sweep` here is the same
resumable one-match-per-process loop, writing to the same cache.

FRAME CONSTANTS ARE SCALED, PHYSICAL ONES ARE NOT

The detectors come from pick_situations.py, which was written against PSF=10. Anything
counted in FRAMES stretches by 2.5 (the velocity gap a header is measured over, the
keeper's minimum hold, the lead-in); anything in METRES or as a cosine is physics and is
left exactly alone. Getting this backwards would silently redefine what a header is.

Run:
    python3 gen_situations_v2.py sweep --shard i/n    # only if a class comes up short
    python3 gen_situations_v2.py pick  <header|gk_throw>
    python3 gen_situations_v2.py gate  <header|gk_throw>
    python3 gen_situations_v2.py build <header|gk_throw>
"""
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen2_lib as G                                    # noqa: E402

G.PSF = 4                                               # before any env is constructed

import numpy as np                                      # noqa: E402
from gen2_lib import (ALL_SLOTS, continuous, render_window,  # noqa: E402
                      shape_spec)
from gen_corners_v2 import (CLIP_FRAMES, FPS, MIN_ONSCREEN, PROBE_EXTRA,  # noqa: E402
                            SWEEP, compose, no_setpiece_inside,
                            onscreen_counts, slide)

PSF_SCALE = 2.5             # 10 / 4 — every frame-denominated constant stretches by this
N_WANT = 10                 # 10 situations per class, as in v1
CACHE = G.CACHE / "situations_v2"

KINDS = {
    "header":   {"out": "01_headers_v2",           "stem": "header",
                 "lead": 50},   # v1 LEAD 20 x 2.5
    "gk_throw": {"out": "03_goalkeeper_throws_v2", "stem": "goalkeeper_throw",
                 "lead": 38},   # v1 LEAD 15 x 2.5
}


# ── detectors, frame constants scaled for 25 fps ───────────────────────────────
def near_count(log, t, radius=0.06):
    pl = np.vstack([log["left"][t], log["right"][t]])
    return int((np.linalg.norm(pl - log["ball"][t][:2], axis=1) < radius).sum())


def find_headers(log, z_min=1.35, z_max=2.9, turn_max=0.35, near=0.035, gap=5):
    """High ball + direction change + a body underneath.

    `gap` is v1's 2-frame velocity window x 2.5: the direction change has to be measured
    over the same amount of TIME, not the same number of frames. Left at 2 the detector
    would compare two 80 ms displacements and start calling ordinary curve a header.
    z_min/z_max/turn_max/near are metres and cosines — physics, unscaled.
    """
    ball = log["ball"]
    out = []
    for i in range(gap, len(ball) - gap):
        z = float(ball[i][2])
        if not (z_min <= z <= z_max):
            continue
        v_in = ball[i][:2] - ball[i - gap][:2]
        v_out = ball[i + gap][:2] - ball[i][:2]
        n_in, n_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
        if n_in < 0.02 or n_out < 0.02:
            continue
        cos = float(np.dot(v_in, v_out) / (n_in * n_out))
        if cos > turn_max:
            continue
        pl = np.vstack([log["left"][i], log["right"][i]])
        if float(np.min(np.linalg.norm(pl - ball[i][:2], axis=1))) > near:
            continue
        contested = near_count(log, i)
        if contested < 1:
            continue
        apex_in = float(ball[max(0, i - 45):i + 1, 2].max())     # v1 18 x 2.5
        out.append({"anchor": i, "score": contested * 10 + apex_in,
                    "detail": round(z, 2), "contested": contested})
    return out


def find_gk_throws(log, hold_min=10, z_lo=1.15, z_hi=1.75, travel_min=0.20):
    """Keeper holds at hand height, then releases a long way. Anchored on the RELEASE.

    hold_min is v1's 4 x 2.5 — the keeper must hold for the same ~0.4 s, which at 25 fps
    is ten frames. travel_min and the z band are metres and stay put.
    """
    ball, ot, op = log["ball"], log["owned_team"], log["owned_player"]
    out = []
    i, n = 0, len(ball)
    while i < n:
        if op[i] == 0 and ot[i] in (0, 1):
            team = ot[i]
            j = i
            while j < n and op[j] == 0 and ot[j] == team:
                j += 1
            hand = [k for k in range(i, j) if z_lo <= ball[k][2] <= z_hi]
            if len(hand) >= hold_min and j < n:
                rel = j - 1
                end = min(n - 1, rel + 75)               # v1 30 x 2.5
                travel = float(np.linalg.norm(ball[end][:2] - ball[rel][:2]))
                if travel >= travel_min:
                    out.append({"anchor": rel, "score": travel * 100 + len(hand),
                                "detail": round(travel, 3), "contested": 0})
            i = j
        else:
            i += 1
    return out


DETECTORS = {"header": find_headers, "gk_throw": find_gk_throws}


# ── shared window handling ─────────────────────────────────────────────────────
def window_ok(log, s, e, n):
    """The log gates that apply to every class. No positional gate: unlike a corner,
    a header or a throw is free to lead anywhere on the pitch."""
    if s < 0 or e > n:
        return False
    return continuous(log["ball"][s:e]) and no_setpiece_inside(log, s, e)


def max_safe_offset(log, s, n, limit=PROBE_EXTRA):
    off = 0
    while off < limit and window_ok(log, s + off + 1, s + off + 1 + CLIP_FRAMES, n):
        off += 1
    return off


def paths(kind):
    return (CACHE / f"shortlist_{kind}.json", CACHE / f"occupancy_{kind}.json",
            HERE / KINDS[kind]["out"])


# ── 1. sweep (shared cache with corners v2) ────────────────────────────────────
def cmd_sweep(argv):
    """Extend the shared PSF=4 sweep. Identical contract to gen_corners_v2 sweep:
    ONE match per process, exit 3 when the shard is empty."""
    import gen_corners_v2 as C
    return C.cmd_sweep(argv)


# ── 2. log gates ───────────────────────────────────────────────────────────────
def cmd_pick(argv):
    kind = argv[0]
    lead = KINDS[kind]["lead"]
    detect = DETECTORS[kind]
    rows = []
    for p in sorted(glob.glob(str(SWEEP / "*.npz"))):
        t = Path(p).stem
        shape, seed = t.rsplit("_s", 1)
        d = np.load(p)
        log = {k: d[k] for k in d.files}
        n = len(log["ball"])
        best = None
        for r in detect(log):
            s = r["anchor"] - lead
            e = s + CLIP_FRAMES
            if not window_ok(log, s, e, n):
                continue
            cand = {"shape": shape, "seed": int(seed), "match": t,
                    "start": int(s), "end": int(e), "score": round(r["score"], 2),
                    "detail": r["detail"], "contested": r["contested"],
                    "max_off": max_safe_offset(log, s, n)}
            if best is None or cand["score"] > best["score"]:
                best = cand
        if best:                       # one per match, for source diversity
            rows.append(best)
    rows.sort(key=lambda r: -r["score"])
    CACHE.mkdir(parents=True, exist_ok=True)
    paths(kind)[0].write_text(json.dumps(rows, indent=1))
    by_shape = {}
    for r in rows:
        by_shape[r["shape"]] = by_shape.get(r["shape"], 0) + 1
    print(f"{kind}: {len(rows)} windows survive the log gates {by_shape}")
    print(f"need {N_WANT} after the >= {MIN_ONSCREEN} occupancy gate")
    return 0


# ── 3. occupancy gate ──────────────────────────────────────────────────────────
def cmd_gate(argv):
    """Render each candidate, measure exact on-screen players by diffing against an
    all-hidden render, and slide the window to its best offset. Resumable."""
    from lib import use_bundle
    from scenario_factory import write_scenario
    kind = argv[0]
    budget = int(argv[1]) if len(argv) > 1 else 3 * N_WANT
    shortlist, occ_path, _ = paths(kind)
    rows = json.loads(shortlist.read_text())
    done = json.loads(occ_path.read_text()) if occ_path.exists() else {}
    n_pass = sum(1 for v in done.values() if v["min"] >= MIN_ONSCREEN)
    for r in rows[:budget]:
        key = f"{r['match']}@{r['start']}"
        if key in done:
            continue
        if n_pass >= N_WANT:
            break                      # enough winners; stop paying for renders
        use_bundle(G.BUNDLE_VIS)
        lvl = write_scenario(shape_spec(r["shape"]), force=False)
        probe_end = r["end"] + r.get("max_off", 0)
        vis, _b = render_window(lvl, r["seed"], r["start"], probe_end)
        empty, _b = render_window(lvl, r["seed"], r["start"], probe_end,
                                  hide_slots=",".join(ALL_SLOTS))
        counts, areas = onscreen_counts(vis, empty)
        del vis, empty
        off, counts, areas = slide(counts, areas)
        done[key] = {"min": int(counts.min()), "med": int(np.median(counts)),
                     "offset": int(off), "per_frame": counts.tolist()}
        occ_path.write_text(json.dumps(done, indent=1))
        ok = counts.min() >= MIN_ONSCREEN
        n_pass += ok
        print(f"  {'PASS' if ok else 'fail'} {key}: min {counts.min():2d}, "
              f"med {int(np.median(counts)):2d}, slid +{off}   [{n_pass}/{N_WANT}]")
    print(f"\n{kind}: {len(done)} measured, {n_pass} pass (need {N_WANT})")
    return 0


# ── 4. build ───────────────────────────────────────────────────────────────────
def cmd_build(argv):
    import csv
    from lib import use_bundle, frames_to_mov
    from scenario_factory import write_scenario
    kind = argv[0]
    shortlist, occ_path, out = paths(kind)
    rows = json.loads(shortlist.read_text())
    occ = json.loads(occ_path.read_text())

    cand = []
    for r in rows:
        key = f"{r['match']}@{r['start']}"
        if key not in occ or occ[key]["min"] < MIN_ONSCREEN:
            continue
        off = occ[key]["offset"]
        r.update(min_onscreen=occ[key]["min"], med_onscreen=occ[key]["med"],
                 offset=off, start=r["start"] + off,
                 end=r["start"] + off + CLIP_FRAMES)
        cand.append(r)
    cand.sort(key=lambda r: (-r["min_onscreen"], -r["score"]))
    kept = cand[:N_WANT]
    if len(kept) < N_WANT:
        print(f"only {len(kept)} of {len(occ)} measured windows clear the "
              f">= {MIN_ONSCREEN} floor; need {N_WANT}.")
        print("Gate more candidates, or sweep more seeds — do not lower the floor.")
        return 1

    for r in kept:
        use_bundle(G.BUNDLE_VIS)
        lvl = write_scenario(shape_spec(r["shape"]), force=False)
        r["vis"], _b = render_window(lvl, r["seed"], r["start"], r["end"])
        use_bundle(G.BUNDLE_INV)
        r["inv"], _b = render_window(lvl, r["seed"], r["start"], r["end"])

    out.mkdir(parents=True, exist_ok=True)
    (out / "full_visibility").mkdir(exist_ok=True)
    (out / "split_1s_4s").mkdir(exist_ok=True)
    stem = KINDS[kind]["stem"]
    with open(out / "ground_truth.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "shape", "seed", "match", "start_frame", "end_frame",
                    "n_frames", "fps", "seconds", "start_px", "start_py",
                    "final_px", "final_py", "min_onscreen", "med_onscreen",
                    "detail", "slid_frames"])
        for i, r in enumerate(kept, 1):
            name = f"{stem}_{i:02d}"
            full, split, (spx, spy), (fpx, fpy) = compose(r["vis"], r["inv"])
            frames_to_mov(full, out / "full_visibility" / f"{name}.mov",
                          fps=FPS, crop_hud=False)
            frames_to_mov(split, out / "split_1s_4s" / f"{name}.mov",
                          fps=FPS, crop_hud=False)
            w.writerow([name, r["shape"], r["seed"], r["match"], r["start"], r["end"],
                        CLIP_FRAMES, FPS, round(CLIP_FRAMES / FPS, 2),
                        round(spx, 1), round(spy, 1), round(fpx, 1), round(fpy, 1),
                        r["min_onscreen"], r["med_onscreen"], r["detail"],
                        r["offset"]])
            print(f"wrote {name}: min on screen {r['min_onscreen']}, slid +{r['offset']}")
    print(f"\n{len(kept)} {kind} clips written to {out}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    sys.exit({"sweep": cmd_sweep, "pick": cmd_pick, "gate": cmd_gate,
              "build": cmd_build}[cmd](sys.argv[2:]))
