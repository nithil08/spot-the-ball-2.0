"""intro_clips.py — two 20-second clips in the locked format, for the experiment intro.

    python3 intro_clips.py scan                 # find 20 s passages in the GEN3 sweep cache
    python3 intro_clips.py render_vis           # 500-frame visible pass  (bundle: gen3)
    python3 intro_clips.py render_inv           # 500-frame invisible pass (bundle: gen3_ball_invisible)
    python3 intro_clips.py compose              # grid + red circle -> .mov + ground truth

WHY THIS IS NOT GEN3 WITH A BIGGER NUMBER
  A benchmark clip is 125 frames and is chosen for its END STATE — an exact player count
  and a target ball cell. An intro clip is chosen for the opposite reason: it has to be
  the most ordinary 20 s of football in the cache, because its job is to teach a viewer
  what the render, the kits and the grid look like before the real trials start. So there
  is no count probe, no camera offset solve and no cell target here; the camera stays at
  offset 0, which is also the widest, cleanest framing the engine gives.

  Everything else is the locked format, deliberately: 25 fps, 16x6 grid, noname bundles,
  per-team keeper kits, invisible officials, red circle on the ball's start position for
  the first 8 frames, and one render pair per clip so the visible and hidden variants are
  the same play frame for frame.

WHAT A 20 S WINDOW COSTS
  Continuity over 500 frames is ~20x rarer than over 125: any goal, any restart awarded
  mid-window and any engine re-spot disqualifies it, and over 20 s of football a free
  kick or a throw-in is close to certain. `scan` reports the funnel so the gates can be
  judged rather than guessed at.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GEN3 = HERE.parent / "GEN3"
sys.path.insert(0, str(GEN3))

import gen3_lib as G                                                  # noqa: E402

CLIP_FRAMES = 500              # 20.0 s at 25 fps
VIS_FRAMES = G.VIS_FRAMES      # the 1 s the hidden variant keeps visible
SKIP_START = 25                # players are still leaving their scenario placement
STRIDE = 5

CACHE = HERE / "_cache"
PICKS = CACHE / "picks.json"
OUT = HERE / "clips"

# The coherence bands in gen3_lib are per-125-frame-window; `path` is a SUM over the
# window, so it is the one that has to be rescaled. The others are a rate, a maximum and
# a median, and carry over as they are — except apex, which is a max over 4x as many
# frames and so is relaxed: one clearance above head height in 20 s is normal football,
# whereas in 5 s it means the passage IS the clearance.
SCALE = CLIP_FRAMES / G.CLIP_FRAMES
BANDS = {"loose_max": 0.55, "apex_max": 4.20, "dnear_max": 3.50,
         "path_min": 0.33 * SCALE, "path_max": 0.95 * SCALE}


def matches_used_by_gen3():
    """GEN3's 24 source matches — an intro clip must not preview a graded item."""
    p = GEN3 / "_cache" / "picks.json"
    if not p.exists():
        return set()
    return {q["match"] for q in json.loads(p.read_text())}


# An intro clip has a job the graded clips do not: it has to SHOW the viewer what the
# material is. The coherence score alone ranks a 20 s spell of one team keeping the ball
# in its own half at the very top — calm, owned, low, close — and that teaches nothing
# about contested play. These three are the gate for "worth watching", checked in `pick`
# rather than in `scan` so the candidate pool stays reusable.
WATCHABLE = {"changes_min": 3,      # possession actually changes hands
             "travel_min": 0.50,    # the ball covers a quarter of the pitch's length
             "ymax_max": 0.38}      # never hugs a touchline, where the stands come in


def _watchable(log, s, e):
    own = log["owned_team"][s:e]
    held = own[own >= 0]
    b = log["ball"][s:e]
    return {"changes": int((np.diff(held) != 0).sum()) if len(held) > 1 else 0,
            "travel": round(float(b[:, 0].max() - b[:, 0].min()), 4),
            "ymax": round(float(np.abs(b[:, 1]).max()), 4)}


def engaged(c):
    return (c["loose"] <= BANDS["loose_max"]
            and c["apex"] <= BANDS["apex_max"]
            and c["dnear"] <= BANDS["dnear_max"]
            and BANDS["path_min"] <= c["path"] <= BANDS["path_max"])


def cmd_scan(argv):
    used = matches_used_by_gen3()
    funnel = {"windows": 0, "continuous": 0, "no_restart": 0, "engaged": 0}
    cands = []
    for f in sorted(G.SWEEP.glob("*.npz")):
        match = f.stem
        if match in used:
            continue
        log = {k: v for k, v in np.load(f).items()}
        n = len(log["ball"])
        for s in range(SKIP_START, n - CLIP_FRAMES + 1, STRIDE):
            e = s + CLIP_FRAMES
            funnel["windows"] += 1
            if not G.continuous(log["ball"][s:e]):
                continue
            funnel["continuous"] += 1
            if not G.no_setpiece_inside(log, s, e):
                continue
            funnel["no_restart"] += 1
            if (log["score"][s] != log["score"][e - 1]).any():
                continue
            c = G.coherence(log, s, e)
            if not engaged(c):
                continue
            funnel["engaged"] += 1
            shape, seed = match.rsplit("_s", 1)
            cands.append({"match": match, "shape": shape, "seed": int(seed),
                          "start": s, "end": e, "coh": round(G.coherence_score(c), 4),
                          **{k: round(v, 4) for k, v in c.items()},
                          **_watchable(log, s, e),
                          "startown": G.start_possessor(log["owned_team"], s, e)})
    print("funnel: " + "  ".join(f"{k}={v}" for k, v in funnel.items()))
    cands.sort(key=lambda d: d["coh"])
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "candidates.json").write_text(json.dumps(cands, indent=2))
    print(f"{len(cands)} windows from {len({c['match'] for c in cands})} matches")
    for c in cands[:15]:
        print(f"  {c['match']:>14} {c['start']:>5}-{c['end']:<5} coh={c['coh']:.3f} "
              f"loose={c['loose']:.2f} apex={c['apex']:.2f} dnear={c['dnear']:.2f} "
              f"path={c['path']:.2f} start={'blue' if c['startown'] == 0 else 'red'}")
    return 0


def cmd_pick(argv):
    """The two calmest windows, from different shapes and opposite starting teams.

    Different shapes so the pair does not show the same formation twice, and opposite
    starting teams for the same reason the graded batch balances them: an intro that
    always opens with blue in possession teaches a bias.
    """
    cands = json.loads((CACHE / "candidates.json").read_text())
    pool = [c for c in cands
            if c["changes"] >= WATCHABLE["changes_min"]
            and c["travel"] >= WATCHABLE["travel_min"]
            and c["ymax"] <= WATCHABLE["ymax_max"]]
    print(f"watchable: {len(pool)} of {len(cands)} windows, "
          f"{len({c['match'] for c in pool})} matches")
    picks = []
    for c in pool:
        if any(p["shape"] == c["shape"] or p["match"] == c["match"] for p in picks):
            continue
        if any(p["startown"] == c["startown"] for p in picks):
            continue
        picks.append(c)
        if len(picks) == 2:
            break
    if len(picks) < 2:
        sys.exit("could not find two windows meeting the shape/team spread rule")
    for i, p in enumerate(picks, 1):
        p["clip"] = f"intro_{i:02d}"
        p["offset_x"] = p["offset_y"] = 0.0
    PICKS.write_text(json.dumps(picks, indent=2))
    for p in picks:
        print(f"  {p['clip']}: {p['match']} frames {p['start']}-{p['end']} "
              f"coh={p['coh']:.3f} start={'blue' if p['startown'] == 0 else 'red'}")
    return 0


def _render(bundle, tag):
    """One pass per clip, frames written straight to a uint8 memmap.

    500 frames of 480x1280x3 is 884 MB, and compose needs the visible and invisible pass
    of the same clip at once. Holding four of those in a Python list is how this falls
    over, so each pass lands on disk as a raw memmap and compose streams them.
    """
    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle(bundle)
    for name, *_ in G.SHAPES:
        write_scenario(G.shape_spec(name), force=False)
    RAW.mkdir(parents=True, exist_ok=True)
    for p in json.loads(PICKS.read_text()):
        dst = RAW / f"{p['clip']}_{tag}.npy"
        if dst.exists():
            print(f"  [{tag}] {p['clip']}: cached")
            continue
        frames, _ = G.render_window(f"g3_{p['shape']}", p["seed"], p["start"], p["end"],
                                    hide_slots="",
                                    offset=(p["offset_x"], p["offset_y"]))
        arr = np.stack(frames)
        np.save(dst, arr)
        print(f"  [{tag}] {p['clip']}: {arr.shape[0]} frames -> {dst.name}")
        del frames, arr
    return 0


RAW = CACHE / "raw"


def cmd_render_vis(argv):
    return _render(G.BUNDLE_VIS, "vis")


def cmd_render_inv(argv):
    return _render(G.BUNDLE_INV, "inv")


HEADER = ("clip,shape,seed,match,start_frame,end_frame,frames,seconds,"
          "start_team,ball_start_cell,ball_start_px,ball_start_py,"
          "ball_final_cell,ball_final_px,ball_final_py,pitch_fraction,coherence")
TEAM = {0: "blue_left", 1: "red_right"}


def green_fraction(frame):
    f = frame.astype(float)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return float(((g > r * 1.04) & (g > b * 1.04) & (g > 12)).mean())


def cmd_compose(argv):
    from grid import build_grid, burn_grid, circle_ball, detect_ball_or_none
    from lib import frames_to_mov
    overlay = build_grid()
    rows = [HEADER]
    for p in json.loads(PICKS.read_text()):
        vis = np.load(RAW / f"{p['clip']}_vis.npy", mmap_mode="r")
        inv = np.load(RAW / f"{p['clip']}_inv.npy", mmap_mode="r")

        start = next((b for b in (detect_ball_or_none(np.asarray(vis[i]),
                                                      np.asarray(inv[i]))
                                  for i in range(G.MARK_FRAMES)) if b), None)
        if start is None:
            sys.exit(f"{p['clip']}: no ball in the first {G.MARK_FRAMES} frames")
        spx, spy = start
        final = detect_ball_or_none(np.asarray(vis[-1]), np.asarray(inv[-1]))
        if final is None:                     # occluded on the last frame; step back
            for k in range(2, 13):
                final = detect_ball_or_none(np.asarray(vis[-k]), np.asarray(inv[-k]))
                if final is not None:
                    print(f"    final ball occluded, measured {k - 1} frame(s) earlier")
                    break
        fpx, fpy = final if final else (float("nan"), float("nan"))

        def stream(hide_after=None):
            for i in range(len(vis)):
                src = inv if (hide_after is not None and i >= hide_after) else vis
                g = burn_grid(np.asarray(src[i]), overlay)
                yield circle_ball(g, spx, spy) if i < G.MARK_FRAMES else g

        greens = []
        full = []
        for i, f in enumerate(stream()):
            full.append(f)
            if i % 50 == 0:
                greens.append(green_fraction(f))
        frames_to_mov(full, OUT_FULL / f"{p['clip']}.mov", fps=G.FPS, crop_hud=False)
        del full
        hidden = list(stream(hide_after=VIS_FRAMES))
        frames_to_mov(hidden, OUT_HIDE / f"{p['clip']}.mov", fps=G.FPS, crop_hud=False)
        del hidden

        pf = float(np.min(greens))
        rows.append(",".join(map(str, [
            p["clip"], p["shape"], p["seed"], p["match"], p["start"], p["end"],
            len(vis), round(len(vis) / G.FPS, 2), TEAM[p["startown"]],
            G.cell_of(spx, spy), round(spx, 1), round(spy, 1),
            G.cell_of(fpx, fpy), round(fpx, 1), round(fpy, 1),
            round(pf, 4), p["coh"]])))
        print(f"  [compose] {p['clip']}: {len(vis)} frames, ball "
              f"{G.cell_of(spx, spy)} -> {G.cell_of(fpx, fpy)}, pitch {pf:.1%}")
    (OUT / "ground_truth.csv").write_text("\n".join(rows) + "\n")
    print(f"compose: {len(rows) - 1} clips -> {OUT}")
    return 0


OUT_FULL = OUT / "full_visibility"
OUT_HIDE = OUT / "split_1s_19s"

if __name__ == "__main__":
    OUT_FULL.mkdir(parents=True, exist_ok=True)
    OUT_HIDE.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "scan"
    table = {"scan": cmd_scan, "pick": cmd_pick, "render_vis": cmd_render_vis,
             "render_inv": cmd_render_inv, "compose": cmd_compose}
    if cmd not in table:
        sys.exit(f"unknown phase {cmd!r}; known: {' '.join(table)}")
    sys.exit(table[cmd](sys.argv[1 + 1:]))
