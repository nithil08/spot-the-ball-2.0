"""gen_quick6.py — 2 clips per situation class (headers / corners / GK throws), fast.

WHY THIS EXISTS
The v2 batch was rejected as "very bouncy" and "the game lacks logic, balls are just
moving to move". Measurement (2026-08-26) cleared the engine and found three causes,
all in v2's SELECTION design. This driver reuses the v2 pipeline verbatim and changes
only those three things:

1. TEAM STRENGTH. Every v2 shape ran a difficulty gap of 0.70-0.95 (pressL 1.00/0.15,
   pressR2 0.05/1.00, boxL 1.00/0.10) with offsides OFF, to manufacture set pieces. The
   weak side sits at or below gfootball's own `11_vs_11_easy_stochastic` preset, so it
   does not press, track runners or hold a line — that is the "lacks logic". The shapes
   here run a 0.40 gap with offsides ON, which is exactly `11_vs_11_stochastic`, the
   level behind the GEN1 clips the user endorsed.

2. NO BALL-COHERENCE GATE. GEN1 (`natural_30_5s_grid`) kept a window only if the ball
   path length fell in a moderate band — "steady passing/possession, not frantic, not
   static". v2 dropped that entirely and gated on player count instead. ENGAGEMENT below
   restores it and adds the two statistics that separated the batches: how long the ball
   is loose, and how far it is from the nearest player.

3. THE OCCUPANCY FLOOR. v2 demanded >= 10 players in every frame, which selects the most
   congested, least legible moments (headers shipped at 20-27 on screen; the reference
   first-batch clip has ~5). The floor here is 6 — enough to keep the ball out of empty
   grass, not enough to drive the selection.

NOT a cause, do not re-investigate: physics_steps_per_frame. It changes the observation
and action rate only — `game_env.cpp` steps a fixed 10 ms physics tick — and ball-height
distributions over the two cached sweeps are identical (airborne ratio 1.000x).

Everything else is v2: 25 fps, 125 frames, 16x6 grid, noname bundle, both visibility
variants cut from one visible + one invisible render.

    python3 gen_quick6.py sweep --shard i/n     # one match per process, exit 3 = empty
    python3 gen_quick6.py pick                  # all three classes, log gates only
    python3 gen_quick6.py gate  <kind>          # render + occupancy, resumable
    python3 gen_quick6.py build <kind>          # render vis/inv, compose, write
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen2_lib as G                                       # noqa: E402

# PSF before any env exists — the engine pools envs and only applies it on the first.
G.PSF = 4

# ── shapes: GEN1 team strength, set-piece-friendly positions ───────────────────
# (name, ball, offsides, (left_diff, right_diff), push_left, push_right)
# The 0.40 gap and offsides=True are the point; the ball position and push keep play in
# the final third so corners and crosses still happen often enough to be affordable.
# Pushes are gentler than v2's (0.60-0.75) because a shape that starts half a team in
# the box plays like a training drill, not a match.
QSHAPES = [
    ("qflankL",  (0.75, 0.34),  True, (1.00, 0.60), 0.50, 0.35),   # crosses -> headers
    ("qboxL",    (0.90, 0.10),  True, (1.00, 0.60), 0.55, 0.40),   # corners, keeper saves
    ("qpressL",  (0.62, 0.05),  True, (1.00, 0.60), 0.45, 0.35),   # general final third
    ("qpressR2", (-0.78, 0.08), True, (0.60, 1.00), 0.35, 0.55),   # mirrored, for variety
]
_known = {s[0] for s in G.SHAPES}
G.SHAPES.extend(s for s in QSHAPES if s[0] not in _known)

import numpy as np                                         # noqa: E402
import gen_corners_v2 as C                                 # noqa: E402
import gen_situations_v2 as S                              # noqa: E402

SHAPES = [s[0] for s in QSHAPES]
SEEDS = list(range(200, 230))          # 4 x 30 = 120 matches
N_WANT = 2
MIN_ONSCREEN = 6

CACHE = G.CACHE / "quick6"
SWEEP = CACHE / "sweep"
FRAMES = CACHE / "build_frames"
OUT = HERE / "V3"

KINDS = {
    "header":   {"out": "01_headers",           "stem": "header",          "lead": 50},
    "corner":   {"out": "02_corner_kicks",      "stem": "corner_kick",     "lead": C.LEAD},
    "gk_throw": {"out": "03_goalkeeper_throws", "stem": "goalkeeper_throw", "lead": 38},
}

# ── the ball-coherence gate ────────────────────────────────────────────────────
# Thresholds are anchored on the two batches actually measured, not invented:
#
#              GEN1 (endorsed)   v2 headers (rejected)
#   loose         39.6%              65.4%
#   apex          1.41               3.16      (GK throws 4.14)
#   d_near        0.8 m              1.0 m
#
# The bounds are deliberately generous — a set piece is legitimately more airborne than
# open play, and over-tight gates on a 120-match sweep yield nothing. They exclude the
# rejected region; the RANKING inside it is what picks the best 2.
ENGAGEMENT = {"loose_max": 0.60, "apex_max": 3.20, "dnear_max": 4.00,
              "path_min": 0.25, "path_max": 1.20}
MX, MY = 52.5, 34.0 / 0.42             # normalised pitch units -> metres


def coherence(log, s, e):
    """The four numbers the gate and the ranking are built from."""
    ball = log["ball"][s:e]
    pl = np.concatenate([log["left"][s:e], log["right"][s:e]], axis=1)
    d = np.linalg.norm((pl - ball[:, None, :2]) * np.array([MX, MY]), axis=2)
    return {"loose": float((log["owned_team"][s:e] == -1).mean()),
            "apex": float(ball[:, 2].max()),
            "dnear": float(np.median(d.min(axis=1))),
            "path": float(np.linalg.norm(np.diff(ball[:, :2], axis=0), axis=1).sum())}


def engaged(c):
    return (c["loose"] <= ENGAGEMENT["loose_max"]
            and c["apex"] <= ENGAGEMENT["apex_max"]
            and c["dnear"] <= ENGAGEMENT["dnear_max"]
            and ENGAGEMENT["path_min"] <= c["path"] <= ENGAGEMENT["path_max"])


def coherence_score(c):
    """Lower is better: mostly-owned ball, kept low, kept close to a player."""
    return c["loose"] * 2.0 + c["apex"] / 3.2 + c["dnear"] / 4.0


# ── wire the v2 modules to this run ────────────────────────────────────────────
def _patch():
    C.SHAPES, C.SEEDS = SHAPES, SEEDS
    C.CACHE, C.SWEEP, C.FRAMES = CACHE, SWEEP, FRAMES
    C.SHORTLIST = CACHE / "shortlist_corner.json"
    C.OCCUPANCY = CACHE / "occupancy_corner.json"
    C.OUT = OUT / KINDS["corner"]["out"]
    C.N_WANT, C.MIN_ONSCREEN = N_WANT, MIN_ONSCREEN
    # gen_situations_v2 imported these by VALUE, so both modules must be set.
    S.SWEEP, S.CACHE, S.FRAMES = SWEEP, CACHE, FRAMES
    S.N_WANT, S.MIN_ONSCREEN = N_WANT, MIN_ONSCREEN
    S.KINDS = {k: v for k, v in KINDS.items() if k != "corner"}


_patch()


# ── 1. sweep ───────────────────────────────────────────────────────────────────
def cmd_sweep(argv):
    return C.cmd_sweep(argv)


# ── 2. log gates + the coherence gate, for all three classes at once ───────────
def _windows(kind, log, n):
    """Candidate (start, end, anchor-info) for one class in one match."""
    if kind == "corner":
        for r in C.find_corners(log):
            s = max(r["anchor"] - KINDS["corner"]["lead"], r["min_start"])
            yield s, s + C.CLIP_FRAMES, r["apex"], r
    else:
        for r in S.DETECTORS[kind](log):
            s = r["anchor"] - KINDS[kind]["lead"]
            yield s, s + C.CLIP_FRAMES, r["score"], r


def cmd_pick(argv):
    import glob
    CACHE.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(SWEEP / "*.npz")))
    print(f"picking over {len(files)} swept matches\n")
    for kind in KINDS:
        ok_fn = C.window_ok if kind == "corner" else S.window_ok
        off_fn = C.max_safe_offset if kind == "corner" else S.max_safe_offset
        rows, rejected = [], 0
        for p in files:
            t = Path(p).stem
            shape, seed = t.rsplit("_s", 1)
            d = np.load(p)
            log = {k: d[k] for k in d.files}
            n = len(log["ball"])
            best = None
            for s, e, score, r in _windows(kind, log, n):
                if not ok_fn(log, s, e, n):
                    continue
                c = coherence(log, s, e)
                if not engaged(c):
                    rejected += 1
                    continue
                cand = {"shape": shape, "seed": int(seed), "match": t,
                        "start": int(s), "end": int(e),
                        "score": round(float(score), 2),
                        "detail": r.get("detail", c["apex"]),
                        "contested": r.get("contested", 0),
                        "apex": round(c["apex"], 2), "loose": round(c["loose"], 3),
                        "dnear": round(c["dnear"], 2), "path": round(c["path"], 3),
                        "coh": round(coherence_score(c), 4),
                        "max_off": off_fn(log, s, n)}
                if best is None or cand["coh"] < best["coh"]:
                    best = cand
            if best:                      # one per match, for source diversity
                rows.append(best)
        rows.sort(key=lambda r: r["coh"])          # most coherent first
        path = (C.SHORTLIST if kind == "corner"
                else CACHE / f"shortlist_{kind}.json")
        path.write_text(json.dumps(rows, indent=1))
        by_shape = {}
        for r in rows:
            by_shape[r["shape"]] = by_shape.get(r["shape"], 0) + 1
        print(f"{kind:<9} {len(rows):>3} windows pass log + coherence gates "
              f"({rejected} rejected as incoherent) {by_shape}")
        for r in rows[:3]:
            print(f"            {r['match']}@{r['start']}  loose {r['loose']:.2f}  "
                  f"apex {r['apex']:.2f}  d_near {r['dnear']:.1f}m  path {r['path']:.2f}")
    return 0


# ── 3. occupancy gate ──────────────────────────────────────────────────────────
def cmd_gate(argv):
    kind = argv[0]
    budget = int(argv[1]) if len(argv) > 1 else 6 * N_WANT
    if kind == "corner":
        # C.cmd_gate has no budget and would render the whole shortlist; trim first so
        # only the most coherent candidates are paid for.
        rows = json.loads(C.SHORTLIST.read_text())[:budget]
        C.SHORTLIST.write_text(json.dumps(rows, indent=1))
        return C.cmd_gate(argv[1:])
    return S.cmd_gate([kind, str(budget)])


# ── 4. build ───────────────────────────────────────────────────────────────────
def cmd_build(argv):
    """Same three-phase split as v2 — a bundle is fixed at import, so the visible and
    invisible renders cannot share a process — but re-entered through THIS file so the
    patches above survive into the subprocesses."""
    import subprocess
    kind = argv[0]
    kept = (C.selected()[1] if kind == "corner" else S.selected(kind)[1])
    if len(kept) < N_WANT:
        print(f"only {len(kept)} {kind} windows clear the >= {MIN_ONSCREEN} floor; "
              f"need {N_WANT}. Gate more candidates or sweep more seeds.")
        return 1
    rc = 0
    for phase in ("render_vis", "render_inv", "compose"):
        print(f"-- {kind} {phase}", flush=True)
        rc = subprocess.run([sys.executable, __file__, phase, kind]).returncode
        if rc != 0 and phase != "compose":
            print(f"{phase} failed rc={rc}")
            return rc
    return rc


def _phase_render(kind, bundle, tag):
    if kind == "corner":
        return C.phase_render(bundle, tag)
    return S.phase_render(kind, bundle, tag)


def _phase_compose(kind):
    return C.phase_compose() if kind == "corner" else S.phase_compose(kind)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pick"
    if cmd == "render_vis":
        sys.exit(_phase_render(sys.argv[2], G.BUNDLE_VIS, "vis"))
    if cmd == "render_inv":
        sys.exit(_phase_render(sys.argv[2], G.BUNDLE_INV, "inv"))
    if cmd == "compose":
        sys.exit(_phase_compose(sys.argv[2]))
    sys.exit({"sweep": cmd_sweep, "pick": cmd_pick, "gate": cmd_gate,
              "build": cmd_build}[cmd](sys.argv[2:]))
