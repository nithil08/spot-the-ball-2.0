"""gen4.py — the GEN4 batch: 24 clips levelled by the OPENING headcount.

    python3 gen4.py plan                     # which matches to probe, at what offset
    python3 gen4.py probe    [--shard i/n]   # EXACT counts on a 5-frame grid, whole match
    python3 gen4.py windows                  # legal windows joined to their counts (free)
    python3 gen4.py ballpix  [--shard i/n]   # ball pixel at the candidate end frames
    python3 gen4.py select                   # the joint assignment -> 24 picks
    python3 gen4.py render_vis / render_inv [--shard i/n]
    python3 gen4.py audit                    # ball at both ends AND in view throughout
    python3 gen4.py compose                  # clips + ground truth
    python3 gen4.py verify

Read GEN3_HARD's gen3.py first: the engine plumbing, the exit-code protocol, the bundle
rule and the leak workaround are all inherited unchanged. What follows is only what GEN4
does differently, and why.

THE PROBE MEASURES THE WHOLE MATCH, NOT A DOZEN END FRAMES
  GEN3 probed only the frames a clip could END on, because the level was the end count.
  GEN4's level is the OPENING count, and a window's start frame is never another window's
  end frame, so none of that data transfers. Since replay is the entire cost of a probe
  and kept frames are free, GEN4 keeps a 5-frame grid across the whole replay instead —
  one probe per match now answers both ends of every window it could ever supply.

THE BALL MUST BE IN THE CAMERA'S VIEW ON EVERY FRAME, AND THAT IS CHECKED IN THE AUDIT
  Not as a phase of its own: measuring it for every candidate window would mean two plate
  replays per match and ~2.7 GB of transient frames to diff, for a constraint that only
  ever binds on the 24 windows that ship. The audit already holds each picked window's
  visible and invisible renders, so a frame showing no ball is free to spot there. Only
  those frames need the plate pair — which separates a ball behind a player (fine) from a
  ball outside the shot (fatal), and costs two replays on the few clips that have a gap.

NO SITUATION QUOTA
  See gen4_lib. The opening count and the situation are not independent quantities, so
  they cannot both be commanded; the level wins and the mix is reported.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen4_lib as G                                                  # noqa: E402

EXIT_DONE = 3

SEEDS = list(range(300, 350))          # 50 seeds x 14 shapes = 700 matches
SWEEP_STEPS = 2750                     # ~110 s of match, matching the GEN2 sweep length

# Windows may not open in the first second of a shape that is not a kick-off: the players
# are still accelerating out of their scenario placement and it does not read as live play.
# Kick-off windows are exempt by construction — that IS the restart.
SKIP_START = 25
# A restart clip has to SHOW its restart, so the window may slide later only until its
# first frame reaches the DELIVERY — `_candidates` caps every offset at `anchor - base`
# on top of this number, which for a restart is exactly `LEAD[kind]`.
#
# The cap used to be this number alone, and that was a bug: the base is already
# `anchor - LEAD`, so a 50-frame slide put the start up to 40 frames PAST the delivery
# and the restart happened before the clip opened. Every corner in the first build was
# cut that way — the ball was in the six-yard box on frame 0 and the clip labelled
# "corner" never showed one. A slide can only ever move the delivery EARLIER in the clip,
# never later, so a cap larger than the lead cannot mean what this comment once claimed.
SLIDE = {"corner": 50, "kickoff": 50, "gk_throw": 50, "open": 10_000}
# 4, not 5, and the number is load-bearing. The probe keeps a grid of frames and a window
# needs BOTH its ends on that grid; the window is 125 frames, so its last frame sits 124
# after its first. 124 is divisible by 4 and not by 5 — on a 5-grid every window would
# have its opening frame measured and its closing frame missed.
STRIDE = 4

CAND = G.CACHE / "shortlist.json"
WINDOWS = G.CACHE / "windows.json"
NATURAL = G.CACHE / "natural"
PLAN = G.CACHE / "plan.json"
COUNTS = G.CACHE / "counts"
PICKS = G.CACHE / "picks.json"
RENDERS = G.CACHE / "renders"
ABSENT = G.CACHE / "ball_absent.json"     # {match: [[lo, hi], ...]} — see cmd_audit

# ── camera calibration (measured; see data/code/gen_30_spread_code.py) ──────────
KX = -26.58                 # px per world unit of offset X, linear over the whole range
CROSS_YX = -2.92            # px of X shift per unit of offset Y (perspective)
OFFY_TABLE = [-20, -16, -14, -10, -6, -3, 0, 3, 6, 9, 12]
DPY_TABLE = [-228.3, -188.7, -168.0, -124.2, -77.3, -39.8, 0.0, 42.2, 87.4, 135.7, 187.0]
OFFX_RANGE, OFFY_RANGE = (-26.0, 26.0), (-12.0, 12.0)
SAFE_PX, SAFE_PY = (55.0, 1225.0), (35.0, 445.0)
# Engine-side bounds on the framing target (match.cpp) and the obs->world scales. The
# linear calibration holds only while the target stays INSIDE these: once the clamp bites,
# the engine's yaw term couples the axes and the ball lands nowhere near the solve.
FRAMED_W, FRAMED_H = 55.0 * 0.95, 36.0 * 0.80
X_FIELD_SCALE, Y_FIELD_SCALE = 54.4, -83.6
CLAMP_MARGIN = 0.88

DOWN = 2                    # the probe renders at half size; a body is still tens of px
MIN_PIX = 40                # changed (half-size) pixels before a player counts as in frame


def parse_shard(argv):
    for i, a in enumerate(argv):
        if a.startswith("--shard"):
            spec = a.split("=", 1)[1] if "=" in a else argv[i + 1]
            n, d = spec.split("/")
            return int(n), int(d)
    return 0, 1


def solve_offsets(px0, py0, tgt_px, tgt_py, world=None):
    """Camera offset that moves the ball from its natural pixel to the target pixel.

    `world` is the ball's (x, y) in world units at the final frame; it limits the offset so
    the framing target stays inside the engine's clamp envelope and the linear calibration
    stays valid. A clip whose ball already sits near a pitch end therefore gets a SMALLER
    offset rather than a wrong one.
    """
    tgt_px = float(np.clip(tgt_px, *SAFE_PX))
    tgt_py = float(np.clip(tgt_py, *SAFE_PY))
    offy = float(np.clip(np.interp(tgt_py - py0, DPY_TABLE, OFFY_TABLE), *OFFY_RANGE))
    offx = float(np.clip((tgt_px - px0 - CROSS_YX * offy) / KX, *OFFX_RANGE))
    if world is not None:
        wx, wy = world
        lx, ly = FRAMED_W * CLAMP_MARGIN, FRAMED_H * CLAMP_MARGIN
        offx = float(np.clip(offx, -lx - wx, lx - wx))
        offy = float(np.clip(offy, -ly - wy, ly - wy))
    return round(offx, 3), round(offy, 3)


def cell_centre(r, c):
    return (c + 0.5) * G.CELL_W, (r + 0.5) * G.CELL_H


def cell_of(px, py):
    c = min(max(int(px // G.CELL_W), 0), G.COLS - 1)
    r = min(max(int(py // G.CELL_H), 0), G.ROWS - 1)
    return f"{chr(ord('A') + r)}{c + 1}"


# ══ phase 1: sweep ══════════════════════════════════════════════════════════════
def jobs():
    return list(G.sweep_jobs(SEEDS))


def _todo(shard_i, shard_n, done_dir, suffix=".npz"):
    return [(sh, sd) for k, (sh, sd) in enumerate(jobs())
            if k % shard_n == shard_i
            and not (done_dir / f"{G.tag(sh, sd)}{suffix}").exists()]


PER_PROCESS = 15            # well inside the ~46-env leak, one env per sweep match


def cmd_sweep(argv):
    from lib import use_bundle
    from scenario_factory import write_scenario
    shard_i, shard_n = parse_shard(argv)
    use_bundle(G.BUNDLE_VIS)
    G.SWEEP.mkdir(parents=True, exist_ok=True)
    for name, *_ in G.SHAPES:
        write_scenario(G.shape_spec(name), force=(shard_i == 0))
    todo = _todo(shard_i, shard_n, G.SWEEP)
    if not todo:
        print(f"shard {shard_i}/{shard_n}: sweep complete")
        return EXIT_DONE
    print(f"shard {shard_i}/{shard_n}: {len(todo)} matches left", flush=True)
    for shape, seed in todo[:PER_PROCESS]:
        log = G.run_log(G.SCEN_PREFIX + shape, seed, SWEEP_STEPS)
        np.savez_compressed(G.SWEEP / f"{G.tag(shape, seed)}.npz", **log)
        print(f"  [sweep] {G.tag(shape, seed)}: {len(log['ball'])} frames, "
              f"score {log['score'][-1].tolist()}", flush=True)
    return 0


# ══ phase 2: shortlist ══════════════════════════════════════════════════════════
def window_ok(log, kind, s, e, n):
    if s < 0 or e > n:
        return False
    if kind != "kickoff" and s < SKIP_START:
        return False
    return G.continuous(log["ball"][s:e]) and G.no_setpiece_inside(log, s, e)


def _candidates(log, kind, n):
    """Every legal (start, slide) for one class in one match, with its coherence."""
    out = []
    if kind == "open":
        anchors = [{"anchor": s + G.LEAD.get("open", 0), "min_start": s, "award": s}
                   for s in range(SKIP_START, n - G.CLIP_FRAMES, STRIDE)]
    else:
        anchors = G.DETECTORS[kind](log)
    for r in anchors:
        base = max(r["anchor"] - G.LEAD.get(kind, 0), r["min_start"])
        # Snap onto the probe grid, upwards: downwards could put the start before
        # `min_start`, which for a restart means opening the clip before the respot.
        base = -(-base // STRIDE) * STRIDE
        # Sliding the window later keeps the situation on screen while changing which
        # frame is the LAST one — which is the frame the count is measured on. For a
        # restart the slide stops at the delivery itself, so the restart is always on
        # camera; open play has nothing to stay near, so it slides freely.
        cap = SLIDE[kind] if kind == "open" else min(SLIDE[kind], r["anchor"] - base)
        offsets = [0] if kind == "open" else range(0, max(cap, 0) + 1, STRIDE)
        for off in offsets:
            s = base + off
            e = s + G.CLIP_FRAMES
            if not window_ok(log, kind, s, e, n):
                continue
            if kind != "open" and s > r["anchor"]:
                continue                       # the restart would precede the clip
            c = G.coherence(log, s, e)
            if not G.engaged(kind, c):
                continue
            out.append({"start": int(s), "end": int(e), "anchor": int(r["anchor"]),
                        "coh": round(G.coherence_score(c), 4),
                        **{k: round(v, 3) for k, v in c.items()}})
    return out


MAX_WINDOWS, MIN_SEP = 40, 25


def _spread(cands, kind):
    """The best windows by coherence, but for open play, forced apart in TIME.

    Open play has hundreds of anchors per match, and taking the top 40 outright would
    cluster them on one passage — every window in a cluster ends on nearly the same frame,
    so they all measure nearly the same player count, which is exactly the freedom this
    list exists to provide. MIN_SEP buys a spread of end frames for a negligible coherence
    cost.

    A RESTART CLASS IS THE OPPOSITE CASE and must not be thinned. Its windows are the
    slide positions around one anchor, only SLIDE frames apart by construction, so a 25
    frame separation rule would discard all but two of the seven — and for corners, which
    run about one usable match in eighty, those slide positions are the only count
    flexibility that match will ever offer.
    """
    if kind != "open":
        return sorted(cands, key=lambda r: r["coh"])[:MAX_WINDOWS]
    kept = []
    for c in sorted(cands, key=lambda r: r["coh"]):
        if all(abs(c["start"] - k["start"]) >= MIN_SEP for k in kept):
            kept.append(c)
        if len(kept) >= MAX_WINDOWS:
            break
    return kept


# Scarcity order: a match that can supply a corner is spent on the corner, because corners
# are ~0.03 per match while open play is ~47. Assigning the other way round would burn the
# rare matches on the abundant class.
CLASS_ORDER = ["corner", "gk_throw", "kickoff", "open"]


def cmd_shortlist(argv):
    """Every cached match that can supply at least one legal window, deduplicated by PLAY.

    GEN3 stopped at the first class a match could supply, because its quota made classes
    scarce. GEN4 has no quota, so a match is kept if it can supply anything at all, and
    what it supplies is worked out later against the counts.

    Deduplication is by `canonical_match`: `mid_bal`, `mid_even` and `mid_even2` are one
    scenario spec, so their identically-seeded sweeps are the same match three times.
    Keeping all three is how GEN3_HARD shipped one kick-off as three clips.
    """
    import glob
    files = sorted(glob.glob(str(G.SWEEP / "*.npz")))
    rows, seen, dropped = [], set(), 0
    for f in files:
        tag = Path(f).stem
        canon = G.canonical_match(tag)
        if canon in seen:
            dropped += 1
            continue
        shape, seed = canon.rsplit("_s", 1)
        d = np.load(G.SWEEP / f"{tag}.npz")
        log = {k: d[k] for k in d.files}
        n = len(log["ball"])
        kinds = {k: len(_candidates(log, k, n)) for k in CLASS_ORDER}
        if not any(kinds.values()):
            continue
        seen.add(canon)
        rows.append({"match": canon, "file": tag, "shape": shape, "seed": int(seed),
                     "kinds": kinds})
    G.CACHE.mkdir(parents=True, exist_ok=True)
    CAND.write_text(json.dumps(rows, indent=1))
    have = {k: sum(1 for r in rows if r["kinds"][k]) for k in CLASS_ORDER}
    print(f"shortlist: {len(rows)} distinct matches ({dropped} dropped as duplicate "
          f"plays of one already kept)")
    print("  matches able to supply each class:", have)
    return 0


# ══ phase 3: plan — which matches to probe, and at what camera offset ═══════════
# One offset per MATCH, from a schedule spanning the reachable framing space; see GEN3's
# note on why the offset is fixed before the probe rather than solved per window.
OFFX_SCHEDULE = [0.0, -12.0, 12.0, -20.0, 20.0, -6.0, 6.0, -24.0, 24.0, -16.0, 16.0]
OFFY_SCHEDULE = [0.0, 6.0, -5.0, 10.0, -9.0, 3.0, -2.0, 8.0, -7.0]

# The probe replays to PROBE_END and its cost is linear in that number. 1600 frames still
# offers ~300 window positions per match. The SCARCE classes are exempt: corners run about
# one usable match in eighty and are spread evenly over the 2750-frame sweep, so capping
# them throws away a third of the few that exist.
PROBE_END = 1600
PROBE_END_SCARCE = 2750
SCARCE = ("corner", "gk_throw", "kickoff")

# How many matches to probe per class. The probe is 23 replays per match and is the whole
# cost of the batch, so the budget goes where the scarcity is — and where the SLOTS are.
# GEN4 is filling gaps in an existing batch, not building 24 clips from nothing, so these
# are sized for the handful of (level, class) slots `needs.json` actually asks for.
# open is 0 on purpose: the 21 matches probed under the first, shape-balanced plan are all
# open-play capable and cover the one or two open slots this batch still needs. They stay
# in the plan because they are already measured; adding more would be paying for windows
# nothing is asking for.
PROBE_BUDGET = {"corner": 999, "gk_throw": 22, "kickoff": 22, "open": 0}


def cmd_plan(argv):
    """Pick the matches to probe and give each one a camera offset.

    Budgeted BY CLASS, not round-robin by shape. A pool assembled for variety of shapes is
    the right pool when the situation mix is free; the moment a quota comes back it is the
    wrong one, because open play is everywhere and corners are not — a shape-balanced 60
    contained zero corner matches while the batch needed five.
    """
    rows = json.loads(CAND.read_text())
    old = {p["match"]: p for p in json.loads(PLAN.read_text())} if PLAN.exists() else {}
    budget = dict(PROBE_BUDGET)
    if "--more" in argv:
        kind = argv[argv.index("--more") + 1]
        budget[kind] = budget.get(kind, 0) + int(argv[argv.index("--more") + 2])
    plan, taken = [], set()
    # Scarcest class first: a match able to supply a corner AND open play must be spent on
    # the corner, because open play has six hundred alternatives and the corner has
    # seventeen.
    for kind in CLASS_ORDER:
        n = 0
        for r in sorted(rows, key=lambda r: (r["shape"], r["seed"])):
            if n >= budget.get(kind, 0):
                break
            if r["match"] in taken or not r["kinds"][kind]:
                continue
            taken.add(r["match"])
            n += 1
            if r["match"] in old:
                plan.append({**old[r["match"]], "kind": kind})
                continue
            k = len(plan)
            plan.append({"match": r["match"], "file": r["file"], "shape": r["shape"],
                         "seed": r["seed"], "kinds": r["kinds"], "kind": kind,
                         "offset_x": OFFX_SCHEDULE[k % len(OFFX_SCHEDULE)],
                         "offset_y": OFFY_SCHEDULE[k % len(OFFY_SCHEDULE)],
                         "probe_end": (PROBE_END_SCARCE if kind in SCARCE
                                       else PROBE_END)})
    # Matches already probed under the old plan stay in it: their counts are on disk and
    # dropping them would throw away work for nothing.
    for m, p_ in old.items():
        if m not in taken:
            plan.append({**p_, "kind": p_.get("kind", "open")})
    PLAN.write_text(json.dumps(plan, indent=1))
    got = {k: sum(1 for p in plan if p["kind"] == k) for k in CLASS_ORDER}
    done = sum(1 for p in plan if (COUNTS / f"{p['match']}.npz").exists())
    print(f"plan: {len(plan)} matches, {done} already probed")
    print("  per class:", got)
    return 0


# ══ phase 4: probe — exact counts on a 5-frame grid across the whole match ═════
def _planned(shard_i, shard_n, done_dir, suffix=".npz"):
    plan = json.loads(PLAN.read_text())
    return [p for i, p in enumerate(plan)
            if i % shard_n == shard_i
            and not (done_dir / f"{p['match']}{suffix}").exists()]


GRID = STRIDE              # the probe grid IS the candidate grid — see STRIDE


def probe_match(p):
    """Per-frame, per-player in-shot map for one match at its shipping offset.

    A plate with all 22 hidden, then 22 passes each showing exactly one player, diffed.
    Hiding is render-only, so all 23 passes replay the identical match and the measurement
    cannot perturb what it measures.

    TWO THINGS DIFFER FROM GEN3'S PROBE, AND BOTH ARE WHY ITS COUNTS WERE WRONG:

      * BODY AND SHADOW ARE SEPARATED. A player outside the frame still casts a shadow
        into it, and a diff cannot tell the two apart by area alone. Shadow pixels are
        strictly darker than the plate and barely change its hue; body pixels are not.
        GEN3 counted a shadow as a person on two of its 24 clips.
      * THE FLOOR IS LOW. At down=2 GEN3 required 40 changed pixels, about 160 at full
        size, which is more than a player clipped by the frame edge leaves behind. Six of
        its counts were short by exactly such a body. MIN_BODY_HALF is 6.

    The 24 that ship are re-measured at full resolution in `verify_final_frame`; this pass
    exists to make selection see the right numbers, and the two agree because the rule is
    the same.
    """
    off = (p["offset_x"], p["offset_y"])
    lvl = G.SCEN_PREFIX + p["shape"]
    end = min(p["probe_end"], G.SWEEP_STEPS if hasattr(G, "SWEEP_STEPS") else 2750)
    frames = list(range(0, end, GRID))
    keep = set(frames)

    plate, _ = G.render_window(lvl, p["seed"], 0, end, keep=keep, down=DOWN,
                               hide_slots=",".join(G.ALL_SLOTS), offset=off)
    plate = np.array(plate, dtype=np.int16)
    body = np.zeros((len(G.ALL_SLOTS), len(plate)), dtype=bool)
    any_ = np.zeros_like(body)
    for si, slot in enumerate(G.ALL_SLOTS):
        hide = ",".join(s for s in G.ALL_SLOTS if s != slot)          # show ONLY this one
        solo, _ = G.render_window(lvl, p["seed"], 0, end, keep=keep, down=DOWN,
                                  hide_slots=hide, offset=off)
        solo = np.array(solo, dtype=np.int16)
        d = np.abs(solo - plate)
        changed = d.sum(axis=3) > 30
        # shadow: darker than the plate everywhere, and not a hue change
        darker = (solo.sum(axis=3) < plate.sum(axis=3)) & (d.max(axis=3) < 60)
        any_[si] = changed.reshape(len(plate), -1).sum(axis=1) > G.MIN_BODY_HALF
        body[si] = (changed & ~darker).reshape(len(plate), -1).sum(axis=1) \
            > G.MIN_BODY_HALF
    COUNTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(COUNTS / f"{p['match']}.npz", body=body, any=any_,
                        frames=np.array(frames))
    n = body.sum(axis=0)
    print(f"  [probe] {p['match']} off=({off[0]:+.0f},{off[1]:+.0f}) "
          f"{len(frames)} frames, counts {n.min()}..{n.max()}, "
          f"levels present {sorted(set(int(v) for v in n) & set(G.START_COUNTS))}",
          flush=True)


def cmd_probe(argv):
    from lib import use_bundle
    from scenario_factory import write_scenario
    shard_i, shard_n = parse_shard(argv)
    use_bundle(G.BUNDLE_VIS)
    for name, *_ in G.SHAPES:
        write_scenario(G.shape_spec(name), force=False)
    todo = _planned(shard_i, shard_n, COUNTS)
    if not todo:
        print(f"shard {shard_i}/{shard_n}: probe complete")
        return EXIT_DONE
    print(f"shard {shard_i}/{shard_n}: {len(todo)} matches left", flush=True)
    probe_match(todo[0])            # ONE match per process: 23 envs, and it dies at ~46
    return 0


# ══ phase 4b: windows — legal windows joined to their measured counts ══════════
def cmd_windows(argv):
    """Every legal window of every probed match, with the count at BOTH ends. Free.

    Kept separate from `select` so the pool can be inspected before any assignment is
    attempted: when a level turns out to be unreachable, the question is always whether
    the windows do not exist or the assignment could not use them.
    """
    plan = json.loads(PLAN.read_text())
    out, missing = [], 0
    for p in plan:
        cf = COUNTS / f"{p['match']}.npz"
        if not cf.exists():
            missing += 1
            continue
        z = np.load(cf)
        at = {int(f): i for i, f in enumerate(z["frames"])}
        body = z["body"]
        d = np.load(G.SWEEP / f"{p['file']}.npz")
        log = {k: d[k] for k in d.files}
        n = len(log["ball"])
        owners = log["owned_team"]
        for kind in CLASS_ORDER:
            for w in _candidates(log, kind, n):
                s, e = w["start"], w["end"] - 1
                if s not in at or e not in at:
                    continue
                own = G.start_possessor(owners, w["start"], w["end"])
                if own < 0:
                    continue          # no starting team: nothing to balance the level on
                out.append({"match": p["match"], "file": p["file"], "shape": p["shape"],
                            "seed": p["seed"], "kind": kind,
                            "offset_x": p["offset_x"], "offset_y": p["offset_y"],
                            "start": w["start"], "end": w["end"], "anchor": w["anchor"],
                            "coh": w["coh"], "loose": w["loose"], "apex": w["apex"],
                            "dnear": w["dnear"], "path": w["path"], "startown": own,
                            "count_first": int(body[:, at[s]].sum()),
                            "count_last": int(body[:, at[e]].sum()),
                            "count": int(body[:, at[e if G.LEVEL_AT == "end"
                                                 else s]].sum())})
    WINDOWS.write_text(json.dumps(out))
    lv = {}
    for w in out:
        lv.setdefault(w["count"], set()).add(w["match"])
    print(f"windows: {len(out)} legal windows over "
          f"{len({w['match'] for w in out})} matches"
          + (f" ({missing} planned matches not probed yet)" if missing else ""))
    print(f"  matches able to supply each level "
          f"(read on the {'LAST' if G.LEVEL_AT == 'end' else 'FIRST'} frame):")
    for n in G.LEVELS:
        ms = len(lv.get(n, ()))
        print(f"    {n:>3}: {ms:>3} matches  {'#' * min(ms, 60)}"
              + ("   <-- SHORT" if ms < 2 else ""))
    extra = {k: len(v) for k, v in sorted(lv.items()) if k not in G.LEVELS}
    if extra:
        print("  (levels outside the spec, for reference:", extra, ")")
    return 0


# ══ phase 5: ballpix — where the ball is, on every candidate end frame ═════════
# Two phases, not one, because the bundle is chosen at gfootball import time: a process
# cannot render both a visible-ball and an invisible-ball pass, and a second `use_bundle`
# inside a live process silently does nothing (measured: a 125-frame pair came back
# bit-identical). The vis pass writes its plate frames; the inv pass diffs against them.
PLATES = G.CACHE / "plates"
PIX = G.CACHE / "ballpix"
MAX_ENDS = 80              # per match; more than selection can use, and the plate is 1 MB
                           # a frame, so this is also the disk cap on the transient npz


def _ends_wanted(match):
    """Candidate end frames worth locating the ball on: those of windows whose OPENING
    count is a level the batch actually wants. Everything else cannot be selected, so
    paying a plate frame for it is waste."""
    rows = [w for w in json.loads(WINDOWS.read_text())
            if w["match"] == match and w["count"] in G.LEVELS]
    ends = sorted({w["end"] - 1 for w in rows})
    if len(ends) <= MAX_ENDS:
        return ends
    step = len(ends) / MAX_ENDS          # thin evenly, keeping the spread over the match
    return [ends[int(i * step)] for i in range(MAX_ENDS)]


def _ballpix_pass(argv, bundle, tag_):
    from lib import use_bundle
    from scenario_factory import write_scenario
    shard_i, shard_n = parse_shard(argv)
    use_bundle(bundle)
    for name, *_ in G.SHAPES:
        write_scenario(G.shape_spec(name), force=False)
    done = PIX if tag_ == "inv" else PLATES
    suffix = ".json" if tag_ == "inv" else ".npz"
    todo = [p for p in _planned(shard_i, shard_n, done, suffix)
            if _ends_wanted(p["match"])
            and (tag_ == "vis" or (PLATES / f"{p['match']}.npz").exists())]
    if not todo:
        print(f"shard {shard_i}/{shard_n}: ballpix {tag_} complete")
        return EXIT_DONE
    p = todo[0]
    ends = _ends_wanted(p["match"])
    frames, _ = G.render_window(G.SCEN_PREFIX + p["shape"], p["seed"], 0, max(ends) + 1,
                                keep=set(ends),
                                hide_slots=",".join(G.ALL_SLOTS),
                                offset=(p["offset_x"], p["offset_y"]))
    frames = np.asarray(frames, dtype=np.uint8)
    if tag_ == "vis":
        PLATES.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(PLATES / f"{p['match']}.npz", frames=frames,
                            ends=np.array(ends))
        print(f"  [ballpix vis] {p['match']}: {len(ends)} plate frames", flush=True)
        return 0
    from grid import detect_ball_or_none
    z = np.load(PLATES / f"{p['match']}.npz")
    vis, kept = z["frames"], list(map(int, z["ends"]))
    out, offscreen = {}, 0
    for i, e in enumerate(kept):
        found = detect_ball_or_none(vis[i], frames[i])
        if found is None:
            offscreen += 1           # camera offset pushed the ball out of shot: no answer
            continue
        out[str(e)] = [round(float(found[0]), 1), round(float(found[1]), 1)]
    PIX.mkdir(parents=True, exist_ok=True)
    (PIX / f"{p['match']}.json").write_text(json.dumps(out))
    (PLATES / f"{p['match']}.npz").unlink()
    print(f"  [ballpix inv] {p['match']}: {len(out)} end frames"
          f"{f' ({offscreen} dropped, ball off screen)' if offscreen else ''}, cells "
          f"{sorted({cell_of(*v) for v in out.values()})}", flush=True)
    return 0


def cmd_ballpix_vis(argv):
    return _ballpix_pass(argv, G.BUNDLE_VIS, "vis")


def cmd_ballpix_inv(argv):
    return _ballpix_pass(argv, G.BUNDLE_INV, "inv")


# ══ phase 6: select — the joint assignment ════════════════════════════════════
def _pool():
    """Every legal window whose ball cell is known, keyed for the assigner by OPENING
    count. `count` is the level; `count_end` rides along for the record."""
    absent = json.loads(ABSENT.read_text()) if ABSENT.exists() else {}
    pool, blocked, nocell = [], 0, 0
    pix = {}
    for w in json.loads(WINDOWS.read_text()):
        if w["count"] not in G.LEVELS:
            continue
        if w["match"] not in pix:
            f = PIX / f"{w['match']}.json"
            pix[w["match"]] = json.loads(f.read_text()) if f.exists() else {}
        e = str(w["end"] - 1)
        if e not in pix[w["match"]]:
            nocell += 1
            continue
        gone = absent.get(w["match"], [])
        if any(lo <= f <= hi for lo, hi in gone
               for f in (w["start"], w["end"] - 1)) or any(
                   lo <= w["start"] and w["end"] - 1 <= hi for lo, hi in gone):
            blocked += 1              # the audit already rejected this stretch of match
            continue
        px, py = pix[w["match"]][e]
        pool.append({**w, "px": px, "py": py, "cell": cell_of(px, py)})
    if nocell:
        print(f"  {nocell} windows excluded: no ball pixel measured at their end frame")
    if blocked:
        print(f"  {blocked} windows excluded: the audit found no ball there")
    return pool


def cmd_select(argv):
    import gen3_assign
    pool = _pool()
    print(f"pool: {len(pool)} windows over {len({c['match'] for c in pool})} matches")
    lv = {}
    for c in pool:
        lv.setdefault(c["count"], set()).add(c["match"])
    print(f"  matches per level ({'closing' if G.LEVEL_AT == 'end' else 'opening'}"
          f" count):", {n: len(lv.get(n, ())) for n in G.LEVELS})

    # No class quota: capacity 24 on every class makes the class layer inert while
    # leaving the network — and so the "one clip per match" and level constraints —
    # exactly as they were.
    quota = {k: G.PER_COUNT * len(G.LEVELS) for k in CLASS_ORDER}
    picks, report = gen3_assign.solve(pool, G.LEVELS, G.PER_COUNT, quota)
    if not picks:
        print("\nINFEASIBLE:")
        print(json.dumps(report, indent=2))
        return 1
    picks.sort(key=lambda c: (c["count"], c["kind"]))
    for i, c in enumerate(picks):
        c["clip"] = f"clip_{i + 1:02d}"
    PICKS.write_text(json.dumps(picks, indent=2))
    mix = {}
    for c in picks:
        mix[c["kind"]] = mix.get(c["kind"], 0) + 1
    print(f"\n{'clip':<8}{'opens':>6}{'ends':>6} {'class':<9}{'cell':>5} "
          f"{'coh':>6}  match      (the level is the "
          f"{'ending' if G.LEVEL_AT == 'end' else 'opening'} count)")
    for c in picks:
        print(f"{c['clip']:<8}{c['count_first']:>6}{c['count_last']:>6} {c['kind']:<9}"
              f"{c['cell']:>5} {c['coh']:>6.2f}  {c['match']}")
    print("\nsituation mix that fell out:", mix)
    return 0


# ══ phase 7: render — vis and inv, nobody hidden ═══════════════════════════════
def rkey(p):
    """Cache key for a render: the WINDOW, never the clip number.

    Clip numbers are positions in a sorted picks list, so re-running `select` after the
    audit rejects a window renumbers everything after it. Keyed by clip number, the
    already-rendered frames of one window would then be served up as another window's —
    silently, since a .npz of the right shape exists at the expected path. Keyed by the
    window itself, a re-selection reuses exactly the renders that are still valid and
    re-renders exactly the ones that changed.
    """
    return f"{p['match']}_{p['start']}_{p['end']}"


def _render(bundle, tag_, shard_i, shard_n):
    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle(bundle)
    for name, *_ in G.SHAPES:
        write_scenario(G.shape_spec(name), force=False)
    picks = json.loads(PICKS.read_text())
    RENDERS.mkdir(parents=True, exist_ok=True)
    todo = [p for i, p in enumerate(picks)
            if i % shard_n == shard_i
            and not (RENDERS / f"{rkey(p)}_{tag_}.npz").exists()]
    if not todo:
        return EXIT_DONE
    for p in todo[:4]:
        # hide_slots deliberately empty: all 22 players render in every shipped frame.
        frames, _ = G.render_window(G.SCEN_PREFIX + p["shape"], p["seed"], p["start"], p["end"],
                                    hide_slots="",
                                    offset=(p["offset_x"], p["offset_y"]))
        np.savez_compressed(RENDERS / f"{rkey(p)}_{tag_}.npz",
                            frames=np.array(frames, dtype=np.uint8))
        print(f"  [{tag_}] {p['clip']} {p['match']} ({len(frames)} frames)", flush=True)
    return 0


def cmd_render_vis(argv):
    return _render(G.BUNDLE_VIS, "vis", *parse_shard(argv))


def cmd_render_inv(argv):
    return _render(G.BUNDLE_INV, "inv", *parse_shard(argv))


# ══ phase 7b: audit — is the ball actually in the clip we rendered? ════════════
MIN_BALL_PX = 3            # matches GEN2's ball_onscreen: fewer than 3 changed pixels is
                           # not a ball you can read, it is the sliver left round a body


def _ball_px(vis_f, inv_f):
    """Changed pixels between the two passes of one frame — i.e. how much ball is showing."""
    d = np.abs(vis_f.astype(np.int16) - inv_f.astype(np.int16)).sum(axis=2)
    return int((d > 40).sum())


def cmd_audit(argv):
    """Reject a picked window unless the ball is in the CAMERA'S VIEW on every frame.

    GEN3 checked the two end frames only, and justified exempting the middle by occlusion:
    a ball behind a defender measures as absent in a visible-versus-invisible diff, and
    that is just football. True — but it is not the only way a frame can come back empty.
    GEN3_HARD's clip_01 shipped with 65 consecutive frames — 2.6 s — in which the ball was
    outside the shot entirely, because the camera offset that composes the shot off-centre
    also pushes a ball rolling towards the touchline off the edge of it. Nothing looked.

    So this checks every frame, and tells the two cases apart:

      * a frame with ball pixels in the shipped pair: in view, nothing to do;
      * a frame without them: re-render that window as a PLATE, both bundles, with all 22
        players hidden. With nobody left to stand in front of it, a ball that still cannot
        be seen is not in the camera's view. Occlusion is accepted and recorded; absence
        is fatal to the window.

    The plate pair costs two replays and is only paid on clips that have a gap at all, so
    in practice it runs on a handful. It is driven from `audit_plate`, which has to be two
    processes: the bundle is fixed at gfootball import time.

    A rejection records the ball-less frames in ABSOLUTE frame numbers, which `_pool` uses
    to exclude every window that starts or ends inside one — and, for an out-of-view span,
    every window that CONTAINS one. Re-run `select` and the render phases afterwards.

    Exit 0 = clean, 4 = rejections were recorded and the pipeline must loop,
    5 = plate evidence is needed first (run `audit_plate_vis` / `audit_plate_inv`).
    """
    picks = json.loads(PICKS.read_text())
    absent = json.loads(ABSENT.read_text()) if ABSENT.exists() else {}
    plates = G.CACHE / "auditplate"
    rejected, need_plate, occluded = [], [], []
    for p in picks:
        vis = np.load(RENDERS / f"{rkey(p)}_vis.npz")["frames"]
        inv = np.load(RENDERS / f"{rkey(p)}_inv.npz")["frames"]
        px = np.array([_ball_px(vis[i], inv[i]) for i in range(len(vis))])
        gaps = np.flatnonzero(px < MIN_BALL_PX)
        if not len(gaps):
            continue
        pf = plates / f"{rkey(p)}.npz"
        if not pf.exists():
            need_plate.append(p["clip"])
            continue
        plate = np.load(pf)["px"]            # ball pixels with all 22 players hidden
        out = np.flatnonzero(plate < MIN_BALL_PX)
        if not len(out):
            occluded.append((p["clip"], len(gaps)))
            continue
        lo, hi = int(out.min()), int(out.max())
        absent.setdefault(p["match"], []).append([p["start"] + lo, p["start"] + hi])
        rejected.append((p["clip"], p["match"],
                         f"ball outside the camera view on {len(out)} frames "
                         f"({lo}-{hi} of {len(px)})"))
        for tag_ in ("vis", "inv"):
            (RENDERS / f"{rkey(p)}_{tag_}.npz").unlink(missing_ok=True)
        pf.unlink()
    if need_plate:
        print(f"audit: {len(need_plate)} clips have frames with no ball and need the "
              f"plate pair: {' '.join(need_plate)}")
        print("  run `audit_plate_vis` then `audit_plate_inv`, then audit again")
        return 5
    ABSENT.write_text(json.dumps(absent, indent=1))
    for clip, match, why in rejected:
        print(f"  rejected {clip} ({match}): {why}")
    for clip, n in occluded:
        print(f"  {clip}: ball behind a player on {n} frames — in view, accepted")
    print(f"audit: {len(picks) - len(rejected)}/{len(picks)} clips hold the ball in view "
          f"on every frame" + (" — re-run select and the render phases" if rejected
                               else ""))
    return 4 if rejected else 0


# ══ phase 7b: audit_plate — is the ball out of shot, or behind someone? ════════
def _audit_plate(argv, bundle, tag_):
    """One window, all 22 players hidden, both bundles. See cmd_audit."""
    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle(bundle)
    for name, *_ in G.SHAPES:
        write_scenario(G.shape_spec(name), force=False)
    plates = G.CACHE / "auditplate"
    plates.mkdir(parents=True, exist_ok=True)
    picks = json.loads(PICKS.read_text())
    todo = []
    for p in picks:
        if (plates / f"{rkey(p)}.npz").exists():
            continue
        vis = np.load(RENDERS / f"{rkey(p)}_vis.npz")["frames"]
        inv = np.load(RENDERS / f"{rkey(p)}_inv.npz")["frames"]
        if any(_ball_px(vis[i], inv[i]) < MIN_BALL_PX for i in range(len(vis))):
            if tag_ == "inv" and not (plates / f"{rkey(p)}_vis.npz").exists():
                continue
            todo.append(p)
    if not todo:
        print(f"audit_plate {tag_}: nothing to do")
        return EXIT_DONE
    p = todo[0]
    frames, _ = G.render_window(G.SCEN_PREFIX + p["shape"], p["seed"], 0, p["end"],
                                keep=set(range(p["start"], p["end"])),
                                hide_slots=",".join(G.ALL_SLOTS),
                                offset=(p["offset_x"], p["offset_y"]))
    frames = np.asarray(frames, dtype=np.uint8)
    if tag_ == "vis":
        np.savez_compressed(plates / f"{rkey(p)}_vis.npz", frames=frames)
        print(f"  [audit_plate vis] {p['clip']}", flush=True)
        return 0
    vis = np.load(plates / f"{rkey(p)}_vis.npz")["frames"]
    px = np.array([_ball_px(vis[i], frames[i]) for i in range(len(vis))])
    np.savez_compressed(plates / f"{rkey(p)}.npz", px=px)
    (plates / f"{rkey(p)}_vis.npz").unlink()
    n = int((px < MIN_BALL_PX).sum())
    print(f"  [audit_plate inv] {p['clip']}: ball outside the camera view on "
          f"{n}/{len(px)} frames", flush=True)
    return 0


def cmd_audit_plate_vis(argv):
    return _audit_plate(argv, G.BUNDLE_VIS, "vis")


def cmd_audit_plate_inv(argv):
    return _audit_plate(argv, G.BUNDLE_INV, "inv")


# ══ phase 7c: trace — the exact count on EVERY frame of a shipped clip ═════════
TRACE = G.CACHE / "trace"


def cmd_trace(argv):
    """Per-frame, per-player in-frame map across a shipped clip's whole window.

    The batch's headline number is the count on the LAST frame, which is all `probe`
    measured — it only ever looked at candidate end frames, because measuring every frame
    of every candidate window in 77 matches would have been absurd. For the 24 windows
    that actually shipped it is affordable, and it answers the question the end count
    cannot: how the number of people in shot CHANGES over the five seconds.

    Same solo-probe mechanism as `probe`, and no cheaper one exists — fitting the camera
    frustum geometrically gets the per-frame count exactly right only 36% of the time, and
    counting kit-coloured blobs is biased +5.8 because the hoardings are saturated blue and
    red. Only plate-versus-solo is exact.

    ~23 replays per clip, bounded to 2 clips per process because the engine leaks.
    """
    from lib import use_bundle
    from scenario_factory import write_scenario
    shard_i, shard_n = parse_shard(argv)
    use_bundle(G.BUNDLE_VIS)
    for name, *_ in G.SHAPES:
        write_scenario(G.shape_spec(name), force=False)
    picks = json.loads(PICKS.read_text())
    TRACE.mkdir(parents=True, exist_ok=True)
    todo = [p for i, p in enumerate(picks)
            if i % shard_n == shard_i
            and not (TRACE / f"{rkey(p)}.npz").exists()]
    if not todo:
        print(f"shard {shard_i}/{shard_n}: trace complete")
        return EXIT_DONE
    print(f"shard {shard_i}/{shard_n}: {len(todo)} clips left", flush=True)
    for p in todo[:2]:
        off = (p["offset_x"], p["offset_y"])
        lvl = G.SCEN_PREFIX + p["shape"]
        keep = set(range(p["start"], p["end"]))
        # down=2 throughout: a body is still tens of pixels at half size, and the full-size
        # window would be 23 passes x 125 frames x 1.8 MB.
        plate, _ = G.render_window(lvl, p["seed"], 0, p["end"], keep=keep, down=DOWN,
                                   hide_slots=",".join(G.ALL_SLOTS), offset=off)
        plate = np.array(plate, dtype=np.int16)
        on = np.zeros((len(G.ALL_SLOTS), len(plate)), dtype=bool)
        for si, slot in enumerate(G.ALL_SLOTS):
            hide = ",".join(s for s in G.ALL_SLOTS if s != slot)      # show ONLY this one
            solo, _ = G.render_window(lvl, p["seed"], 0, p["end"], keep=keep, down=DOWN,
                                      hide_slots=hide, offset=off)
            d = np.abs(np.array(solo, dtype=np.int16) - plate).sum(axis=3)
            on[si] = (d > 30).reshape(len(plate), -1).sum(axis=1) > MIN_PIX
        np.savez_compressed(TRACE / f"{rkey(p)}.npz", on=on,
                            frames=np.arange(p["start"], p["end"]))
        n = on.sum(axis=0)
        print(f"  [trace] {p['clip']} {p['match']}: {n[0]} -> {n[-1]} "
              f"(min {n.min()}, max {n.max()})", flush=True)
    return 0


# ══ phase 8: compose ═══════════════════════════════════════════════════════════
# Both ends are published and both are named for the frame they are read from. The LEVEL
# is `players_in_frame_last` (see gen4_lib.LEVEL_AT) — the same frame the ball answer is
# read from, so the count and the answer describe one instant.
HEADER = ("clip,situation,players_in_frame_first,players_in_frame_last,players_in_play,"
          "players_hidden,start_team,start_team_colour,"
          "ball_final_cell,final_px,final_py,ball_start_cell,start_px,start_py,"
          "offset_x,offset_y,shape,seed,match,start_frame,end_frame,n_frames,seconds,"
          "loose,apex,dnear,ball_path,coherence")

# Which side starts the clip with the ball. The engine's left team is A, and
# apply_gen3_kits paints A blue and B red — so this column names what a viewer sees, not
# an engine index, and it moves if the kit palette ever does.
TEAM_NAME = {0: ("A", "blue"), 1: ("B", "red")}


def cmd_compose(argv):
    picks = json.loads(PICKS.read_text())
    rows = [HEADER]
    G.OUT.mkdir(parents=True, exist_ok=True)
    for p in picks:
        vis = np.load(RENDERS / f"{rkey(p)}_vis.npz")["frames"]
        inv = np.load(RENDERS / f"{rkey(p)}_inv.npz")["frames"]
        # The plate measurement of this clip's own final frame, at its own offset: the
        # ground truth when a player is standing in front of the ball on that frame.
        full, split, (spx, spy), (fpx, fpy) = G.compose_pair(
            vis, inv, final_fallback=(p["px"], p["py"]))
        G.write_clip(full, G.OUT / "full_visibility" / f"{p['clip']}.mov")
        G.write_clip(split, G.OUT / "split_1s_4s" / f"{p['clip']}.mov")
        p["final_cell_measured"] = G.cell_of(fpx, fpy)
        rows.append(",".join(map(str, [
            p["clip"], p["kind"], p["count_first"], p["count_last"], 22, 0,
            *TEAM_NAME[p["startown"]],
            G.cell_of(fpx, fpy), round(fpx, 1), round(fpy, 1),
            G.cell_of(spx, spy), round(spx, 1), round(spy, 1),
            p["offset_x"], p["offset_y"], p["shape"], p["seed"], p["match"],
            p["start"], p["end"], len(full), round(len(full) / G.FPS, 2),
            p["loose"], p["apex"], p["dnear"], p["path"], p["coh"]])))
        print(f"  [compose] {p['clip']}: opens with {p['count_first']} people, ends "
              f"with {p['count_last']}, {p['kind']}, "
              f"ball {G.cell_of(spx, spy)} -> {G.cell_of(fpx, fpy)}", flush=True)
    (G.OUT / "ground_truth.csv").write_text("\n".join(rows) + "\n")
    PICKS.write_text(json.dumps(picks, indent=2))
    print(f"compose: {len(picks)} situations -> {len(picks) * 2} clips in {G.OUT}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "shortlist"
    table = {"shortlist": cmd_shortlist, "plan": cmd_plan, "probe": cmd_probe,
             "windows": cmd_windows,
             "ballpix_vis": cmd_ballpix_vis, "ballpix_inv": cmd_ballpix_inv,
             "select": cmd_select,
             "render_vis": cmd_render_vis, "render_inv": cmd_render_inv,
             "audit": cmd_audit, "audit_plate_vis": cmd_audit_plate_vis,
             "audit_plate_inv": cmd_audit_plate_inv,
             "trace": cmd_trace, "compose": cmd_compose}
    if cmd not in table:
        sys.exit(f"unknown phase {cmd!r}; known: {' '.join(table)}")
    sys.exit(table[cmd](sys.argv[2:]))
