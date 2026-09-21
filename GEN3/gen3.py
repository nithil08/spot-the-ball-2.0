"""gen3.py — the GEN3 pilot batch, end to end.

    python3 gen3.py sweep    [--shard i/n]   # log-only match sweep, resumable
    python3 gen3.py shortlist                # detect situations, apply coherence gates
    python3 gen3.py natural  [--shard i/n]   # ball pixel at offset 0, per match
    python3 gen3.py plan                     # assign target cells -> per-match offset
    python3 gen3.py probe    [--shard i/n]   # EXACT player counts at that offset
    python3 gen3.py select                   # the joint assignment -> 24 picks
    python3 gen3.py render   [--shard i/n]   # vis + inv renders of the picks
    python3 gen3.py compose                  # clips + ground truth
    python3 gen3.py verify                   # spec checks + contact sheets

WHY THE PHASES ARE SPLIT THIS WAY
  An asset bundle is chosen by an env var read at gfootball import time, so the visible
  and invisible renders can never share a process. And the engine leaks: a process dies
  at roughly its 46th FootballEnv, with no traceback (four independent shards each
  completed exactly 2 matches of a 23-env probe and died). So every engine phase does a
  BOUNDED amount of work and exits, and a shell loop re-invokes it. Exit codes carry the
  distinction the loop needs:

      0            did some work, call me again
      EXIT_DONE(3) this shard has nothing left, stop looping
      anything else — including being killed outright — the process DIED, retry it

  Using 1 for "nothing left" was a real bug once: it is indistinguishable from a crash, so
  every shard stopped after its first match while reporting success.

ORDER OF THE HARD CONSTRAINTS
  The count is the least controllable quantity, so everything is arranged to give the
  count the most freedom at the end. Each match gets ONE camera offset (chosen for its
  ball cell), one 23-pass probe, and then the count is picked by SLIDING the window inside
  that match — because the probe returns the count for every frame at no extra cost.
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

import gen3_lib as G                                                  # noqa: E402

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
STRIDE = 5

CAND = G.CACHE / "shortlist.json"
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
        log = G.run_log(f"g3_{shape}", seed, SWEEP_STEPS)
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
    import glob
    rows = []
    files = sorted(glob.glob(str(G.SWEEP / "*.npz")))
    tally = {k: 0 for k in CLASS_ORDER}
    for p in files:
        t = Path(p).stem
        shape, seed = t.rsplit("_s", 1)
        d = np.load(p)
        log = {k: d[k] for k in d.files}
        n = len(log["ball"])
        for kind in CLASS_ORDER:
            cands = _candidates(log, kind, n)
            if not cands:
                continue
            rows.append({"match": t, "shape": shape, "seed": int(seed), "kind": kind,
                         "windows": _spread(cands, kind)})
            tally[kind] += 1
            break                      # one class per match, scarcity-first
    G.CACHE.mkdir(parents=True, exist_ok=True)
    CAND.write_text(json.dumps(rows, indent=1))
    print(f"shortlist over {len(files)} matches -> {len(rows)} usable matches")
    for k in CLASS_ORDER:
        print(f"  {k:<9} {tally[k]:>4} matches")
    return 0


# ══ phase 3: plan — which matches to probe, and at what camera offset ═══════════
# One offset per MATCH, drawn from a schedule that spans the reachable framing space.
#
# Why a schedule instead of solving each offset for a chosen target cell: the solve needs
# the ball's natural pixel at the window's final frame, which needs a rendered
# visible/invisible pair BEFORE the probe — a whole extra two-process stage over hundreds
# of matches. The probe already yields the exact ball pixel for free (see cmd_ballpix), so
# it is cheaper to fix the offset first, measure what cell it actually produced, and let
# SELECTION do the spreading across a large candidate pool.
#
# Row A is barely reachable and is not chased: putting the ball in the top row means aiming
# the camera from beyond the near touchline, which fills the bottom of the shot with
# hoardings and seating. The framing audit in cmd_verify enforces that independently.
OFFX_SCHEDULE = [0.0, -12.0, 12.0, -20.0, 20.0, -6.0, 6.0, -24.0, 24.0, -16.0, 16.0]
OFFY_SCHEDULE = [0.0, 6.0, -5.0, 10.0, -9.0, 3.0, -2.0, 8.0, -7.0]

# How many matches to probe per class. The probe is 23 replays per match — by far the most
# expensive stage — so the budget goes where the scarcity is. Corners run ~0.03 per match,
# so every corner match the sweep found gets probed; open play runs ~47 per match and is
# never the binding constraint.
PROBE_BUDGET = {"corner": 999, "gk_throw": 18, "kickoff": 16, "open": 26}
MAX_END = 1900             # cap the probe replay length; cost is linear in the end frame
SCARCE = ("corner", "gk_throw")   # exempt from MAX_END — too few of them to discard any


def cmd_plan(argv):
    rows = json.loads(CAND.read_text())
    by_kind = {k: [] for k in CLASS_ORDER}
    for r in rows:
        # MAX_END caps how far the probe has to replay, and cost is linear in the end
        # frame. But corners are uniformly spread over the 2750-frame sweep, so applying
        # the cap to them throws away roughly a third of the few that exist. The scarce
        # classes therefore pay the longer replay; only open play, which has hundreds of
        # candidates per match, is capped.
        w = (r["windows"] if r["kind"] in SCARCE
             else [x for x in r["windows"] if x["end"] <= MAX_END])
        if not w:
            continue
        by_kind[r["kind"]].append({**r, "windows": w})
    plan, k = [], 0
    for kind in CLASS_ORDER:
        # Best-coherence matches first, but spread across shapes and seeds so the batch
        # does not come from one corner of the scenario space.
        cands = sorted(by_kind[kind], key=lambda r: r["windows"][0]["coh"])
        for r in cands[:PROBE_BUDGET[kind]]:
            plan.append({"match": r["match"], "shape": r["shape"], "seed": r["seed"],
                         "kind": kind,
                         "offset_x": OFFX_SCHEDULE[k % len(OFFX_SCHEDULE)],
                         "offset_y": OFFY_SCHEDULE[k % len(OFFY_SCHEDULE)],
                         "windows": r["windows"]})
            k += 1
    PLAN.write_text(json.dumps(plan, indent=1))
    got = {kind: sum(1 for p in plan if p["kind"] == kind) for kind in CLASS_ORDER}
    print(f"plan: {len(plan)} matches to probe  {got}")
    short = {kind: G.SITUATION_QUOTA[kind] for kind in CLASS_ORDER
             if got[kind] < G.SITUATION_QUOTA[kind]}
    if short:
        print(f"  *** only {got} matches available against a quota of "
              f"{G.SITUATION_QUOTA} — sweep more seeds for {list(short)}")
    return 0


# ══ phase 4: probe — the exact player count, at the offset that ships ═══════════
def _planned(shard_i, shard_n, done_dir, suffix=".npz"):
    plan = json.loads(PLAN.read_text())
    return [p for i, p in enumerate(plan)
            if i % shard_n == shard_i
            and not (done_dir / f"{p['match']}{suffix}").exists()]


PLATES = G.CACHE / "plates"


def probe_match(p):
    """Exact per-frame, per-player in-frame map for one match at its shipping offset.

    Render a plate with all 22 players hidden, then 22 more passes each showing exactly
    one player, and diff. A non-trivial pixel difference means that player's body is
    inside the camera frustum on that frame. Hiding is render-only, so all 23 passes
    replay the identical match and the measurement cannot perturb what it measures.

    The plate frames at the candidate end frames are kept for cmd_ballpix, which needs
    them to locate the ball. They are deleted as soon as that runs.
    """
    off = (p["offset_x"], p["offset_y"])
    lvl = f"g3_{p['shape']}"
    # Only the candidate END frames are ever read — the count is a property of the last
    # frame of a clip. Keeping the whole replay would cost ~10 GB on a corner match.
    ends = sorted({w["end"] - 1 for w in p["windows"]})
    keep = set(ends)
    end = max(ends) + 1

    plate_full, _ = G.render_window(lvl, p["seed"], 0, end, keep=keep,
                                    hide_slots=",".join(G.ALL_SLOTS), offset=off)
    plate_full = np.array(plate_full, dtype=np.uint8)          # (len(ends), H, W, 3)
    small = plate_full[:, ::DOWN, ::DOWN].astype(np.int16)

    on = np.zeros((len(G.ALL_SLOTS), len(ends)), dtype=bool)
    for si, slot in enumerate(G.ALL_SLOTS):
        hide = ",".join(s for s in G.ALL_SLOTS if s != slot)          # show ONLY this one
        solo, _ = G.render_window(lvl, p["seed"], 0, end, keep=keep, down=DOWN,
                                  hide_slots=hide, offset=off)
        solo = np.array(solo, dtype=np.int16)
        d = np.abs(solo - small).sum(axis=3)
        on[si] = (d > 30).reshape(len(ends), -1).sum(axis=1) > MIN_PIX

    COUNTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(COUNTS / f"{p['match']}.npz", on=on, ends=np.array(ends))
    PLATES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PLATES / f"{p['match']}.npz", frames=plate_full,
                        ends=np.array(ends))
    at_ends = on.sum(axis=0)
    print(f"  [probe] {p['match']} off=({off[0]:+.0f},{off[1]:+.0f}) "
          f"{len(ends)} end frames, counts {sorted(set(map(int, at_ends)))}", flush=True)


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


# ══ phase 5: ballpix — where the ball actually is, at that same offset ══════════
def cmd_ballpix(argv):
    """Ball pixel at every candidate end frame, from the plate pair.

    The plate has all 22 players hidden, so between a visible-ball plate and an
    invisible-ball plate the ONLY thing that differs is the ball — no shirts, no socks,
    nothing else that could be mistaken for it. That makes this the cleanest ball
    detection in the pipeline, and it costs one extra replay on top of the probe's 23.

    Ground truth is still re-measured from the real shipped render in cmd_compose; this
    exists so SELECTION can see each candidate's grid cell before committing.

    An end frame whose ball cannot be located is DROPPED, not fatal. With every player
    hidden nothing can occlude the ball, so the only way to lose it is for the camera
    offset to have pushed it out of shot — at +-24 px of reframing that happens, and such
    a window has no answer cell and must not ship. `_pool` already skips ends missing
    from this file, so dropping them here is all the exclusion that is needed.
    """
    from grid import detect_ball_or_none
    from lib import use_bundle
    from scenario_factory import write_scenario
    shard_i, shard_n = parse_shard(argv)
    use_bundle(G.BUNDLE_INV)
    for name, *_ in G.SHAPES:
        write_scenario(G.shape_spec(name), force=False)
    PIX = G.CACHE / "ballpix"
    PIX.mkdir(parents=True, exist_ok=True)
    todo = [p for p in _planned(shard_i, shard_n, PIX, ".json")
            if (PLATES / f"{p['match']}.npz").exists()]
    if not todo:
        print(f"shard {shard_i}/{shard_n}: ballpix complete")
        return EXIT_DONE
    p = todo[0]
    off = (p["offset_x"], p["offset_y"])
    z = np.load(PLATES / f"{p['match']}.npz")
    vis_plate, ends = z["frames"], list(map(int, z["ends"]))
    inv, _ = G.render_window(f"g3_{p['shape']}", p["seed"], 0, max(ends) + 1,
                             hide_slots=",".join(G.ALL_SLOTS), offset=off)
    out, offscreen = {}, 0
    for i, e in enumerate(ends):
        found = detect_ball_or_none(vis_plate[i], np.asarray(inv[e]))
        if found is None:
            offscreen += 1
            continue
        out[str(e)] = [round(float(found[0]), 1), round(float(found[1]), 1)]
    (PIX / f"{p['match']}.json").write_text(json.dumps(out))
    (PLATES / f"{p['match']}.npz").unlink()
    print(f"  [ballpix] {p['match']}: {len(out)} end frames"
          f"{f' ({offscreen} dropped, ball off screen)' if offscreen else ''}, cells "
          f"{sorted({cell_of(*v) for v in out.values()})}", flush=True)
    return 0


# ══ phase 6: select — the joint assignment ═════════════════════════════════════
def _pool():
    """Every probed candidate window, with its EXACT end count and measured ball cell."""
    import gen3_assign                                                # noqa: F401
    plan = json.loads(PLAN.read_text())
    PIX = G.CACHE / "ballpix"
    absent = json.loads(ABSENT.read_text()) if ABSENT.exists() else {}
    pool, skipped, blocked, teamless = [], 0, 0, 0
    owners = {}          # match -> owned_team track, read once from the sweep log
    for p in plan:
        cf, pf = COUNTS / f"{p['match']}.npz", PIX / f"{p['match']}.json"
        if not (cf.exists() and pf.exists()):
            skipped += 1
            continue
        gone = absent.get(p["match"], [])
        z = np.load(cf)
        on = z["on"]
        # `on` is indexed by POSITION in the probed end-frame list, not by frame number:
        # the probe only measures the frames a clip could actually end on.
        at = {int(f): i for i, f in enumerate(z["ends"])}
        pix = json.loads(pf.read_text())
        for w in p["windows"]:
            e = w["end"] - 1
            if e not in at or str(e) not in pix:
                continue
            # The audit measures which frames of a match actually show the ball, and a
            # window is dead if either END of it lands on one that does not: the red
            # start-circle would have nothing to mark, or the answer would be unreadable
            # on the frame it is read from.
            if any(lo <= f <= hi for lo, hi in gone for f in (w["start"], e)):
                blocked += 1
                continue
            px, py = pix[str(e)]
            # Which team the clip starts with the ball. Read from the sweep log, which is
            # camera-independent, so this costs nothing beyond one file read per match.
            if p["match"] not in owners:
                owners[p["match"]] = np.load(G.SWEEP / f"{p['match']}.npz")["owned_team"]
            own = G.start_possessor(owners[p["match"]], w["start"], w["end"])
            if own < 0:
                teamless += 1
                continue
            pool.append({"startown": own,
                         "match": p["match"], "shape": p["shape"], "seed": p["seed"],
                         "kind": p["kind"], "offset_x": p["offset_x"],
                         "offset_y": p["offset_y"], "start": w["start"], "end": w["end"],
                         "anchor": w["anchor"], "coh": w["coh"],
                         "loose": w["loose"], "apex": w["apex"], "dnear": w["dnear"],
                         "path": w["path"],
                         "count": int(on[:, at[e]].sum()),
                         "px": px, "py": py, "cell": cell_of(px, py)})
    if blocked:
        print(f"  {blocked} windows excluded: the audit measured no ball at their start")
    if teamless:
        print(f"  {teamless} windows excluded: ball never owned, so no starting team")
    return pool, skipped


def cmd_select(argv):
    import gen3_assign
    pool, skipped = _pool()
    print(f"pool: {len(pool)} probed windows ({skipped} planned matches not yet probed)")
    hist = {}
    for c in pool:
        hist.setdefault(c["kind"], set()).add(c["match"])
    print("  matches per class:", {k: len(v) for k, v in sorted(hist.items())})
    avail = {n: len({c['match'] for c in pool if c['count'] == n}) for n in G.END_COUNTS}
    print("  matches per count:", avail)

    picks, report = gen3_assign.solve(pool, G.END_COUNTS, G.PER_COUNT, G.SITUATION_QUOTA)
    if not picks:
        print("\nINFEASIBLE:")
        print(json.dumps(report, indent=2))
        return 1
    picks.sort(key=lambda c: (c["count"], c["kind"]))
    for i, c in enumerate(picks):
        c["clip"] = f"clip_{i + 1:02d}"
    PICKS.write_text(json.dumps(picks, indent=2))
    print(f"\n{'clip':<8}{'count':>6} {'class':<9}{'cell':>5} {'coh':>6}  match")
    for c in picks:
        print(f"{c['clip']:<8}{c['count']:>6} {c['kind']:<9}{c['cell']:>5} "
              f"{c['coh']:>6.2f}  {c['match']}")
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
        frames, _ = G.render_window(f"g3_{p['shape']}", p["seed"], p["start"], p["end"],
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
    """Reject any picked window that does not hold the ball at BOTH of its ends.

    Both ends, and only the ends — the same rule GEN2 settled on. Those are the two frames
    ground truth is read from: the red circle marks the first, and the final cell, the
    answer, is read off the last. Interior frames are deliberately exempt, because the
    visible and invisible passes are identical wherever a player stands in front of the
    ball, so a ball going behind a body for half a second measures as absent — and that is
    just football, not a defect.

    Nothing upstream can do this check. `ballpix` measured only END frames, and it measured
    them on a PLATE, with all 22 players hidden, so it can see a ball that is out of shot
    but not one that is behind a defender. This runs on the pair that actually ships, which
    is both the honest place for it and free: those renders already exist.

    A rejection records the ball-less frames in ABSOLUTE frame numbers, which `_pool` then
    uses to exclude every window of that match that starts or ends inside one. Re-run
    `select` and the render phases afterwards: renders are keyed by window, so the ones
    that survived are reused and only the replacements are rendered.

    Exit 0 = clean, 4 = rejections were recorded and the pipeline must loop.
    """
    picks = json.loads(PICKS.read_text())
    absent = json.loads(ABSENT.read_text()) if ABSENT.exists() else {}
    rejected = []
    for p in picks:
        vis = np.load(RENDERS / f"{rkey(p)}_vis.npz")["frames"]
        inv = np.load(RENDERS / f"{rkey(p)}_inv.npz")["frames"]
        spans, why = [], None
        first = next((i for i in range(len(vis))
                      if _ball_px(vis[i], inv[i]) >= MIN_BALL_PX), None)
        if first is None or first >= G.MARK_FRAMES:
            # No ball at all, or not until frame `first`: that whole opening stretch is
            # ball-less, so record it rather than only the one frame that was checked.
            span = len(vis) if first is None else first
            spans.append([p["start"], p["start"] + span - 1])
            why = f"no ball for the first {span}/{len(vis)} frames"
        end_px = _ball_px(vis[-1], inv[-1])
        if end_px < MIN_BALL_PX:
            spans.append([p["end"] - 1, p["end"] - 1])
            why = (why + "; " if why else "") + f"only {end_px} ball px on the final frame"
        if not spans:
            continue
        absent.setdefault(p["match"], []).extend(spans)
        rejected.append((p["clip"], p["match"], why))
        for tag_ in ("vis", "inv"):
            (RENDERS / f"{rkey(p)}_{tag_}.npz").unlink(missing_ok=True)
    ABSENT.write_text(json.dumps(absent, indent=1))
    for clip, match, why in rejected:
        print(f"  rejected {clip} ({match}): {why}")
    print(f"audit: {len(picks) - len(rejected)}/{len(picks)} clips hold the ball at both "
          f"ends" + (" — re-run select and the render phases" if rejected else ""))
    return 4 if rejected else 0


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
        lvl = f"g3_{p['shape']}"
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
HEADER = ("clip,situation,players_in_frame_last,players_in_play,players_hidden,"
          "start_team,start_team_colour,"
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
            p["clip"], p["kind"], p["count"], 22, 0,
            *TEAM_NAME[p["startown"]],
            G.cell_of(fpx, fpy), round(fpx, 1), round(fpy, 1),
            G.cell_of(spx, spy), round(spx, 1), round(spy, 1),
            p["offset_x"], p["offset_y"], p["shape"], p["seed"], p["match"],
            p["start"], p["end"], len(full), round(len(full) / G.FPS, 2),
            p["loose"], p["apex"], p["dnear"], p["path"], p["coh"]])))
        print(f"  [compose] {p['clip']}: {p['count']} players, {p['kind']}, "
              f"ball {G.cell_of(spx, spy)} -> {G.cell_of(fpx, fpy)}", flush=True)
    (G.OUT / "ground_truth.csv").write_text("\n".join(rows) + "\n")
    PICKS.write_text(json.dumps(picks, indent=2))
    print(f"compose: {len(picks)} situations -> {len(picks) * 2} clips in {G.OUT}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "shortlist"
    table = {"sweep": cmd_sweep, "shortlist": cmd_shortlist, "plan": cmd_plan,
             "probe": cmd_probe, "ballpix": cmd_ballpix, "select": cmd_select,
             "render_vis": cmd_render_vis, "render_inv": cmd_render_inv,
             "audit": cmd_audit, "trace": cmd_trace, "compose": cmd_compose}
    if cmd not in table:
        sys.exit(f"unknown phase {cmd!r}; known: {' '.join(table)}")
    sys.exit(table[cmd](sys.argv[2:]))
