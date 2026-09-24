"""fill_gaps.py — repair the GEN3_HARD batch by re-solving it, keeping what still fits.

THE BATCH IS 24 CLIPS SATISFYING ALL OF THIS AT ONCE
  * two clips at each end-frame headcount 8..19, measured at full resolution
  * one starting with blue and one with red at every level, so 12/12 overall
  * 5 corners, 5 keeper distributions, 5 kick-offs, 9 open play
  * one clip per PASSAGE OF PLAY, not per match name
  * the ball inside the camera's view on every frame
  * final-frame ball cells spread over the 16x6 grid

WHY THIS RE-SOLVES RATHER THAN PATCHES
  A greedy repair — fill the short levels, leave everything else — cannot see that levels
  11, 14 and 16 each hold two clips that start with the SAME side. Fixing those by hand
  then breaks the class quota, and fixing that by hand breaks something else: the
  constraints are joint, so the repair has to be joint too.

  So every existing clip and every probed candidate goes into ONE assignment, with the
  existing clips priced far below the candidates. The solver keeps an existing clip
  wherever one fits and asks for a new one only where none does. That is the minimum
  rebuild by construction, not by argument, and it reports which constraint bound when
  something is impossible.

WHAT IS TRUSTED, AND WHAT IS RE-MEASURED
  Existing clips carry a full-resolution count (verify_final_frame) and a proven
  ball-in-view record, so they enter the pool as facts. Candidates carry the count the
  ORIGINAL probe measured — half resolution, no body/shadow split, the rule that put nine
  of the 24 counts wrong — so they enter as a SHORTLIST. Whatever the solver wants is
  re-measured at full resolution before it can ship, and the solve is repeated until
  every chosen candidate's measured level is the level it was chosen for.

    python3 fill_gaps.py solve            # assign over existing + shortlisted candidates
    python3 fill_gaps.py verify [i n]     # full-res counts for whatever the solve wants
    python3 fill_gaps.py solve            # again, now on measured counts — repeat to fix
    python3 fill_gaps.py emit             # write the new picks.json for render/compose
    python3 fill_gaps.py scanlist         # candidates worth screening for ball-in-view
    python3 fill_gaps.py scan_vis|scan_inv [i n]     # screen them, 2 replays each
    python3 fill_gaps.py audit            # ball in view on EVERY frame of every clip
    python3 fill_gaps.py plate_vis|plate_inv [i n]    # only for clips the audit flags
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen3_lib as G                                                  # noqa: E402
import gen3_assign                                                    # noqa: E402

CACHE = G.CACHE
PICKS = CACHE / "picks.json"
PLAN = CACHE / "plan.json"
COUNTS = CACHE / "counts"
PIX = CACHE / "ballpix"
FINAL = CACHE / "finalframe"
GAPS = CACHE / "gaps"
CLIPS = HERE / "clips"

LEVELS = list(range(8, 20))
PER_LEVEL = 2
QUOTA = {"corner": 5, "gk_throw": 5, "kickoff": 5, "open": 9}
MIN_BODY = 20

# Existing clips proven to fail a HARD constraint, so they cannot enter the pool at all.
# Both were measured with all 22 players hidden, where nothing can occlude the ball:
# these are frames in which the ball is outside the camera's view, not behind anyone.
BALL_LEFT_SHOT = {"clip_01": 65, "clip_08": 3}

# An existing clip costs nothing to keep and a new one costs a render and a measurement,
# so an existing clip must win every tie and most non-ties. This is subtracted from the
# coherence the solver minimises; it is larger than the whole coherence range, which is
# what makes "keep it unless it cannot fit" a property of the solution rather than a hope.
KEEP_BONUS = 100.0

ALIAS = {"mid_even": "mid_bal", "mid_even2": "mid_bal"}


def play_of(match):
    """One name per actual match: the three identical shapes replay bit-identically."""
    shape, seed = match.rsplit("_s", 1)
    return f"{ALIAS.get(shape, shape)}_s{seed}"


BEFORE = CACHE / "picks_before_repair.json"


def existing():
    """The clips of the batch AS IT WAS, with their full-resolution measured level.

    Read from the pre-repair snapshot, never from picks.json. Once a repaired picks.json
    is installed, its clip numbers refer to different windows than the measurement cache
    and the ground truth do — they are keyed by the OLD names — and a solve built on the
    mixture is meaningless. Keeping the source of "what exists today" fixed also makes the
    solve idempotent: running it twice cannot drift.
    """
    src = BEFORE if BEFORE.exists() else PICKS
    gt = {r["clip"]: r for r in csv.DictReader(open(CLIPS / "ground_truth.csv"))}
    picks = {p["clip"]: p for p in json.loads(src.read_text())}
    out = []
    for c, p in sorted(picks.items()):
        if c in BALL_LEFT_SHOT:
            continue
        rec = json.loads(str(np.load(FINAL / f"{c}.npz", allow_pickle=True)["rec"]))
        level = sum(1 for s in rec["slots"]
                    if s["pix"] >= MIN_BODY and s.get("body_pix", 0) >= MIN_BODY)
        if level not in LEVELS:
            continue                      # clip_04 measures 7, outside the spec
        out.append({**p, "count": level, "cell": gt[c]["ball_final_cell"],
                    "match": play_of(p["match"]), "real_match": p["match"],
                    "keep": c, "coh": p["coh"] - KEEP_BONUS})
    return out


ABSENT = CACHE / "ball_absent.json"          # audit: ball unreadable at a window END
OFFSCREEN = CACHE / "ball_offscreen.json"    # plate: ball outside the camera's VIEW


def candidates(verified):
    """Probed windows that could fill a slot, from the plan's own cache.

    A window enters with the level the ORIGINAL probe gave it unless `verified` holds a
    full-resolution measurement, in which case that wins. Nothing ships on the old number.

    Two recorded kinds of ball trouble bind here, and they bind differently:

      * `ball_absent.json` — frames where the audit could not read the ball in a render
        that HAS players, so it may be occlusion. It matters at the two ends, which are
        the frames ground truth is read from, so a window is dropped if either end lands
        in one. (This file was written by the original build's audit and then never
        consulted again when candidates were re-derived — three of the first solve's
        picks sat squarely inside it.)
      * `ball_offscreen.json` — frames proven on a PLATE, with all 22 players hidden, to
        have no ball in the camera's view at all. Nothing can occlude a ball on an empty
        pitch, so this is absence, and it is fatal anywhere in the window, not just at
        its ends.
    """
    absent = json.loads(ABSENT.read_text()) if ABSENT.exists() else {}
    offscreen = json.loads(OFFSCREEN.read_text()) if OFFSCREEN.exists() else {}
    out = []
    for p in json.loads(PLAN.read_text()):
        cf, pf = COUNTS / f"{p['match']}.npz", PIX / f"{p['match']}.json"
        if not (cf.exists() and pf.exists()):
            continue
        z = np.load(cf)
        on, ends = z["on"], list(map(int, z["ends"]))
        pix = json.loads(pf.read_text())
        owners = np.load(G.SWEEP / f"{p['match']}.npz")["owned_team"]
        for i, e in enumerate(ends):
            if str(e) not in pix:
                continue
            w = next((w for w in p["windows"] if w["end"] - 1 == e), None)
            if w is None:
                continue
            if p["kind"] != "open" and w["start"] > w["anchor"]:
                continue              # the restart would happen before the clip opens
            w_ends = (w["start"], w["end"] - 1)      # not `ends`: that is the loop's
            if any(lo <= f <= hi for lo, hi in absent.get(p["match"], [])
                   for f in w_ends):
                continue              # ground truth is read from those two frames
            if any(w["start"] <= hi and lo <= w["end"] - 1
                   for lo, hi in offscreen.get(p["match"], [])):
                continue              # the ball leaves the shot inside this window
            own = G.start_possessor(owners, w["start"], w["end"])
            if own < 0:
                continue
            key = f"{p['match']}_{w['start']}"
            level = verified.get(key, {}).get("count_last", int(on[:, i].sum()))
            if level not in LEVELS:
                continue
            px, py = pix[str(e)]
            out.append({"startown": own, "match": play_of(p["match"]),
                        "real_match": p["match"], "shape": p["shape"],
                        "seed": p["seed"], "kind": p["kind"],
                        "offset_x": p["offset_x"], "offset_y": p["offset_y"],
                        "start": w["start"], "end": w["end"], "anchor": w["anchor"],
                        "coh": w["coh"], "loose": w["loose"], "apex": w["apex"],
                        "dnear": w["dnear"], "path": w["path"], "count": level,
                        "px": px, "py": py, "cell": G.cell_of(px, py),
                        "verified": key in verified, "key": key})
    return out


def load_verified():
    d = {}
    for f in (GAPS / "verified").glob("*.json") if (GAPS / "verified").exists() else []:
        r = json.loads(f.read_text())
        d[r["key"]] = r
    return d


def cmd_solve(argv):
    GAPS.mkdir(parents=True, exist_ok=True)
    ver = load_verified()
    keep = existing()
    cand = candidates(ver)
    pool = keep + cand
    print(f"pool: {len(keep)} existing clips + {len(cand)} probed windows "
          f"({sum(1 for c in cand if c['verified'])} of them re-measured)")
    picks, report = gen3_assign.solve(pool, LEVELS, PER_LEVEL, QUOTA, verbose=False)
    if not picks:
        print("INFEASIBLE:")
        print(json.dumps(report, indent=2))
        return 1
    kept = [c for c in picks if c.get("keep")]
    new = [c for c in picks if not c.get("keep")]
    unver = [c for c in new if not c.get("verified")]
    picks.sort(key=lambda c: (c["count"], c.get("keep") or "zzz"))
    (GAPS / "solution.json").write_text(json.dumps(picks, indent=1))
    print(f"solution: keep {len(kept)}, build {len(new)} "
          f"({len(unver)} of the new ones still on the old probe's count)")
    mix = defaultdict(int)
    teams = defaultdict(int)
    for c in picks:
        mix[c["kind"]] += 1
        teams[c["startown"]] += 1
    print(f"  mix {dict(mix)}   quota {QUOTA}")
    print(f"  starting team: blue {teams[0]}, red {teams[1]}")
    print(f"  distinct cells {len({c['cell'] for c in picks})}/24")
    print(f"\n{'level':>5}  {'keep / build':<34}{'class':<9}{'start':<6}cell")
    for c in picks:
        who = (c["keep"].replace("clip_", "clip ") if c.get("keep")
               else f"BUILD {c['real_match']} {c['start']}-{c['end']}"
                    + ("" if c.get("verified") else "  (level unconfirmed)"))
        print(f"{c['count']:>5}  {who:<34}{c['kind']:<9}"
              f"{'blue' if c['startown'] == 0 else 'red':<6}{c['cell']}")
    # Everything the solve wants but has not measured, plus a couple of alternates per
    # slot so a candidate that misses its level does not force another whole round.
    want = {c["key"] for c in unver}
    by_slot = defaultdict(list)
    for c in cand:
        if not c["verified"]:
            by_slot[(c["count"], c["startown"], c["kind"])].append(c)
    for c in unver:
        alts = sorted(by_slot[(c["count"], c["startown"], c["kind"])],
                      key=lambda x: x["coh"])
        for a in alts[:3]:
            want.add(a["key"])
    todo = [c for c in cand if c["key"] in want]
    (GAPS / "toverify.json").write_text(json.dumps(todo, indent=1))
    print(f"\nto re-measure at full resolution: {len(todo)} windows "
          f"({len(unver)} chosen + alternates)")
    return 0


def _count_ends(p):
    """Full-resolution people-in-shot on the window's first and last frame."""
    off = (p["offset_x"], p["offset_y"])
    lvl = f"g3h_{p['shape']}"
    s, e = p["start"], p["end"]

    def render(hide):
        f, _ = G.render_window(lvl, p["seed"], 0, e, keep={s, e - 1}, hide_slots=hide,
                               offset=off)
        return np.asarray(f, dtype=np.int16)

    plate = render(",".join(G.ALL_SLOTS))
    n = [0, 0]
    for slot in G.ALL_SLOTS:
        solo = render(",".join(x for x in G.ALL_SLOTS if x != slot))
        for i in (0, 1):
            d = np.abs(solo[i] - plate[i])
            m = d.sum(axis=2) > 30
            if not m.any():
                continue
            sp, pp = solo[i][m].astype(np.int16), plate[i][m].astype(np.int16)
            # A body replaces the grass; a shadow only darkens it.
            dark = (sp.sum(axis=1) < pp.sum(axis=1)) & (np.abs(sp - pp).max(axis=1) < 60)
            if int((~dark).sum()) >= MIN_BODY:
                n[i] += 1
    return n[0], n[1]


def cmd_verify(argv):
    from lib import use_bundle
    from scenario_factory import write_scenario
    i, n = (int(argv[0]), int(argv[1])) if len(argv) >= 2 else (0, 1)
    use_bundle(G.BUNDLE_VIS)
    for name, *_ in G.SHAPES:
        write_scenario(G.shape_spec(name), force=False)
    todo = json.loads((GAPS / "toverify.json").read_text())
    out = GAPS / "verified"
    out.mkdir(parents=True, exist_ok=True)
    left = [c for k, c in enumerate(todo)
            if k % n == i and not (out / f"{c['key']}.json").exists()]
    if not left:
        print(f"shard {i}/{n}: verify complete")
        return 3
    c = left[0]
    first, last = _count_ends(c)
    (out / f"{c['key']}.json").write_text(
        json.dumps({**c, "count_first": first, "count_last": last}))
    flag = "" if last == c["count"] else f"  <- shortlisted as {c['count']}"
    print(f"  [verify] {c['real_match']} {c['start']}-{c['end']} {c['kind']}: "
          f"ends with {last}, opens with {first}{flag}", flush=True)
    return 0


def cmd_emit(argv):
    """Write the repaired batch as picks.json, numbered by level, plus the mapping.

    NUMBERS ARE NOT RECYCLED. Nine clips leave the batch and nine join it, so there are
    nine free numbers; giving them to the new clips would make `clip_05` mean one thing in
    yesterday's scene graph and another in today's, with nothing to notice it by. The
    batch is renumbered by level instead — the convention it was built with, where
    clip_01 and clip_02 are the level-8 pair — and `clips/clip_renumbering.csv` records
    where every old number went, including the seven that went nowhere.
    """
    sol = json.loads((GAPS / "solution.json").read_text())
    ordered = sorted(sol, key=lambda c: (c["count"], c["kind"], c.get("keep") or "zzz"))
    out, mapping = [], []
    for i, c in enumerate(ordered, 1):
        c = dict(c)
        was = c.pop("keep", None)
        c["clip"] = f"clip_{i:02d}"
        c["match"] = c.pop("real_match")
        c.pop("verified", None)
        c.pop("key", None)
        if was is None:
            c["coh"] = round(c["coh"], 4)
        else:
            c["coh"] = round(c["coh"] + KEEP_BONUS, 4)      # undo the keep bonus
        c["was"] = was
        out.append(c)
        mapping.append((was or "", c["clip"], c["count"], c["kind"],
                        "kept, renumbered" if was else "newly built"))
    (GAPS / "picks_new.json").write_text(json.dumps(out, indent=2))
    gone = {f"clip_{i:02d}" for i in range(1, 25)} - {m[0] for m in mapping}
    reasons = {"clip_01": "ball outside the camera view for 65 frames",
               "clip_08": "ball outside the camera view at the corner delivery",
               "clip_04": "measured 7 people — outside the 8..19 range",
               "clip_15": "same passage of play as clip_06 and clip_19",
               "clip_06": "same passage of play as clip_15 and clip_19",
               "clip_05": "level 11 already held two, and its side started red",
               "clip_09": "level 13 already held two, and its side started blue",
               "clip_14": "level 14 already held two, and its side started red",
               "clip_16": "level 16 already held two, and its side started red"}
    # Written with the csv module, not f-strings: the notes contain commas.
    with open(CLIPS / "clip_renumbering.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["old_clip", "new_clip", "level", "situation", "note"])
        for was, now, lvl, kind, note in sorted(mapping, key=lambda m: m[1]):
            w.writerow([was, now, lvl, kind, note])
        for c in sorted(gone):
            w.writerow([c, "", "", "", f"dropped: {reasons.get(c, 'not selected')}"])
    n_new = sum(1 for m in mapping if not m[0])
    print(f"wrote {(GAPS / 'picks_new.json').relative_to(HERE)}: 24 clips, "
          f"{n_new} newly built")
    print(f"wrote {(CLIPS / 'clip_renumbering.csv').relative_to(HERE)}: "
          f"{len(gone)} old clips dropped, {24 - n_new} kept and renumbered")
    return 0


RENDERS = CACHE / "renders"
PLATES = CACHE / "auditplate"
SCANNED = CACHE / "ballscan"          # the screening pass writes the same measurement
MIN_BALL_PX = 3


def _rkey(p):
    return f"{p['match']}_{p['start']}_{p['end']}"


def _ball_px(a, b):
    return int((np.abs(a.astype(np.int16) - b.astype(np.int16)).sum(axis=2) > 30).sum())


def _shipped_gaps(p):
    """Frames of a picked window where the shipped pair shows no ball. Either the ball is
    behind a player or it is not in the shot; this cannot tell which, which is the whole
    reason the plate pass exists."""
    vis = np.load(RENDERS / f"{_rkey(p)}_vis.npz")["frames"]
    inv = np.load(RENDERS / f"{_rkey(p)}_inv.npz")["frames"]
    px = np.array([_ball_px(vis[i], inv[i]) for i in range(len(vis))])
    return np.flatnonzero(px < MIN_BALL_PX)


def cmd_audit(argv):
    """Every frame of every clip must have the ball inside the camera's view.

    The old audit checked the first and last frames and exempted the middle, on the
    grounds that a ball behind a defender reads as absent and that is just football. It is
    — but it is not the only way a frame comes back empty, and clip_01 shipped with 65
    consecutive frames in which the ball was outside the shot entirely.

    Frames with no ball in the shipped pair are referred to the plate pass, which renders
    the same window with all 22 players hidden. On an empty pitch nothing can occlude the
    ball, so a frame still showing none has none: that window is rejected and the span
    recorded in ball_offscreen.json, which `candidates` then avoids.

    0 = every clip clean, 4 = rejections recorded (re-solve), 5 = plate evidence needed.
    """
    picks = json.loads((GAPS / "picks_new.json").read_text())
    off = json.loads(OFFSCREEN.read_text()) if OFFSCREEN.exists() else {}
    need, rejected, occluded, clean = [], [], [], 0
    for p in picks:
        gaps = _shipped_gaps(p)
        if not len(gaps):
            clean += 1
            continue
        # The screening pass measures exactly this — the same window, the same plate pair —
        # so a scanned window needs no second rendering.
        pf = PLATES / f"{_rkey(p)}.npz"
        sf = SCANNED / f"{_rkey(p)}.npz"
        if not pf.exists() and not sf.exists():
            need.append(p["clip"])
            continue
        plate = np.load(pf if pf.exists() else sf)["px"]
        out = np.flatnonzero(plate < MIN_BALL_PX)
        if not len(out):
            occluded.append((p["clip"], len(gaps)))
            continue
        lo, hi = int(out.min()), int(out.max())
        off.setdefault(p["match"], []).append([p["start"] + lo, p["start"] + hi])
        rejected.append((p["clip"], p["match"], len(out), lo, hi))
    if need:
        print(f"audit: {len(need)} clips have frames with no ball and need the plate "
              f"pair: {' '.join(need)}")
        print("  run `plate_vis` then `plate_inv`, then audit again")
        return 5
    OFFSCREEN.write_text(json.dumps(off, indent=1))
    for clip, match, n, lo, hi in rejected:
        print(f"  REJECT {clip} ({match}): ball outside the camera view on {n} frames "
              f"({lo}-{hi})")
    for clip, n in occluded:
        print(f"  {clip}: ball behind a player on {n} frames — in view, accepted")
    print(f"audit: {clean + len(occluded)}/{len(picks)} clips hold the ball in view on "
          f"every frame" + ("  — re-solve" if rejected else ""))
    return 4 if rejected else 0


def _plate(argv, bundle, tag_):
    from lib import use_bundle
    from scenario_factory import write_scenario
    i, n = (int(argv[0]), int(argv[1])) if len(argv) >= 2 else (0, 1)
    use_bundle(bundle)
    for name, *_ in G.SHAPES:
        write_scenario(G.shape_spec(name), force=False)
    PLATES.mkdir(parents=True, exist_ok=True)
    picks = json.loads((GAPS / "picks_new.json").read_text())
    todo = []
    for k, p in enumerate(picks):
        if k % n != i or (PLATES / f"{_rkey(p)}.npz").exists():
            continue
        if (SCANNED / f"{_rkey(p)}.npz").exists():
            continue                      # already screened; see cmd_audit
        if tag_ == "inv" and not (PLATES / f"{_rkey(p)}_vis.npz").exists():
            continue
        if tag_ == "vis" and (PLATES / f"{_rkey(p)}_vis.npz").exists():
            continue
        if len(_shipped_gaps(p)):
            todo.append(p)
    if not todo:
        print(f"plate {tag_} shard {i}/{n}: nothing to do")
        return 3
    p = todo[0]
    frames, _ = G.render_window(f"g3h_{p['shape']}", p["seed"], 0, p["end"],
                                keep=set(range(p["start"], p["end"])),
                                hide_slots=",".join(G.ALL_SLOTS),
                                offset=(p["offset_x"], p["offset_y"]))
    frames = np.asarray(frames, dtype=np.uint8)
    if tag_ == "vis":
        np.savez_compressed(PLATES / f"{_rkey(p)}_vis.npz", frames=frames)
        print(f"  [plate vis] {p['clip']}", flush=True)
        return 0
    vis = np.load(PLATES / f"{_rkey(p)}_vis.npz")["frames"]
    px = np.array([_ball_px(vis[i], frames[i]) for i in range(len(vis))])
    np.savez_compressed(PLATES / f"{_rkey(p)}.npz", px=px)
    (PLATES / f"{_rkey(p)}_vis.npz").unlink()
    print(f"  [plate inv] {p['clip']}: ball outside the camera view on "
          f"{int((px < MIN_BALL_PX).sum())}/{len(px)} frames", flush=True)
    return 0


# ── screening: find the out-of-view windows BEFORE the solve commits to one ─────
# Measured on this batch: about a third of candidate windows have the ball leave the shot
# at some point, because the camera offset that composes the frame off-centre also pushes
# a ball drifting to a touchline off the edge. Discovering that in the audit costs a
# render, a solve and another verification round; discovering it here costs two replays.
# Screen first, then spend the 23-replay count measurement only on windows that can ship.
SCANLIST = GAPS / "scanlist.json"


def cmd_scanlist(argv):
    """The candidates a future solve could plausibly choose, per unfilled slot."""
    per_slot = int(argv[0]) if argv else 6
    ver = load_verified()
    keep = existing()
    cand = candidates(ver)
    picks, _ = gen3_assign.solve(keep + cand, LEVELS, PER_LEVEL, QUOTA, verbose=False)
    need = {(c["count"], c["startown"], c["kind"]) for c in (picks or []) 
            if not c.get("keep")}
    by_slot = defaultdict(list)
    for c in cand:
        by_slot[(c["count"], c["startown"], c["kind"])].append(c)
    out, seen = [], set()
    for slot in sorted(need):
        for c in sorted(by_slot.get(slot, []), key=lambda x: x["coh"])[:per_slot]:
            k = f"{c['real_match']}_{c['start']}_{c['end']}"
            if k in seen or (SCANNED / f"{k}.npz").exists():
                continue
            seen.add(k)
            out.append(c)
    SCANLIST.write_text(json.dumps(out, indent=1))
    print(f"scanlist: {len(out)} windows across {len(need)} unfilled slots "
          f"({per_slot} candidates each, already-scanned ones skipped)")
    return 0


def _scan(argv, bundle, tag_):
    from lib import use_bundle
    from scenario_factory import write_scenario
    i, n = (int(argv[0]), int(argv[1])) if len(argv) >= 2 else (0, 1)
    use_bundle(bundle)
    for name, *_ in G.SHAPES:
        write_scenario(G.shape_spec(name), force=False)
    SCANNED.mkdir(parents=True, exist_ok=True)
    todo = []
    for k, c in enumerate(json.loads(SCANLIST.read_text())):
        if k % n != i:
            continue
        key = f"{c['real_match']}_{c['start']}_{c['end']}"
        if (SCANNED / f"{key}.npz").exists():
            continue
        if tag_ == "inv" and not (SCANNED / f"{key}_vis.npz").exists():
            continue
        if tag_ == "vis" and (SCANNED / f"{key}_vis.npz").exists():
            continue
        todo.append((key, c))
    if not todo:
        print(f"scan {tag_} shard {i}/{n}: nothing to do")
        return 3
    key, c = todo[0]
    frames, _ = G.render_window(f"g3h_{c['shape']}", c["seed"], 0, c["end"],
                                keep=set(range(c["start"], c["end"])),
                                hide_slots=",".join(G.ALL_SLOTS),
                                offset=(c["offset_x"], c["offset_y"]))
    frames = np.asarray(frames, dtype=np.uint8)
    if tag_ == "vis":
        np.savez_compressed(SCANNED / f"{key}_vis.npz", frames=frames)
        return 0
    vis = np.load(SCANNED / f"{key}_vis.npz")["frames"]
    px = np.array([_ball_px(vis[j], frames[j]) for j in range(len(vis))])
    np.savez_compressed(SCANNED / f"{key}.npz", px=px)
    (SCANNED / f"{key}_vis.npz").unlink()
    out = np.flatnonzero(px < MIN_BALL_PX)
    if len(out):
        off = json.loads(OFFSCREEN.read_text()) if OFFSCREEN.exists() else {}
        off.setdefault(c["real_match"], []).append(
            [int(c["start"] + out.min()), int(c["start"] + out.max())])
        OFFSCREEN.write_text(json.dumps(off, indent=1))
    print(f"  [scan] {key} {c['kind']}: "
          + (f"ball outside the camera view on {len(out)} frames — EXCLUDED"
             if len(out) else "ball in view throughout"), flush=True)
    return 0


def cmd_scan_vis(argv):
    return _scan(argv, G.BUNDLE_VIS, "vis")


def cmd_scan_inv(argv):
    return _scan(argv, G.BUNDLE_INV, "inv")


def cmd_plate_vis(argv):
    return _plate(argv, G.BUNDLE_VIS, "vis")


def cmd_plate_inv(argv):
    return _plate(argv, G.BUNDLE_INV, "inv")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "solve"
    sys.exit({"solve": cmd_solve, "verify": cmd_verify, "emit": cmd_emit,
              "audit": cmd_audit, "plate_vis": cmd_plate_vis,
              "plate_inv": cmd_plate_inv, "scanlist": cmd_scanlist,
              "scan_vis": cmd_scan_vis, "scan_inv": cmd_scan_inv}[cmd](sys.argv[2:]))
