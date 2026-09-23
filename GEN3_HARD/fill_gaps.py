"""fill_gaps.py — repair the GEN3_HARD batch in place, by replacing only what is broken.

The batch is 24 clips, two ENDING with each headcount 8..19, 5 corners / 5 keeper
distributions / 5 kick-offs / 9 open play. Re-measurement broke that claim; this rebuilds
the claim without rebuilding the batch. 17 clips are kept exactly as they are and 7 are
replaced.

WHY EACH OF THE SEVEN IS BEING REPLACED — all measured, none by taste
  clip_01  the ball is outside the camera's view for 65 frames (2.6 s). The camera offset
           that composes the shot off-centre pushes a ball rolling to the touchline off
           the edge; confirmed by re-rendering with all 22 players hidden.
  clip_08  same fault, 3 frames, at the corner delivery — so the red start-marker sits on
           a frame with no ball in it.
  clip_04  measured at 7 people, outside the 8..19 range.
  clip_15  the same passage of play as clip_06 and clip_19 (`mid_bal`, `mid_even` and
  clip_19  `mid_even2` are one scenario spec under three names). clip_06 is kept because
           it is the only clip at level 10; these two go.
  clip_11  surplus: level 13 already has two clips without them.
  clip_18  surplus: level 16 already has two.

THE POOL IS THE PROBE CACHE THAT IS ALREADY ON DISK
  104 matches were probed at their shipping offsets when the batch was first built, and
  their candidate end frames carry a count and a measured ball pixel. That is a shortlist,
  not an answer: those counts came from the half-resolution probe whose rule was wrong.
  So every finalist is re-measured at FULL resolution, body pixels only, before it can
  ship — the same rule `verify_final_frame.py` applies to the batch it is joining.

    python3 fill_gaps.py plan                  # keep-set, slots, shortlist  (free)
    python3 fill_gaps.py verify [i n]          # full-res counts for the shortlist
    python3 fill_gaps.py pick                  # the 7 replacements
    python3 fill_gaps.py render_vis|render_inv [i n]
    python3 fill_gaps.py finish                # audit, compose, merge the key
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

CACHE = G.CACHE
PICKS = CACHE / "picks.json"
PLAN = CACHE / "plan.json"
COUNTS = CACHE / "counts"
PIX = CACHE / "ballpix"
GAPS = CACHE / "gaps"
RENDERS = CACHE / "renders"
CLIPS = HERE / "clips"

QUOTA = {"corner": 5, "gk_throw": 5, "kickoff": 5, "open": 9}
LEVELS = list(range(8, 20))
MIN_BODY = 20              # full-resolution body pixels before a person is in shot

DROP = ["clip_01", "clip_04", "clip_08", "clip_11", "clip_15", "clip_18", "clip_19"]


def keep_set():
    """The 17 clips that survive, with their measured level and class."""
    gt = {r["clip"]: r for r in csv.DictReader(open(CLIPS / "ground_truth.csv"))}
    picks = {p["clip"]: p for p in json.loads(PICKS.read_text())}
    keep = []
    for c, r in sorted(gt.items()):
        if c in DROP:
            continue
        keep.append({"clip": c, "level": int(r["players_in_frame_last"]),
                     "kind": picks[c]["kind"], "match": picks[c]["match"],
                     "cell": r["ball_final_cell"], "startown": picks[c]["startown"]})
    return keep, picks


def slots(keep):
    """What has to be built: (level, how many), and the class quota still unfilled."""
    have = defaultdict(int)
    mix = defaultdict(int)
    for k in keep:
        have[k["level"]] += 1
        mix[k["kind"]] += 1
    need_level = {lv: 2 - have[lv] for lv in LEVELS if have[lv] < 2}
    need_kind = {k: QUOTA[k] - mix[k] for k in QUOTA if QUOTA[k] > mix[k]}
    return need_level, need_kind, dict(mix)


def cmd_plan(argv):
    keep, picks = keep_set()
    need_level, need_kind, mix = slots(keep)
    used_match = {k["match"] for k in keep}
    used_cell = {k["cell"] for k in keep}
    # A play, not a name: the three identical shapes replay bit-identically, so a
    # replacement drawn from `mid_even_s340` would be the kick-off clip_06 already holds.
    used_play = {(G.SHAPE_ALIASES.get(m.rsplit("_s", 1)[0], m.rsplit("_s", 1)[0])
                  + "_s" + m.rsplit("_s", 1)[1]) if hasattr(G, "SHAPE_ALIASES")
                 else m for m in used_match}
    alias = {"mid_even": "mid_bal", "mid_even2": "mid_bal"}

    def play_of(m):
        shape, seed = m.rsplit("_s", 1)
        return f"{alias.get(shape, shape)}_s{seed}"

    used_play = {play_of(m) for m in used_match}

    plan = json.loads(PLAN.read_text())
    cands = []
    for p in plan:
        cf, pf = COUNTS / f"{p['match']}.npz", PIX / f"{p['match']}.json"
        if not (cf.exists() and pf.exists()):
            continue
        if p["match"] in used_match or play_of(p["match"]) in used_play:
            continue
        if p["kind"] not in need_kind:
            continue
        z = np.load(cf)
        on, ends = z["on"], list(map(int, z["ends"]))
        pix = json.loads(pf.read_text())
        owners = np.load(G.SWEEP / f"{p['match']}.npz")["owned_team"]
        for i, e in enumerate(ends):
            if str(e) not in pix:
                continue
            lvl = int(on[:, i].sum())
            if lvl not in need_level:
                continue
            w = next((w for w in p["windows"] if w["end"] - 1 == e), None)
            if w is None:
                continue
            own = G.start_possessor(owners, w["start"], w["end"])
            if own < 0:
                continue
            px, py = pix[str(e)]
            cell = G.cell_of(px, py)
            cands.append({"match": p["match"], "shape": p["shape"], "seed": p["seed"],
                          "kind": p["kind"], "offset_x": p["offset_x"],
                          "offset_y": p["offset_y"], "start": w["start"],
                          "end": w["end"], "anchor": w["anchor"], "coh": w["coh"],
                          "loose": w["loose"], "apex": w["apex"], "dnear": w["dnear"],
                          "path": w["path"], "startown": own, "level_guess": lvl,
                          "px": px, "py": py, "cell": cell,
                          "fresh_cell": cell not in used_cell})
    # Shortlist: the most coherent few per (level, class), preferring an unused grid cell
    # and one match per entry, because a match can supply only one clip in the end.
    short, seen = [], defaultdict(set)
    for c in sorted(cands, key=lambda c: (not c["fresh_cell"], c["coh"])):
        key = (c["level_guess"], c["kind"])
        if len(seen[key]) >= 4 or c["match"] in {m for v in seen.values() for m in v}:
            continue
        seen[key].add(c["match"])
        short.append(c)
    GAPS.mkdir(parents=True, exist_ok=True)
    (GAPS / "shortlist.json").write_text(json.dumps(short, indent=1))
    (GAPS / "keep.json").write_text(json.dumps(keep, indent=1))
    print(f"keep {len(keep)} clips, build {24 - len(keep)}")
    print(f"  levels still needed: {need_level}")
    print(f"  classes still needed: {need_kind}   (kept mix {mix})")
    print(f"  shortlist: {len(short)} windows")
    for (lvl, kind), ms in sorted(seen.items()):
        print(f"    level {lvl:>2} {kind:<9} {len(ms)} candidates")
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
    first = last = 0
    for slot in G.ALL_SLOTS:
        solo = render(",".join(x for x in G.ALL_SLOTS if x != slot))
        for i, which in ((0, "first"), (1, "last")):
            d = np.abs(solo[i] - plate[i])
            m = d.sum(axis=2) > 30
            if not m.any():
                continue
            sp, pp = solo[i][m].astype(np.int16), plate[i][m].astype(np.int16)
            darker = (sp.sum(axis=1) < pp.sum(axis=1)) & (np.abs(sp - pp).max(axis=1) < 60)
            body = int((~darker).sum())
            if body >= MIN_BODY:
                if i == 0:
                    first += 1
                else:
                    last += 1
    return first, last


def cmd_verify(argv):
    from lib import use_bundle
    from scenario_factory import write_scenario
    i, n = (int(argv[0]), int(argv[1])) if len(argv) >= 2 else (0, 1)
    use_bundle(G.BUNDLE_VIS)
    for name, *_ in G.SHAPES:
        write_scenario(G.shape_spec(name), force=False)
    short = json.loads((GAPS / "shortlist.json").read_text())
    out = GAPS / "verified"
    out.mkdir(exist_ok=True)
    todo = [c for k, c in enumerate(short) if k % n == i
            and not (out / f"{c['match']}_{c['start']}.json").exists()]
    if not todo:
        print(f"shard {i}/{n}: verify complete")
        return 3
    c = todo[0]
    first, last = _count_ends(c)
    (out / f"{c['match']}_{c['start']}.json").write_text(
        json.dumps({**c, "count_first": first, "count_last": last}))
    print(f"  [verify] {c['match']} {c['start']}-{c['end']} {c['kind']}: "
          f"ends with {last} (shortlisted as {c['level_guess']}), opens with {first}",
          flush=True)
    return 0


def cmd_pick(argv):
    keep, _ = keep_set()
    need_level, need_kind, _ = slots(keep)
    ver = [json.loads(f.read_text()) for f in (GAPS / "verified").glob("*.json")]
    ok = [c for c in ver if c["count_last"] in need_level]
    print(f"verified {len(ver)} candidates, {len(ok)} landed on a level that is short")
    # Greedy under the two hard constraints: the level must still be short, and the class
    # must still be owed. Most coherent first; a fresh grid cell breaks ties.
    picked, lv_left, kd_left, used = [], dict(need_level), dict(need_kind), set()
    for c in sorted(ok, key=lambda c: (not c["fresh_cell"], c["coh"])):
        if c["match"] in used:
            continue
        if lv_left.get(c["count_last"], 0) <= 0 or kd_left.get(c["kind"], 0) <= 0:
            continue
        picked.append(c)
        used.add(c["match"])
        lv_left[c["count_last"]] -= 1
        kd_left[c["kind"]] -= 1
    (GAPS / "picks.json").write_text(json.dumps(picked, indent=1))
    print(f"picked {len(picked)} of {sum(need_level.values())}")
    for c in sorted(picked, key=lambda c: c["count_last"]):
        print(f"  level {c['count_last']:>2}  {c['kind']:<9} {c['match']:<16} "
              f"{c['start']}-{c['end']}  cell {c['cell']}")
    if any(v > 0 for v in lv_left.values()):
        print(f"  STILL SHORT: levels {dict((k, v) for k, v in lv_left.items() if v > 0)}"
              f", classes {dict((k, v) for k, v in kd_left.items() if v > 0)}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    table = {"plan": cmd_plan, "verify": cmd_verify, "pick": cmd_pick}
    sys.exit(table[cmd](sys.argv[2:]))
