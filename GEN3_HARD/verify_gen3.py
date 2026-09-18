"""verify_gen3.py — check the shipped batch against the spec, and prove the checks.

Every assertion here is made against the RENDERED CLIPS or the frames they were cut from,
not against the plan that produced them. A pipeline that reports its own intentions back
is worth nothing; the point is to catch the case where intent and output diverged.

    python3 verify_gen3.py            # all checks + contact sheets
    python3 verify_gen3.py recheck N  # re-probe N clips end to end (23 replays each)

WHAT `recheck` IS FOR
  The player count is the batch's headline claim, and it comes from a probe that ran
  BEFORE the clip was rendered. `recheck` re-runs the solo probe on the exact shipped
  window — same shape, seed, start, end and camera offset — and compares. If the render
  and the probe ever disagree (a stale scenario file, an env var not set on one path, a
  cache keyed wrongly), this is what finds it.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen3_lib as G                                                  # noqa: E402

PICKS = G.CACHE / "picks.json"
RENDERS = G.CACHE / "renders"


def rkey(p):
    """Renders are cached by WINDOW, not by clip number — see gen3.rkey."""
    return f"{p['match']}_{p['start']}_{p['end']}"
REVIEW = HERE / "_review"

PITCH_FLOOR = 0.80          # fraction of the frame that must still be pitch
MAX_BALL_DIFF = 400         # changed pixels between the vis and inv renders of one frame


def _ok(flag):
    return "PASS" if flag else "FAIL"


def green_fraction(frame):
    """How much of the shot is turf.

    A blunt proxy, but it caught every badly framed clip in the spread batch: a third of
    the frame as empty seating scores 53-72%, every clean shot scores above 86%. Compared
    as RATIOS rather than absolute channel gaps, because half a pitch can sit in deep
    stand shadow and an absolute test reads that shadow as non-pitch.
    """
    f = frame.astype(float)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return float(((g > r * 1.04) & (g > b * 1.04) & (g > 12)).mean())


def check(picks):
    fails = []

    def want(name, flag, detail=""):
        print(f"  [{_ok(flag)}] {name}{'  ' + detail if detail else ''}")
        if not flag:
            fails.append(name)

    print("\n== spec ==")
    want("24 situations", len(picks) == 24, f"got {len(picks)}")

    counts = sorted(p["count"] for p in picks)
    expect = sorted(c for c in G.END_COUNTS for _ in range(G.PER_COUNT))
    want("counts are 8..19, two each", counts == expect, f"got {counts}")

    kinds = {}
    for p in picks:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    want("situation quota", kinds == G.SITUATION_QUOTA,
         f"got {kinds} want {G.SITUATION_QUOTA}")

    # Starting team, checked per LEVEL rather than only in total. A 12/12 grand total is
    # easy to hit while still leaving a level with two blue starts and another with two
    # red, which is exactly the imbalance this constraint exists to remove.
    by_count = {}
    for p in picks:
        by_count.setdefault(p["count"], []).append(p.get("startown", -1))
    uneven = {c: v for c, v in by_count.items() if sorted(v) != [0, 1]}
    want("each count has one blue-start and one red-start clip", not uneven,
         f"uneven levels {uneven}" if uneven else "")
    tot = sorted(p.get("startown", -1) for p in picks)
    want("starting team is 12 blue / 12 red",
         tot.count(0) == 12 and tot.count(1) == 12,
         f"got blue {tot.count(0)} red {tot.count(1)}")

    matches = {p["match"] for p in picks}
    want("one clip per match", len(matches) == len(picks),
         f"{len(matches)} distinct matches")

    want("nobody hidden", all(p.get("players_hidden", 0) == 0 for p in picks))

    print("\n== the rendered frames ==")
    green, balldiff, cont = [], [], []
    for p in picks:
        vis = np.load(RENDERS / f"{rkey(p)}_vis.npz")["frames"]
        inv = np.load(RENDERS / f"{rkey(p)}_inv.npz")["frames"]
        if len(vis) != G.CLIP_FRAMES or len(inv) != G.CLIP_FRAMES:
            fails.append(f"{p['clip']} frame count")
        green.append((green_fraction(vis[-1]), p["clip"]))
        d = np.abs(vis[-1].astype(np.int16) - inv[-1].astype(np.int16)).sum(axis=2)
        balldiff.append((int((d > 40).sum()), p["clip"]))
    green.sort()
    balldiff.sort()
    want(f"framing: pitch >= {PITCH_FLOOR:.0%}", green[0][0] >= PITCH_FLOOR,
         f"worst {green[0][1]} at {green[0][0]:.1%}")
    # The visible and invisible renders must differ ONLY by the ball. A large diff means
    # something else changed between the two passes — a kit env var not set on one path,
    # say — and then the "ball" ground truth is not the ball.
    want(f"vis/inv differ only in the ball (<{MAX_BALL_DIFF}px)",
         balldiff[-1][0] <= MAX_BALL_DIFF,
         f"largest {balldiff[-1][1]} at {balldiff[-1][0]} px")
    want("ball found in every final frame", balldiff[0][0] >= 3,
         f"smallest {balldiff[0][1]} at {balldiff[0][0]} px")

    print("\n== ball spread ==")
    cells = [p.get("final_cell_measured", p["cell"]) for p in picks]
    rows = {}
    cols = {}
    for c in cells:
        rows[c[0]] = rows.get(c[0], 0) + 1
        cols[int(c[1:])] = cols.get(int(c[1:]), 0) + 1
    print(f"  distinct cells {len(set(cells))}/24   rows {dict(sorted(rows.items()))}")
    print(f"  columns {dict(sorted(cols.items()))}")
    # The old benchmark used 15 of 96 cells and put 20 of 30 finals in one row. Those are
    # the numbers to beat, so they are the bar rather than a notional uniform ideal.
    want("no row holds more than half the clips", max(rows.values()) <= 12,
         f"worst row {max(rows, key=rows.get)} with {max(rows.values())}")
    want(">= 12 distinct cells", len(set(cells)) >= 12, f"got {len(set(cells))}")
    want("spans >= 4 rows", len(rows) >= 4, f"got {len(rows)}")
    want("spans >= 8 columns", len(cols) >= 8, f"got {len(cols)}")
    return fails


def contact_sheets(picks):
    """Frames 0 / 62 / 124 of every clip, so the batch can be reviewed without a player."""
    from PIL import Image, ImageDraw
    REVIEW.mkdir(exist_ok=True)
    for kind in sorted({p["kind"] for p in picks}):
        rows = [p for p in picks if p["kind"] == kind]
        tiles = []
        for p in rows:
            vis = np.load(RENDERS / f"{rkey(p)}_vis.npz")["frames"]
            strip = np.concatenate([vis[0], vis[G.CLIP_FRAMES // 2], vis[-1]], axis=1)
            tiles.append((p, strip))
        h, w = tiles[0][1].shape[:2]
        scale = 3
        sheet = Image.new("RGB", (w // scale, (h // scale + 18) * len(tiles)), (20, 20, 20))
        dr = ImageDraw.Draw(sheet)
        for i, (p, strip) in enumerate(tiles):
            im = Image.fromarray(strip).resize((w // scale, h // scale), Image.LANCZOS)
            y = i * (h // scale + 18)
            sheet.paste(im, (0, y))
            dr.text((4, y + h // scale + 3),
                    f"{p['clip']}  {p['kind']}  {p['count']} players  "
                    f"ball {p.get('final_cell_measured', p['cell'])}  "
                    f"off=({p['offset_x']:+.0f},{p['offset_y']:+.0f})  {p['match']}",
                    fill=(255, 255, 120))
        out = REVIEW / f"gen3_{kind}.png"
        sheet.save(out)
        print(f"  {out.name}: {len(tiles)} clips")


def recheck(picks, n):
    """Re-probe the exact shipped window and compare to the published count."""
    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle(G.BUNDLE_VIS)
    for name, *_ in G.SHAPES:
        write_scenario(G.shape_spec(name), force=False)
    bad = 0
    for p in picks[:n]:
        off = (p["offset_x"], p["offset_y"])
        e = p["end"]
        plate, _ = G.render_window(f"g3h_{p['shape']}", p["seed"], e - 1, e,
                                   hide_slots=",".join(G.ALL_SLOTS), offset=off)
        plate = np.asarray(plate[0], dtype=np.int16)[::2, ::2]
        seen = 0
        for slot in G.ALL_SLOTS:
            hide = ",".join(s for s in G.ALL_SLOTS if s != slot)
            solo, _ = G.render_window(f"g3h_{p['shape']}", p["seed"], e - 1, e,
                                      hide_slots=hide, offset=off)
            solo = np.asarray(solo[0], dtype=np.int16)[::2, ::2]
            d = np.abs(solo - plate).sum(axis=2)
            if int((d > 30).sum()) > 40:
                seen += 1
        flag = "PASS" if seen == p["count"] else "FAIL"
        print(f"  [{flag}] {p['clip']}: published {p['count']}, re-probed {seen}")
        bad += seen != p["count"]
    return bad


if __name__ == "__main__":
    picks = json.loads(PICKS.read_text())
    if len(sys.argv) > 1 and sys.argv[1] == "recheck":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        sys.exit(1 if recheck(picks, n) else 0)
    fails = check(picks)
    print("\n== contact sheets ==")
    contact_sheets(picks)
    print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)
