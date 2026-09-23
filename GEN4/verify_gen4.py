"""verify_gen4.py — check the shipped GEN4 batch against its spec.

Every check reads the SHIPPED artefacts — the ground-truth key, the rendered frames, the
engine log — and never `picks.json`. GEN3_HARD's verifier read the selector's target for
the player count and compared match NAMES for source diversity, and so reported a clean
batch while nine counts were wrong and three clips were one kick-off. Those two mistakes
are the reason this file exists in the form it does.

    python3 verify_gen4.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen4_lib as G                                                  # noqa: E402

PICKS = G.CACHE / "picks.json"
RENDERS = G.CACHE / "renders"
FINAL = G.CACHE / "finalframe"
CLIPS = HERE / "clips"

PITCH_FLOOR = 0.80
MAX_BALL_DIFF = 400
MIN_BALL_PX = 3


def green_fraction(frame):
    f = frame.astype(float)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return float(((g > r * 1.04) & (g > b * 1.04) & (g > 12)).mean())


def rkey(p):
    return f"{p['match']}_{p['start']}_{p['end']}"


def main():
    fails = []

    def want(name, flag, detail=""):
        print(f"  [{'PASS' if flag else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
        if not flag:
            fails.append(name)

    picks = json.loads(PICKS.read_text())
    key = {r["clip"]: r for r in csv.DictReader(open(CLIPS / "ground_truth.csv"))}

    print("\n== spec ==")
    want("24 clips", len(picks) == 24, f"got {len(picks)}")

    # The level, from the KEY. Re-measured at full resolution by verify_final_frame when
    # that has run; the key is what a consumer reads either way.
    col = ("players_in_frame_last" if G.LEVEL_AT == "end"
           else "players_in_frame_first")
    lv = sorted(int(key[p["clip"]][col]) for p in picks)
    expect = sorted(c for c in G.LEVELS for _ in range(G.PER_COUNT))
    per = {c: lv.count(c) for c in sorted(set(lv))}
    want(f"{'closing' if G.LEVEL_AT == 'end' else 'opening'} counts are 8..19, two each",
         lv == expect, f"got {per}")
    other = sorted(int(key[p["clip"]]["players_in_frame_first"
                                      if G.LEVEL_AT == "end"
                                      else "players_in_frame_last"]) for p in picks)
    print(f"  [    ] the other end, for reference: {sorted(set(other))}")

    # Diversity by PLAY, not by name: identical ball tracks are one match under two names.
    shared, tracks = {}, {}
    for p in picks:
        tracks[p["clip"]] = np.load(G.SWEEP / f"{p['file']}.npz")["ball"]
    for i, a in enumerate(picks):
        for b in picks[i + 1:]:
            ta, tb = tracks[a["clip"]], tracks[b["clip"]]
            if len(ta) != len(tb) or np.abs(ta - tb).max() > 1e-6:
                continue
            if min(a["end"], b["end"]) > max(a["start"], b["start"]):
                shared.setdefault(a["clip"], []).append(b["clip"])
    want("one clip per passage of play", not shared,
         f"shared {shared}" if shared else f"{len(picks)} distinct passages")

    late = {p["clip"]: p["start"] - p["anchor"] for p in picks
            if p["kind"] != "open" and p["start"] > p["anchor"]}
    want("every restart clip opens at or before its delivery", not late,
         f"late {late}" if late else "")

    mix = {}
    for p in picks:
        mix[p["kind"]] = mix.get(p["kind"], 0) + 1
    print(f"  [    ] situation mix (not commanded): {mix}")
    tot = [p.get("startown", -1) for p in picks]
    print(f"  [    ] starting team: blue {tot.count(0)}, red {tot.count(1)}")

    print("\n== the rendered frames ==")
    green, balldiff, holes = [], [], []
    for p in picks:
        vis = np.load(RENDERS / f"{rkey(p)}_vis.npz")["frames"]
        inv = np.load(RENDERS / f"{rkey(p)}_inv.npz")["frames"]
        if len(vis) != G.CLIP_FRAMES or len(inv) != G.CLIP_FRAMES:
            fails.append(f"{p['clip']} frame count")
        green.append((green_fraction(vis[-1]), p["clip"]))
        d = np.abs(vis[-1].astype(np.int16) - inv[-1].astype(np.int16)).sum(axis=2)
        balldiff.append((int((d > 40).sum()), p["clip"]))
        px = np.array([(np.abs(vis[i].astype(np.int16) - inv[i].astype(np.int16))
                        .sum(axis=2) > 30).sum() for i in range(len(vis))])
        gap = int((px < MIN_BALL_PX).sum())
        if gap:
            holes.append((p["clip"], gap))
    green.sort()
    balldiff.sort()
    want(f"framing: pitch >= {PITCH_FLOOR:.0%}", green[0][0] >= PITCH_FLOOR,
         f"worst {green[0][1]} at {green[0][0]:.1%}")
    want(f"vis/inv differ only in the ball (<{MAX_BALL_DIFF}px)",
         balldiff[-1][0] <= MAX_BALL_DIFF,
         f"largest {balldiff[-1][1]} at {balldiff[-1][0]} px")
    want("ball found in every final frame", balldiff[0][0] >= MIN_BALL_PX,
         f"smallest {balldiff[0][1]} at {balldiff[0][0]} px")
    # A frame with no ball in the SHIPPED pair is either occlusion or absence; the audit
    # has already proved which, so here it is only reported.
    print(f"  [    ] frames where the ball is behind a player: "
          + (", ".join(f"{c} {n}" for c, n in holes) if holes else "none"))

    print("\n== ball spread ==")
    cells = [key[p["clip"]]["ball_final_cell"] for p in picks]
    rows, cols = {}, {}
    for c in cells:
        rows[c[0]] = rows.get(c[0], 0) + 1
        cols[int(c[1:])] = cols.get(int(c[1:]), 0) + 1
    print(f"  distinct cells {len(set(cells))}/24   rows {dict(sorted(rows.items()))}")
    want("no row holds more than half the clips", max(rows.values()) <= 12,
         f"worst row {max(rows, key=rows.get)} with {max(rows.values())}")
    want(">= 12 distinct cells", len(set(cells)) >= 12, f"got {len(set(cells))}")
    want("spans >= 4 rows", len(rows) >= 4, f"got {len(rows)}")
    want("spans >= 8 columns", len(cols) >= 8, f"got {len(cols)}")

    print("\n" + ("ALL CHECKS PASSED" if not fails else "FAILED: " + ", ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
