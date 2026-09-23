"""review_rebuild.py — the batch as it will be: 17 clips kept, 7 slots still to build.

Writes `_review/player_counts_rebuild.png`. It does NOT touch `player_counts.png`, which
is the record of the batch as it stands today.

WHAT THE CHART HAS TO SAY, WHICH THE OTHER ONE CANNOT
  The counts chart is one row per clip. This one is one row per SLOT — twelve levels,
  two clips each, twenty-four slots — because the subject is the spec rather than the
  clips, and an empty slot has to be as visible as a filled one. A level short of a clip
  is the whole point of the picture; drawn as one row per existing clip it would be
  invisible, being the row that is not there.

  Slots are therefore always drawn in pairs, and an unfilled one is drawn as an empty
  outline carrying the reason it is unfilled. State is never colour alone: a pending slot
  is dashed, hollow, and labelled in words.

  The level IS the closing count, so every orange dot sits at its own row's level and the
  chart reads as a staircase. That is a property of the spec being met, and a dot off the
  staircase would be a clip in the wrong slot.
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

REVIEW = HERE / "_review"
CLIPS = HERE / "clips"
TRACE = G.CACHE / "trace"
PICKS = G.CACHE / "picks.json"

# ── the reference palette, light surface (categorical slots 1 and 2, unchanged) ──
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
S1 = "#2a78d6"          # categorical slot 1 — the first frame
S2 = "#eb6834"          # categorical slot 2 — the final frame, which is the level
SEQ0 = "#cde2fb"        # blue ramp step 100 — the range bar, a recessive backdrop

LEVELS = list(range(8, 20))
KIND_NAME = {"corner": "corner kick", "gk_throw": "goalkeeper distribution",
             "kickoff": "kick-off", "open": "open play"}

# Why each replaced clip is going. Written here rather than derived because the reason is
# a finding, not a measurement — each one is stated in _review/CLIP_REVIEW.md.
DROPPED = {
    "clip_01": (8, "ball left the shot for 2.6 s"),
    "clip_08": (12, "ball left the shot at the corner"),
    "clip_19": (17, "duplicate of clip 06"),
    "clip_04": (None, "measured 7 — outside 8-19"),
    "clip_15": (None, "same kick-off as clip 06"),
    "clip_11": (None, "surplus at level 13"),
    "clip_18": (None, "surplus at level 16"),
}
NEVER_FILLED = "level was never filled"


def load():
    gt = {r["clip"]: r for r in csv.DictReader(open(CLIPS / "ground_truth.csv"))}
    picks = {p["clip"]: p for p in json.loads(PICKS.read_text())}
    kept = []
    for c, r in sorted(gt.items()):
        if c in DROPPED:
            continue
        t = TRACE / f"{picks[c]['match']}_{picks[c]['start']}_{picks[c]['end']}.npz"
        n = np.load(t)["on"].sum(axis=0) if t.exists() else None
        kept.append({"clip": c, "level": int(r["players_in_frame_last"]),
                     "kind": picks[c]["kind"],
                     "first": int(n[0]) if n is not None else None,
                     "lo": int(n.min()) if n is not None else None,
                     "hi": int(n.max()) if n is not None else None})
    return kept


def rows(kept):
    """One entry per SLOT, two per level, filled ones first."""
    by = defaultdict(list)
    for k in kept:
        by[k["level"]].append(k)
    out = []
    for lv in LEVELS:
        have = sorted(by.get(lv, []), key=lambda k: k["clip"])
        for k in have[:2]:
            out.append({"level": lv, "clip": k})
        for _ in range(2 - len(have[:2])):
            repl = next((c for c, (l, _w) in DROPPED.items() if l == lv), None)
            out.append({"level": lv, "clip": None,
                        "why": (f"replaces {repl.replace('clip_', 'clip ')}: "
                                f"{DROPPED[repl][1]}") if repl else NEVER_FILLED})
    return out


def chart(kept):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "text.color": INK,
                         "axes.facecolor": SURFACE})
    data = rows(kept)
    fig = plt.figure(figsize=(11.2, 10.2), dpi=150)
    ax = fig.add_axes([0.075, 0.075, 0.535, 0.735])
    y = np.arange(len(data))[::-1]
    # x in axes fraction, y in data units: the label gutter to the right of the plot.
    tr = ax.get_yaxis_transform()

    lo_x = min([k["clip"]["lo"] for k in data if k["clip"]] + [3]) - 1.8
    hi_x = max([k["clip"]["hi"] for k in data if k["clip"]] + [20]) + 0.8

    for i, d in zip(y, data):
        # A band behind each LEVEL, so the two slots of a level read as a pair.
        if d is data[0] or d["level"] != data[list(y).index(i) - 1]["level"]:
            pass
        k = d["clip"]
        if k is None:
            # Pending: hollow, dashed, and labelled. Never state by colour alone.
            ax.add_patch(Rectangle((d["level"] - 0.42, i - 0.30), 0.84, 0.60,
                                   fill=False, ec=MUTED, lw=1.3, ls=(0, (3, 2)),
                                   zorder=3))
            ax.text(1.02, i, "to build", va="center", ha="left", fontsize=8.5,
                    color=INK2, weight="bold", transform=tr)
            ax.text(1.115, i, d["why"], va="center", ha="left", fontsize=8,
                    color=MUTED, transform=tr)
            continue
        ax.plot([k["lo"], k["hi"]], [i, i], color=SEQ0, lw=7, solid_capstyle="round",
                zorder=1)
        ax.plot([k["first"], k["level"]], [i, i], color=AXIS, lw=2,
                solid_capstyle="round", zorder=2)
        same = k["first"] == k["level"]
        ax.plot(k["first"], i, "o", ms=13 if same else 9, mfc=S1, mec=SURFACE, mew=1.6,
                zorder=3)
        ax.plot(k["level"], i, "o", ms=9, mfc=S2, mec=SURFACE if not same else S2,
                mew=1.6 if not same else 0, zorder=4)
        lo, hi = sorted((k["first"], k["level"]))
        if lo != hi:
            ax.text(lo - 0.42, i, str(lo), va="center", ha="right", fontsize=8,
                    color=INK2)
        ax.text(1.02, i, k["clip"].replace("clip_", "clip "), va="center", ha="left",
                fontsize=8.5, color=INK2, transform=tr)
        ax.text(1.115, i, KIND_NAME[k["kind"]], va="center", ha="left", fontsize=8,
                color=MUTED, transform=tr)

    ax.set_yticks(y, [f"level {d['level']}" if j % 2 == 0 else ""
                      for j, d in enumerate(data)], fontsize=9.5)
    for t in ax.get_yticklabels():
        t.set_color(INK2)
    # A hairline between levels, so the pairing is structural rather than remembered.
    for j in range(2, len(data), 2):
        ax.axhline(len(data) - j - 0.5, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("people in shot   (the level is the count on the final frame)",
                  fontsize=9, color=INK2, labelpad=8)
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(-0.8, len(data) - 0.2)
    ax.set_xticks(range(int(np.ceil(lo_x)), 21, 2))
    ax.tick_params(axis="y", pad=6)
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc=S1, mec=SURFACE, mew=1.6,
                   label="first frame"),
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc=S2, mec=SURFACE, mew=1.6,
                   label="final frame — the level"),
        plt.Line2D([], [], color=SEQ0, lw=7, solid_capstyle="round",
                   label="range during the clip"),
        plt.Line2D([], [], marker="s", ls="", ms=9, mfc="none", mec=MUTED, mew=1.3,
                   label="slot still to build"),
    ]
    ax.legend(handles=handles, ncol=2, loc="lower left", frameon=False, fontsize=8.5,
              bbox_to_anchor=(-0.005, 1.012), handletextpad=0.6, columnspacing=2.2,
              labelcolor=INK2)

    n_keep = sum(1 for d in data if d["clip"])
    fig.text(0.04, 0.973, "GEN3_HARD after the repair: 12 levels, two clips each",
             fontsize=16, color=INK, weight="bold")
    fig.text(0.04, 0.942,
             f"{n_keep} clips are kept exactly as they are and {len(data) - n_keep} "
             "slots are being rebuilt. Every count re-measured at full resolution, "
             "body pixels only.", fontsize=9, color=INK2)
    fig.text(0.04, 0.921,
             "The seven still to build must come out as 2 corners, 2 keeper "
             "distributions, 2 kick-offs and 1 open play — which is what completes "
             "5 / 5 / 5 / 9.", fontsize=9, color=INK2)
    fig.text(0.04, 0.020,
             "Also dropped, without leaving a gap of their own: clip 04 (measured 7, "
             "outside the range), clip 11 and clip 18 (surplus at a level that already "
             "held two), clip 15 (the same kick-off as clip 06).",
             fontsize=7.5, color=MUTED)
    out = REVIEW / "player_counts_rebuild.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(HERE)}  ({n_keep} kept, {len(data) - n_keep} to build)")


if __name__ == "__main__":
    REVIEW.mkdir(exist_ok=True)
    chart(load())
