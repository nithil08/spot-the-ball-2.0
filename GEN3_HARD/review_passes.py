"""review_passes.py — how often the ball changes hands while it is hidden.

Writes `_review/passes.png` and `_review/passes.csv`.

WHY THE HIDDEN WINDOW AND NOT THE WHOLE CLIP
  The benchmark shows one second of visible ball and hides it for four. Whatever a model
  or a person has to reason about happens in those four seconds, so that is the interval
  the count is taken over. Over the whole five it is a different and less useful number.

WHAT COUNTS AS A CHANGE OF HANDS
  The engine log names the player in possession on every frame. A change is that name
  moving to a different player: to a TEAM-MATE is a pass, to the other side is a turnover.
  A possession run has to last two frames before it counts, or a ball brushing past a
  player registers as a touch and the count inflates.

THE NUMBER THIS IS REALLY REPORTING
  Passing is almost absent: over the hidden window the batch completes 4 passes in total.
  If passing is meant to be an independent variable — does more of it change where the
  ball can be inferred to be — this batch cannot carry the experiment, and the histogram
  is the clearest way to see why.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen3_lib as G                                                  # noqa: E402

REVIEW = HERE / "_review"
PICKS = G.CACHE / "picks.json"
VIS = G.VIS_FRAMES          # visible for frames 0..VIS-1, hidden after
MIN_HOLD = 2                # frames a player must hold the ball for it to count

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
S1 = "#2a78d6"          # categorical slot 1 — passes, to a team-mate
S2 = "#eb6834"          # categorical slot 2 — turnovers, to the other side


def measure():
    out = []
    for p in json.loads(PICKS.read_text()):
        z = np.load(G.SWEEP / f"{p['match']}.npz")
        s, e = p["start"], p["end"]
        ot, op = z["owned_team"][s:e], z["owned_player"][s:e]
        runs, cur, n = [], None, 0
        for t, pl in zip(ot[VIS - 1:], op[VIS - 1:]):
            if t < 0:
                cur, n = None, 0
                continue
            key = (int(t), int(pl))
            if key == cur:
                n += 1
                if n == MIN_HOLD:
                    runs.append(key)
            else:
                cur, n = key, 1
        pairs = [(a, b) for a, b in zip(runs, runs[1:]) if a != b]
        out.append({"clip": p["clip"], "situation": p["kind"],
                    "passes": sum(1 for a, b in pairs if a[0] == b[0]),
                    "turnovers": sum(1 for a, b in pairs if a[0] != b[0]),
                    "changes": len(pairs)})
    return sorted(out, key=lambda r: r["clip"])


def chart(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "text.color": INK,
                         "axes.facecolor": SURFACE})
    xs = list(range(0, max(r["changes"] for r in rows) + 1))
    # Each bar is split by what the changes in those clips were, so the bar height answers
    # "how often does the ball change hands" and the colour answers "to whom" — which is
    # the distinction the research question turns on.
    passes = [sum(r["passes"] for r in rows if r["changes"] == x) for x in xs]
    turns = [sum(r["turnovers"] for r in rows if r["changes"] == x) for x in xs]
    clips = [sum(1 for r in rows if r["changes"] == x) for x in xs]

    fig = plt.figure(figsize=(9.6, 5.6), dpi=150)
    ax = fig.add_axes([0.085, 0.225, 0.88, 0.545])
    ax.bar(xs, clips, width=0.6, color=S1)
    for x, c in zip(xs, clips):
        ax.text(x, c + 0.3, str(c), ha="center", fontsize=12, color=INK, weight="bold")
    # The breakdown rides on the tick label rather than floating near the bar: a second
    # line of text beside a bar collides with the axis the moment the bars are short.
    labels = []
    for x, np_, nt in zip(xs, passes, turns):
        bits = []
        if np_:
            bits.append(f"{np_} pass" + ("es" if np_ > 1 else ""))
        if nt:
            bits.append(f"{nt} turnover" + ("s" if nt > 1 else ""))
        # parenthesised, because "4 passes" under the bar at x=1 otherwise reads as
        # "these clips had four passes each" rather than as the bar's composition
        labels.append(f"{x}\n({', '.join(bits)})" if bits else str(x))
    ax.set_xticks(xs, labels, fontsize=11)
    ax.set_yticks(range(0, max(clips) + 3, 4))
    ax.set_ylim(0, max(clips) * 1.18)
    ax.set_xlim(-0.6, max(xs) + 0.6)
    ax.set_xlabel("times the ball changes hands while it is hidden",
                  fontsize=10, color=INK2, labelpad=12)
    ax.set_ylabel("clips", fontsize=10, color=INK2)
    ax.yaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=10)
    for sp in ax.spines.values():
        sp.set_visible(False)

    tot_p = sum(r["passes"] for r in rows)
    tot_t = sum(r["turnovers"] for r in rows)
    fig.text(0.085, 0.935, "How often the ball changes hands while it is hidden",
             fontsize=16, color=INK, weight="bold")
    fig.text(0.085, 0.888,
             f"GEN3_HARD, 24 clips, over the four seconds the ball is invisible. "
             f"{tot_p} passes and {tot_t} turnovers in the whole batch.",
             fontsize=9.5, color=INK2)
    fig.text(0.085, 0.855,
             f"{clips[0]} of {len(rows)} clips: the ball never leaves the player who has "
             "it.", fontsize=9.5, color=S2)
    fig.text(0.085, 0.072,
             "A change of hands is the player in possession becoming a different one, "
             "read from the engine log, so it counts whether or not it happens",
             fontsize=7.5, color=MUTED)
    fig.text(0.085, 0.050,
             "on screen. A run of possession must last two frames to count.",
             fontsize=7.5, color=MUTED)
    fig.text(0.085, 0.022,
             "If passing is meant to be the variable — whether more of it changes where "
             "the ball can be inferred to be — this batch cannot carry it.",
             fontsize=7.5, color=INK2)
    out = REVIEW / "passes.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(HERE)}  ({tot_p} passes, {tot_t} turnovers, "
          f"{clips[0]}/{len(rows)} clips with no change of hands)")


if __name__ == "__main__":
    REVIEW.mkdir(exist_ok=True)
    rows = measure()
    chart(rows)
    with open(REVIEW / "passes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote _review/passes.csv")
