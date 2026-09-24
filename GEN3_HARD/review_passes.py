"""review_passes.py — how much passing happens inside a five-second clip.

Writes `_review/passes.png`.

WHAT A PASS IS HERE
  The sweep log names the player in possession on every frame. A PASS is that name moving
  between two players of the SAME side; a TURNOVER is it moving to the other side. Both
  are read off the log, which is camera-independent, so a pass counts whether or not it
  happens on screen.

  A run of possession is only counted once it has lasted two frames. Without that, a ball
  brushing past a player registers as a touch and the count inflates.

WHY THE NUMBERS ARE SMALL, AND WHY THAT IS THE ANSWER RATHER THAN A BUG
  Five seconds is about one phase of play. Measured across the batch, the ball is usually
  carried rather than exchanged — clip_22 has a single player in possession for all 125
  frames — so most clips complete no pass at all and none completes more than one. The
  chart therefore leads with the distribution, which is the honest shape of it, and puts
  the per-clip detail beside it with turnovers included: a clip with no pass is not
  necessarily a clip where nothing happened.
"""
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
MIN_HOLD = 2                # frames of possession before a touch is real

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
S1 = "#2a78d6"          # categorical slot 1 — passes
S2 = "#eb6834"          # categorical slot 2 — turnovers
KIND_NAME = {"corner": "corner", "gk_throw": "keeper", "kickoff": "kick-off",
             "open": "open play"}


def possession_runs(log, s, e):
    """(team, player) in possession, in order, once each run has held MIN_HOLD frames."""
    runs, cur, n = [], None, 0
    for t, p in zip(log["owned_team"][s:e], log["owned_player"][s:e]):
        if t < 0:
            cur, n = None, 0            # loose ball: the run ends, nothing is recorded
            continue
        key = (int(t), int(p))
        if key == cur:
            n += 1
            if n == MIN_HOLD:
                runs.append(key)
        else:
            cur, n = key, 1
    return runs


def measure():
    out = []
    for p in json.loads(PICKS.read_text()):
        z = np.load(G.SWEEP / f"{p['match']}.npz")
        runs = possession_runs({k: z[k] for k in z.files}, p["start"], p["end"])
        pairs = list(zip(runs, runs[1:]))
        out.append({
            "clip": p["clip"], "kind": p["kind"],
            "passes": sum(1 for a, b in pairs if a[0] == b[0] and a[1] != b[1]),
            "turnovers": sum(1 for a, b in pairs if a[0] != b[0]),
            "touches": len(runs),
        })
    return sorted(out, key=lambda r: r["clip"])


def chart(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "text.color": INK,
                         "axes.facecolor": SURFACE})
    fig = plt.figure(figsize=(11.4, 6.6), dpi=150)

    # ── the distribution: how many clips complete how many passes ──────────────
    ax = fig.add_axes([0.055, 0.135, 0.26, 0.56])
    hist = Counter(r["passes"] for r in rows)
    xs = list(range(0, max(hist) + 1))
    ys = [hist.get(x, 0) for x in xs]
    ax.bar(xs, ys, width=0.62, color=S1)
    for x, y in zip(xs, ys):
        if y:
            ax.text(x, y + 0.4, str(y), ha="center", fontsize=9, color=INK2)
    ax.set_xticks(xs, [str(x) for x in xs], fontsize=9.5)
    ax.set_xlabel("passes completed in the clip", fontsize=9, color=INK2, labelpad=6)
    ax.set_ylabel("clips", fontsize=9, color=INK2)
    ax.set_ylim(0, max(ys) * 1.22)
    ax.set_yticks(range(0, max(ys) + 2, 4))     # clips are whole things
    ax.yaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # ── the detail: which clips, and what else happened in them ────────────────
    ax2 = fig.add_axes([0.50, 0.135, 0.455, 0.56])
    y = np.arange(len(rows))[::-1]
    for i, r in zip(y, rows):
        ax2.barh(i + 0.16, r["passes"], height=0.30, color=S1)
        ax2.barh(i - 0.16, r["turnovers"], height=0.30, color=S2)
        if not r["passes"] and not r["turnovers"]:
            ax2.text(0.06, i, "ball never changed hands", va="center", ha="left",
                     fontsize=7.5, color=MUTED)
    ax2.set_yticks(y, [f"{r['clip'].replace('clip_', 'clip ')}  {KIND_NAME[r['kind']]}"
                       for r in rows], fontsize=7.5)
    for t in ax2.get_yticklabels():
        t.set_color(INK2)
    ax2.set_xlim(0, max(max(r["passes"], r["turnovers"]) for r in rows) + 0.6)
    ax2.set_ylim(-0.8, len(rows) - 0.2)
    ax2.set_xticks(range(0, max(max(r["passes"], r["turnovers"]) for r in rows) + 1))
    ax2.set_xlabel("events in the clip", fontsize=9, color=INK2, labelpad=6)
    ax2.xaxis.grid(True, color=GRID, lw=0.7)
    ax2.set_axisbelow(True)
    ax2.tick_params(length=0, labelsize=9)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    handles = [plt.Line2D([], [], marker="s", ls="", ms=9, color=S1,
                          label="pass — to a team-mate"),
               plt.Line2D([], [], marker="s", ls="", ms=9, color=S2,
                          label="turnover — to the other side")]
    ax2.legend(handles=handles, ncol=2, loc="lower left", frameon=False, fontsize=8.5,
               bbox_to_anchor=(-0.30, 1.01), labelcolor=INK2, handletextpad=0.6,
               columnspacing=1.8)

    tot = sum(r["passes"] for r in rows)
    none = sum(1 for r in rows if not r["passes"])
    fig.text(0.045, 0.945, "Passes completed inside each five-second clip",
             fontsize=16, color=INK, weight="bold")
    fig.text(0.045, 0.905,
             f"GEN3_HARD, 24 clips. {tot} passes in the batch; {none} clips complete "
             "none and no clip completes more than one.",
             fontsize=9.5, color=INK2)
    fig.text(0.045, 0.043,
             "A pass is the player in possession changing to a team-mate, read from the "
             "engine log, so it counts whether or not it happens on screen. Five seconds "
             "is about one phase of play:",
             fontsize=7.5, color=MUTED)
    fig.text(0.045, 0.022,
             "the ball is usually carried rather than exchanged — in clip 22 one player "
             "holds it for all 125 frames. Turnovers are shown beside each clip so a "
             "quiet passing count is not mistaken for a quiet clip.",
             fontsize=7.5, color=MUTED)
    out = REVIEW / "passes.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(HERE)}  ({tot} passes over {len(rows)} clips)")


if __name__ == "__main__":
    REVIEW.mkdir(exist_ok=True)
    rows = measure()
    chart(rows)
    import csv
    with open(REVIEW / "passes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip", "kind", "passes", "turnovers",
                                           "touches"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote _review/passes.csv")
