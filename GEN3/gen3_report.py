"""gen3_report.py — the three review artefacts for the shipped GEN3 batch.

    python3 gen3_report.py            # all three
    python3 gen3_report.py heatmap    # _review/ball_final_heatmap.png
    python3 gen3_report.py counts     # _review/player_counts.png
    python3 gen3_report.py classify   # clips/clip_classification.csv

Everything here is DERIVED — it reads the cached sweep log, the trace probe and the
shipped ground truth, and runs no engine. Re-running it cannot change the batch.

WHERE EACH NUMBER COMES FROM, because they are not equally hard-won
  ball cell / pixel   measured in `compose` from the shipped render pair
  players in frame    measured by the 23-pass solo probe (`trace`) at the shipping
                      camera offset — the only exact method; see gen3.py
  everything else     the sweep log, which is camera-independent: ball xyz, per-player
                      xy, possession, game mode, score, at 25 fps

PITCH CONVENTIONS (engine coordinates, fixed by the scenarios)
  x runs -1 .. +1 goal line to goal line, y runs about -0.42 .. +0.42 touchline to
  touchline, z is metres. TEAM A is the LEFT team and wears BLUE; it defends x = -1 and
  attacks x = +1. TEAM B is the RIGHT team and wears RED; it defends x = +1. So a ball at
  x = +0.8 is deep in RED's defensive third, whichever side has it.

COLOUR
  The dataviz reference palette, used unchanged — slots 1 (blue) and 2 (orange) for the
  two-series charts, the blue sequential ramp for magnitude. Values are not substituted,
  so the published validation for that palette stands as-is.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen3_lib as G                                                  # noqa: E402

PICKS = G.CACHE / "picks.json"
SWEEP = G.CACHE / "sweep"
TRACE = G.CACHE / "trace"
REVIEW = HERE / "_review"
CLIPS = HERE / "clips"

# ── the reference palette, light surface ────────────────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
S1 = "#2a78d6"          # categorical slot 1 — blue
S2 = "#eb6834"          # categorical slot 2 — orange
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]   # blue ramp, light->dark

# metres per unit of engine coordinate (gfootball's 108 x 68 m pitch)
MX, MY = 54.4, 83.6
NEAR = 0.06             # "near the ball" radius, the same one the coherence gate uses


def rkey(p):
    return f"{p['match']}_{p['start']}_{p['end']}"


def load():
    """Every shipped clip, with its window of log, its trace, and its ground truth row."""
    picks = json.loads(PICKS.read_text())
    gt = {r["clip"]: r for r in csv.DictReader(open(CLIPS / "ground_truth.csv"))}
    out = []
    for p in picks:
        z = np.load(SWEEP / f"{p['match']}.npz")
        s, e = p["start"], p["end"]
        row = dict(p)
        row["gt"] = gt[p["clip"]]
        row["ball"] = z["ball"][s:e]
        row["owned"] = z["owned_team"][s:e]
        row["mode"] = z["game_mode"][s:e]
        row["left"] = z["left"][s:e]
        row["right"] = z["right"][s:e]
        tf = TRACE / f"{rkey(p)}.npz"
        row["trace"] = np.load(tf)["on"].sum(axis=0) if tf.exists() else None
        out.append(row)
    return out


# ══ what is happening ══════════════════════════════════════════════════════════
KIND_NAME = {"corner": "corner kick", "gk_throw": "goalkeeper distribution",
             "kickoff": "kick-off", "open": "open play"}
TEAM_NAME = {0: "blue", 1: "red", -1: "loose"}


def third(x):
    """Which third of the pitch, named by whose goal it is nearest.

    Named by the DEFENDING side rather than by "attacking third", because possession
    changes hands inside these windows and a label that flips with it would describe the
    same patch of grass two different ways in one row.
    """
    if x < -1 / 3:
        return "blue defensive third"
    if x > 1 / 3:
        return "red defensive third"
    return "middle third"


def near_ball(row, i):
    pl = np.vstack([row["left"][i], row["right"][i]])
    return int((np.linalg.norm(pl - row["ball"][i][:2], axis=1) < NEAR).sum())


def crowding(n):
    return "isolated" if n < 2 else ("contested" if n < 5 else "crowded")


RESTART_VERB = {"corner": "corner kick taken", "gk_throw": "keeper releases the ball",
                "kickoff": "kick-off restart"}


def _restart_phrase(r):
    """When the restart happens relative to the clip — and it is often BEFORE it.

    `start = anchor - LEAD + slide`, and the slide runs to 50 frames against a LEAD of 10
    for a corner, so the most-slid corner window opens 1.6 s AFTER the ball was struck.
    Said plainly rather than as a negative number, because a reader scanning the column
    should not have to work out what "-1.6 s in" means.
    """
    what = RESTART_VERB[r["kind"]]
    t = r["restart_s"]
    if r["restart_visible"]:
        return f"{what} {t:.1f} s into the clip"
    return f"{what} {abs(t):.1f} s before the clip opens, so the clip shows what followed"


def _n_players(n):
    return f"{n} player" + ("" if n == 1 else "s")


def describe(r):
    """One sentence, built from the measured fields rather than written per clip."""
    bits = []
    if r["kind"] == "open":
        bits.append(f"Open play in the {r['zone_start']}")
    else:
        bits.append(_restart_phrase(r).capitalize())
    # Direction comes from NET displacement, distance from the path actually walked —
    # a ball can travel 47 m and finish 6 m from where it started.
    bits.append(f"net movement {r['net_m']:.0f} m {r['direction']} over "
                f"{r['travel_m']:.0f} m of travel, finishing in the {r['zone_end']}")
    if r["apex_m"] >= 2.0:
        bits.append(f"lofted to {r['apex_m']:.1f} m")
    bits.append(f"{_n_players(r['near_end'])} around it at the end "
                f"({r['crowding_end']})")
    who = TEAM_NAME[r["poss_end"]]
    bits.append("nobody in possession at the end" if who == "loose"
                else f"{who} in possession at the end")
    if r["turnovers"]:
        bits.append(f"{r['turnovers']} change{'s' if r['turnovers'] > 1 else ''} of "
                    f"possession")
    if r["count_start"] is not None:
        d = r["count_end"] - r["count_start"]
        change = ("unchanged" if d == 0 else f"{d:+d}")
        bits.append(f"{r['count_start']} people in shot at the start and "
                    f"{r['count_end']} at the end ({change})")
    return "; ".join(bits) + "."


def classify(rows):
    for r in rows:
        b, own = r["ball"], r["owned"]
        step = np.diff(b[:, :2], axis=0) * (MX, MY)
        r["travel_m"] = float(np.linalg.norm(step, axis=1).sum())
        net = (b[-1, :2] - b[0, :2]) * (MX, MY)
        r["net_m"] = float(np.linalg.norm(net))
        # Direction is named by the goal it heads for, matching `third`.
        if abs(net[0]) < 5:
            r["direction"] = "across the pitch"
        else:
            r["direction"] = ("towards the red goal" if net[0] > 0
                              else "towards the blue goal")
        r["apex_m"] = float(b[:, 2].max())
        r["airborne_frac"] = float((b[:, 2] > 0.5).mean())
        r["zone_start"], r["zone_end"] = third(b[0, 0]), third(b[-1, 0])
        # Possession is read at the two ends as it actually stands there — -1 means the
        # ball is genuinely loose, which is a fact about the frame, not missing data.
        r["poss_start"], r["poss_end"] = int(own[0]), int(own[-1])
        held = own[own >= 0]
        r["turnovers"] = int((np.diff(held) != 0).sum()) if len(held) > 1 else 0
        r["near_start"], r["near_end"] = near_ball(r, 0), near_ball(r, len(b) - 1)
        r["crowding_end"] = crowding(r["near_end"])
        r["restart_s"] = (r["anchor"] - r["start"]) / G.FPS
        r["restart_visible"] = bool(r["start"] <= r["anchor"] < r["end"])
        t = r["trace"]
        r["count_start"] = int(t[0]) if t is not None else None
        r["count_end"] = int(t[-1]) if t is not None else int(r["count"])
        r["count_min"] = int(t.min()) if t is not None else None
        r["count_max"] = int(t.max()) if t is not None else None
        r["description"] = describe(r)
    return rows


FIELDS = [
    ("clip", lambda r: r["clip"]),
    ("situation", lambda r: KIND_NAME[r["kind"]]),
    ("description", lambda r: r["description"]),
    ("players_in_frame_start", lambda r: r["count_start"]),
    ("players_in_frame_end", lambda r: r["count_end"]),
    ("players_in_frame_delta", lambda r: (None if r["count_start"] is None
                                          else r["count_end"] - r["count_start"])),
    ("players_in_frame_min", lambda r: r["count_min"]),
    ("players_in_frame_max", lambda r: r["count_max"]),
    ("ball_start_cell", lambda r: r["gt"]["ball_start_cell"]),
    ("ball_final_cell", lambda r: r["gt"]["ball_final_cell"]),
    ("zone_start", lambda r: r["zone_start"]),
    ("zone_end", lambda r: r["zone_end"]),
    ("ball_direction", lambda r: r["direction"]),
    ("ball_travel_m", lambda r: round(r["travel_m"], 1)),
    ("ball_net_m", lambda r: round(r["net_m"], 1)),
    ("ball_apex_m", lambda r: round(r["apex_m"], 2)),
    ("airborne_fraction", lambda r: round(r["airborne_frac"], 3)),
    ("possession_start", lambda r: TEAM_NAME[r["poss_start"]]),
    ("possession_end", lambda r: TEAM_NAME[r["poss_end"]]),
    ("possession_changes", lambda r: r["turnovers"]),
    ("players_near_ball_start", lambda r: r["near_start"]),
    ("players_near_ball_end", lambda r: r["near_end"]),
    ("crowding_end", lambda r: r["crowding_end"]),
    # Signed: negative means the restart happened BEFORE the clip opens, which is the
    # case for every corner in this batch. See _restart_phrase.
    ("restart_seconds_into_clip", lambda r: (round(r["restart_s"], 2)
                                             if r["kind"] != "open" else "")),
    ("restart_visible_in_clip", lambda r: ("" if r["kind"] == "open"
                                           else str(r["restart_visible"]).lower())),
    ("shape", lambda r: r["shape"]),
    ("seed", lambda r: r["seed"]),
    ("match", lambda r: r["match"]),
    ("start_frame", lambda r: r["start"]),
    ("end_frame", lambda r: r["end"]),
    ("camera_offset_x", lambda r: r["offset_x"]),
    ("camera_offset_y", lambda r: r["offset_y"]),
]


def cmd_classify(rows):
    out = CLIPS / "clip_classification.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([n for n, _ in FIELDS])
        for r in rows:
            w.writerow(["" if (v := f(r)) is None else v for _, f in FIELDS])
    print(f"wrote {out.relative_to(HERE)}  ({len(rows)} clips x {len(FIELDS)} fields)")


# ══ the charts ═════════════════════════════════════════════════════════════════
def _fig(w, h):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        # The system sans, named concretely: matplotlib resolves font FAMILIES, and
        # "system-ui" is a CSS generic it cannot look up — asking for it prints a
        # findfont warning per text object and silently falls back.
        "font.family": ["Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.weight": "regular",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })
    return plt, plt.figure(figsize=(w, h), dpi=200)


def cmd_heatmap(rows):
    """Where the ball ends, on the 16x6 answer grid.

    NOT a smoothed density: 24 finals over 96 cells is far too sparse for a kernel, and
    smoothing would draw structure that is not in the data. The cells carry coverage, and
    the two marginals carry the distribution that actually matters — how the finals spread
    across columns and rows against a uniform expectation.
    """
    plt, fig = _fig(11, 6.4)
    gs = fig.add_gridspec(2, 2, width_ratios=[16, 3.1], height_ratios=[3.0, 6],
                          wspace=0.045, hspace=0.05,
                          left=0.055, right=0.975, top=0.845, bottom=0.075)
    ax = fig.add_subplot(gs[1, 0])
    axc = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)

    grid = np.zeros((G.ROWS, G.COLS), dtype=int)
    for r in rows:
        c = r["gt"]["ball_final_cell"]
        grid[ord(c[0]) - 65, int(c[1:]) - 1] += 1

    ax.imshow(np.where(grid > 0, grid, np.nan), cmap=_cmap(plt, grid.max()),
              vmin=0.5, vmax=max(grid.max(), 1) + 0.5,
              extent=[0, G.COLS, G.ROWS, 0], aspect="auto", interpolation="nearest")
    for x in range(G.COLS + 1):
        ax.axvline(x, color=GRID, lw=0.6, zorder=2)
    for y in range(G.ROWS + 1):
        ax.axhline(y, color=GRID, lw=0.6, zorder=2)

    # The exact measured pixel, and the clip it belongs to. The label goes in the CELL's
    # own corner rather than beside its dot: a dot can sit hard against a cell edge, and
    # a label offset from it then lands in the neighbouring cell and reads as belonging
    # to that one. Every cell holds at most one clip, so the corner is unambiguous.
    for r in rows:
        px, py = float(r["gt"]["final_px"]), float(r["gt"]["final_py"])
        gx, gy = px / G.CELL_W, py / G.CELL_H
        ax.plot(gx, gy, "o", ms=6, mfc=S1, mec=SURFACE, mew=1.5, zorder=4)
        # ...in whichever top corner the dot is not near, so the two never overlap.
        left = (gx - int(gx)) > 0.5
        ax.text(int(gx) + (0.07 if left else 0.93), int(gy) + 0.055,
                r["clip"].replace("clip_", ""), fontsize=6.6, color=INK2,
                ha="left" if left else "right", va="top", zorder=5)

    ax.set_xticks(np.arange(G.COLS) + 0.5, [str(i + 1) for i in range(G.COLS)],
                  fontsize=8)
    ax.set_yticks(np.arange(G.ROWS) + 0.5, [chr(65 + i) for i in range(G.ROWS)],
                  fontsize=9)
    ax.set_xlabel("grid column", fontsize=9, color=INK2, labelpad=6)
    ax.set_ylabel("grid row", fontsize=9, color=INK2, labelpad=6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    _marginal(axc, grid.sum(axis=0), len(rows) / G.COLS, "column", horizontal=False)
    _marginal(axr, grid.sum(axis=1), len(rows) / G.ROWS, "row", horizontal=True)

    fig.text(0.055, 0.955, "Where the ball ends", fontsize=16, color=INK, weight="bold")
    fig.text(0.055, 0.905,
             f"GEN3 pilot — final-frame ball cell for all {len(rows)} clips, on the "
             f"{G.COLS}x{G.ROWS} answer grid. Every clip ends in a different cell "
             f"({int((grid > 0).sum())} of {G.ROWS * G.COLS}); dots are the measured "
             f"pixel, dashed lines a uniform spread.",
             fontsize=9, color=INK2)
    out = REVIEW / "ball_final_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(HERE)}")


def _cmap(plt, top):
    from matplotlib.colors import ListedColormap
    return ListedColormap(SEQ[1:2] if top <= 1 else SEQ[1:1 + max(top, 2)])


def _marginal(ax, vals, expect, what, horizontal):
    n = len(vals)
    pos = np.arange(n) + 0.5
    if horizontal:
        ax.barh(pos, vals, height=0.62, color=SEQ[2], edgecolor=SURFACE, lw=1.2)
        ax.axvline(expect, color=MUTED, lw=1, ls=(0, (3, 2)))
        for p, v in zip(pos, vals):
            ax.text(v + 0.30, p, str(v), va="center", fontsize=7.5, color=INK2)
        ax.set_xlim(0, max(vals.max(), expect) * 1.55)
        ax.tick_params(labelleft=False, labelbottom=False, length=0)
    else:
        ax.bar(pos, vals, width=0.62, color=SEQ[2], edgecolor=SURFACE, lw=1.2)
        ax.axhline(expect, color=MUTED, lw=1, ls=(0, (3, 2)))
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.1, str(v), ha="center", fontsize=7.5, color=INK2)
        ax.set_ylim(0, max(vals.max(), expect) * 1.42)
        ax.tick_params(labelleft=False, labelbottom=False, length=0)
    ax.text(0.0, 1.0, f"clips per {what}", transform=ax.transAxes, fontsize=8,
            color=MUTED, va="bottom", ha="left")
    for s in ax.spines.values():
        s.set_visible(False)


def cmd_counts(rows):
    """Players in frame at the first frame against the last, per clip.

    A dumbbell rather than paired bars: the quantity that matters is the CHANGE, and a
    dumbbell puts the change on the page as the length of the connector. The pale bar
    behind each row is the full range the count took DURING the clip, which is what says
    whether a flat start-to-end pair means "nothing moved" or "it went out and came back".
    """
    if any(r["count_start"] is None for r in rows):
        sys.exit("counts chart needs the trace probe: run `python3 gen3.py trace`")
    plt, fig = _fig(10, 9)
    ax = fig.add_axes([0.20, 0.065, 0.755, 0.80])

    rows = sorted(rows, key=lambda r: (r["count_end"], r["count_start"]))
    y = np.arange(len(rows))[::-1]
    for i, r in zip(y, rows):
        ax.plot([r["count_min"], r["count_max"]], [i, i], color=SEQ[0], lw=7,
                solid_capstyle="round", zorder=1)
        ax.plot([r["count_start"], r["count_end"]], [i, i], color=AXIS, lw=2,
                solid_capstyle="round", zorder=2)
        ax.plot(r["count_start"], i, "o", ms=9, mfc=S1, mec=SURFACE, mew=1.6, zorder=3)
        ax.plot(r["count_end"], i, "o", ms=9, mfc=S2, mec=SURFACE, mew=1.6, zorder=3)
        lo, hi = sorted((r["count_start"], r["count_end"]))
        ax.text(lo - 0.42, i, str(lo), va="center", ha="right", fontsize=8, color=INK2)
        ax.text(hi + 0.42, i, str(hi), va="center", ha="left", fontsize=8, color=INK2)

    ax.set_yticks(y, [f"{r['clip'].replace('clip_', 'clip ')}   {KIND_NAME[r['kind']]}"
                      for r in rows], fontsize=8.5)
    for t, r in zip(ax.get_yticklabels(), rows):
        t.set_color(INK2)
    ax.set_xlabel("people in frame  (all 22 players are in play; officials are not "
                  "rendered)", fontsize=9, color=INK2, labelpad=8)
    lo = min(r["count_min"] for r in rows) - 1.6
    hi = max(r["count_max"] for r in rows) + 1.6
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc=S1, mec=SURFACE, mew=1.6,
                   label="first frame"),
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc=S2, mec=SURFACE, mew=1.6,
                   label="final frame  (the published count)"),
        plt.Line2D([], [], color=SEQ[0], lw=7, solid_capstyle="round",
                   label="range during the clip"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5,
                    handletextpad=0.6, borderaxespad=0.8, labelcolor=INK2)
    leg.set_zorder(6)

    fig.text(0.055, 0.955, "People in frame: first frame to last",
             fontsize=16, color=INK, weight="bold")
    fig.text(0.055, 0.906,
             "GEN3 pilot — exact counts from the 23-pass solo probe at each clip's own "
             "camera offset. Sorted by the final count, which is the batch's level.",
             fontsize=9, color=INK2)
    out = REVIEW / "player_counts.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(HERE)}")


if __name__ == "__main__":
    REVIEW.mkdir(exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    data = classify(load())
    if what in ("all", "classify"):
        cmd_classify(data)
    if what in ("all", "heatmap"):
        cmd_heatmap(data)
    if what in ("all", "counts"):
        cmd_counts(data)
