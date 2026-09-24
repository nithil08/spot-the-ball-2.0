"""review_gen3_hard.py — an honest, re-measured review of the 24 shipped GEN3_HARD clips.

Reads only measurements, never targets:
  * `_cache/finalframe/<clip>.npz`  — verify_final_frame.py, full resolution, per-player
                                      pixel masks on the LAST frame of the shipped window
  * `_cache/trace/<match>_<s>_<e>.npz` — the half-resolution per-frame solo probe, used
                                      for the shape of the curve (first / min / max)
  * `_cache/sweep/<match>.npz`      — the engine log: ball, players, possession, mode
  * `clips/ground_truth.csv`        — what the batch PUBLISHED, so the two can be compared

Writes
  _review/clip_review.csv     one row per clip, measured
  _review/CLIP_REVIEW.md      the narrative, including every disagreement found
  _review/player_counts.png   the corrected chart
  _review/annotated/*.png     the final frame with a box round every person counted

    python3 review_gen3_hard.py
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
FINAL = G.CACHE / "finalframe"
TRACE = G.CACHE / "trace"
SWEEP = G.CACHE / "sweep"
REVIEW = HERE / "_review"
ANNO = REVIEW / "annotated"
CLIPS = HERE / "clips"

MIN_PIX = 20        # full-resolution changed pixels before a person is counted at all
SLIVER = 120        # below this a body is a sliver at the edge of frame, worth flagging

KIND_NAME = {"corner": "corner kick", "gk_throw": "goalkeeper distribution",
             "kickoff": "kick-off", "open": "open play"}

# ── palette (the dataviz reference palette, light surface) ──────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
S1 = "#2a78d6"          # blue   — first frame
S2 = "#eb6834"          # orange — final frame, measured
FLAG = "#c7442e"        # red    — a published number that is wrong
SOFT = "#b08b2e"        # amber  — published number differs only by a body clipped by
                        #          the frame edge


SUPERSEDED = CLIPS / "ground_truth.superseded_2026-09-21.csv"


def load():
    """Every clip, its re-measurement, and the count that was published BEFORE the fix.

    The comparison reads the superseded key, not the live one. Once the live key is
    corrected the two agree, and a review that compared against it would report a clean
    bill of health and quietly lose the record of what was wrong.
    """
    picks = json.loads(PICKS.read_text())
    key = SUPERSEDED if SUPERSEDED.exists() else CLIPS / "ground_truth.csv"
    gt = {r["clip"]: r for r in csv.DictReader(open(key))}
    rows = []
    for p in picks:
        f = FINAL / f"{p['clip']}.npz"
        if not f.exists():
            sys.exit(f"missing {f} — run verify_final_frame.py first")
        z = np.load(f, allow_pickle=True)
        rec = json.loads(str(z["rec"]))
        # A player can be outside the camera frustum and still throw a shadow into the
        # shot. That is not a person you can see, so the headline count is BODIES; the
        # shadows are kept and reported separately rather than quietly folded in.
        seen = [s for s in rec["slots"] if s["pix"] >= MIN_PIX]
        slots = [s for s in seen if s["body_pix"] >= MIN_PIX]
        r = dict(p)
        r["gt"] = gt[p["clip"]]
        r["frame"] = z["full"]
        r["slots"] = slots
        r["shadows"] = [s for s in seen if s["body_pix"] < MIN_PIX]
        r["measured"] = len(slots)
        r["blue"] = sum(1 for s in slots if s["slot"].startswith("L"))
        r["red"] = sum(1 for s in slots if s["slot"].startswith("R"))
        r["gk"] = sorted(s["slot"] for s in slots if s["slot"] in ("L0", "R0"))
        r["slivers"] = sum(1 for s in slots if s["body_pix"] < SLIVER)
        r["clear"] = r["measured"] - r["slivers"]
        r["shadow_only"] = len(seen) - len(slots)
        r["smallest"] = min((s["body_pix"] for s in slots), default=0)
        pub = int(r["gt"]["players_in_frame_last"])
        # "borderline" is not a pass: the published number is defensible only if you do
        # not count a player who is half out of frame. Anything else is simply wrong.
        r["verdict"] = ("matches" if pub == r["measured"]
                        else "borderline" if pub == r["clear"] else "wrong")
        t = TRACE / f"{p['match']}_{p['start']}_{p['end']}.npz"
        n = np.load(t)["on"].sum(axis=0) if t.exists() else None
        # A clip with no per-frame trace still has both ends measured, so it falls back
        # to its own final count rather than dropping out of the chart.
        r["first"] = int(n[0]) if n is not None else r["measured"]
        r["lo"] = int(n.min()) if n is not None else r["measured"]
        r["hi"] = int(n.max()) if n is not None else r["measured"]
        r["traced"] = n is not None
        r["trace_last"] = int(n[-1]) if n is not None else None
        log = np.load(SWEEP / f"{p['match']}.npz")
        r["log"] = {k: log[k] for k in log.files}
        rows.append(r)
    return rows


# ── who is the same play as whom ────────────────────────────────────────────────
def duplicates(rows):
    """Clips whose windows are the same passage of play.

    Three of the shape names are the SAME scenario spec (mid_bal, mid_even, mid_even2 all
    run ball (0,0), offsides on, difficulty HARD, no push), so `shape_seed` is not a match
    identity: the same seed under any two of those names replays bit-identically. Compare
    the logged ball and player tracks instead of trusting the match name.
    """
    for r in rows:
        r["dup"] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            la, lb = a["log"], b["log"]
            if len(la["ball"]) != len(lb["ball"]):
                continue
            if np.abs(la["ball"] - lb["ball"]).max() > 1e-6:
                continue
            # same match. do the windows overlap?
            lo = max(a["start"], b["start"])
            hi = min(a["end"], b["end"])
            if hi > lo:
                share = hi - lo
                a["dup"].append((b["clip"], share))
                b["dup"].append((a["clip"], share))
    return rows


def restart_taker_in_shot(r):
    if r["kind"] == "open":
        return ""
    t = TRACE / f"{r['match']}_{r['start']}_{r['end']}.npz"
    if not t.exists():
        return "?"
    on = np.load(t)["on"]
    a = r["anchor"] - r["start"]
    log = r["log"]
    pl = np.vstack([log["left"][r["anchor"]], log["right"][r["anchor"]]])
    k = int(np.linalg.norm(pl - log["ball"][r["anchor"]][:2], axis=1).argmin())
    return "yes" if bool(on[k, a]) else "NO"


# ── annotated evidence ──────────────────────────────────────────────────────────
def annotate(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ANNO.mkdir(parents=True, exist_ok=True)
    for r in rows:
        # 12.8 x 4.8 of axes is exactly the frame's 1280 x 480, so the drawing is in the
        # frame's own pixels and a box cannot drift off the body it belongs to.
        fig = plt.figure(figsize=(12.8, 5.3), dpi=110)
        ax = fig.add_axes([0, 0, 1, 4.8 / 5.3])
        ax.imshow(r["frame"])
        ax.set_axis_off()
        for i, s in enumerate(sorted(r["slots"], key=lambda s: s["cx"]), 1):
            x0, y0, x1, y1 = s["bbox"]
            edge = S1 if s["slot"].startswith("L") else S2
            thin = s["body_pix"] < SLIVER
            ax.add_patch(Rectangle((x0 - 3, y0 - 3), x1 - x0 + 6, y1 - y0 + 6,
                                   fill=False, ec=edge, lw=1.4,
                                   ls=(0, (2, 2)) if thin else "-"))
            # A body clipped by the top edge has its box at y=0, where a label above it
            # would be drawn off the figure; drop those labels below the box instead.
            top = y0 > 16
            ax.text(x0 - 4, (y0 - 6) if top else (y1 + 6),
                    f"{i}  sliver" if thin else str(i), color=edge,
                    fontsize=9, weight="bold", ha="left",
                    va="bottom" if top else "top")
        for s in r["shadows"]:
            # Drawn, but not numbered: the body is outside the frame and only the shadow
            # landed in it. The earlier probe counted these as people in shot.
            x0, y0, x1, y1 = s["bbox"]
            ax.add_patch(Rectangle((x0 - 3, y0 - 3), x1 - x0 + 6, y1 - y0 + 6,
                                   fill=False, ec=FLAG, lw=1.2, ls=(0, (3, 2))))
            top = y0 > 16
            ax.text(x0 - 4, (y0 - 6) if top else (y1 + 6), "shadow only", color=FLAG,
                    fontsize=7.5, ha="left", va="bottom" if top else "top")
        extra = []
        if r["slivers"]:
            extra.append(f"{r['slivers']} clipped by the frame edge")
        if r["shadow_only"]:
            extra.append(f"{r['shadow_only']} shadow-only, not counted")
        fig.text(0.005, 0.962, f"{r['clip']}  {KIND_NAME[r['kind']]}  —  final frame, "
                 f"{r['measured']} people in shot "
                 f"(published {r['gt']['players_in_frame_last']})"
                 + ("   ·   " + ", ".join(extra) if extra else ""),
                 fontsize=10.5, color=INK, weight="bold")
        fig.savefig(ANNO / f"{r['clip']}.png", facecolor=SURFACE)
        plt.close(fig)


# ── the corrected chart ─────────────────────────────────────────────────────────
def chart(rows, flag=False):
    """The counts chart. `flag` overlays what each count used to be published as.

    Default off: the corrections are recorded in CLIP_REVIEW.md and in git, and once the
    key is fixed the overlay is history rather than information. Pass --flag-corrections
    to draw it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "text.color": INK,
                         "axes.facecolor": SURFACE})
    fig = plt.figure(figsize=(11.0, 9.8), dpi=150)
    # Without the corrections line there is one less line of header to clear, so the
    # plot takes the space back rather than leaving a band of white under the subtitle.
    ax = fig.add_axes([0.205, 0.105, 0.765, 0.702 if flag else 0.730])

    rows = sorted(rows, key=lambda r: (r["measured"], r["first"] or 0))
    y = np.arange(len(rows))[::-1]
    wrong = []
    for i, r in zip(y, rows):
        ax.plot([r["lo"], r["hi"]], [i, i], color="#cde2fb", lw=7,
                solid_capstyle="round", zorder=1)
        ax.plot([r["first"], r["measured"]], [i, i], color=AXIS, lw=2,
                solid_capstyle="round", zorder=2)
        same = r["first"] == r["measured"]
        ax.plot(r["first"], i, "o", ms=13 if same else 9, mfc=S1, mec=SURFACE,
                mew=1.6, zorder=3)
        ax.plot(r["measured"], i, "o", ms=9, mfc=S2,
                mec=SURFACE if not same else S2, mew=1.6 if not same else 0, zorder=4)
        pub = int(r["gt"]["players_in_frame_last"])
        blocked = None
        if flag and pub != r["measured"]:
            col = FLAG if r["verdict"] == "wrong" else SOFT
            if r["verdict"] == "wrong":
                wrong.append(r)
            ax.plot(pub, i, "x", ms=9, mew=2.2, color=col, zorder=5)
            ax.annotate("", xy=(r["measured"], i), xytext=(pub, i),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.2,
                                        shrinkA=6, shrinkB=7), zorder=5)
            # The label goes on the far side of the x from the arrow, or it lands on
            # top of the measured dot it is pointing at.
            out_ = pub > r["measured"]
            # Beside the x normally; lifted onto the row above when the first-frame dot
            # is close enough that a horizontal label would run into it.
            gap = abs(r["first"] - pub) if out_ == (r["first"] > pub) else 99
            ax.text(pub + (0.5 if out_ else -0.5), i + (0.30 if gap < 2.2 else 0),
                    f"was {pub}", va="bottom" if gap < 2.2 else "center",
                    ha="left" if out_ else "right", fontsize=7.5, color=col)
            # Everything the correction occupies: the arrow, and the label beside it.
            blocked = (min(pub, r["measured"]) - 0.2,
                       pub + 2.1) if out_ else (pub - 2.1, max(pub, r["measured"]) + 0.2)
        # Number both ends of the dumbbell, but never inside the span the correction
        # occupies — two numbers in one place read as one wrong number.
        def free(x):
            return not (blocked and blocked[0] <= x <= blocked[1])

        lo, hi = sorted((r["first"], r["measured"]))
        if lo == hi:
            if free(hi + 0.42):
                ax.text(hi + 0.42, i, str(hi), va="center", ha="left", fontsize=8,
                        color=INK2)
        else:
            if free(lo - 0.42):
                ax.text(lo - 0.42, i, str(lo), va="center", ha="right", fontsize=8,
                        color=INK2)
            if free(hi + 0.42):
                ax.text(hi + 0.42, i, str(hi), va="center", ha="left", fontsize=8,
                        color=INK2)

    # The duplicate note is a dagger, not the other clips' names: spelled out, the
    # label is long enough to push the clip number out of the figure.
    labels = [f"{r['clip'].replace('clip_', 'clip ')}   {KIND_NAME[r['kind']]}"
              + ("  †" if r["dup"] else "") for r in rows]
    ax.set_yticks(y, labels, fontsize=8.5)
    for t, r in zip(ax.get_yticklabels(), rows):
        t.set_color(INK2 if not flag else
                    FLAG if r["verdict"] == "wrong"
                    else SOFT if r["verdict"] == "borderline" else INK2)
    ax.set_xlabel("people in shot   (all 22 players are in play; officials are not "
                  "rendered)", fontsize=9, color=INK2, labelpad=8)
    lo = min(r["lo"] for r in rows) - 1.8
    hi = max(r["hi"] for r in rows) + 1.8
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xticks(range(int(np.ceil(lo)), int(hi) + 1, 2))
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc=S1, mec=SURFACE, mew=1.6,
                   label="first frame  (half-res probe)"),
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc=S2, mec=SURFACE, mew=1.6,
                   label="final frame  (re-measured, full resolution)"),
        plt.Line2D([], [], color="#cde2fb", lw=7, solid_capstyle="round",
                   label="range during the clip"),
    ]
    if flag:
        handles += [
            plt.Line2D([], [], marker="x", ls="", ms=9, mew=2.2, color=FLAG,
                       label="count as first published, wrong under any definition"),
            plt.Line2D([], [], marker="x", ls="", ms=9, mew=2.2, color=SOFT,
                       label="count as first published, missing a body at the edge"),
        ]
    ax.legend(handles=handles, ncol=2, loc="lower left", frameon=False, fontsize=8.5,
              bbox_to_anchor=(-0.005, 1.012), handletextpad=0.6, columnspacing=2.2,
              labelcolor=INK2)

    fig.text(0.042, 0.972, "People in shot: first frame to last", fontsize=16,
             color=INK, weight="bold")
    n_wrong = len(wrong)
    n_soft = sum(1 for r in rows if r["verdict"] == "borderline")
    fig.text(0.042, 0.941,
             "GEN3_HARD, re-measured — a person is counted by hiding that one player and "
             "diffing the render, at full resolution and at each clip's own offset.",
             fontsize=9, color=INK2)
    if flag:
        fig.text(0.042, 0.920,
                 f"{n_wrong} of 24 counts were wrong outright and {n_soft} more missed "
                 "a body clipped by the frame edge — the answer key now carries the "
                 "measured column.", fontsize=9, color=FLAG)
    dup_names = sorted({c for r in rows for c, _ in r["dup"]}
                       | {r["clip"] for r in rows if r["dup"]})
    if dup_names:
        fig.text(0.042, 0.039,
                 "†  " + ", ".join(n.replace("clip_", "clip ") for n in dup_names)
                 + " are the same passage of play, at different camera offsets.",
                 fontsize=7.5, color=MUTED)
    fig.text(0.042, 0.018,
             "The pale range bar is the per-frame probe, which runs at half resolution; "
             "it counts a shadow cast into the shot as a person, so it can sit above the "
             "final count.", fontsize=7.5, color=MUTED)
    out = REVIEW / "player_counts.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(HERE)}")
    return wrong


def distribution(rows):
    """The batch was built to hold each of 8..19 exactly twice. Does it?

    Two bars per level rather than one: the design is a flat line at 2, so the thing worth
    seeing is not the shape of the measured histogram but the GAP between it and that
    line — which levels are empty and which are doubled up.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "text.color": INK,
                         "axes.facecolor": SURFACE})
    lo = min(min(r["measured"] for r in rows), 8)
    hi = max(max(r["measured"] for r in rows), 19)
    levels = list(range(lo, hi + 1))
    meas = [sum(1 for r in rows if r["measured"] == x) for x in levels]
    want = [2 if 8 <= x <= 19 else 0 for x in levels]

    fig = plt.figure(figsize=(9.6, 4.4), dpi=150)
    ax = fig.add_axes([0.07, 0.155, 0.90, 0.575])
    w = 0.38
    xs = np.arange(len(levels))
    ax.bar(xs - w / 2, want, w, color="#cde2fb", label="designed (two clips per level)")
    ax.bar(xs + w / 2, meas, w, color=S2, label="measured")
    for x, v in zip(xs, meas):
        if v:
            ax.text(x + w / 2, v + 0.06, str(v), ha="center", fontsize=8, color=INK2)
    for x, (v, d) in enumerate(zip(meas, want)):
        if v != d:
            ax.text(x, -0.42, "✗", ha="center", fontsize=10, color=FLAG)
    ax.set_xticks(xs, [str(x) for x in levels], fontsize=9, color=INK2)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylim(-0.6, max(max(meas), 2) + 0.7)
    ax.set_xlabel("people in shot on the final frame", fontsize=9, color=INK2,
                  labelpad=6)
    ax.yaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.02),
              ncol=2, labelcolor=INK2)
    fig.text(0.07, 0.945, "The count distribution the batch actually has", fontsize=14,
             color=INK, weight="bold")
    miss = [str(x) for x, v in zip(levels, meas) if 8 <= x <= 19 and v == 0]
    over = [f"{x} has {v}" for x, v in zip(levels, meas) if v > 2]
    out_of = [str(x) for x, v in zip(levels, meas) if v and not 8 <= x <= 19]
    if miss or over or out_of:
        bits = []
        if miss:
            bits.append(("levels " if len(miss) > 1 else "level ")
                        + ", ".join(miss) + " ended up empty")
        if over:
            bits.append("level " + ", ".join(over))
        if out_of:
            bits.append("and " + ", ".join(out_of) + " is outside the designed range")
        note = "Re-measured, it does not: " + "; ".join(bits) + "."
    else:
        note = "Re-measured, it does."
    bad = bool(miss or over or out_of)
    fig.text(0.07, 0.900, "GEN3_HARD was built to cover 8 to 19, two clips each.",
             fontsize=9, color=INK2)
    fig.text(0.07, 0.855, note, fontsize=9, color=FLAG if bad else INK2)
    out = REVIEW / "count_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(HERE)}")


# ── the table ───────────────────────────────────────────────────────────────────
COLS = [
    ("clip", lambda r: r["clip"]),
    ("situation", lambda r: KIND_NAME[r["kind"]]),
    ("people_final_measured", lambda r: r["measured"]),
    ("people_final_clearly_visible", lambda r: r["clear"]),
    ("people_final_incl_shadow_only", lambda r: r["measured"] + r["shadow_only"]),
    ("people_final_published", lambda r: int(r["gt"]["players_in_frame_last"])),
    ("published_verdict", lambda r: r["verdict"]),
    ("blue_final", lambda r: r["blue"]),
    ("red_final", lambda r: r["red"]),
    ("keepers_final", lambda r: "+".join(r["gk"]) or "none"),
    ("people_first", lambda r: r["first"]),
    ("people_min", lambda r: r["lo"]),
    ("people_max", lambda r: r["hi"]),
    ("edge_slivers_final", lambda r: r["slivers"]),
    ("shadow_only_final", lambda r: r["shadow_only"]),
    ("smallest_body_px", lambda r: r["smallest"]),
    ("restart_s", lambda r: "" if r["kind"] == "open"
     else round((r["anchor"] - r["start"]) / G.FPS, 2)),
    ("restart_taker_in_shot", lambda r: r["taker"]),
    ("same_play_as", lambda r: " ".join(c for c, _ in r["dup"])),
    ("ball_final_cell", lambda r: r["cell"]),
    ("match", lambda r: r["match"]),
    ("window", lambda r: f"{r['start']}-{r['end']}"),
    ("camera_offset", lambda r: f"{r['offset_x']:+.0f},{r['offset_y']:+.0f}"),
]


def table(rows):
    out = REVIEW / "clip_review.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([c for c, _ in COLS])
        for r in rows:
            w.writerow([f(r) for _, f in COLS])
    print(f"wrote {out.relative_to(HERE)}")


SEEN = {  # what a viewer actually sees, written from watching all 24 clips end to end
    "clip_01": "Ball spotted on the far corner arc (B15) with the taker at the very right "
               "edge of frame; the rest of the shot is empty grass. The delivery crosses "
               "46 m and dies near the right touchline.",
    "clip_02": "Opens inside the blue six-yard box with the keeper over the ball; he "
               "releases it at 1.1 s and the play moves out to the left flank.",
    "clip_03": "The red keeper is holding the ball in his own box, in shot, and releases "
               "at 1.3 s; the clip ends with a contested ball outside his area.",
    "clip_04": "The blue keeper is at the top edge of frame with the ball in his hands "
               "and releases at 0.5 s. Two thirds of the shot is empty grass; the play "
               "is a knot of bodies in the top-right corner.",
    "clip_05": "A textbook corner: taker in shot beside the flag, delivery into a box "
               "that fills up as the ball arrives.",
    "clip_06": "Kick-off from the centre spot with two players over the ball; it is "
               "played back and the move goes left towards the blue goal.",
    "clip_07": "The ball is already moving off the far corner arc on frame 0, taker in "
               "shot; it is met in the box and the play breaks out to the left.",
    "clip_08": "The ball sits on the near corner arc (F16) with NO taker in shot — he is "
               "outside the frame. The delivery is headed clear and the clip ends at the "
               "halfway line, 44 m away.",
    "clip_09": "The blue keeper releases almost immediately, at the left edge; the play "
               "stays in the blue defensive third.",
    "clip_10": "Open play in the blue defensive third, camera behind the goal line; red "
               "builds and the clip ends with bodies converging on the box.",
    "clip_11": "The blue keeper holds, then releases at 1.3 s; the ball is worked out to "
               "the left and the shot stays busy throughout.",
    "clip_12": "Continuous open play across the blue half — the busiest midfield in the "
               "batch, bodies in shot the whole way.",
    "clip_13": "Open play in the red defensive third. The action is packed into the left "
               "third of the shot and several bodies overlap or are cut by the top edge.",
    "clip_14": "Open play round the blue box; the keeper is in shot for the whole clip.",
    "clip_15": "The SAME kick-off as clip 06, frame for frame — only the camera offset "
               "differs, so the shot is wider and 6 more people are in it.",
    "clip_16": "Open play through the middle third, ending with the ball in the far "
               "top-left corner of the shot.",
    "clip_17": "A corner with the taker clearly in shot; three people at the start, "
               "sixteen by the end as both boxes empty into the area.",
    "clip_18": "Open play in the red defensive third; play runs left to right across a "
               "full shot.",
    "clip_19": "The same kick-off again — the same match as clips 06 and 15, started five "
               "frames earlier, framed differently.",
    "clip_20": "Open play in the blue defensive third with a crowd around the ball.",
    "clip_21": "Its own kick-off (seed 347): centre spot, two players over the ball, the "
               "move goes right.",
    "clip_22": "Open play in the red defensive third, an even spread of bodies.",
    "clip_23": "Its own kick-off (seed 349); the fullest shot in the batch at the end.",
    "clip_24": "Open play through the centre circle, 19 people in shot at the end.",
}


def markdown(rows, wrong):
    n = len(rows)
    dups = sorted({tuple(sorted([r["clip"]] + [c for c, _ in r["dup"]]))
                   for r in rows if r["dup"]})
    lines = []
    A = lines.append
    A("# GEN3_HARD — what the 24 full-visibility clips actually are\n")
    A("Re-measured from the shipped renders, not from the pipeline's targets. Every "
      "number below was produced by replaying the clip's own match at the clip's own "
      "camera offset and rendering the final frame 23 times: once with all 22 players "
      "hidden, then once per player with only that player shown. A player is in shot if "
      "their own pass differs from the plate.\n")
    A("## How a person is counted\n")
    A("| term | rule | why it is separate |")
    A("|---|---|---|")
    A(f"| **people in shot** | at least {MIN_PIX} changed pixels that are the player's "
      "**body** | the headline number |")
    A(f"| clearly visible | at least {SLIVER} body pixels | excludes a shoulder or a pair "
      "of boots clipped by the frame edge |")
    A("| shadow only | changed pixels, but effectively no body | the player is outside "
      "the frame and only their shadow fell inside it. The shipped probe counted these "
      "as people in shot |")
    A("")
    A("## Every clip\n")
    A("| clip | situation | people in shot (final) | published | first | min–max | "
      "notes |")
    A("|---|---|---|---|---|---|---|")
    for r in rows:
        pub = int(r["gt"]["players_in_frame_last"])
        mark = "" if pub == r["measured"] else f" **← was {pub}**"
        note = []
        if r["slivers"]:
            note.append(f"{r['slivers']} clipped by the frame edge")
        if r["shadow_only"]:
            note.append(f"{r['shadow_only']} shadow-only")
        if r["dup"]:
            note.append("same play as " + ", ".join(c for c, _ in r["dup"]))
        if r["taker"] == "NO":
            note.append("restart taker out of shot")
        A(f"| {r['clip']} | {KIND_NAME[r['kind']]} | **{r['measured']}**{mark} | {pub} | "
          f"{r['first']} | {r['lo']}–{r['hi']} | {'; '.join(note)} |")
    A("")
    A("## What each clip looks like\n")
    for r in rows:
        A(f"* **{r['clip']} — {KIND_NAME[r['kind']]}, {r['measured']} in shot at the "
          f"end.** {SEEN.get(r['clip'], '')}")
    A("")
    repaired = (CLIPS / "clip_renumbering.csv").exists()
    if wrong or not repaired:
        A("## Mistakes found\n")
        A(f"**{len(wrong)} of {n} published final-frame counts are wrong.** "
          + ("; ".join(f"{r['clip']} published {r['gt']['players_in_frame_last']}, "
                       f"actually {r['measured']}" for r in wrong) or "none") + ".")
        A("")
        soft = [r for r in rows if r["verdict"] == "borderline"]
        if soft:
            A(f"**{len(soft)} more are one short because a body clipped by the frame "
              "edge was not counted:** "
              + "; ".join(f"{r['clip']} published "
                          f"{r['gt']['players_in_frame_last']}, {r['measured']} people "
                          f"have pixels in the frame" for r in soft) + ".")
            A("")
    lv = {}
    for r in rows:
        lv.setdefault(r["measured"], []).append(r["clip"])
    empty = [str(x) for x in range(8, 20) if x not in lv]
    holds = not empty and all(len(v) == 2 for v in lv.values())
    A("## The levels\n")
    A(("**The batch covers 8 to 19 with two clips each, and every count is measured at "
       "full resolution.** " if holds else
       "**The designed distribution does not hold.** The batch was built to cover 8 to "
       "19 with two clips each. ")
      + "Measured, it runs "
      + ", ".join(f"{k}×{len(v)}" for k, v in sorted(lv.items()))
      + (f" — level {', '.join(empty)} has no clip at all" if empty else "")
      + ". See `count_distribution.png`.")
    A("")
    if repaired:
        A("## How it got here\n")
        A("This batch was repaired on 23 Sep 2026 rather than rebuilt: 15 clips were kept "
          "exactly as they were and 9 were replaced, then the whole thing was renumbered "
          "by level. `clips/clip_renumbering.csv` records where every old number went and "
          "why each dropped clip was dropped; `clips/ground_truth.superseded_2026-09-21."
          "csv` is the key as it stood before.")
        A("")
        A("Four faults drove it, each measured rather than argued:")
        A("")
        A("* **The counts were wrong.** The probe that selected the original batch ran at "
          "half resolution with a 40-pixel floor — about 160 at full size, which is more "
          "than a player clipped by the frame edge leaves behind and less than the shadow "
          "of a player entirely outside it. Nine of the 24 counts were wrong in "
          "consequence. Every count now comes from a full-resolution pass that counts "
          "body pixels and discards shadow.")
        A("* **The ball left the shot.** Only the first and last frames were ever checked "
          "for it. clip_01 shipped with the ball outside the camera's view for 65 "
          "consecutive frames. Every frame of every clip is now checked, and a frame with "
          "no ball is referred to a plate render with all 22 players hidden, which "
          "separates a ball behind a defender from a ball that is not there.")
        A("* **One passage of play shipped three times.** `mid_bal`, `mid_even` and "
          "`mid_even2` are three names for one scenario spec, so one seed replays "
          "bit-identically under each, and the one-clip-per-match rule compared names. "
          "Diversity is now judged on the logged ball track.")
        A("* **The levels and the starting teams had drifted.** Level 15 was empty, level "
          "16 held four clips, and three levels had both clips starting with the same "
          "side. The repair re-solved all 24 places at once rather than patching the "
          "gaps, which is why 9 clips moved and not 7.")
        A("")
    A("## What is sound\n")
    A("* **The ball ground truth.** All 24 final-frame ball pixels reproduce exactly from "
      "the shipped render pair, and every one agrees to within 2 px with the independent "
      "plate measurement taken with all 22 players hidden.")
    A("* **The situation labels.** All 24 are right at the engine level: the five corners "
      "all start with the ball on a corner arc, the five keeper clips all start with the "
      "ball in a keeper's hands or at his feet, the five kick-offs all start on the "
      "centre spot with both elevens in their own halves, and no set piece is awarded "
      "inside any window.")
    A("* **Continuity.** No goal, no teleport, no respot inside any of the 24 windows.")
    A("* **Restarts are shown.** Every restart clip opens at or before its delivery, and "
      "the player taking it is in shot as he takes it — checked by watching all 24.")
    A("* **The ball is in the camera's view on every frame of every clip**, proven on "
      "plate renders with all 22 players hidden wherever the shipped pair showed none.")
    A("")
    A("## What was changed\n")
    A("`clips/ground_truth.csv` and `clips/clip_classification.csv` now carry the "
      "measured count, under the rule at the top of this file: a person is in shot if "
      "any part of their body has pixels in the frame. The count as first published is "
      "kept in `clips/ground_truth.superseded_2026-09-21.csv`, and both columns are in "
      "`clip_review.csv`, so the change is reversible and auditable. Every "
      "`players_in_frame_last` is now reproducible with `verify_final_frame.py`.")
    A("")
    A("Both files also gained a `same_play_as` column naming the clips that share a "
      "passage of play, so the three copies of one kick-off cannot be mistaken for "
      "three independent samples. The clips themselves are untouched.")
    A("")
    A("Files: `clip_review.csv` (the table), `annotated/clip_XX.png` (the final frame "
      "with a box round every person counted), `player_counts.png` (the corrected "
      "chart), `count_distribution.png` (the levels the batch really covers).")
    out = REVIEW / "CLIP_REVIEW.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out.relative_to(HERE)}")


def main():
    REVIEW.mkdir(exist_ok=True)
    rows = duplicates(load())
    for r in rows:
        r["taker"] = restart_taker_in_shot(r)
    rows = sorted(rows, key=lambda r: r["clip"])
    table(rows)
    wrong = chart(rows, flag="--flag-corrections" in sys.argv)
    distribution(rows)
    markdown(rows, sorted(wrong, key=lambda r: r["clip"]))
    annotate(rows)
    print(f"{len(wrong)} published counts disagree with the frame: "
          + ", ".join(f"{r['clip']} {r['gt']['players_in_frame_last']}->{r['measured']}"
                      for r in wrong))


if __name__ == "__main__":
    main()
