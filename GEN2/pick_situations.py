"""pick_situations.py — turn the cached sweep into 20 clean 5 s windows per situation.

Reads _cache/sweep/*.npz (see sweep_events.py) and writes _cache/picks.json. No engine
work happens here, so the scoring can be re-tuned and re-run in seconds.

THE THREE SITUATIONS, AND WHAT MAKES ONE "REAL"
  header       A ball above chest height that CHANGES DIRECTION next to a player. The
               direction change is what separates a header from a ball merely bouncing
               past someone, and 2+ bodies under it is what makes it a contested aerial
               duel rather than an uncontested nod.
  corner       Law 17, awarded by the engine's own referee because a defender put the ball
               behind — never staged by dropping a ball near the flag. After the award the
               engine re-spots the ball on the arc and PrepareSetPiece walks the box into
               shape, which takes ~2 s of nobody doing anything. The clip therefore opens
               on the DELIVERY (first real ball movement after the award), not the award,
               so all 5 s are live football.
  gk_throw     Law 12 distribution BY HAND. The engine tell is unambiguous: while the
               keeper holds the ball it parks at ~1.35 m (his hands, the holdball retain
               state) and the release then plays the overarm throw animation. A keeper
               with the ball at ~0.11 m is dribbling and about to PUNT it — height is what
               separates the two, which is why hold height is a hard filter, not a score.

EVERY WINDOW IS EXACTLY 50 FRAMES (5.0 s) and is placed so the action sits ~15-20 frames
in: enough lead-in to read the build-up, enough after to see the outcome.

NO GOALS, NO RESTARTS — a goal teleports the ball to the centre spot for the kickoff,
which splits the clip into two unrelated passages of play. gen2_lib.continuous() rejects
those, and it is checked again after rendering against the real ball track.

DIVERSITY: at most one clip per match, so 20 headers are 20 different matches rather than
20 views of one busy game.

Run:  python3 pick_situations.py [n_per_kind]
"""
import json
import sys
from collections import Counter

import numpy as np

from gen2_lib import (CACHE, CLIP_FRAMES, GM_CORNER, GM_FREEKICK, GM_GOALKICK,
                      GM_KICKOFF, GM_PENALTY, GM_THROWIN, SWEEP, continuous)

PICKS = CACHE / "picks.json"
N_PER_KIND = 20
# Keep one shape from supplying the whole set. Set at 8 rather than 6 because corners
# concentrate hard in the flank/box shapes — a tighter cap starves the corner set before
# it reaches 20, since only about four of the twelve shapes produce corners at all.
MAX_PER_SHAPE = 8

# Where the action sits inside the 50-frame window.
# Corners get a short lead deliberately: the window must open AFTER the award (see
# find_corners), and the ball only sits on the arc for a moment before it is struck, so
# a long lead would just clamp back to the award frame and waste seconds on a static ball.
LEAD = {"header": 20, "corner": 4, "gk_throw": 15}

SETPIECE_MODES = (GM_KICKOFF, GM_GOALKICK, GM_FREEKICK, GM_CORNER, GM_THROWIN, GM_PENALTY)


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def window_for(r, kind, n_frames):
    """The 50-frame window placing the anchor LEAD frames in, clamped into the log.

    `min_start` is a hard floor a detector can impose. Corners need one: the referee's
    award RE-SPOTS the ball onto the corner arc, and a window opening before the award
    shows the ball jump there. That jump is small when the ball went out near the flag,
    so it slips under continuous()'s teleport threshold and produces a clip whose red
    start-circle marks a position the ball instantly leaves. Measured on flankR_s001:
    award at f816, delivery at f819, and a LEAD of 15 opened the window at f806 — ten
    frames of dead ball and a respot before the corner even exists.
    """
    start = max(r["anchor"] - LEAD[kind], r.get("min_start", 0))
    if start < 0 or start + CLIP_FRAMES > n_frames:
        return None
    return start, start + CLIP_FRAMES


def no_setpiece_inside(log, start, end):
    """No NEW set piece is awarded inside the window.

    A restart mid-clip re-spots the ball and splits the clip into two unrelated passages
    of play. Transitions before `start` are fine — that is exactly how a corner clip is
    built — so only transitions strictly inside the window are rejected.
    """
    gm = log["game_mode"][start:end]
    return not any(gm[i] in SETPIECE_MODES and gm[i] != gm[i - 1]
                   for i in range(1, len(gm)))


def near_count(log, t, radius=0.06):
    pl = np.vstack([log["left"][t], log["right"][t]])
    return int((np.linalg.norm(pl - log["ball"][t][:2], axis=1) < radius).sum())


# ── detectors ──────────────────────────────────────────────────────────────────
def find_headers(log, z_min=1.35, z_max=2.9, turn_max=0.35, near=0.035):
    """High ball + direction change + a body underneath.

    turn_max is the cosine of the incoming/outgoing direction, so a genuine redirection
    rather than a bounce-through; `near` requires a player within ~3.5% of pitch length.

    Contest count is SCORED, not filtered. GEN1 required 2+ bodies under the ball, which
    on the GEN2 match shapes rejects essentially everything: measured over 31 matches,
    21 headers clear the physical tests but only 1 is contested by more than one player.
    Those shapes lean on difficulty mismatches, which spreads play out, so aerial duels
    are mostly one attacker rising alone. An uncontested header is still a header, so the
    gate is >= 1 and the score prefers the contested ones — with ~490 headers across the
    full sweep, any genuinely contested duels rise to the top on their own.
    """
    ball = log["ball"]
    out = []
    for i in range(2, len(ball) - 2):
        z = float(ball[i][2])
        if not (z_min <= z <= z_max):
            continue
        v_in = ball[i][:2] - ball[i - 2][:2]
        v_out = ball[i + 2][:2] - ball[i][:2]
        n_in, n_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
        if n_in < 0.02 or n_out < 0.02:
            continue
        cos = float(np.dot(v_in, v_out) / (n_in * n_out))
        if cos > turn_max:
            continue
        pl = np.vstack([log["left"][i], log["right"][i]])
        if float(np.min(np.linalg.norm(pl - ball[i][:2], axis=1))) > near:
            continue
        contested = near_count(log, i)
        if contested < 1:
            continue
        apex_in = float(ball[max(0, i - 18):i + 1, 2].max())
        out.append({"anchor": i, "z": round(z, 2), "contested_by": contested,
                    "apex_in": round(apex_in, 2), "turn_cos": round(cos, 2),
                    "score": contested * 10 + apex_in})
    return out


def find_corners(log, quiet=0.012):
    """Law 17 awards, anchored on the DELIVERY rather than the award.

    After the award the ball is re-spotted and sits still while PrepareSetPiece arranges
    the box; the delivery is the first frame the ball actually moves again. Anchoring
    there is what keeps all 5 s of the clip live instead of spending 2 s on a static shot
    of a ball on the arc.
    """
    gm = log["game_mode"]
    ball = log["ball"]
    awards = [i for i in range(1, len(gm)) if gm[i] == GM_CORNER and gm[i - 1] != GM_CORNER]
    out = []
    for f in awards:
        speed = np.linalg.norm(np.diff(ball[f:f + 90, :2], axis=0), axis=1)
        moving = np.flatnonzero(speed > quiet)
        if len(moving) == 0:
            continue
        delivery = f + int(moving[0])
        end = min(len(ball) - 1, delivery + 45)
        apex = float(ball[delivery:end, 2].max())
        # the contest is where the delivery drops back into the box
        drop = [i for i in range(delivery + 5, end)
                if ball[i][2] < 1.5 and abs(ball[i][0]) > 0.82]
        contested = max((near_count(log, i) for i in drop[:6]), default=0)
        out.append({"anchor": delivery, "award": f, "min_start": f + 1,
                    "apex": round(apex, 2), "contested_by": contested,
                    "score": contested * 10 + apex})
    return out


def find_gk_throws(log, hold_min=4, z_lo=1.15, z_hi=1.75, travel_min=0.20):
    """Keeper holds the ball at hand height, then releases it a long way.

    Anchored on the RELEASE: that is the frame the throw animation fires, and it is the
    moment the clip is about.
    """
    ball, ot, op = log["ball"], log["owned_team"], log["owned_player"]
    out = []
    i, n = 0, len(ball)
    while i < n:
        if op[i] == 0 and ot[i] in (0, 1):
            team = ot[i]
            j = i
            while j < n and op[j] == 0 and ot[j] == team:
                j += 1
            hand = [k for k in range(i, j) if z_lo <= ball[k][2] <= z_hi]
            if len(hand) >= hold_min and j < n:
                rel = j - 1
                end = min(n - 1, rel + 30)
                apex = float(ball[rel:end, 2].max())
                travel = float(np.linalg.norm(ball[end][:2] - ball[rel][:2]))
                if travel >= travel_min:
                    out.append({"anchor": rel, "catch": int(i),
                                "hand_frames": len(hand), "apex": round(apex, 2),
                                "travel": round(travel, 3),
                                "score": travel * 100 + len(hand)})
            i = j
        else:
            i += 1
    return out


DETECTORS = {"header": find_headers, "corner": find_corners, "gk_throw": find_gk_throws}


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else N_PER_KIND
    files = sorted(SWEEP.glob("*.npz"))
    if not files:
        sys.exit(f"no sweep cache in {SWEEP} — run sweep_events.py first")

    cands = {k: [] for k in DETECTORS}
    for path in files:
        match = path.stem                      # e.g. flankL_s002
        shape, seed = match.rsplit("_s", 1)
        log = load(path)
        n = len(log["ball"])
        for kind, detect in DETECTORS.items():
            for r in detect(log):
                w = window_for(r, kind, n)
                if w is None:
                    continue
                start, end = w
                if not continuous(log["ball"][start:end]):
                    continue
                if not no_setpiece_inside(log, start, end):
                    continue
                cands[kind].append({"match": match, "shape": shape, "seed": int(seed),
                                    "start": start, "end": end, "kind": kind, **r})

    picks = {}
    for kind, rows in cands.items():
        rows.sort(key=lambda d: -d["score"])
        chosen, per_match, per_shape = [], Counter(), Counter()
        for r in rows:
            if len(chosen) >= want:
                break
            # one clip per match, and no single shape may dominate the set
            if per_match[r["match"]] >= 1 or per_shape[r["shape"]] >= MAX_PER_SHAPE:
                continue
            per_match[r["match"]] += 1
            per_shape[r["shape"]] += 1
            chosen.append(r)
        picks[kind] = chosen
        shapes = Counter(r["shape"] for r in chosen)
        print(f"{kind:9s} {len(chosen):2d}/{want} chosen from {len(rows):5d} candidates "
              f"in {len(files)} matches   shapes={dict(shapes)}")

    CACHE.mkdir(parents=True, exist_ok=True)
    PICKS.write_text(json.dumps(picks, indent=2, default=float))
    print(f"\n-> {PICKS}")
    return picks


if __name__ == "__main__":
    main()
