"""gen_player_delta.py — clips that END with an exact number of players in frame,
with NOBODY hidden.

THE REWRITE, AND WHY
  The first version hit the target count by hiding players (renderScale, the trick used on
  referees). Every one of those 20 clips was rejected. Four things were wrong with them,
  and the fourth kills the whole mechanism:

    1. The ball was stranded in empty grass. Hiding chose victims by shortest dwell in
       frame, and the players moving fastest through frame are precisely the ones chasing
       the ball — so it systematically deleted the players involved in the play and kept
       distant bystanders.
    2. The frames were too empty to read as a match.
    3. Windows opened on kickoff clumps (the probe only ever covered frames 0-119, which
       is the restart), so players were bunched on the centre spot and nothing was
       contested.
    4. Hiding players is the wrong approach, full stop.

  So the count is now obtained by SELECTION, not subtraction. All 22 players are rendered
  in every frame of every clip. We search a large pool of real match windows for ones the
  camera happens to frame with exactly the target number of bodies on the final frame.
  Nothing is removed, so nothing can look removed.

HOW THE COUNT IS KNOWN WITHOUT RENDERING
  Counting by rendering costs 23 passes per match and only ever covered 120 frames of 25
  matches — far too small a pool to also demand good football. calibrate_inframe.py fits
  the camera's field of view once against those rendered labels, and the fitted model then
  gives the on-screen count for any frame of any of the 600+ cached matches for free.
  Ground truth for the published count is still the model, but the clips are chosen from
  a pool large enough that we can insist on much more than the count.

WHAT ELSE A WINDOW MUST SATISFY  (this is what fixes 1-3)
  * the ball is genuinely in play near people — at least MIN_NEAR players within
    NEAR_RADIUS of it on both the first and last frame, so it is never sitting alone
  * the ball actually travels, and is not parked
  * the visible players are near the ball rather than scattered to the edges
  * no set piece, no goal, no restart inside the window
  * nothing from the opening SKIP_START frames, which is the kickoff

Run:  python3 gen_player_delta.py all
      (phases: pick -> vis -> inv -> compose; no probe phase any more)
Out:  GEN2/04_player_delta/{full_visibility,split_1s_4s}/
"""
import json
import subprocess
import sys
from collections import Counter

import numpy as np

from gen2_lib import (CACHE, CLIP_FRAMES, BUNDLE_INV, BUNDLE_VIS, HERE, SHAPES, SWEEP,
                      compose_pair, continuous, render_window, shape_spec, write_clip)
from calibrate_inframe import MODEL, in_frame_mask

OUT = HERE / "04_player_delta"
PD = CACHE / "player_delta"
PICKS = PD / "picks_natural.json"

END_COUNTS = [6, 10, 12, 16]
PER_COUNT = 5

SKIP_START = 150       # frames of kickoff / settling to ignore at the start of a match
NEAR_RADIUS = 0.13     # "near the ball" in pitch units (~13% of the 105 m length)
MIN_NEAR = 3           # bodies that must be near the ball on the first and last frame
MIN_TRAVEL = 0.12      # the ball must cover at least this much ground across the window
STRIDE = 7             # window start stride when scanning a match

SETPIECE_MODES = (1, 2, 3, 4, 5, 6)   # anything that is not GM_NORMAL


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def near_ball(ball_xy, left, right, radius):
    pl = np.concatenate([left, right], axis=0)
    return int((np.linalg.norm(pl - ball_xy, axis=1) < radius).sum())


def score_window(log, on, s, e, want):
    """Return a quality score for [s, e] if it is usable at `want` players, else None.

    Everything here exists to answer one of the four rejections: the ball must be with
    people (1), the frame must be busy enough to read as football (2), and the play must
    be live rather than a restart or a jog (3).
    """
    if int(on[e].sum()) != want:
        return None

    ball = log["ball"][s:e + 1]
    if not continuous(ball):
        return None

    gm = log["game_mode"][s:e + 1]
    if any(m in SETPIECE_MODES for m in gm):
        return None                      # no set piece, and no restart, anywhere inside

    # 1. the ball is with people, at BOTH ends of the clip — never stranded
    near_first = near_ball(ball[0][:2], log["left"][s], log["right"][s], NEAR_RADIUS)
    near_last = near_ball(ball[-1][:2], log["left"][e], log["right"][e], NEAR_RADIUS)
    if near_first < MIN_NEAR or near_last < MIN_NEAR:
        return None

    # 2. the ball is actually played, not parked at someone's feet for five seconds
    travel = float(np.linalg.norm(ball[-1][:2] - ball[0][:2]))
    path = float(np.linalg.norm(np.diff(ball[:, :2], axis=0), axis=1).sum())
    if travel < MIN_TRAVEL:
        return None

    # 3. the visible players are gathered around the play rather than strung out. Mean
    #    distance from the ball to the on-screen players, lower is tighter.
    pl = np.concatenate([log["left"][e], log["right"][e]], axis=0)[on[e]]
    spread = float(np.linalg.norm(pl - ball[-1][:2], axis=1).mean()) if len(pl) else 9.9

    # Prefer: more bodies around the ball, more of the ball moving, tighter grouping.
    return {"near_first": near_first, "near_last": near_last,
            "travel": round(travel, 3), "path": round(path, 3),
            "spread": round(spread, 3),
            "score": near_last * 3 + near_first * 2 + path * 6 - spread * 8}


def phase_pick():
    if not MODEL.exists():
        sys.exit("no in-frame model — run: python3 calibrate_inframe.py")
    model = json.loads(MODEL.read_text())
    print(f"in-frame model: per-frame count exact {model['count_exact']*100:.1f}%, "
          f"within +-1 {model['count_within1']*100:.1f}%")

    files = sorted(SWEEP.glob("*.npz"))
    if not files:
        sys.exit(f"no sweep cache in {SWEEP} — run sweep_events.py")

    cands = {w: [] for w in END_COUNTS}
    for path in files:
        log = load(path)
        on = in_frame_mask(log["ball"], log["left"], log["right"], model)
        T = len(log["ball"])
        for s in range(SKIP_START, T - CLIP_FRAMES, STRIDE):
            e = s + CLIP_FRAMES - 1
            for want in END_COUNTS:
                r = score_window(log, on, s, e, want)
                if r is None:
                    continue
                cands[want].append({"match": path.stem, "shape": path.stem.rsplit("_s", 1)[0],
                                    "seed": int(path.stem.rsplit("_s", 1)[1]),
                                    "start": s, "end": e + 1, "end_count": want, **r})

    picks, used = [], set()
    # Hardest first: 16 needs a crowded frame and few windows offer it.
    for want in sorted(END_COUNTS, reverse=True):
        rows = sorted(cands[want], key=lambda d: -d["score"])
        got = 0
        for r in rows:
            if got >= PER_COUNT:
                break
            if r["match"] in used:
                continue            # one clip per match — 20 clips from 20 matches
            used.add(r["match"])
            picks.append(r)
            got += 1
        print(f"  end_count={want:2d}: {got}/{PER_COUNT} chosen from "
              f"{len(cands[want])} candidate windows")
    PD.mkdir(parents=True, exist_ok=True)
    PICKS.write_text(json.dumps(picks, indent=2, default=float))
    print(f"phase pick: {len(picks)} windows from {len({p['match'] for p in picks})} "
          f"distinct matches -> {PICKS}")


def _levels():
    from scenario_factory import write_scenario
    return {name: write_scenario(shape_spec(name), force=True) for name, *_ in SHAPES}


def _render(bundle, tag):
    from lib import use_bundle
    use_bundle(bundle)
    levels = _levels()
    picks = json.loads(PICKS.read_text())
    for i, p in enumerate(picks):
        # hide_slots stays empty: every one of the 22 players is rendered.
        frames, balls = render_window(levels[p["shape"]], p["seed"], p["start"], p["end"])
        np.savez_compressed(PD / f"nd{i:02d}_{tag}.npz", frames=np.array(frames))
        print(f"  [{tag}] {p['match']} end={p['end_count']} ({len(frames)} frames)",
              flush=True)


def phase_vis():
    _render(BUNDLE_VIS, "vis")


def phase_inv():
    _render(BUNDLE_INV, "inv")


def phase_compose():
    from grid import cell_of
    picks = json.loads(PICKS.read_text())
    rows = [("clip", "end_count", "players_in_frame_last", "players_in_play",
             "players_hidden", "near_ball_first", "near_ball_last", "ball_path",
             "shape", "seed", "match", "start_frame", "end_frame", "n_frames", "seconds",
             "ball_start_cell", "start_px", "start_py",
             "ball_final_cell", "final_px", "final_py")]
    counters = {}
    for i, p in enumerate(picks):
        vis = np.load(PD / f"nd{i:02d}_vis.npz")["frames"]
        inv = np.load(PD / f"nd{i:02d}_inv.npz")["frames"]
        full, split, (spx, spy), (fpx, fpy) = compose_pair(vis, inv)
        n = counters.get(p["end_count"], 0) + 1
        counters[p["end_count"]] = n
        clip = f"end{p['end_count']:02d}_{n:02d}"
        write_clip(full, OUT / "full_visibility" / f"{clip}.mov")
        write_clip(split, OUT / "split_1s_4s" / f"{clip}.mov")
        rows.append((clip, p["end_count"], p["end_count"], 22, 0,
                     p["near_first"], p["near_last"], p["path"],
                     p["shape"], p["seed"], p["match"], p["start"], p["end"],
                     len(full), round(len(full) / 10, 1),
                     cell_of(spx, spy), round(spx, 1), round(spy, 1),
                     cell_of(fpx, fpy), round(fpx, 1), round(fpy, 1)))
        print(f"  [compose] {clip}: {p['end_count']} in frame, "
              f"{p['near_last']} near the ball, ball {cell_of(spx, spy)} -> "
              f"{cell_of(fpx, fpy)}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ground_truth.csv").write_text(
        "\n".join(",".join(map(str, r)) for r in rows) + "\n")
    print(f"phase compose: {len(picks)} situations -> {len(picks) * 2} clips in {OUT}")


def phase_all():
    for mode in ("pick", "vis", "inv", "compose"):
        print(f"\n===== phase {mode} =====", flush=True)
        subprocess.run([sys.executable, __file__, mode], check=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    {"pick": phase_pick, "vis": phase_vis, "inv": phase_inv,
     "compose": phase_compose, "all": phase_all}[mode]()
