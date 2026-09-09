"""scene_graph.py — export the full game state of every GEN3 clip, frame by frame, as JSON.

WHY THIS IS ESSENTIALLY FREE
  The sweep already logged every match's complete observation — ball position, all 22
  player positions, possession, game mode and score at every step — and those logs are
  cached. A clip is a window into one of them. So the entire world-space scene graph is a
  slice and a re-labelling: no engine, no replay, seconds for the whole batch. It is also
  exact rather than inferred, which a scene graph recovered from pixels could never be.

WHAT A MODEL COULD BE ASKED, WITH THIS
  It is the symbolic counterpart of the video: the same 5 seconds as structured state
  instead of frames. That supports the comparison the benchmark is really after — can a
  model that is TOLD the game state do the task the video asks, and how much of the gap
  is perception versus reasoning. The ball is in here, so for ball-prediction work the
  caller must withhold it; `--hide-ball-after` does that at the same 1 s cut the video
  uses, leaving the ball's opening second and nothing after it.

FRAME INDEXING — the thing to get right
  Both the sweep and the render loop observe AFTER stepping, and the render loop keeps a
  frame once its loop index reaches `start`. So loop index i carries log[i], the clip's
  first rendered frame is log[start], and its last is log[end - 1]. Frame 0 of the JSON is
  frame 0 of the .mov. Getting this wrong by one is invisible in most frames and wrong
  exactly at a possession change on the cut, which is where it matters.

WHAT IS AND IS NOT IN HERE
  IN, exactly: every player's world position each frame, the ball's world position,
  possession, game mode, score, roles, teams, kits, and the derived facts that follow from
  those (speeds, who is nearest the ball, team centroids).

  NOT in here: PIXEL positions of players, and therefore which players are actually in
  shot. That is not an oversight and it cannot be derived — the camera tracks the ball and
  the repo's only calibration maps the BALL between world and pixel space, not arbitrary
  points. Per-frame per-player pixel data needs the 23-pass hide-and-diff probe extended
  from end frames to all frames, ~23 replays per clip. The end-frame player COUNT, which
  is the benchmark's ground truth, is already measured that way and is carried through
  here as clip metadata.

USAGE
  python3 scene_graph.py                       # all 24 -> clips/scene_graphs/
  python3 scene_graph.py --clips clip_09       # just one
  python3 scene_graph.py --hide-ball-after 1.0 # drop ball state past the 1 s cut
  python3 scene_graph.py --pretty              # indented, much larger
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen3_lib as G  # noqa: E402

GT = HERE / "clips" / "ground_truth.csv"
REPRO = HERE / "clips" / "reproducibility.csv"
OUT = HERE / "clips" / "scene_graphs"

# Slot -> role, from gen3_lib.eleven(). Both sides use the same 4-3-3-ish shape, and slot 0
# is the goalkeeper — which is what GFOOTBALL_HIDE_SLOTS addresses as L0..L10 / R0..R10.
ROLES = ["GK", "RM", "CF", "LB", "CB", "CB", "RB", "CM", "CM", "CM", "LM"]

# On-screen kit colours, from experiments/apply_gen3_kits.py. Stated here so a consumer of
# the JSON can tie a player to what it sees without reading the bundle build script.
KITS = {
    "A": {"outfield": "blue", "outfield_rgb": [28, 72, 224],
          "goalkeeper": "pale sky blue", "goalkeeper_rgb": [125, 195, 255]},
    "B": {"outfield": "red", "outfield_rgb": [214, 32, 32],
          "goalkeeper": "pale coral", "goalkeeper_rgb": [255, 145, 130]},
}

GAME_MODES = {0: "normal", 1: "kickoff", 2: "goalkick", 3: "freekick",
              4: "corner", 5: "throwin", 6: "penalty"}

# The pitch in gfootball world units: x spans [-1, 1] goal to goal, y spans [-0.42, 0.42]
# touchline to touchline. Team A (left) attacks +x. Included in the JSON so distances and
# directions in it can be interpreted without outside knowledge.
PITCH = {"x_range": [-1.0, 1.0], "y_range": [-0.42, 0.42],
         "team_A_attacks": "+x", "team_B_attacks": "-x",
         "note": "world units; multiply x by 52.5 and y by 34 for approximate metres"}


def _rows(path):
    with path.open() as f:
        return {r["clip"]: r for r in csv.DictReader(f)}


def _r3(v):
    return [round(float(x), 4) for x in v]


def build(clip, gt, repro, hide_ball_after=None):
    """One clip's scene graph. `hide_ball_after` is seconds; None keeps the ball throughout."""
    match, start, end = gt["match"], int(gt["start_frame"]), int(gt["end_frame"])
    log = dict(np.load(G.SWEEP / f"{match}.npz"))
    n = end - start

    cut = None if hide_ball_after is None else int(round(hide_ball_after * G.FPS))

    frames = []
    for j in range(n):
        i = start + j                      # loop index i carries log[i]; see module docstring
        ball = log["ball"][i]
        owned_t, owned_p = int(log["owned_team"][i]), int(log["owned_player"][i])

        players = []
        for team, key in (("A", "left"), ("B", "right")):
            pos = log[key][i]
            for slot in range(len(pos)):
                players.append({
                    "id": f"{team}{slot}",
                    "team": team,
                    "slot": slot,
                    "role": ROLES[slot] if slot < len(ROLES) else "CM",
                    "is_goalkeeper": slot == 0,
                    "x": round(float(pos[slot][0]), 4),
                    "y": round(float(pos[slot][1]), 4),
                })

        # Velocity by finite difference against the previous logged frame, in world units
        # per second. The first frame has no predecessor inside the clip, so it borrows the
        # step before it — that frame exists in the match even though it is not in the clip,
        # which is better than reporting a spurious zero.
        prev = max(i - 1, 0)
        for p in players:
            key = "left" if p["team"] == "A" else "right"
            q = log[key][prev][p["slot"]]
            p["vx"] = round((p["x"] - float(q[0])) * G.FPS, 4)
            p["vy"] = round((p["y"] - float(q[1])) * G.FPS, 4)

        bprev = log["ball"][prev]
        bvel = [(float(ball[k]) - float(bprev[k])) * G.FPS for k in range(3)]

        # Nearest player to the ball, over both sides. Cheap to compute and the single most
        # asked-for derived fact, so it ships rather than being left to every consumer.
        d = [((p["x"] - float(ball[0])) ** 2 + (p["y"] - float(ball[1])) ** 2) ** 0.5
             for p in players]
        k_near = int(np.argmin(d))

        fr = {
            "frame": j,
            "engine_frame": i,
            "t": round(j / G.FPS, 3),
            "game_mode": GAME_MODES.get(int(log["game_mode"][i]), str(int(log["game_mode"][i]))),
            "score": [int(x) for x in log["score"][i]],
            "possession": {
                "team": None if owned_t < 0 else ("A" if owned_t == 0 else "B"),
                "player": None if owned_t < 0 or owned_p < 0 else
                          f"{'A' if owned_t == 0 else 'B'}{owned_p}",
                "loose": owned_t < 0,
            },
            "players": players,
            "nearest_player_to_ball": {"id": players[k_near]["id"],
                                       "distance": round(float(d[k_near]), 4)},
            "team_centroid": {
                t: _r3(np.mean([[p["x"], p["y"]] for p in players if p["team"] == t], axis=0))
                for t in ("A", "B")},
        }
        if cut is None or j < cut:
            fr["ball"] = {"x": round(float(ball[0]), 4), "y": round(float(ball[1]), 4),
                          "z": round(float(ball[2]), 4),
                          "vx": round(bvel[0], 4), "vy": round(bvel[1], 4),
                          "vz": round(bvel[2], 4)}
        else:
            fr["ball"] = None
        frames.append(fr)

    doc = {
        "clip": clip,
        "schema": "gen3-scene-graph/1",
        "video": {
            "full_visibility": f"clips/full_visibility/{clip}.mov",
            "split_1s_4s": f"clips/split_1s_4s/{clip}.mov",
            "fps": G.FPS, "n_frames": n, "seconds": round(n / G.FPS, 2),
            "grid": {"cols": G.COLS, "rows": G.ROWS,
                     "labels": "rows A-F top to bottom, columns 1-16 left to right"},
        },
        "situation": gt["situation"],
        "start_team": {"team": gt["start_team"], "colour": gt["start_team_colour"]},
        "ground_truth": {
            "players_in_frame_last": int(gt["players_in_frame_last"]),
            "players_in_play": int(gt["players_in_play"]),
            "ball_final_cell": gt["ball_final_cell"],
            "ball_start_cell": gt["ball_start_cell"],
            "final_px": float(gt["final_px"]), "final_py": float(gt["final_py"]),
            "start_px": float(gt["start_px"]), "start_py": float(gt["start_py"]),
        },
        "reproduce": {
            "shape": gt["shape"], "seed": int(gt["seed"]), "match": gt["match"],
            "start_frame": start, "end_frame": end,
            "camera_offset": [float(gt["offset_x"]), float(gt["offset_y"])],
            "scenario_sha": repro.get("scenario_sha") if repro else None,
            "play_sha": repro.get("play_sha") if repro else None,
        },
        "pitch": PITCH,
        "kits": KITS,
        "notes": {
            "coordinates": "world units, NOT pixels; player pixel positions are not "
                           "derivable from these logs (the camera tracks the ball)",
            "in_frame": "this file describes every player ON THE PITCH. Which of them is "
                        "in shot is a camera property; the measured end-frame count is "
                        "ground_truth.players_in_frame_last",
            "officials": "referee and assistants render at 2% scale and are invisible; "
                         "they are not in this graph and never count as players",
            "ball_hidden_after": None if hide_ball_after is None else hide_ball_after,
        },
        "frames": frames,
    }
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--hide-ball-after", type=float, default=None,
                    help="seconds; drop ball state from this point on (1.0 matches split_1s_4s)")
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    gts = _rows(GT)
    repro = _rows(REPRO) if REPRO.exists() else {}
    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)

    want = a.clips or sorted(gts)
    total = 0
    for clip in want:
        if clip not in gts:
            print(f"  unknown clip {clip}")
            continue
        doc = build(clip, gts[clip], repro.get(clip), a.hide_ball_after)
        p = out / f"{clip}.json"
        p.write_text(json.dumps(doc, indent=2) if a.pretty else json.dumps(doc))
        total += p.stat().st_size
        print(f"  {clip}: {len(doc['frames'])} frames, "
              f"{len(doc['frames'][0]['players'])} players, {p.stat().st_size / 1024:.0f} KB")
    print(f"{len(want)} scene graphs -> {out}  ({total / 1024 / 1024:.1f} MB total)")


if __name__ == "__main__":
    main()
