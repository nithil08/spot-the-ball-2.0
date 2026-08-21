"""gen_situation_clips.py — the real-football situation clips, on the benchmark grid.

Three situations, 5 clips each, every one of them a window cut out of a real deterministic
11 v 11 match (22 players, the engine's own referee running). Nothing is staged:

  01_header            a ball above chest height redirected by a player with other bodies
                       under it — a genuine contested aerial header, not a bounce.
  02_corner_kick       Law 17. The referee awarded it because a defender put the ball
                       behind: ball on the corner arc, attackers loaded into the box,
                       defenders marking, whipped in, contested, cleared.
  03_goalkeeper_throw  Law 12 distribution BY HAND. The keeper catches the ball, the engine
                       parks it at ~1.35 m (in his hands — the holdball retain state), and
                       his release plays the overarm THROW animation, carrying the ball
                       downfield to start an attack. A keeper with the ball at ~0.11 m is
                       dribbling and about to punt it; that is not a throw and is rejected.

NO GOALS, NO RESTARTS. A goal teleports the ball to the centre spot for the kickoff, which
breaks the clip into two passages of play. Every window is rejected unless the ball track
across it is continuous (no teleport, never over a goal line). This is checked twice: when
the window is picked (pick_situations.py) and again here against the rendered ball track.

Free kicks are deliberately NOT in this set. Worth recording why: with offsides on, almost
every free kick the engine awards is an OFFSIDE award, which by definition goes to the
defending side deep in their own half and just gets hoofed clear — the boring kind. The
dangerous 25-yard free kick needs a defender to foul an attacker in the final third, and a
dedicated sweep (hunt_free_kick.py, offsides off, strong attack vs weak defence) turned up
none in 51 match shapes.

Format (locked benchmark): 1280x480, 10 fps, 16x6 yellow grid, no player names, red circle
around the ball on the first frame. Ball ground truth per clip comes from pixel-diffing the
identical play rendered with the ball visible vs invisible.

Run:  python3 gen_situation_clips.py all
      (phases: vis [bundle noname] -> inv [noname_ball_invisible] -> compose)
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = "/Users/nithilbalamurugan/gfootball_src"
# Repo root = the nearest ancestor holding experiments/; keeps working if the
# checkout is moved (it was, from code/spot-the-ball-2.0 to the Desktop root).
EXP = str(next(p for p in Path(__file__).resolve().parents
               if (p / "experiments").is_dir()) / "experiments")
for _p in (SRC, SRC + "/third_party", EXP, str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from situations import all_variants  # noqa: E402
from render_window import render_window  # noqa: E402
from grid import build_grid, burn_grid, cell_of, circle_ball, detect_ball, FPS  # noqa: E402

SEED = 42
CACHE = HERE / "_frames"
PICKS = HERE / "_events" / "picks.json"
PLAN = CACHE / "situation_plan.json"
N_PER_KIND = 5
# Plan a couple of spares per kind: a window is only rejected once it has been rendered
# and its ball track checked, so spares are what keep us at 5 clips instead of 3.
N_PLANNED = 7

# folder, kind key in picks.json, frames before / after the key frame
KINDS = [
    ("01_header", "header", "header", 22, 44),
    ("02_corner_kick", "corner", "corner_kick", 0, 80),
    ("03_goalkeeper_throw", "gk_throw", "goalkeeper_throw", 10, 62),
]


def continuous(balls):
    """One unbroken passage of play: no goal, no kickoff restart, no mid-clip respot.

    Two independent tests:
      * no teleport — any single-frame ball jump >= 0.35 means the engine moved the ball
        rather than the players did: a goal sending it to the centre spot for the kickoff,
        or a set piece being re-spotted part-way through the clip.
      * no goal — the ball never crosses a goal line BETWEEN THE POSTS (|x| > 1.0 with
        |y| < 0.05, the goal mouth being 7.32 m of the 68 m pitch width).

    "|x| > 1.0" on its own is NOT a goal and must not be used as the test: a corner is
    taken from x = 1.011 (the ball sits on the arc, past the goal line by definition of
    Law 17) and balls run out for goal kicks all the time. Using it rejected every real
    corner, which is how this was found.
    """
    import numpy as np
    if len(balls) < 3:
        return False
    steps = np.linalg.norm(np.diff(balls, axis=0), axis=1)
    if steps.max() >= 0.35:
        return False
    in_goal = (np.abs(balls[:, 0]) > 1.0) & (np.abs(balls[:, 1]) < 0.05)
    return not bool(in_goal.any())


def build_plan():
    """Turn the ranked picks into concrete (level, start, end) windows."""
    picks = json.loads(PICKS.read_text())
    plan = []
    for folder, key, name, before, after in KINDS:
        chosen = []
        for r in picks.get(key, []):
            if len(chosen) >= N_PLANNED:
                break
            f = r.get("frame", r.get("catch"))
            if key == "corner":
                start = f                     # open ON the award: hides the reposition
            else:
                start = max(0, f - before)
            end = f + after
            chosen.append({"folder": folder, "name": name, "kind": key,
                           "level": r["level"], "key_frame": f,
                           "start": start, "end": end, "detail": r})
        plan.extend(chosen)
    return plan


def phase_vis():
    import numpy as np
    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle("noname")
    for name, spec in all_variants():
        write_scenario(spec, force=True)      # levels must match what was scanned
    CACHE.mkdir(parents=True, exist_ok=True)
    plan = build_plan()
    kept = []
    per_kind = {}
    for i, p in enumerate(plan):
        if per_kind.get(p["kind"], 0) >= N_PER_KIND:
            continue                       # already have five good ones of this kind
        frames, log = render_window(p["level"], SEED, p["start"], p["end"])
        balls = np.array([s["ball"][:2] for s in log])
        if not continuous(balls):
            print(f"  SKIP {p['level']}@{p['key_frame']} ({p['kind']}): "
                  f"play breaks in the window (goal / restart)")
            continue
        idx = len(kept)
        np.savez_compressed(CACHE / f"sit{idx:02d}_vis.npz", frames=np.array(frames))
        p["index"] = idx
        kept.append(p)
        per_kind[p["kind"]] = per_kind.get(p["kind"], 0) + 1
        print(f"  [vis] {p['kind']:9s} {p['level']} f{p['start']}-{p['end']} "
              f"({len(frames)} frames)", flush=True)
    PLAN.write_text(json.dumps(kept, indent=2, default=float))
    print(f"phase vis: {len(kept)} clips kept of {len(plan)} planned")


def phase_inv():
    import numpy as np
    from lib import use_bundle
    use_bundle("noname_ball_invisible")
    plan = json.loads(PLAN.read_text())
    for p in plan:
        frames, _ = render_window(p["level"], SEED, p["start"], p["end"])
        np.savez_compressed(CACHE / f"sit{p['index']:02d}_inv.npz",
                            frames=np.array(frames))
        print(f"  [inv] {p['kind']:9s} {p['level']}", flush=True)
    print(f"phase inv: {len(plan)} renders")


def phase_compose():
    import numpy as np
    from lib import frames_to_mov
    plan = json.loads(PLAN.read_text())
    overlay = build_grid()
    per_folder = {}
    rows = [("clip", "situation", "level", "key_frame", "start_frame", "end_frame",
             "n_frames", "seconds", "ball_start_cell", "start_px", "start_py",
             "ball_final_cell", "final_px", "final_py", "detail")]
    counters = {}
    for p in plan:
        vis = np.load(CACHE / f"sit{p['index']:02d}_vis.npz")["frames"]
        inv = np.load(CACHE / f"sit{p['index']:02d}_inv.npz")["frames"]
        spx, spy = detect_ball(vis[0], inv[0])
        fpx, fpy = detect_ball(vis[-1], inv[-1])
        out = []
        for k, f in enumerate(vis):
            g = burn_grid(f, overlay)
            if k == 0:
                g = circle_ball(g, spx, spy)
            out.append(g)
        n = counters.get(p["kind"], 0) + 1
        counters[p["kind"]] = n
        clip = f"{p['name']}_{n:02d}"
        d = HERE / p["folder"]
        d.mkdir(parents=True, exist_ok=True)
        frames_to_mov(out, d / f"{clip}.mov", fps=FPS, crop_hud=False)
        row = (clip, p["kind"], p["level"], p["key_frame"], p["start"], p["end"],
               len(out), round(len(out) / FPS, 1), cell_of(spx, spy),
               round(spx, 1), round(spy, 1), cell_of(fpx, fpy),
               round(fpx, 1), round(fpy, 1), json.dumps(p["detail"]).replace(",", ";"))
        rows.append(row)
        per_folder.setdefault(p["folder"], []).append(row)
        print(f"  [compose] {clip}: ball {cell_of(spx, spy)} -> {cell_of(fpx, fpy)}")

    header = ",".join(rows[0])
    for folder, rs in per_folder.items():
        (HERE / folder / "ground_truth.csv").write_text(
            header + "\n" + "\n".join(",".join(map(str, r)) for r in rs) + "\n")
    (HERE / "ALL_GROUND_TRUTH.csv").write_text(
        header + "\n" + "\n".join(",".join(map(str, r)) for r in rows[1:]) + "\n")
    print(f"phase compose: wrote {len(plan)} clips")


def phase_all():
    py = sys.executable
    for mode in ("vis", "inv", "compose"):
        print(f"\n===== phase {mode} =====")
        subprocess.run([py, __file__, mode], check=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    {"vis": phase_vis, "inv": phase_inv, "compose": phase_compose,
     "all": phase_all}[mode]()
