"""gen_frame_visibility_clips.py — 11 v 11 clips that START with 8 players in frame and
END with 6, while all 22 players are in the game the whole time.

THE POINT: the player count on screen changes because of what the CAMERA sees, not
because anybody left the match. All 22 players are in the simulation for every frame —
passing, marking, closing down. Some of them are simply outside the broadcast frame, and
which ones are outside changes as the camera tracks the ball. That is exactly what a real
11 v 11 broadcast looks like: you never see all 22 at once.

HOW THE COUNT IS MADE EXACT
  "In frame" is measured, not guessed. For each base match we render:
      plate  — every player hidden (GFOOTBALL_HIDE_SLOTS = all 22 slots)
      solo_i — only player i shown, for each of the 22 players
  and diff solo_i against the plate frame by frame. A non-trivial pixel difference means
  player i's body is inside the camera frustum on that frame. Hiding is render-only
  (renderScale, the same engine trick used on the referees), so every one of these passes
  replays the identical match — the ball and all 22 players move the same way in all 23
  renders. That gives an exact per-frame, per-player on-screen map.

  With that map we choose ONE fixed set of players to hide for the whole clip such that,
  of the players still shown, exactly 8 are inside the frame on the first frame and
  exactly 6 on the last. Players who are never on camera during the window are left shown
  — they need no hiding, they are simply out of shot.

  So each clip is a SINGLE continuous render with a SINGLE hide set: no splice, no
  mid-clip pop-in or pop-out. The 8 -> 6 change happens because players ran and the camera
  moved, which is the situation the clips are meant to show.

Grid/format matches the locked benchmark: 16x6 yellow grid, no player names, and a red
circle around the ball on the very first frame. The ball's cell is ground truth, obtained
by pixel-diffing the identical play rendered with the ball visible vs invisible.

Run:  python3 gen_frame_visibility_clips.py all
      (phases: probe -> vis  [bundle noname],  inv  [bundle noname_ball_invisible],
       compose [no engine])
Out:  MATCH_SITUATIONS/05_frame_visibility_8to6/
"""
import json
import os
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

from situations import match_spec  # noqa: E402
from grid import (build_grid, burn_grid, cell_of, circle_ball, detect_ball,  # noqa: E402
                  W, H, FPS)

OUT = HERE / "05_frame_visibility_8to6"
CACHE = HERE / "_frames"
PROBE_STEPS = 190          # how far into each match the on-screen map is measured
CLIP_LEN = 50              # 5.0 s at 10 fps
START_COUNT, END_COUNT = 8, 6
N_CLIPS = 5
DOWN = 2                   # probe frames are half-size; a body is still tens of pixels
MIN_PIX = 40               # changed (half-size) pixels before we call a player "in frame"

ALL_SLOTS = [f"{t}{i}" for t in ("L", "R") for i in range(11)]

HUD_TOP, HUD_BOT, FRAME_H = 60, 180, 720


# ── base matches: ordinary balanced 11 v 11, nothing staged ─────────────────────
def bases():
    shapes = [
        ("fv_base1", (0.0, 0.0), (0.00, 0.00), (0.8, 0.8)),
        ("fv_base2", (0.20, 0.12), (0.25, 0.15), (0.8, 0.8)),
        ("fv_base3", (-0.15, -0.10), (0.15, 0.30), (0.9, 0.7)),
        ("fv_base4", (0.35, -0.20), (0.40, 0.20), (0.8, 0.9)),
        ("fv_base5", (-0.05, 0.25), (0.30, 0.35), (0.7, 0.9)),
        ("fv_base6", (0.45, 0.05), (0.50, 0.25), (0.9, 0.8)),
        ("fv_base7", (0.10, -0.28), (0.20, 0.10), (0.85, 0.85)),
    ]
    for name, ball, (pl, pr), diff in shapes:
        yield name, match_spec(name, ball=ball, offsides=True, difficulty=diff,
                               push_left=pl, push_right=pr)


def render_play(level, steps, hide_slots, down=1, keep_from=0):
    """Render `level` for `steps` steps with `hide_slots` hidden. Hiding is render-only,
    so every call replays the identical match."""
    import numpy as np
    from gfootball.env import config as cfg, football_env
    os.environ["GFOOTBALL_HIDE_SLOTS"] = hide_slots
    bundle = os.environ.get("GFOOTBALL_DATA_DIR", "")
    blank = Path(bundle) / "media/fonts/alegreya/AlegreyaSansSC-ExtraBold.ttf"
    if blank.exists():
        os.environ["GFOOTBALL_FONT"] = str(blank)
    work = CACHE / "_work"
    work.mkdir(parents=True, exist_ok=True)
    values = {
        "level": level, "players": [], "action_set": "full",
        "write_video": False, "dump_full_episodes": False, "dump_scores": False,
        "tracesdir": str(work), "real_time": False,
        "game_engine_random_seed": 42, "video_quality_level": 2,
        "display_game_stats": False,
    }
    env = football_env.FootballEnv(cfg.Config(values))
    env.render("rgb_array")
    obs = env.reset()
    frames, balls = [], []
    for i in range(steps):
        obs, r, done, _ = env.step([])
        o = obs[0] if isinstance(obs, list) else obs
        if i >= keep_from:
            f = np.array(env.render("rgb_array"))[HUD_TOP:FRAME_H - HUD_BOT, :]
            frames.append(f[::down, ::down] if down > 1 else f)
            balls.append(np.array(o["ball"][:2], dtype=float))
        if done:
            break
    env.close()
    return frames, np.array(balls)


def play_is_clean(balls):
    """No goal, no kickoff reset, no teleport — one unbroken passage of play."""
    import numpy as np
    if len(balls) < 3:
        return False
    steps = np.linalg.norm(np.diff(balls, axis=0), axis=1)
    if steps.max() >= 0.35:
        return False
    return float(np.abs(balls[:, 0]).max()) <= 1.0


# ── PHASE probe: measure, per player per frame, whether the body is inside the frame ──
def phase_probe():
    import numpy as np
    from lib import use_bundle
    from scenario_factory import write_scenario
    use_bundle("noname")
    CACHE.mkdir(parents=True, exist_ok=True)
    maps = {}
    for name, spec in bases():
        level = write_scenario(spec, force=True)
        plate, balls = render_play(level, PROBE_STEPS, ",".join(ALL_SLOTS), down=DOWN)
        if not play_is_clean(balls):
            print(f"  skip {level}: play breaks (goal / restart)")
            continue
        plate = np.array(plate, dtype=np.int16)
        on = np.zeros((len(ALL_SLOTS), len(plate)), dtype=bool)
        for si, slot in enumerate(ALL_SLOTS):
            hide = ",".join(s for s in ALL_SLOTS if s != slot)   # show ONLY this player
            solo, _ = render_play(level, PROBE_STEPS, hide, down=DOWN)
            solo = np.array(solo, dtype=np.int16)
            d = np.abs(solo - plate).sum(axis=3)                 # per-pixel change
            on[si] = (d > 30).reshape(len(plate), -1).sum(axis=1) > MIN_PIX
        maps[level] = on.tolist()
        print(f"  [probe] {level}: on-screen per frame "
              f"min={int(on.sum(axis=0).min())} max={int(on.sum(axis=0).max())}",
              flush=True)
    (CACHE / "fv_onscreen.json").write_text(json.dumps(maps))
    print(f"phase probe: measured {len(maps)} matches")


# ── PHASE pick: choose window + a single fixed hide set giving 8 then 6 in frame ──
def choose_hide_set(on, s, e):
    """on: (22, T) bool. Window [s, e]. Return the slots to hide, or None.

    A = players in frame on the first frame, B = players in frame on the last frame.
    Keep `a` players from A&B, `b` from A-B and `c` from B-A with a+b == 8 and a+c == 6,
    plus every player who is never in frame during the window (they need no hiding —
    they are out of shot anyway). Everyone else is hidden for the WHOLE clip, so the
    count changes only because players moved and the camera moved with the ball.
    """
    A = {i for i in range(len(on)) if on[i][s]}
    B = {i for i in range(len(on)) if on[i][e]}
    never = {i for i in range(len(on)) if not any(on[i][s:e + 1])}
    both, onlyA, onlyB = A & B, A - B, B - A
    # Prefer players who are on camera for most of the window: it keeps the count
    # steady at 8 until the ones who are leaving actually leave, instead of flickering.
    dwell = {i: sum(on[i][s:e + 1]) for i in range(len(on))}
    rank = lambda g: sorted(g, key=lambda i: -dwell[i])          # noqa: E731
    for a in range(min(END_COUNT, len(both)), -1, -1):
        b, c = START_COUNT - a, END_COUNT - a
        if b < 0 or c < 0 or b > len(onlyA) or c > len(onlyB):
            continue
        shown = set(rank(both)[:a]) | set(rank(onlyA)[:b]) | set(rank(onlyB)[:c]) | never
        hide = [ALL_SLOTS[i] for i in range(len(on)) if i not in shown]
        n0 = sum(1 for i in shown if on[i][s])
        n1 = sum(1 for i in shown if on[i][e])
        if n0 == START_COUNT and n1 == END_COUNT:
            per_frame = [sum(1 for i in shown if on[i][t]) for t in range(s, e + 1)]
            return hide, n0, n1, sorted(shown), per_frame
    return None


def phase_pick():
    maps = json.loads((CACHE / "fv_onscreen.json").read_text())
    picks = []
    for level, on in maps.items():
        T = len(on[0])
        for s in range(0, T - CLIP_LEN, 10):
            e = s + CLIP_LEN - 1
            got = choose_hide_set(on, s, e)
            if got is None:
                continue
            hide, n0, n1, shown, per_frame = got
            picks.append({"level": level, "start": s, "end": e + 1,
                          "hide": ",".join(hide), "n_first": n0, "n_last": n1,
                          "shown_slots": [ALL_SLOTS[i] for i in shown],
                          "n_hidden": len(hide), "in_frame_per_frame": per_frame})
            break                      # one window per base match keeps the clips distinct
    picks = picks[:N_CLIPS]
    (CACHE / "fv_picks.json").write_text(json.dumps(picks, indent=2))
    for p in picks:
        print(f"  [pick] {p['level']} frames {p['start']}-{p['end']} "
              f"in-frame {p['n_first']} -> {p['n_last']}, {p['n_hidden']} hidden")
    print(f"phase pick: {len(picks)} clips")


# ── PHASE vis / inv: render the chosen windows with the chosen fixed hide set ────
def _render_picks(bundle, tag):
    import numpy as np
    from lib import use_bundle
    use_bundle(bundle)
    picks = json.loads((CACHE / "fv_picks.json").read_text())
    for i, p in enumerate(picks):
        frames, balls = render_play(p["level"], p["end"], p["hide"],
                                    keep_from=p["start"])
        np.savez_compressed(CACHE / f"fv{i:02d}_{tag}.npz", frames=np.array(frames))
        print(f"  [{tag}] {p['level']} {len(frames)} frames", flush=True)


def phase_vis():
    _render_picks("noname", "vis")


def phase_inv():
    _render_picks("noname_ball_invisible", "inv")


# ── PHASE compose: grid + red circle + movs + ground truth ──────────────────────
def phase_compose():
    import numpy as np
    from lib import frames_to_mov
    picks = json.loads((CACHE / "fv_picks.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    overlay = build_grid()
    rows = [("clip", "level", "start_frame", "end_frame", "players_in_frame_first",
             "players_in_frame_last", "players_in_play", "hidden_slots",
             "ball_start_cell", "start_px", "start_py", "ball_final_cell",
             "final_px", "final_py")]
    for i, p in enumerate(picks):
        vis = np.load(CACHE / f"fv{i:02d}_vis.npz")["frames"]
        inv = np.load(CACHE / f"fv{i:02d}_inv.npz")["frames"]
        spx, spy = detect_ball(vis[0], inv[0])
        fpx, fpy = detect_ball(vis[-1], inv[-1])
        out = []
        for k, f in enumerate(vis):
            g = burn_grid(f, overlay)
            if k == 0:
                g = circle_ball(g, spx, spy)
            out.append(g)
        name = f"frame_visibility_{START_COUNT}to{END_COUNT}_{i + 1:02d}"
        frames_to_mov(out, OUT / f"{name}.mov", fps=FPS, crop_hud=False)
        rows.append((name, p["level"], p["start"], p["end"], p["n_first"], p["n_last"],
                     22, p["n_hidden"], cell_of(spx, spy), round(spx, 1), round(spy, 1),
                     cell_of(fpx, fpy), round(fpx, 1), round(fpy, 1)))
        print(f"  [compose] {name}: in frame {p['n_first']} -> {p['n_last']}, "
              f"ball {cell_of(spx, spy)} -> {cell_of(fpx, fpy)}")
    (OUT / "ground_truth.csv").write_text(
        "\n".join(",".join(map(str, r)) for r in rows) + "\n")
    print(f"phase compose: wrote {len(picks)} clips -> {OUT}")


def phase_all():
    py = sys.executable
    for mode in ("stage1", "inv", "compose"):
        print(f"\n===== phase {mode} =====")
        subprocess.run([py, __file__, mode], check=True)


def phase_stage1():
    phase_probe()
    phase_pick()
    phase_vis()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    {"probe": phase_probe, "pick": phase_pick, "vis": phase_vis, "inv": phase_inv,
     "compose": phase_compose, "all": phase_all, "stage1": phase_stage1}[mode]()
