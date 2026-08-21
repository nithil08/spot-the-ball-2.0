"""gen2_lib.py — shared plumbing for the GEN2 benchmark batch.

GEN2 is the second-generation classification set. Every clip is EXACTLY 5.0 s
(50 frames at 10 fps) on the locked 16x6 grid, and every situation ships in two
visibility variants cut from the SAME underlying render pair:

    full_visibility/   ball visible for all 50 frames
    split_1s_4s/       ball visible for 10 frames, then invisible for 40

Both come from one visible render + one invisible render of the identical
deterministic state, so they are the same play — only the ball's visibility
differs. That is also what makes ball ground truth exact: the pixels that differ
between the two renders ARE the ball.

KITS: GEN2 uses the `gen2` / `gen2_ball_invisible` bundles — team A blue, team B
red, goalkeepers yellow (see experiments/apply_gen2_kits.py).

MATCH VARIETY COMES FROM SEEDS, NOT SHAPES
  The older MATCH_SITUATIONS sweep set `deterministic=True`, which makes the
  engine ignore game_engine_random_seed: every seed replays one identical match,
  so variety had to come from hand-written scenario shapes. That capped the yield
  at 3 corners in 48 matches, nowhere near the 20 GEN2 needs.

  Measured here instead: with `deterministic=False`, the same seed still replays
  BIT-IDENTICALLY (max ball-track difference 0.0 over 120 steps) while a different
  seed gives a genuinely different match (max difference 0.95). Identical replay is
  the only property the visible/invisible diff actually requires, so GEN2 keeps a
  small set of shapes and sweeps SEEDS across them. That is effectively unlimited
  match variety, and it is what makes 20 distinct corners reachable.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "experiments").is_dir())
SRC = "/Users/nithilbalamurugan/gfootball_src"
for _p in (SRC, SRC + "/third_party", str(REPO / "experiments"),
           str(REPO / "MATCH_SITUATIONS"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scenario_factory import ScenarioSpec, PlayerSpec  # noqa: E402

# ── locked format ───────────────────────────────────────────────────────────────
CLIP_FRAMES = 50           # 5.0 s at 10 fps — every GEN2 clip, no exceptions
VIS_FRAMES = 10            # the "1 s visible" half of the split variant
FPS = 10
MARK_FRAMES = 3            # red start-circle is drawn on the first N frames
HUD_TOP, HUD_BOT, FRAME_H = 60, 180, 720   # crop the score strip + radar panel

BUNDLE_VIS = "gen2"
BUNDLE_INV = "gen2_ball_invisible"

CACHE = HERE / "_cache"
SWEEP = CACHE / "sweep"

# game_mode enum (engine src/defines.hpp)
GM_NORMAL, GM_KICKOFF, GM_GOALKICK, GM_FREEKICK, GM_CORNER, GM_THROWIN, GM_PENALTY = range(7)

ALL_SLOTS = [f"{t}{i}" for t in ("L", "R") for i in range(11)]


def P(x, y, role="CM"):
    return PlayerSpec(x, y, role)


def eleven(push_up=0.0):
    """4-3-3-ish starting eleven. Index order is the AddPlayer order, which is what
    GFOOTBALL_HIDE_SLOTS addresses as L0..L10 / R0..R10."""
    return [
        P(-1.000, 0.000, "GK"),                 # 0
        P(-0.010 + push_up, 0.020, "RM"),       # 1
        P(-0.010 + push_up, -0.020, "CF"),      # 2
        P(-0.422 + push_up, -0.196, "LB"),      # 3
        P(-0.500 + push_up, -0.064, "CB"),      # 4
        P(-0.500 + push_up, 0.064, "CB"),       # 5
        P(-0.422 + push_up, 0.196, "RB"),       # 6
        P(-0.184 + push_up, -0.106, "CM"),      # 7
        P(-0.268 + push_up, 0.000, "CM"),       # 8
        P(-0.184 + push_up, 0.106, "CM"),       # 9
        P(-0.010 + push_up, -0.216, "LM"),      # 10
    ]


def match_spec(name, ball=(0.0, 0.0), offsides=True, difficulty=(0.8, 0.8),
               push_left=0.0, push_right=0.0, duration=3000):
    """A full 11 v 11 with the engine's own referee running and nothing ending the
    episode early, so awarded set pieces actually get played out on camera.

    deterministic=False is deliberate — see the module docstring. It is what lets
    game_engine_random_seed generate different matches while keeping each seed's
    replay exact.
    """
    return ScenarioSpec(
        name=name, ball=ball,
        left=eleven(push_left), right=eleven(push_right),
        game_duration=duration, deterministic=False, offsides=offsides,
        end_on_score=False, end_on_out=False, end_on_possession_change=False,
        left_difficulty=difficulty[0], right_difficulty=difficulty[1],
    )


# ── the shapes we sweep seeds across ────────────────────────────────────────────
# Weighted towards the situations that actually produce set pieces: a strong side
# camped on a weak side's box concedes corners (deflections behind) and commits
# fouls, and a ball out on the flank in the final third is where crosses (and the
# headers that meet them) come from. `balanced` supplies ordinary midfield play.
SHAPES = [
    # name,        ball,           offsides, difficulty,   push_l, push_r
    ("balanced",   (0.00, 0.00),   True,  (0.80, 0.80), 0.00, 0.00),
    ("midpush",    (0.20, 0.12),   True,  (0.85, 0.75), 0.30, 0.20),
    ("pressL",     (0.62, 0.05),   False, (1.00, 0.15), 0.60, 0.35),
    ("pressL2",    (0.78, -0.08),  False, (1.00, 0.05), 0.72, 0.40),
    ("pressR",     (-0.62, -0.05), False, (0.15, 1.00), 0.35, 0.60),
    ("pressR2",    (-0.78, 0.08),  False, (0.05, 1.00), 0.40, 0.72),
    ("flankL",     (0.75, 0.34),   False, (1.00, 0.30), 0.65, 0.40),
    ("flankL2",    (0.85, -0.32),  False, (1.00, 0.20), 0.70, 0.45),
    ("flankR",     (-0.75, -0.34), False, (0.30, 1.00), 0.40, 0.65),
    ("boxL",       (0.90, 0.10),   False, (1.00, 0.10), 0.75, 0.45),
    ("boxR",       (-0.90, -0.10), False, (0.10, 1.00), 0.45, 0.75),
    ("endtoend",   (0.10, -0.28),  True,  (0.95, 0.95), 0.45, 0.45),
]


def shape_spec(shape_name):
    for name, ball, offs, diff, pl, pr in SHAPES:
        if name == shape_name:
            return match_spec(f"g2_{name}", ball=ball, offsides=offs,
                              difficulty=diff, push_left=pl, push_right=pr)
    raise KeyError(shape_name)


def sweep_jobs(seeds):
    """(shape, seed) pairs, seed-major so an interrupted sweep still has all shapes."""
    for seed in seeds:
        for shape, *_ in SHAPES:
            yield shape, seed


def tag(shape, seed):
    return f"{shape}_s{seed:03d}"


# ── engine passes ───────────────────────────────────────────────────────────────
def _env(level, seed, workdir):
    from gfootball.env import config as cfg, football_env
    workdir.mkdir(parents=True, exist_ok=True)
    values = {
        "level": level, "players": [], "action_set": "full",
        "write_video": False, "dump_full_episodes": False, "dump_scores": False,
        "tracesdir": str(workdir), "real_time": False,
        "game_engine_random_seed": seed, "video_quality_level": 2,
        "display_game_stats": False,
    }
    return football_env.FootballEnv(cfg.Config(values))


def _blank_font():
    """The engine keeps a second font handle the bundle swap misses; point it at the
    bundle's blanked font or player-name captions leak back in."""
    bundle = os.environ.get("GFOOTBALL_DATA_DIR", "")
    blank = Path(bundle) / "media/fonts/alegreya/AlegreyaSansSC-ExtraBold.ttf"
    if blank.exists():
        os.environ["GFOOTBALL_FONT"] = str(blank)


def run_log(level, seed, steps):
    """Play deterministically and return the observation log — no frames kept.

    render("rgb_array") is called once up front so the sim advances on the same
    timing as the render pass; without it the logged play and the rendered play
    can drift apart.
    """
    import numpy as np
    _blank_font()
    os.environ["GFOOTBALL_HIDE_SLOTS"] = ""
    env = _env(level, seed, CACHE / "_work_log")
    env.render("rgb_array")
    env.reset()
    ball, ot, op, gm, lt, rt, sc = [], [], [], [], [], [], []
    for _ in range(steps):
        obs, _r, done, _i = env.step([])
        o = obs[0] if isinstance(obs, list) else obs
        ball.append(np.asarray(o["ball"], dtype=np.float32))
        ot.append(int(o["ball_owned_team"]))
        op.append(int(o["ball_owned_player"]))
        gm.append(int(o["game_mode"]))
        lt.append(np.asarray(o["left_team"], dtype=np.float32))
        rt.append(np.asarray(o["right_team"], dtype=np.float32))
        sc.append(list(map(int, o["score"])))
        if done:
            break
    env.close()
    return {"ball": np.array(ball), "owned_team": np.array(ot),
            "owned_player": np.array(op), "game_mode": np.array(gm),
            "left": np.array(lt), "right": np.array(rt), "score": np.array(sc)}


def render_window(level, seed, start, end, hide_slots=""):
    """Replay and keep frames [start, end). Returns (frames, ball_xy).

    Hiding is render-only (renderScale), so the hidden set never changes the play.
    """
    import numpy as np
    _blank_font()
    os.environ["GFOOTBALL_HIDE_SLOTS"] = hide_slots
    env = _env(level, seed, CACHE / "_work_render")
    env.render("rgb_array")
    env.reset()
    frames, balls = [], []
    for i in range(end):
        obs, _r, done, _i = env.step([])
        o = obs[0] if isinstance(obs, list) else obs
        if i >= start:
            f = np.asarray(env.render("rgb_array"))[HUD_TOP:FRAME_H - HUD_BOT, :]
            frames.append(f)
            balls.append(np.asarray(o["ball"][:2], dtype=float))
        if done:
            break
    env.close()
    return frames, np.array(balls)


# ── continuity: one unbroken passage of play ────────────────────────────────────
def continuous(balls):
    """No goal, no kickoff restart, no mid-clip respot.

    Two independent tests:
      * no teleport — a single-frame ball jump >= 0.35 means the ENGINE moved the
        ball (a goal sending it to the centre spot, or a set piece being re-spotted
        part-way through), not the players.
      * no goal — the ball never crosses a goal line BETWEEN THE POSTS
        (|x| > 1.0 with |y| < 0.05).

    "|x| > 1.0" alone is NOT a goal test and must never be used as one: a corner is
    taken from x = 1.011, on the arc and past the goal line by definition of Law 17,
    so that test rejects every real corner.
    """
    import numpy as np
    balls = np.asarray(balls)
    if len(balls) < 3:
        return False
    if float(np.linalg.norm(np.diff(balls[:, :2], axis=0), axis=1).max()) >= 0.35:
        return False
    in_goal = (np.abs(balls[:, 0]) > 1.0) & (np.abs(balls[:, 1]) < 0.05)
    return not bool(in_goal.any())


# ── composition: the two visibility variants from one render pair ───────────────
def compose_pair(vis, inv):
    """(full_visibility_frames, split_frames) with the grid burned in and a red
    circle on the ball's START position for the first MARK_FRAMES frames.

    Returns the frames plus the measured start/final ball pixels, which are the
    ground truth. Both variants share identical frames for the first VIS_FRAMES,
    so a model sees the same opening either way.
    """
    import numpy as np
    from grid import build_grid, burn_grid, circle_ball, detect_ball
    overlay = build_grid()
    spx, spy = detect_ball(vis[0], inv[0])
    fpx, fpy = detect_ball(vis[-1], inv[-1])

    def finish(seq):
        out = []
        for i, f in enumerate(seq):
            g = burn_grid(np.asarray(f), overlay)
            if i < MARK_FRAMES:
                g = circle_ball(g, spx, spy)
            out.append(g)
        return out

    full = finish(list(vis))
    split = finish(list(vis[:VIS_FRAMES]) + list(inv[VIS_FRAMES:]))
    return full, split, (spx, spy), (fpx, fpy)


def write_clip(frames, path):
    from lib import frames_to_mov
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_to_mov(frames, path, fps=FPS, crop_hud=False)
    return path
