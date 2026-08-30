"""gen3_lib.py — plumbing for the GEN3 pilot batch: 24 clips, exact player counts,
spread ball, and play that holds up as football.

WHAT GEN3 HAS TO SATISFY, ALL AT ONCE
  * the LAST FRAME of each clip contains exactly N players, for N = 8..19, two clips each
  * 5 corner kicks, 5 goalkeeper throws, 5 kick-offs, 9 open play, scattered over the Ns
  * final-frame ball cells spread across the 16x6 grid instead of piling into the middle
  * nothing hidden, no goals, no restart mid-clip, and play that reads as a real match

THREE THINGS THIS FILE GETS RIGHT THAT EARLIER BATCHES GOT WRONG
  Each was paid for once already; the reasoning is kept here so it is not re-litigated.

  1. TEAM STRENGTH. The v2 batch was rejected as "very bouncy", with "the game lacks
     logic, balls are just moving to move". The cause was measured and it was not the
     engine: v2's shapes ran difficulty gaps of 0.70-0.95 with offsides OFF, to manufacture
     set pieces. The weak side sits at or below gfootball's own `11_vs_11_easy_stochastic`
     preset, so it does not press, track runners or hold a line. Every shape below runs a
     gap of at most 0.40 with offsides ON — the stock `11_vs_11_stochastic` setting, which
     is what the endorsed GEN1 clips used. Set pieces are found by SEARCHING more matches,
     never by crippling one side.

     Do NOT re-investigate `physics_steps_per_frame` as a cause. It changes the observation
     and action rate only; ball-height distributions over two cached sweeps are identical
     (airborne-time ratio 1.000x).

  2. THE COUNT COMES FROM SELECTION, NEVER FROM HIDING. An earlier player-count batch hit
     its targets with renderScale and all 20 clips were rejected — the ball ended up
     stranded in empty grass, because the players who leave frame fastest are the ones
     chasing the ball, so hiding deleted the play and kept the bystanders. Every GEN3 clip
     renders all 22 players; the count is found by searching for windows the camera
     naturally frames at the target.

  3. COUNTING IS EXACT OR IT IS NOTHING. Two cheap counters were measured and both are
     useless as ground truth: fitting the camera frustum geometrically is 93.6% right per
     player but exactly right per FRAME only 36% of the time (it cannot see occlusion),
     and counting kit-coloured blobs runs +5.8 players because the advertising hoardings
     are saturated blue and red. Only the solo probe — a plate with all 22 hidden, plus 22
     passes each showing one player, diffed — is exact. Do not retry the other two.

THE CAMERA OFFSET IS WHY THIS IS ONE PIPELINE AND NOT TWO
  The stock camera centres on the ball, so the grid answer collapses to the middle: the
  original 30-clip benchmark used 15 of 96 cells and put 20 of 30 finals in row C. That is
  camera geometry, not scenario design — moving play around the pitch does not help,
  because the camera follows it. `GFOOTBALL_CAM_OFFSET_X/_Y` shifts the framing target so
  the shot is composed off-centre, the way a real operator frames play.

  But the offset also changes WHICH PLAYERS ARE IN FRAME. So the count and the ball cell
  are not independent knobs, and a count measured at offset 0 is worthless for a clip that
  ships at offset -12. The count must be probed AT THE OFFSET THAT SHIPS.

  That is affordable because of one measured fact: `env.render("rgb_array")` costs
  ~0.1 ms against ~8.3 ms for `env.step`, i.e. rendering is free and REPLAY is the whole
  cost. Probing one frame therefore costs the same as probing every frame, so a single
  23-pass probe of a match at one offset yields the exact count for all ~1600 of its
  candidate windows. The offset is consequently a per-MATCH parameter — which costs
  nothing, because source diversity already limits us to one clip per match.
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "experiments").is_dir())
SRC = "/Users/nithilbalamurugan/gfootball_src"
for _p in (SRC, SRC + "/third_party", str(REPO / "experiments"),
           str(REPO / "MATCH_SITUATIONS"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scenario_factory import ScenarioSpec, PlayerSpec  # noqa: E402

# ── locked format ───────────────────────────────────────────────────────────────
PSF = 4                    # observation rate = 100 / PSF, so 25 fps
FPS = 25
CLIP_FRAMES = 125          # 5.0 s
VIS_FRAMES = 25            # the "1 s visible" half of the split variant
MARK_FRAMES = 8            # red start-circle, ~0.3 s
HUD_TOP, HUD_BOT, FRAME_H = 60, 180, 720   # crop the score strip + radar panel

BUNDLE_VIS = "gen3"
BUNDLE_INV = "gen3_ball_invisible"

CACHE = HERE / "_cache"
SWEEP = CACHE / "sweep"
OUT = HERE / "clips"

# game_mode enum (engine src/defines.hpp)
GM_NORMAL, GM_KICKOFF, GM_GOALKICK, GM_FREEKICK, GM_CORNER, GM_THROWIN, GM_PENALTY = range(7)
SETPIECE_MODES = (GM_KICKOFF, GM_GOALKICK, GM_FREEKICK, GM_CORNER, GM_THROWIN, GM_PENALTY)

ALL_SLOTS = [f"{t}{i}" for t in ("L", "R") for i in range(11)]

# ── the count targets ───────────────────────────────────────────────────────────
END_COUNTS = list(range(8, 20))        # 12 levels, one exact count each
PER_COUNT = 2                          # 24 clips

SITUATION_QUOTA = {"corner": 5, "gk_throw": 5, "kickoff": 5, "open": 9}

# ── grid (locked 16x6, see MATCH_SITUATIONS/grid.py) ────────────────────────────
W, H = 1280, 480
COLS, ROWS = 16, 6
CELL_W, CELL_H = W / COLS, H / ROWS

# ── scenario ────────────────────────────────────────────────────────────────────


def P(x, y, role="CM"):
    return PlayerSpec(x, y, role)


def eleven(push_up=0.0):
    """4-3-3-ish starting eleven. Index order is the AddPlayer order, which is what
    GFOOTBALL_HIDE_SLOTS addresses as L0..L10 / R0..R10 — so slot 0 is the keeper."""
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

    `deterministic=False` is deliberate. With it True the engine ignores
    game_engine_random_seed and every seed replays ONE identical match, which caps the
    variety a sweep can reach. With it False the same seed still replays bit-identically
    (max ball-track difference 0.0 over 120 steps) — which is all the visible/invisible
    ball diff requires — while a different seed gives a genuinely different match.
    """
    return ScenarioSpec(
        name=name, ball=ball,
        left=eleven(push_left), right=eleven(push_right),
        game_duration=duration, deterministic=False, offsides=offsides,
        end_on_score=False, end_on_out=False, end_on_possession_change=False,
        left_difficulty=difficulty[0], right_difficulty=difficulty[1],
    )


# ── shapes ──────────────────────────────────────────────────────────────────────
# Every shape runs offsides ON and a difficulty gap of at most 0.40 (see note 1 in the
# module docstring). What varies is WHERE play tends to happen, because that is what
# drives both the situation mix and the number of bodies the camera frames:
#
#   midfield        both banks of players in shot        -> the high counts, 16-19
#   flank / box     attacking pressure in the final third-> corners and keeper throws
#   wide + even     contested wide play that stays alive -> mid counts, 11-15
#   counterattack   a break into open ground             -> the low counts, 8-10
#
# The counterattack shapes matter more than they look. With the stock tracking camera
# "few players on screen" almost always means play is jammed into a corner, which is
# exactly when the ball is about to go out — few-players and stoppage are the same event.
# A break the other way is the one situation where live, attended play genuinely frames a
# handful of bodies: one side committed high (push 0.72-0.78), the other sitting deep, and
# balanced difficulty so turnovers actually happen.
#
# The three push=0 shapes with the ball on the centre spot are ALSO the only source of
# kick-offs. A restart resets both teams to the scenario's own formation, so a shape whose
# formation is pushed 0.55 up the pitch restarts with five players in the opponent's half
# — measured — which is not a kick-off and is correctly rejected by find_kickoffs. Only a
# formation that is genuinely two banks behind the halfway line restarts as a kick-off.
SHAPES = [
    # name,     ball,            offsides, difficulty,   push_l, push_r
    ("mid_bal",  (0.00, 0.00),   True, (0.80, 0.80), 0.00, 0.00),
    ("mid_even", (0.00, 0.00),   True, (0.85, 0.75), 0.00, 0.00),
    ("mid_even2", (0.00, 0.00),  True, (0.75, 0.85), 0.00, 0.00),
    ("mid_push", (0.20, 0.12),   True, (0.85, 0.75), 0.30, 0.20),
    ("mid_r",    (-0.15, -0.10), True, (0.75, 0.85), 0.15, 0.30),
    ("flankL",   (0.75, 0.34),   True, (1.00, 0.60), 0.50, 0.35),
    ("flankL2",  (0.85, -0.32),  True, (1.00, 0.60), 0.55, 0.40),
    ("flankR",   (-0.75, -0.34), True, (0.60, 1.00), 0.35, 0.50),
    ("boxL",     (0.90, 0.10),   True, (1.00, 0.60), 0.55, 0.40),
    ("boxR",     (-0.90, -0.10), True, (0.60, 1.00), 0.40, 0.55),
    ("wideL",    (0.70, 0.36),   True, (0.80, 0.80), 0.50, 0.45),
    ("wideR",    (-0.70, -0.36), True, (0.80, 0.80), 0.45, 0.50),
    ("countL",   (0.30, 0.05),   True, (0.80, 0.80), 0.78, 0.00),
    ("countR",   (-0.30, -0.05), True, (0.80, 0.80), 0.00, 0.78),
]


def shape_spec(shape_name):
    for name, ball, offs, diff, pl, pr in SHAPES:
        if name == shape_name:
            return match_spec(f"g3_{name}", ball=ball, offsides=offs,
                              difficulty=diff, push_left=pl, push_right=pr)
    raise KeyError(shape_name)


def tag(shape, seed):
    return f"{shape}_s{seed:03d}"


def sweep_jobs(seeds):
    """(shape, seed) pairs, seed-major so an interrupted sweep still has all shapes."""
    for seed in seeds:
        for shape, *_ in SHAPES:
            yield shape, seed


# ── engine passes ───────────────────────────────────────────────────────────────
def _blank_font():
    """The engine keeps a second font handle the bundle swap misses; point it at the
    bundle's blanked font or player-name captions leak back in — and the locked format
    is noname."""
    bundle = os.environ.get("GFOOTBALL_DATA_DIR", "")
    blank = Path(bundle) / "media/fonts/alegreya/AlegreyaSansSC-ExtraBold.ttf"
    if blank.exists():
        os.environ["GFOOTBALL_FONT"] = str(blank)


def set_offset(offx, offy):
    """Set the camera framing offset for the next render.

    Setting these to 0 is NOT the same as leaving them unset: the unset path skips the
    offset branch entirely and is byte-identical to the stock camera, while "0" still
    applies the branch's pitch bound, which is a little wider than the stock follow clamp.
    Every GEN3 pass sets them — 0 during the natural-pixel probe — precisely so the probe
    baseline and the reframed render go through the SAME camera bounds. Otherwise the
    calibration would be measured against one camera and applied to another.
    """
    os.environ["GFOOTBALL_CAM_OFFSET_X"] = str(float(offx))
    os.environ["GFOOTBALL_CAM_OFFSET_Y"] = str(float(offy))


def _env(level, seed, workdir):
    from gfootball.env import config as cfg, football_env
    workdir.mkdir(parents=True, exist_ok=True)
    values = {
        "level": level, "players": [], "action_set": "full",
        "write_video": False, "dump_full_episodes": False, "dump_scores": False,
        "tracesdir": str(workdir), "real_time": False,
        "game_engine_random_seed": seed, "video_quality_level": 2,
        "display_game_stats": False, "physics_steps_per_frame": PSF,
    }
    return football_env.FootballEnv(cfg.Config(values))


def _prep(hide_slots, offset):
    """Environment state every engine pass needs, in one place.

    GFOOTBALL_TEAM_GK_KITS is set on EVERY pass: it selects the per-team keeper kits, and
    a pass that forgot it would render a different-looking keeper, which would show up in
    the visible/invisible diff as a false ball.
    """
    _blank_font()
    os.environ["GFOOTBALL_TEAM_GK_KITS"] = "1"
    os.environ["GFOOTBALL_HIDE_SLOTS"] = hide_slots
    set_offset(*offset)


def run_log(level, seed, steps, offset=(0.0, 0.0)):
    """Play deterministically and return the observation log — no frames kept.

    render("rgb_array") is called once up front so the sim advances on the same timing as
    a render pass; without it the logged play and the rendered play can drift apart.

    The log does not depend on the camera at all, so ONE sweep serves every offset.
    """
    _prep("", offset)
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


def render_window(level, seed, start, end, hide_slots="", offset=(0.0, 0.0),
                  keep=None, down=1):
    """Replay and keep frames [start, end). Returns (frames, ball_xy).

    Hiding is render-only (renderScale), so the hidden set never changes the play — which
    is what lets the solo probe count players without perturbing what it is counting.

    `keep` and `down` exist for the probe, and they are not an optimisation — without them
    it does not run at all. A probe replays to the last candidate frame, which for a corner
    is up to 2750, and holding that many 480x1280x3 frames costs ~5 GB as uint8 and ~10 GB
    once stacked as int16 for the diff. Six shards of that take the machine down.

    The probe only ever reads the CANDIDATE END FRAMES — at most a few dozen per match —
    so it passes `keep` (the frame indices it wants) and `down=2`. Everything else is
    rendered, because the engine renders every step regardless, and simply dropped.
    Memory falls to tens of megabytes with no change in replay cost.
    """
    _prep(hide_slots, offset)
    env = _env(level, seed, CACHE / "_work_render")
    env.render("rgb_array")
    env.reset()
    frames, balls = [], []
    for i in range(end):
        obs, _r, done, _i = env.step([])
        o = obs[0] if isinstance(obs, list) else obs
        if i >= start and (keep is None or i in keep):
            f = np.asarray(env.render("rgb_array"))[HUD_TOP:FRAME_H - HUD_BOT, :]
            frames.append(f[::down, ::down] if down > 1 else f)
            balls.append(np.asarray(o["ball"][:2], dtype=float))
        if done:
            break
    env.close()
    return frames, np.array(balls)


# ── continuity: one unbroken passage of play ────────────────────────────────────
def continuous(balls):
    """No goal, no restart, no mid-clip respot.

    Two independent tests:
      * no teleport — a single-frame ball jump >= 0.35 means the ENGINE moved the ball (a
        goal sending it to the centre spot, or a set piece being re-spotted part-way
        through), not the players.
      * no goal — the ball never crosses a goal line BETWEEN THE POSTS
        (|x| > 1.0 with |y| < 0.05).

    "|x| > 1.0" alone is NOT a goal test and must never be used as one: a corner is taken
    from x = 1.011, on the arc and past the goal line by definition of Law 17, so that
    test rejects every real corner.
    """
    balls = np.asarray(balls)
    if len(balls) < 3:
        return False
    if float(np.linalg.norm(np.diff(balls[:, :2], axis=0), axis=1).max()) >= 0.35:
        return False
    in_goal = (np.abs(balls[:, 0]) > 1.0) & (np.abs(balls[:, 1]) < 0.05)
    return not bool(in_goal.any())


def no_setpiece_inside(log, s, e):
    """No restart AWARDED inside the window.

    A window that merely OPENS while a restart is already pending is fine — that is just
    play resuming, which is ordinary football, and it is exactly what a corner or kick-off
    clip is. What is not fine is a restart being awarded part-way through, because that
    re-spots the ball and splits the clip into two unrelated passages.
    """
    gm = log["game_mode"][s:e]
    return not any(gm[i] in SETPIECE_MODES and gm[i] != gm[i - 1] for i in range(1, len(gm)))


# ── coherence: does this passage read as football? ──────────────────────────────
# GEN1 kept a window only if the ball path length fell in a moderate band — "steady
# passing/possession, not frantic, not static". v2 dropped that entirely and gated on
# player count instead, and that is a large part of why it was rejected. It is restored
# here, with the two statistics that actually separated the two batches: how long the ball
# is loose (nobody in possession) and how far it is from the nearest player.
#
#              GEN1 (endorsed)   v2 headers (rejected)
#   loose         39.6%              65.4%
#   apex          1.41               3.16
#   d_near        0.8 m              1.0 m
#
# PER-CLASS, because one cap cannot be honest about all four. A keeper's delivery goes up
# and a corner is an unowned ball in flight by definition; capping those at open play's
# numbers yields zero clips rather than better ones. Each class gets the bound its own
# physics allows, and the RANKING inside it picks the calmest instance.
ENGAGEMENT = {
    "open":     {"loose_max": 0.55, "apex_max": 2.60, "dnear_max": 3.50,
                 "path_min": 0.33, "path_max": 0.95},   # GEN1's own movement band
    "corner":   {"loose_max": 0.85, "apex_max": 3.60, "dnear_max": 4.00,
                 "path_min": 0.25, "path_max": 1.20},
    "gk_throw": {"loose_max": 0.65, "apex_max": 5.20, "dnear_max": 4.00,
                 "path_min": 0.25, "path_max": 1.20},
    "kickoff":  {"loose_max": 0.60, "apex_max": 3.00, "dnear_max": 4.00,
                 "path_min": 0.25, "path_max": 1.10},
}
MX, MY = 52.5, 34.0 / 0.42             # normalised pitch units -> metres


def coherence(log, s, e):
    """The four numbers the gate and the ranking are built from."""
    ball = log["ball"][s:e]
    pl = np.concatenate([log["left"][s:e], log["right"][s:e]], axis=1)
    d = np.linalg.norm((pl - ball[:, None, :2]) * np.array([MX, MY]), axis=2)
    return {"loose": float((log["owned_team"][s:e] == -1).mean()),
            "apex": float(ball[:, 2].max()),
            "dnear": float(np.median(d.min(axis=1))),
            "path": float(np.linalg.norm(np.diff(ball[:, :2], axis=0), axis=1).sum())}


def engaged(kind, c):
    e = ENGAGEMENT[kind]
    return (c["loose"] <= e["loose_max"]
            and c["apex"] <= e["apex_max"]
            and c["dnear"] <= e["dnear_max"]
            and e["path_min"] <= c["path"] <= e["path_max"])


def coherence_score(c):
    """Lower is better: mostly-owned ball, kept low, kept close to a player."""
    return c["loose"] * 2.0 + c["apex"] / 3.2 + c["dnear"] / 4.0


# ── situation detectors ─────────────────────────────────────────────────────────
# Restart classes (corner, kick-off) are anchored on the DELIVERY — the first frame the
# ball moves again after the referee re-spots it — not on the award. Anchoring on the
# award spends the opening seconds of the clip on a static ball, and opening BEFORE the
# award drags in the re-spot itself: a teleport small enough to slip under the continuity
# threshold, which would leave the red start-circle marking a spot the ball instantly
# leaves. `min_start` enforces the "never before award + 1" rule.
LEAD = {"corner": 10, "kickoff": 10, "gk_throw": 38}


def _restart_awards(gm, mode):
    """Frames where `mode` is newly awarded. Frame 0 counts when the match opens in it,
    which is how the genuine opening kick-off is found — it has no transition to detect."""
    awards = [i for i in range(1, len(gm)) if gm[i] == mode and gm[i - 1] != mode]
    if len(gm) and gm[0] == mode:
        awards.insert(0, 0)
    return awards


def _deliveries(log, mode, quiet=0.0048, search=225):
    """(delivery, min_start) for each award of `mode`.

    `quiet` is a per-frame ball speed at 25 fps: below it the ball is still being spotted.
    """
    gm, ball = log["game_mode"], log["ball"]
    out = []
    for f in _restart_awards(gm, mode):
        seg = ball[f:f + search, :2]
        if len(seg) < 3:
            continue
        speed = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        moving = np.flatnonzero(speed > quiet)
        if len(moving) == 0:
            continue
        out.append({"anchor": f + int(moving[0]), "award": f, "min_start": f + 1})
    return out


def find_corners(log):
    return _deliveries(log, GM_CORNER)


def find_kickoffs(log, spot=0.03, own_half_min=10):
    """Kick-offs — found as BEHAVIOUR, not as a game_mode.

    GM_KICKOFF never fires in these scenarios: measured over 120 cached matches
    (330,000 frames) it was awarded exactly 0 times, while goal kicks and free kicks fired
    constantly. Restarts still happen, the engine just does not label them. So detect the
    thing itself — the ball respotted on the centre mark with both sides behind the
    halfway line — which is the definition of a kick-off anyway:

      * frame 0, where the scenario itself opens with the ball on the centre spot and the
        two elevens in their own halves. That IS a kick-off, and it is the one every match
        supplies for free.
      * every frame the score changes, where the engine teleports the ball back to the
        centre spot (measured: a 1.017 single-frame jump) and resets both teams.

    Both are validated against the same geometric test rather than trusted, so a scenario
    whose ball does not start on the centre spot contributes no kick-off.

    `min_start` is the restart frame itself: opening a window even one frame earlier would
    drag in the respot teleport, and on a post-goal restart that means showing the goal.
    """
    ball, sc = log["ball"], log["score"]
    n = len(ball)
    restarts = [0] + [i for i in range(1, n) if tuple(sc[i]) != tuple(sc[i - 1])]
    out = []
    for f in restarts:
        if float(np.linalg.norm(ball[f][:2])) > spot:
            continue                                   # not on the centre mark
        left_home = int((log["left"][f][:, 0] < 0).sum())
        right_home = int((log["right"][f][:, 0] > 0).sum())
        if left_home < own_half_min or right_home < own_half_min:
            continue                                   # not a kick-off shape
        seg = ball[f:f + 225, :2]
        if len(seg) < 3:
            continue
        moving = np.flatnonzero(np.linalg.norm(np.diff(seg, axis=0), axis=1) > 0.0015)
        if len(moving) == 0:
            continue
        out.append({"anchor": f + int(moving[0]), "award": f, "min_start": f,
                    "opening": f == 0})
    return out


def find_gk_throws(log, hold_min=10, z_lo=1.15, z_hi=1.75, travel_min=0.20):
    """Keeper holds the ball at hand height, then releases it a long way. Anchored on the
    RELEASE, so the clip opens on the throw rather than on a keeper standing still.

    `hold_min` is ten frames = ~0.4 s at 25 fps. The z band and travel are metres and
    pitch units — physics, not frame counts, so they do not scale with fps.
    """
    ball, ot, op = log["ball"], log["owned_team"], log["owned_player"]
    out = []
    i, n = 0, len(ball)
    while i < n:
        if op[i] == 0 and ot[i] in (0, 1):          # slot 0 is the keeper
            team = ot[i]
            j = i
            while j < n and op[j] == 0 and ot[j] == team:
                j += 1
            hand = [k for k in range(i, j) if z_lo <= ball[k][2] <= z_hi]
            if len(hand) >= hold_min and j < n:
                rel = j - 1
                end = min(n - 1, rel + 75)
                travel = float(np.linalg.norm(ball[end][:2] - ball[rel][:2]))
                if travel >= travel_min:
                    out.append({"anchor": rel, "award": rel, "min_start": 0,
                                "travel": round(travel, 3)})
            i = j
        else:
            i += 1
    return out


DETECTORS = {"corner": find_corners, "kickoff": find_kickoffs, "gk_throw": find_gk_throws}


# ── composition: the two visibility variants from one render pair ───────────────
def compose_pair(vis, inv):
    """(full_visibility_frames, split_frames), grid burned in and a red circle on the
    ball's START position for the first MARK_FRAMES frames.

    Both variants share identical frames for the first VIS_FRAMES, so a model sees the
    same opening either way and the split variant is the same play, not a different clip.

    Returns the measured start/final ball pixels too: those come from THIS render pair,
    the one that actually ships, so the ground truth cannot drift from the video.
    """
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


def cell_of(px, py):
    c = min(max(int(px // CELL_W), 0), COLS - 1)
    r = min(max(int(py // CELL_H), 0), ROWS - 1)
    return f"{chr(ord('A') + r)}{c + 1}"
