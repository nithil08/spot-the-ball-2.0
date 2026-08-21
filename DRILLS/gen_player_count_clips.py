"""gen_player_count_clips.py — plain demo clips whose VISIBLE player count changes mid-clip.

15 clips (5s, 1280x480, 10 fps, ball visible, NO grid / NO ground-truth — plain demo):
  * 5 clips: start with 4 players on screen, end with 6
  * 5 clips: start with 6, end with 8
  * 5 clips: start with 8, end with 4

"Players on screen" = total visible bodies across BOTH teams. Every player stays FULLY in
the simulation the whole time — hidden ones are only shrunk out of the render (renderScale,
the same engine trick used for referees), so ball/player motion is unchanged. That is what
lets us change the visible count without any teleport or reset.

HOW A SINGLE CLIP IS MADE (seamless count change):
  The play is deterministic, and hiding is render-only, so we render the SAME (scenario,
  seed) twice — once with the START count visible, once with the END count visible — and
  splice: frames [0:T] from the start render, frames [T:] from the end render (T = midpoint).
  Because both renders share identical motion, the extra players simply pop into (or out of)
  view at the exact positions they were already playing in. No teleport, fully continuous.

  Visibility is controlled by the engine env var GFOOTBALL_HIDE_SLOTS (added to team.cpp):
  a comma list of tokens "L<i>"/"R<i>" (team + index-within-team) to shrink to invisible.

Run:  python3 gen_player_count_clips.py
Output: DRILLS/player_count_clips/<name>.mov  + manifest.csv
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = "/Users/nithilbalamurugan/gfootball_src"
EXP = "/Users/nithilbalamurugan/Desktop/Nithil Research/code/spot-the-ball-2.0/experiments"
for _p in (SRC, SRC + "/third_party", EXP, str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scenario_factory import ScenarioSpec, PlayerSpec, write_scenario  # noqa: E402


def P(x, y, role="CM"):
    return PlayerSpec(x, y, role)


# ── 3 balanced 5v5 bases (GK + 4 outfield per side), all kept around midfield so play
#    stays continuous with no goals. gfootball convention: both teams in the left frame. ──
def base_spec(base_idx, name):
    if base_idx == 0:          # central open play
        ball = (0.0, 0.0)
        left = [P(-1.0, 0.0, "GK"), P(-0.15, 0.12, "CM"), P(-0.15, -0.12, "CM"),
                P(-0.35, 0.0, "CB"), P(0.05, 0.0, "CF")]
        right = [P(-1.0, 0.0, "GK"), P(-0.15, 0.12, "CM"), P(-0.15, -0.12, "CM"),
                 P(-0.35, 0.0, "CB"), P(0.05, 0.0, "CF")]
    elif base_idx == 1:        # stretched wide
        ball = (-0.05, -0.10)
        left = [P(-1.0, 0.0, "GK"), P(-0.10, -0.28, "LM"), P(-0.10, 0.28, "RM"),
                P(-0.22, 0.0, "CM"), P(0.10, 0.05, "CF")]
        right = [P(-1.0, 0.0, "GK"), P(-0.05, -0.20, "LM"), P(-0.05, 0.20, "RM"),
                 P(-0.15, 0.0, "CM"), P(0.02, 0.0, "CB")]
    else:                      # compact, slightly higher up the pitch
        ball = (0.05, 0.0)
        left = [P(-1.0, 0.0, "GK"), P(0.0, 0.10, "CM"), P(0.0, -0.10, "CM"),
                P(0.15, 0.0, "AM"), P(0.30, 0.0, "CF")]
        right = [P(-1.0, 0.0, "GK"), P(0.10, 0.05, "CB"), P(0.10, -0.05, "CB"),
                 P(-0.05, 0.15, "CM"), P(-0.05, -0.15, "CM")]
    return ScenarioSpec(
        name=name, ball=ball, left=left, right=right,
        game_duration=600, deterministic=True, offsides=False,
        end_on_score=False, end_on_out=False, end_on_possession_change=False,
        left_difficulty=0.6, right_difficulty=0.6,
    )


# Reveal order across both teams, outfield-first so keepers (L0/R0) stay hidden for counts
# <=8 and each shown count is balanced per team: 4 -> 2+2, 6 -> 3+3, 8 -> 4+4.
REVEAL = ["L1", "R1", "L2", "R2", "L3", "R3", "L4", "R4"]
ALL_SLOTS = [f"{t}{i}" for t in ("L", "R") for i in range(5)]   # L0..L4, R0..R4


def hide_slots_for(visible_count):
    """Comma list of (team,index) slots to HIDE so exactly `visible_count` remain shown."""
    visible = set(REVEAL[:visible_count])
    return ",".join(s for s in ALL_SLOTS if s not in visible)


# 5 clips per transition; each clip = (start_count, end_count).
PROFILES = [(4, 6), (6, 8), (8, 4)]
CLIPS_PER_PROFILE = 5

# ── geometry / timing (match the DRILLS crop; no grid) ──────────────────────────────
HUD_TOP, HUD_BOT, FRAME_H = 60, 180, 720
N_STEPS = 50           # 5.0 s @ 10 fps
FPS = 10
TRANSITION = 25        # midpoint: frames [0:25] show start count, [25:50] show end count
SEEDS_TRY = [42, 7, 11, 23, 3, 5, 17, 31, 55, 71, 13, 99, 101, 202, 2, 8, 19, 27]

OUT = HERE / "player_count_clips"


def render_play(level, seed, hide_slots):
    """Render `level` deterministically for N_STEPS with the given hidden slots.
    Returns (frames[list of HxWx3], balls[np Nx2]). Hiding is render-only."""
    import numpy as np
    from gfootball.env import config as cfg, football_env
    os.environ["GFOOTBALL_HIDE_SLOTS"] = hide_slots      # read live at env.reset()
    work = HERE / "_frames" / "_work"
    work.mkdir(parents=True, exist_ok=True)
    values = {
        "level": level, "players": [], "action_set": "full",
        "write_video": False, "dump_full_episodes": False, "dump_scores": False,
        "tracesdir": str(work), "real_time": False,
        "game_engine_random_seed": seed, "video_quality_level": 2,
        "display_game_stats": False,
    }
    env = football_env.FootballEnv(cfg.Config(values))
    env.render("rgb_array")
    obs = env.reset()
    frames, balls = [], []
    for _ in range(N_STEPS):
        obs, r, done, _ = env.step([])
        o = obs[0] if isinstance(obs, list) else obs
        frames.append(np.array(env.render("rgb_array"))[HUD_TOP:FRAME_H - HUD_BOT, :])
        balls.append(o["ball"][:2].copy())
        if done:
            break
    env.close()
    return frames, np.array(balls)


def _clean(balls):
    """Continuous play only: no teleport/goal/reset, no goal-line crossing (same rule as
    generate_drills.py)."""
    import numpy as np
    if len(balls) < N_STEPS:
        return False
    steps = np.linalg.norm(np.diff(balls, axis=0), axis=1)
    if steps.max() >= 0.35:
        return False
    if float(np.abs(balls[:, 0]).max()) > 1.0:
        return False
    return 0.12 < float(steps.sum()) < 4.0


def main():
    import numpy as np
    from lib import use_bundle, frames_to_mov
    use_bundle("noname")                      # ball visible, no player names
    OUT.mkdir(parents=True, exist_ok=True)

    rows = [("clip", "base", "seed", "start_count", "end_count", "transition_frame")]
    clip_num = 0
    for start_c, end_c in PROFILES:
        hide_start = hide_slots_for(start_c)
        hide_end = hide_slots_for(end_c)
        for k in range(CLIPS_PER_PROFILE):
            clip_num += 1
            base_idx = k % 3
            name = f"pc_{start_c}to{end_c}_{k + 1:02d}"
            level = write_scenario(base_spec(base_idx, name), force=True)

            # find a seed whose play is continuous (rendered with the START hide set)
            chosen = None
            for seed in SEEDS_TRY:
                fr_s, balls_s = render_play(level, seed, hide_start)
                if _clean(balls_s):
                    chosen = (seed, fr_s, balls_s)
                    break
            if chosen is None:
                print(f"  SKIP {name}: no continuous seed in {len(SEEDS_TRY)} tries")
                continue
            seed, fr_s, balls_s = chosen

            # same play, END hide set — identical motion (hiding is render-only)
            fr_e, balls_e = render_play(level, seed, hide_end)
            drift = float(np.abs(balls_s[:len(balls_e)] - balls_e[:len(balls_s)]).max())

            # splice: start count for [0:T], end count for [T:]
            spliced = list(fr_s[:TRANSITION]) + list(fr_e[TRANSITION:])
            frames_to_mov(spliced, OUT / f"{name}.mov", fps=FPS, crop_hud=False)

            rows.append((name, base_idx, seed, start_c, end_c, TRANSITION))
            print(f"  [{clip_num:2d}/15] {name} seed={seed} "
                  f"{start_c}->{end_c} players  (motion drift={drift:.4f})")

    csv = "\n".join(",".join(map(str, r)) for r in rows) + "\n"
    (OUT / "manifest.csv").write_text(csv)
    print(f"\nwrote {len(rows) - 1} clips -> {OUT}")


if __name__ == "__main__":
    main()
