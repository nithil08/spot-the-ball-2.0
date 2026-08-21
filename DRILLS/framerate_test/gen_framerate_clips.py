"""gen_framerate_clips.py — 3 normal 11v11 clips, each at 3 frame rates (5 / 10 / 20 fps).

Purpose: pick a final frame rate for the benchmark. These are the OG-style natural 11v11
clips (11_vs_11_stochastic, goal-free window, moderate movement, ball visible throughout,
32x12 yellow grid, noname bundle, tracking camera) — NOT the small-sided drills.

To keep the comparison fair, each clip is rendered ONCE at the 20 fps master
(physics_steps_per_frame = 100/fps = 5), then the 10 fps and 5 fps versions are made by
dropping frames. So the gameplay is byte-for-byte identical across the three rates and the
ONLY difference is motion smoothness. All three play back over the same 5.0 s.

  fps 20 -> 100 frames (master)        psf=5,  real engine temporal resolution
  fps 10 -> every 2nd frame (50)       matches the current default rate
  fps  5 -> every 4th frame (25)       choppier / smaller files

Output: DRILLS/framerate_test/clipN_{20,10,05}fps.mov  (9 movs) + notes below.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = "/Users/nithilbalamurugan/gfootball_src"
# Repo root = the nearest ancestor holding experiments/; keeps working if the
# checkout is moved (it was, from code/spot-the-ball-2.0 to the Desktop root).
EXP = str(next(p for p in Path(__file__).resolve().parents
               if (p / "experiments").is_dir()) / "experiments")
for _p in (SRC, SRC + "/third_party", EXP, EXP + "/nithil_work"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 16x6 grid — the locked-in benchmark grid (env must be set before importing G).
os.environ["GRID_COLS"] = "16"
os.environ["GRID_ROWS"] = "6"

# Activate the noname bundle (no player-name captions, ball visible, ref handled) and
# point the engine at the blanked font BEFORE gfootball is ever imported — otherwise
# names leak in. Everything below (make_env) then just builds the env.
from lib import use_bundle, frames_to_mov  # noqa: E402
use_bundle("noname")
_bundle = os.environ.get("GFOOTBALL_DATA_DIR", "")
_blank = Path(_bundle) / "media/fonts/alegreya/AlegreyaSansSC-ExtraBold.ttf"
if _blank.exists():
    os.environ["GFOOTBALL_FONT"] = str(_blank)

MASTER_FPS = 20
RATES = [20, 10, 5]                      # master + frame-dropped derivatives
STEP = {r: MASTER_FPS // r for r in RATES}   # 20->1, 10->2, 5->4
SEC = 5.0
WARMUP = MASTER_FPS                      # 1.0 s to skip the static kick-off formation
N = int(SEC * MASTER_FPS)               # 100 master frames
SEEDS = [42, 7, 11, 23, 3, 5, 17, 31]
N_CLIPS = 3
MOVE_LOW, MOVE_HIGH = 0.25, 1.20        # moderate ball movement over the window
HUD_TOP, HUD_BOT, FRAME_H = 60, 180, 720


def make_env(seed):
    from gfootball.env import config as cfg, football_env
    work = HERE / "_work"
    work.mkdir(parents=True, exist_ok=True)
    values = {
        "level": "11_vs_11_stochastic", "players": [], "action_set": "full",
        "write_video": False, "dump_full_episodes": False, "dump_scores": False,
        "tracesdir": str(work), "real_time": False, "game_engine_random_seed": seed,
        "video_quality_level": 2, "display_game_stats": False,
        "physics_steps_per_frame": int(100 / MASTER_FPS),
    }
    env = football_env.FootballEnv(cfg.Config(values))
    env.render("rgb_array")
    return env


def capture(seed):
    """Warm up then grab N master frames. Return (frames, pathlen, goal)."""
    import numpy as np
    env = make_env(seed)
    env.reset()
    frames, balls, goal = [], [], False
    for step in range(WARMUP + N):
        obs, r, done, _ = env.step([])
        o = obs[0] if isinstance(obs, list) else obs
        if float(np.sum(np.asarray(r))) != 0.0:
            goal = True
        fr = np.array(env.render("rgb_array"))[HUD_TOP:FRAME_H - HUD_BOT, :]
        if step >= WARMUP:
            frames.append(fr)
            balls.append(o["ball"][:2].copy())
        if done:
            break
    env.close()
    balls = np.array(balls)
    path = float(np.linalg.norm(np.diff(balls, axis=0), axis=1).sum()) if len(balls) > 1 else 0.0
    return frames, path, goal


def main():
    import gen_natural_30_clips as G
    overlay = G.build_grid()
    kept = 0
    notes = ["clip,seed,pathlen,rates_fps,frames_per_rate"]
    for seed in SEEDS:
        if kept >= N_CLIPS:
            break
        frames, path, goal = capture(seed)
        if goal or not (MOVE_LOW <= path <= MOVE_HIGH) or len(frames) < N:
            print(f"  seed {seed}: skip (goal={goal} path={path:.3f} n={len(frames)})")
            continue
        kept += 1
        gframes = [G.burn_grid(f, overlay) for f in frames]        # burn grid once
        for r in RATES:
            sub = gframes[::STEP[r]]
            name = f"clip{kept}_{r:02d}fps.mov"
            frames_to_mov(sub, HERE / name, fps=r, crop_hud=False)
            print(f"  clip{kept} seed={seed} {r:>2}fps -> {len(sub)} frames  {name}")
        notes.append(f"{kept},{seed},{round(path,3)},{'/'.join(map(str,RATES))},"
                     f"{'/'.join(str(len(gframes[::STEP[r]])) for r in RATES)}")
    (HERE / "clips_index.csv").write_text("\n".join(notes) + "\n")
    print(f"\nwrote {kept} clips x {len(RATES)} rates = {kept*len(RATES)} movs -> {HERE}")
    if kept < N_CLIPS:
        print(f"WARNING: only {kept}/{N_CLIPS} clips — add more SEEDS.")


if __name__ == "__main__":
    main()
