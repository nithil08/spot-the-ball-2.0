"""Plain scenario videos for C:\\Users\\nithi\\OneDrive\\Desktop\\Nithil Research\\UPDATED.

No ground truth is computed here -- the user will produce grid-cell ground
truth themselves from the with-ball video. Per scenario we write three
videos:

  full_clip_with_ball.mp4      ball VISIBLE,   grid burned in
  full_clip_without_ball.mp4   ball INVISIBLE, grid burned in
  short_clip_without_ball.mp4  the without-ball clip, genuinely ffmpeg-trimmed
                                 (real cut of the encoded video, NOT a separate
                                 render with a different frame count) down to
                                 the first SHORT_S seconds

No fps/step-count calibration: render_worker.py runs the engine with
`real_time=True` and just lets it play for a genuine FULL_S-second
wall-clock window, recording whatever frames come out at whatever native
rate that is (varies slightly run to run, e.g. ~42 frames / ~8.4fps for a
5s window -- that's expected and fine). Each video is encoded at its own
measured native fps so file duration matches real elapsed capture time.

(Earlier attempts here tried to hand-calibrate physics_steps_per_frame
against the engine's in-game clock to force an exact frame count/fps --
that produced a correct-on-paper but very choppy, unnatural-looking
result. This is simpler and looks right: just record real time.)

Grid: reuses the 32x12 square-cell (40x40px) grid overlay already built in
gen_ball_hidden_grid_video.py, burned into every frame of both with-ball and
without-ball videos so cell labels line up across all three files.

Scenario/seed choices (same as before -- these reliably survive well past
5 real seconds of play):
  academy_3_vs_1_with_keeper              seed=23
  academy_counterattack_easy              seed=331
  academy_counterattack_hard              seed=3
  academy_run_pass_and_shoot_with_keeper  seed=3
  5_vs_5                                  seed=11
  5_vs_5                                  seed=47
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lib
from lib import frames_to_mov

from gen_ball_hidden_grid_video import crop_frame, burn_grid
from gen_prediction import trim_clip

WORKER = Path(__file__).resolve().parent / "render_worker.py"

OUT = Path(__file__).resolve().parents[3] / "UPDATED"

FULL_S = 5.0
SHORT_S = 4.0

SCENARIOS = [
    ("academy_3_vs_1_with_keeper", 23),
    ("academy_counterattack_easy", 331),
    ("academy_counterattack_hard", 3),
    ("academy_run_pass_and_shoot_with_keeper", 3),
    ("5_vs_5", 11),
    ("5_vs_5", 47),
]


def render(level: str, seed: int, bundle: str, work_dir: Path):
    """Render in a fresh subprocess -- the gfootball engine only reads
    GFOOTBALL_DATA_DIR once per process, so switching bundles in-process
    silently reuses whichever bundle loaded first (verified: same-process
    default vs ball_tiny produced byte-identical frames).

    Returns (frames, native_fps) -- native_fps is whatever the real-time
    capture actually measured, not a precomputed value.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    out_npy = work_dir / "frames.npy"
    result = subprocess.run(
        [sys.executable, str(WORKER), level, str(seed), bundle, str(FULL_S), str(out_npy)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not out_npy.exists():
        raise RuntimeError(
            f"{level} seed={seed} bundle={bundle}: render_worker failed\n"
            f"stdout={result.stdout}\nstderr={result.stderr[-2000:]}"
        )
    frames = np.load(out_npy)
    native_fps = float(out_npy.with_suffix(".fps.txt").read_text())
    return list(frames), native_fps


def make_scenario(idx: int, level: str, seed: int):
    sdir = OUT / f"scenario_{idx}"
    sdir.mkdir(parents=True, exist_ok=True)
    print(f"scenario_{idx}: {level} seed={seed}")

    frames_visible, fps_visible = render(level, seed, "default", sdir / "_work_with_ball")
    frames_hidden, fps_hidden = render(level, seed, "ball_tiny", sdir / "_work_without_ball")

    visible_gridded = [burn_grid(crop_frame(f)) for f in frames_visible]
    hidden_gridded = [burn_grid(crop_frame(f)) for f in frames_hidden]

    with_ball_mp4 = sdir / "full_clip_with_ball.mp4"
    without_ball_mp4 = sdir / "full_clip_without_ball.mp4"
    short_mp4 = sdir / "short_clip_without_ball.mp4"

    frames_to_mov(visible_gridded, with_ball_mp4, fps=fps_visible, crop_hud=False)
    frames_to_mov(hidden_gridded, without_ball_mp4, fps=fps_hidden, crop_hud=False)
    trim_clip(without_ball_mp4, short_mp4, duration_s=SHORT_S)

    shutil.rmtree(sdir / "_work_with_ball", ignore_errors=True)
    shutil.rmtree(sdir / "_work_without_ball", ignore_errors=True)

    print(f"  {with_ball_mp4.name}: {len(visible_gridded)} frames @ {fps_visible:.2f}fps = {len(visible_gridded)/fps_visible:.2f}s")
    print(f"  {without_ball_mp4.name}: {len(hidden_gridded)} frames @ {fps_hidden:.2f}fps = {len(hidden_gridded)/fps_hidden:.2f}s")
    print(f"  {short_mp4.name}: trimmed to {SHORT_S}s")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (level, seed) in enumerate(SCENARIOS, start=1):
        make_scenario(i, level, seed)
    print(f"\ndone -> {OUT}")


if __name__ == "__main__":
    main()
