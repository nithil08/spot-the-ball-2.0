"""Generate 20-second human-sampler clips across all 4 visual modes.

One clip per asset bundle (default, ball_tiny, ball_transparent, uniform_jerseys),
same scenario and seed so physics are identical across modes -- humans can be shown
any mode and the ground truth ball trajectory is the same.

Output: HUMAN_SAMPLER/<scenario>/<mode>.mp4  (HUD-cropped, H.264)

Usage:
  python gen_human_sampler_clips.py

Writes to <repo-root>/HUMAN_SAMPLER/.
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parent.parent
REPO = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))

from lib import frames_to_mov

WORKER = Path(__file__).resolve().parent / "render_worker.py"
OUT = REPO / "HUMAN_SAMPLER"
DURATION_S = 20.0

MODES = ["default", "ball_tiny", "ball_transparent", "uniform_jerseys"]

SCENARIOS = [
    ("academy_3_vs_1_with_keeper", 23),
    ("academy_counterattack_easy", 331),
    ("academy_counterattack_hard", 3),
    ("academy_run_pass_and_shoot_with_keeper", 3),
]


def render(level: str, seed: int, bundle: str, work_dir: Path) -> tuple[list, float]:
    work_dir.mkdir(parents=True, exist_ok=True)
    out_npy = work_dir / "frames.npy"
    result = subprocess.run(
        [
            sys.executable, str(WORKER),
            level, str(seed), bundle, str(DURATION_S), str(out_npy),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not out_npy.exists():
        raise RuntimeError(
            f"{level} seed={seed} bundle={bundle} failed\n"
            f"stdout={result.stdout}\nstderr={result.stderr[-2000:]}"
        )
    frames = list(np.load(out_npy))
    fps = float(out_npy.with_suffix(".fps.txt").read_text())
    return frames, fps


def make_scenario(idx: int, level: str, seed: int):
    sdir = OUT / f"scenario_{idx}_{level}"
    sdir.mkdir(parents=True, exist_ok=True)
    print(f"\nscenario_{idx}: {level}  seed={seed}")

    for mode in MODES:
        mp4 = sdir / f"{mode}.mp4"
        if mp4.exists():
            print(f"  {mode}: already exists, skipping")
            continue

        work = sdir / f"_work_{mode}"
        t0 = time.time()
        print(f"  {mode}: rendering...", end="", flush=True)
        frames, fps = render(level, seed, mode, work)
        elapsed = time.time() - t0

        frames_to_mov(frames, mp4, fps=round(fps), crop_hud=True)

        # clean up raw numpy cache
        for f in work.glob("*"):
            f.unlink()
        work.rmdir()

        print(f" {len(frames)} frames @ {fps:.1f}fps = {len(frames)/fps:.1f}s  [{elapsed:.0f}s wall]")
        print(f"    -> {mp4.relative_to(REPO)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (level, seed) in enumerate(SCENARIOS, start=1):
        make_scenario(i, level, seed)
    print(f"\ndone -> {OUT}")


if __name__ == "__main__":
    main()
