"""Shared helpers for stimulus generation.

All generators import from here so the env/clip/asset-bundle plumbing
lives in one place.
"""

import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
BUNDLES = REPO / "experiments/asset_bundles"
DEFAULT_BUNDLE = BUNDLES / "default" / "data"

# Default clip length: 10 s at the engine's 10 fps == 100 env steps.
CLIP_STEPS = 100
FPS = 10


def _discover_engine_build() -> Optional[Path]:
    """Find a vendored gfootball build dir with a working engine binary.

    On Neha's Mac the engine is vendored at football/build/lib.macosx-*/.
    On Windows / fresh clones, this returns None — and we rely on
    `pip install gfootball` having put the engine on the normal Python path.
    """
    build_root = REPO / "football/build"
    if not build_root.exists():
        return None
    for c in sorted(build_root.glob("lib.*"), reverse=True):  # newest cpython first
        engine_dir = c / "gfootball_engine"
        if not engine_dir.exists():
            continue
        # Require an actual compiled binary, not just the copied source tree.
        has_binary = (
            any(engine_dir.glob("_gameplayfootball*.so"))
            or any(engine_dir.glob("_gameplayfootball*.pyd"))
            or any(c.glob("brainball_cpp_engine*"))
        )
        if has_binary:
            return c
    return None


ENGINE_BUILD = _discover_engine_build()


def gfootball_engine_data_dir() -> Path:
    """Locate the engine's stock `data/` directory (used as the source for
    asset bundles). Works on both vendored builds and pip-installed gfootball."""
    if ENGINE_BUILD is not None:
        p = ENGINE_BUILD / "gfootball_engine/data"
        if p.exists():
            return p
    # Pip-installed: gfootball_engine ships data/ inside the package.
    import importlib.util
    spec = importlib.util.find_spec("gfootball_engine")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Cannot find gfootball_engine. Either vendor a build under "
            "football/build/lib.*/ or `pip install gfootball`."
        )
    return Path(spec.origin).parent / "data"


def use_bundle(bundle: str = "default") -> Path:
    """Activate the named asset bundle for this process. Call BEFORE import gfootball."""
    p = BUNDLES / bundle / "data"
    if not p.exists():
        raise FileNotFoundError(f"bundle not built: {p}. Run build_bundles.py.")
    os.environ["GFOOTBALL_DATA_DIR"] = str(p)
    if ENGINE_BUILD is not None and str(ENGINE_BUILD) not in sys.path:
        sys.path.insert(0, str(ENGINE_BUILD))
    return p


def make_env(
    level: str,
    seed: int,
    out_dir: Optional[Path] = None,
    write_video: bool = True,
    hud: bool = False,
):
    """Build a bot-vs-bot football env with HUD off by default.

    Only one bot per side gets a controllable slot — this hides the on-field
    player-name captions for everyone else (those appear only for externally
    controllable players in the C++ engine).
    """
    from gfootball.env import config as cfg
    from gfootball.env import football_env

    values = {
        "level": level,
        # exactly 2 controllable slots (one bot per side) so we get clean visuals
        "players": [
            "bot:left_players=1,right_players=0",
            "bot:left_players=0,right_players=1",
        ],
        "action_set": "full",
        "write_video": write_video,
        "dump_full_episodes": True,
        "dump_scores": False,
        "tracesdir": str(out_dir) if out_dir else "/tmp/gfootball",
        "real_time": False,
        "game_engine_random_seed": seed,
        "video_quality_level": 2,
        "display_game_stats": hud,
    }
    c = cfg.Config(values)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    env = football_env.FootballEnv(c)
    env.render("rgb_array")
    return env


def run_clip(env, steps: int = CLIP_STEPS, dump_name: str = "clip"):
    """Drive the env for `steps` steps. Writes .avi and .dump under env.tracesdir."""
    env.reset()
    last_obs = None
    done = False
    for i in range(steps):
        if done:
            break
        obs, _, done, _ = env.step([])
        last_obs = obs
    if not done:
        env.write_dump(dump_name)
    return {"steps_run": i + (0 if done else 1), "done": done, "obs": last_obs}


def read_dump(dump_path: Path):
    """Iterator over (frame_idx, observation) from a gfootball .dump file."""
    with open(dump_path, "rb") as f:
        idx = 0
        while True:
            try:
                step = pickle.load(f)
            except EOFError:
                break
            yield idx, step
            idx += 1


# Engine still renders a score/time strip at the top and a radar + sticky-action
# panel at the bottom even with display_game_stats=False. We crop those out in
# post so the stimulus is just the playing field.
# Calibrated from sample 1280x720 frames.
HUD_TOP_PX = 60
HUD_BOTTOM_PX = 180


def _hud_crop_filter(w: int = 1280, h: int = 720) -> str:
    return f"crop={w}:{h - HUD_TOP_PX - HUD_BOTTOM_PX}:0:{HUD_TOP_PX}"


def avi_to_mov(avi_path: Path, mov_path: Optional[Path] = None,
               crop_hud: bool = True) -> Path:
    """Transcode AVI (mjpeg, big) → MOV (H.264, fast-start). HUD-cropped by default."""
    mov_path = mov_path or avi_path.with_suffix(".mov")
    vf_args = ["-vf", _hud_crop_filter()] if crop_hud else []
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(avi_path),
            *vf_args,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(mov_path),
        ],
        check=True,
    )
    return mov_path


def extract_frame(avi_path: Path, frame_idx: int, out_png: Path,
                  crop_hud: bool = True) -> Path:
    """Pull one frame (0-indexed) from a video file as PNG. HUD-cropped by default."""
    vf = f"select=eq(n\\,{frame_idx})"
    if crop_hud:
        vf += "," + _hud_crop_filter()
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(avi_path),
            "-vf", vf,
            "-vframes", "1",
            str(out_png),
        ],
        check=True,
    )
    return out_png


def latest_avi(dir_path: Path) -> Path:
    """Most recent .avi in a directory (write_dump timestamps them)."""
    avis = sorted(dir_path.glob("*.avi"), key=lambda p: p.stat().st_mtime)
    if not avis:
        raise FileNotFoundError(f"no .avi in {dir_path}")
    return avis[-1]


def latest_dump(dir_path: Path) -> Path:
    dumps = sorted(dir_path.glob("*.dump"), key=lambda p: p.stat().st_mtime)
    if not dumps:
        raise FileNotFoundError(f"no .dump in {dir_path}")
    return dumps[-1]


# Scenario-coordinate → pixel mapping for 1280x720 top-down render.
# These are approximate, calibrated from a few sample frames. Refine if needed.
PITCH_X_RANGE = (-1.0, 1.0)
PITCH_Y_RANGE = (-0.42, 0.42)
FRAME_W = 1280
FRAME_H = 720
# Visible play region in the rendered frame (camera shows some margin).
FRAME_X_PAD = 60   # pixels from left/right edge to goal lines
FRAME_Y_PAD = 80   # pixels from top/bottom edge to sidelines


def pitch_to_pixel(x: float, y: float) -> tuple:
    """Map gfootball pitch coords to pixel coords in the rendered video frame."""
    px = FRAME_X_PAD + (x - PITCH_X_RANGE[0]) / (PITCH_X_RANGE[1] - PITCH_X_RANGE[0]) \
        * (FRAME_W - 2 * FRAME_X_PAD)
    py = FRAME_Y_PAD + (y - PITCH_Y_RANGE[0]) / (PITCH_Y_RANGE[1] - PITCH_Y_RANGE[0]) \
        * (FRAME_H - 2 * FRAME_Y_PAD)
    return int(px), int(py)
