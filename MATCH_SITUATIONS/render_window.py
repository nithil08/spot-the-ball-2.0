"""render_window.py — phase 2: re-run a scanned match and keep only a window of frames.

The match is deterministic, so replaying the same (scenario, seed) reproduces the exact
play that scan_events.py analysed; we simply throw away every frame outside [start, end)
and write the window out as a clip. Frames are cached as .npz so composing / re-cutting
is free.

Usage (as a library):
    from render_window import render_window
    frames, log = render_window(level, seed, start, end)

CLI:  python3 render_window.py <level> <seed> <start> <end> <out.mov> [--png]
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

HUD_TOP, HUD_BOT, FRAME_H = 60, 180, 720
FPS = 10


def render_window(level, seed, start, end, hide_slots=""):
    """Play `level` deterministically; return (frames[start:end], log[start:end])."""
    import numpy as np
    from gfootball.env import config as cfg, football_env
    os.environ["GFOOTBALL_HIDE_SLOTS"] = hide_slots
    # The engine keeps a second font handle that the bundle swap misses, so point it at
    # the bundle's blanked font too — otherwise player-name captions leak back in.
    bundle = os.environ.get("GFOOTBALL_DATA_DIR", "")
    blank = Path(bundle) / "media/fonts/alegreya/AlegreyaSansSC-ExtraBold.ttf"
    if blank.exists():
        os.environ["GFOOTBALL_FONT"] = str(blank)
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
    frames, log = [], []
    for i in range(end):
        obs, r, done, _ = env.step([])
        o = obs[0] if isinstance(obs, list) else obs
        if i >= start:
            frames.append(np.array(env.render("rgb_array"))[HUD_TOP:FRAME_H - HUD_BOT, :])
            log.append({
                "ball": np.array(o["ball"], dtype=float),
                "ball_owned_team": int(o["ball_owned_team"]),
                "ball_owned_player": int(o["ball_owned_player"]),
                "game_mode": int(o["game_mode"]),
                "left_team": np.array(o["left_team"], dtype=float),
                "right_team": np.array(o["right_team"], dtype=float),
            })
        if done:
            break
    env.close()
    return frames, log


def contact_sheet(frames, out_png, every=6, cols=5, label_from=0):
    """Save a grid of every Nth frame so a clip can be eyeballed without playing it."""
    from PIL import Image, ImageDraw
    import numpy as np
    picks = list(range(0, len(frames), every))
    rows = (len(picks) + cols - 1) // cols
    h, w = frames[0].shape[:2]
    sw, sh = w // 2, h // 2
    sheet = Image.new("RGB", (cols * sw, rows * sh), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    for n, idx in enumerate(picks):
        im = Image.fromarray(frames[idx]).resize((sw, sh))
        x, y = (n % cols) * sw, (n // cols) * sh
        sheet.paste(im, (x, y))
        dr.text((x + 4, y + 4), f"f{label_from + idx}", fill=(255, 255, 0))
    sheet.save(out_png)
    return out_png


if __name__ == "__main__":
    from lib import use_bundle, frames_to_mov
    use_bundle("noname")
    level, seed, start, end, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), \
        int(sys.argv[4]), sys.argv[5]
    frames, log = render_window(level, seed, start, end)
    frames_to_mov(frames, Path(out), fps=FPS, crop_hud=False)
    print(f"wrote {out}: {len(frames)} frames")
    if "--png" in sys.argv:
        contact_sheet(frames, Path(out).with_suffix(".sheet.png"), label_from=start)
        print("wrote contact sheet")
