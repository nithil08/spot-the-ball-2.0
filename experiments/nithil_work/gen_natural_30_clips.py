"""gen_natural_30_clips.py — 30 natural 5-second passing clips, 32x12 grid, ball visible.

Follows the user's "game rules" for natural gameplay clips:
  * 11_vs_11_stochastic, continuous play, NO goal scoring (goal-free windows only).
  * Engine's standard ball-tracking camera (the "og" render) — NOT wide/half crops.
  * Moderate-movement windows (steady passing/possession), not frantic, not static.
  * noname bundle: ball VISIBLE at normal size, no player-name captions.
  * 32 x 12 yellow alphanumeric grid burned on every frame.
  * NO visibility variants — one clip per window, ball visible throughout.

30 clips are cut from several goal-free seeds of one continuous match each. Each
seed's run is sliced into back-to-back non-overlapping 50-frame windows (5.0 s @
10 fps); a window is kept only if it has no goal and moderate ball movement.

Modes (bundle must be chosen BEFORE importing gfootball):
  visible     -> select 30 windows, write grid clips, save visible final frames,
                 write manifest.json
  invisible   -> read manifest, replay each seed with the ball made invisible,
                 save the invisible final frame for every clip (for ground truth)
  groundtruth -> no engine: diff visible vs invisible final frames -> ball pixel ->
                 grid cell; write ground_truth.{json,csv,md} + _debug markers

Output: results/natural_30_5s_grid/
"""

import os
import sys
import json
from pathlib import Path

OUT_NAME = os.environ.get("OUT_NAME", "natural_30_5s_grid")

# ── geometry (1280x480 HUD-cropped render; grid configurable via env) ──────────
W, H = 1280, 480
COLS = int(os.environ.get("GRID_COLS", 32))   # default 32 x 12
ROWS = int(os.environ.get("GRID_ROWS", 12))
CELL_W = W / COLS
CELL_H = H / ROWS
STEPS = 50                                 # 50 frames @ 10 fps = 5.0 s
FPS = 10

LEVEL = "11_vs_11_stochastic"
WARMUP = 10                                # skip the static kick-off formation
MAX_WINDOWS_SCAN = 8                       # windows probed per seed
CAP_PER_SEED = 6                           # keep at most this many per seed (variety)
MOVE_LOW, MOVE_HIGH = 0.33, 0.95           # moderate-movement band (ball path length)
TARGET = 30
SEEDS = [42, 7, 11, 23, 3, 5, 17, 31, 55, 71]

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"

# HUD crop (from lib.py): top 60px, bottom 180px on the 1280x720 render -> 480 rows
HUD_TOP_PX, HUD_BOTTOM_PX, FRAME_H = 60, 180, 720


def _paths():
    here = Path(__file__).resolve().parent
    out = here / "results" / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    return here, out


# ── grid overlay ──────────────────────────────────────────────────────────────

def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def build_grid():
    from PIL import Image, ImageDraw
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line = (255, 255, 0, 130)
    for c in range(COLS + 1):
        x = int(round(c * CELL_W))
        draw.line([(x, 0), (x, H)], fill=line, width=1)
    for r in range(ROWS + 1):
        y = int(round(r * CELL_H))
        draw.line([(0, y), (W, y)], fill=line, width=1)
    font = _font(11)
    for c in range(COLS):
        draw.text((int(round(c * CELL_W)) + 2, 0), str(c + 1),
                  fill=(255, 255, 0, 255), font=font)
    for r in range(ROWS):
        draw.text((1, int(round(r * CELL_H)) + 1), chr(ord("A") + r),
                  fill=(255, 255, 0, 255), font=font)
    return overlay


def burn_grid(frame_rgb, overlay):
    from PIL import Image
    import numpy as np
    base = Image.fromarray(frame_rgb).convert("RGBA")
    return np.array(Image.alpha_composite(base, overlay).convert("RGB"))


def cell_of(px, py):
    """Pixel (in cropped 1280x480 space) -> grid cell label like 'F17'."""
    c = min(max(int(px // CELL_W), 0), COLS - 1)
    r = min(max(int(py // CELL_H), 0), ROWS - 1)
    return f"{chr(ord('A') + r)}{c + 1}", r, c


# ── engine ────────────────────────────────────────────────────────────────────

def make_env(seed, work_dir):
    import sys as _sys
    for p in ["/Users/nithilbalamurugan/gfootball_src",
              "/Users/nithilbalamurugan/gfootball_src/third_party"]:
        if p not in _sys.path:
            _sys.path.insert(0, p)
    bundle = os.environ.get("GFOOTBALL_DATA_DIR", "")
    blank = Path(bundle) / "media/fonts/alegreya/AlegreyaSansSC-ExtraBold.ttf"
    if blank.exists():
        os.environ["GFOOTBALL_FONT"] = str(blank)
    from gfootball.env import config as cfg, football_env
    values = {
        "level": LEVEL, "players": [], "action_set": "full",
        "write_video": False, "dump_full_episodes": False, "dump_scores": False,
        "tracesdir": str(work_dir), "real_time": False,
        "game_engine_random_seed": seed, "video_quality_level": 2,
        "display_game_stats": False,
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    env = football_env.FootballEnv(cfg.Config(values))
    env.render("rgb_array")
    return env


def crop(frame):
    import numpy as np
    return np.array(frame)[HUD_TOP_PX: FRAME_H - HUD_BOTTOM_PX, :]


def run_seed(seed, work_dir, want_starts=None):
    """Play one continuous match; return dict window_start -> (frames, balls, goal).

    If want_starts is given, only those window starts are materialised (frames kept);
    otherwise every scanned window is returned.
    """
    import numpy as np
    env = make_env(seed, work_dir)
    obs = env.reset()
    windows = {}
    n_windows = MAX_WINDOWS_SCAN if want_starts is None else (max(
        (s - WARMUP) // STEPS for s in want_starts) + 1)
    end = WARMUP + n_windows * STEPS
    buf_frames, buf_balls, goal = [], [], False
    cur_start = WARMUP
    for step in range(end):
        obs, r, done, _ = env.step([])
        o = obs[0] if isinstance(obs, list) else obs
        if step < WARMUP:
            continue
        if float(np.sum(np.asarray(r))) != 0.0:
            goal = True
        keep = (want_starts is None) or (cur_start in want_starts)
        if keep:
            buf_frames.append(crop(env.render("rgb_array")))
        else:
            env.render("rgb_array")  # keep renderer in sync every step
        buf_balls.append(o["ball"][:2].copy())
        if len(buf_balls) == STEPS:
            windows[cur_start] = (buf_frames if keep else None,
                                  np.array(buf_balls), goal)
            buf_frames, buf_balls, goal = [], [], False
            cur_start += STEPS
        if done:
            break
    env.close()
    return windows


def movement(balls):
    import numpy as np
    return float(np.sum(np.linalg.norm(np.diff(balls, axis=0), axis=1)))


# ── mode: visible ─────────────────────────────────────────────────────────────

def mode_visible():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lib import use_bundle, frames_to_mov
    use_bundle("noname")
    _, out = _paths()
    frames_dir = out / "_final_frames"
    frames_dir.mkdir(exist_ok=True)
    from PIL import Image
    import numpy as np

    overlay = build_grid()
    manifest, clip_no = [], 0

    for seed in SEEDS:
        if clip_no >= TARGET:
            break
        wins = run_seed(seed, out / "_work")
        kept = 0
        for start in sorted(wins):
            if clip_no >= TARGET or kept >= CAP_PER_SEED:
                break
            frames, balls, goal = wins[start]
            mv = movement(balls)
            if goal or not (MOVE_LOW <= mv <= MOVE_HIGH):
                continue
            clip_no += 1
            kept += 1
            gframes = [burn_grid(f, overlay) for f in frames]
            name = f"clip_{clip_no:02d}.mov"
            frames_to_mov(gframes, out / name, fps=FPS, crop_hud=False)
            # save the visible final frame (cropped, pre-grid) for ground-truth diff
            Image.fromarray(frames[-1]).save(frames_dir / f"clip_{clip_no:02d}_vis.png")
            manifest.append({
                "clip": clip_no, "file": name, "seed": seed,
                "window_start": start, "final_step": start + STEPS - 1,
                "movement": round(mv, 4),
            })
            print(f"  clip_{clip_no:02d}  seed={seed} start={start} mv={mv:.3f}  {name}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nvisible: wrote {clip_no} clips + manifest -> {out}")
    if clip_no < TARGET:
        print(f"WARNING: only {clip_no}/{TARGET} clips — add more SEEDS.")


# ── mode: invisible ───────────────────────────────────────────────────────────

def mode_invisible():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lib import use_bundle
    use_bundle("noname_ball_invisible")
    _, out = _paths()
    frames_dir = out / "_final_frames"
    from PIL import Image

    manifest = json.loads((out / "manifest.json").read_text())
    by_seed = {}
    for m in manifest:
        by_seed.setdefault(m["seed"], []).append(m)

    for seed, items in by_seed.items():
        starts = {m["window_start"] for m in items}
        wins = run_seed(seed, out / "_work_inv", want_starts=starts)
        for m in items:
            frames, _, _ = wins[m["window_start"]]
            Image.fromarray(frames[-1]).save(
                frames_dir / f"clip_{m['clip']:02d}_inv.png")
            print(f"  clip_{m['clip']:02d}  seed={seed} start={m['window_start']}  invisible final saved")
    print(f"\ninvisible: saved final frames -> {frames_dir}")


# ── mode: groundtruth (no engine) ─────────────────────────────────────────────

def detect_ball(vis, inv):
    """Return (px, py) of the ball in the cropped frame via visible-vs-invisible diff."""
    import numpy as np
    v = vis.astype(np.int16)
    i = inv.astype(np.int16)
    d = np.abs(v - i).sum(axis=2)                 # per-pixel change
    thr = max(40, d.max() * 0.35)
    mask = d > thr
    if mask.sum() == 0:
        mask = d > (d.max() * 0.5)
    bright = vis.astype(np.int32).sum(axis=2)     # ball is near-white
    # among changed pixels, prefer the bright ball highlight over its dark shadow
    ballmask = mask & (bright > 600)
    if ballmask.sum() < 3:
        ballmask = mask
    ys, xs = np.nonzero(ballmask)
    w = d[ys, xs].astype(float)
    px = float(np.average(xs, weights=w))
    py = float(np.average(ys, weights=w))
    return px, py, int(mask.sum())


def mode_groundtruth():
    from PIL import Image, ImageDraw
    import numpy as np
    _, out = _paths()
    frames_dir = out / "_final_frames"
    debug_dir = out / "_debug"
    debug_dir.mkdir(exist_ok=True)
    manifest = json.loads((out / "manifest.json").read_text())
    overlay = build_grid()

    rows = []
    for m in manifest:
        cid = m["clip"]
        vis = np.array(Image.open(frames_dir / f"clip_{cid:02d}_vis.png").convert("RGB"))
        inv = np.array(Image.open(frames_dir / f"clip_{cid:02d}_inv.png").convert("RGB"))
        px, py, npix = detect_ball(vis, inv)
        cell, r, c = cell_of(px, py)
        rows.append({
            "clip": cid, "file": m["file"], "seed": m["seed"],
            "ball_cell": cell, "grid_row": chr(ord('A') + r), "grid_col": c + 1,
            "px": round(px, 1), "py": round(py, 1), "changed_pixels": npix,
        })
        # debug: grid frame with a red crosshair at the detected ball
        dbg = Image.fromarray(burn_grid(vis, overlay))
        dr = ImageDraw.Draw(dbg)
        dr.line([(px - 12, py), (px + 12, py)], fill=(255, 0, 0), width=2)
        dr.line([(px, py - 12), (px, py + 12)], fill=(255, 0, 0), width=2)
        dr.text((px + 8, py + 8), cell, fill=(255, 0, 0), font=_font(16))
        dbg.save(debug_dir / f"clip_{cid:02d}_gt.png")
        print(f"  clip_{cid:02d}  ball -> {cell}  (px={px:.0f},py={py:.0f}, dpix={npix})")

    (out / "ground_truth.json").write_text(json.dumps(rows, indent=2))
    # csv
    csv = ["clip,file,seed,ball_cell,grid_row,grid_col,px,py"]
    for r in rows:
        csv.append(f"{r['clip']},{r['file']},{r['seed']},{r['ball_cell']},"
                   f"{r['grid_row']},{r['grid_col']},{r['px']},{r['py']}")
    (out / "ground_truth.csv").write_text("\n".join(csv) + "\n")
    # markdown
    md = ["# Ground truth — ball location on final frame (32x12 grid)", "",
          "Grid: columns 1-32 left->right, rows A-L top->bottom. Frame 1280x480.", "",
          "| Clip | Seed | Ball cell | Row | Col | px | py |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['clip']:02d} | {r['seed']} | **{r['ball_cell']}** | "
                  f"{r['grid_row']} | {r['grid_col']} | {r['px']} | {r['py']} |")
    (out / "ground_truth.md").write_text("\n".join(md) + "\n")
    print(f"\ngroundtruth: wrote ground_truth.(json|csv|md) + _debug -> {out}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "visible"
    {"visible": mode_visible, "invisible": mode_invisible,
     "groundtruth": mode_groundtruth}[mode]()
