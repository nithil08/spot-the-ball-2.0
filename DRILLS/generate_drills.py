"""generate_drills.py — render all 36 drill clips (6 drills x 3 variants x 2 versions).

For every (drill, variant) we deterministically render the SAME play twice — once with
the ball VISIBLE (noname bundle) and once with the ball INVISIBLE (noname_ball_invisible).
Diffing the two isolates the exact ball pixel (ground truth) with no engine hacks.

Two clips are composed per variant, both on a 16x6 yellow grid with a RED CIRCLE around
the ball's start on the VERY FIRST FRAME:

  * <name>_visible.mov  — ball visible for all 5 s.
  * <name>_split.mov    — ball visible 1 s (10 frames) then invisible 4 s (40 frames);
                          the model must infer the final-frame cell.

Because the engine caches its asset bundle at import, this runs in three phases (like
gen_30_1s4s.py). Run all three in order:

    python3 generate_drills.py vis        # bundle=noname          -> _frames/*_vis.npz
    python3 generate_drills.py inv        # bundle=noname_invisible -> _frames/*_inv.npz
    python3 generate_drills.py compose    # no engine -> 36 .mov + ground_truth.csv
    python3 generate_drills.py all        # convenience: re-execs the three phases

Output: DRILLS/<NN>_<drill_name>/ with the 6 movs for that drill + ground_truth.csv,
plus DRILLS/ALL_GROUND_TRUTH.csv.
"""
import os
import sys
import json
import subprocess
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

from drills import all_variants as _all_variants, variant_spec, DRILL_NAMES  # noqa: E402


def all_variants():
    """Iteration order; DRILL_ONLY / VAR_ONLY env vars narrow it for smoke tests."""
    only_d = os.environ.get("DRILL_ONLY")
    only_v = os.environ.get("VAR_ONLY")
    for d, v in _all_variants():
        if only_d and d != int(only_d):
            continue
        if only_v and v != int(only_v):
            continue
        yield d, v

# ── geometry / grid (16x6, matching the real invisible-ball benchmark) ───────────
W, H = 1280, 480
COLS, ROWS = 16, 6
CELL_W, CELL_H = W / COLS, H / ROWS
N_STEPS = 50          # 5.0 s @ 10 fps
FPS = 10
VIS_FRAMES = 10       # first 1 s stays visible in the split clip
CIRCLE_R = 18
HUD_TOP, HUD_BOT, FRAME_H = 60, 180, 720
SEEDS_TRY = [42, 7, 11, 23, 3, 5, 17, 31, 55, 71, 13, 99, 101, 202,
             2, 8, 19, 27, 37, 44, 63, 88, 123, 256, 314, 512, 777, 1001]

FRAMES = HERE / "_frames"
MANIFEST = HERE / "_frames" / "manifest.json"
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"


def drill_dir(drill_idx):
    d = HERE / f"{drill_idx:02d}_{DRILL_NAMES[drill_idx]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── grid helpers ────────────────────────────────────────────────────────────────
def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def build_grid():
    from PIL import Image, ImageDraw
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    line = (255, 255, 0, 130)
    for c in range(COLS + 1):
        x = int(round(c * CELL_W))
        dr.line([(x, 0), (x, H)], fill=line, width=1)
    for r in range(ROWS + 1):
        y = int(round(r * CELL_H))
        dr.line([(0, y), (W, y)], fill=line, width=1)
    font = _font(13)
    for c in range(COLS):
        dr.text((int(round(c * CELL_W)) + 2, 0), str(c + 1), fill=(255, 255, 0, 255), font=font)
    for r in range(ROWS):
        dr.text((1, int(round(r * CELL_H)) + 1), chr(ord("A") + r), fill=(255, 255, 0, 255), font=font)
    return ov


def burn_grid(frame_rgb, overlay):
    from PIL import Image
    import numpy as np
    base = Image.fromarray(frame_rgb).convert("RGBA")
    return np.array(Image.alpha_composite(base, overlay).convert("RGB"))


def cell_of(px, py):
    c = min(max(int(px // CELL_W), 0), COLS - 1)
    r = min(max(int(py // CELL_H), 0), ROWS - 1)
    return f"{chr(ord('A') + r)}{c + 1}", chr(ord('A') + r), c + 1


# ── engine render (single scenario, deterministic) ───────────────────────────────
def render_scenario(level, seed):
    """Play `level` deterministically for N_STEPS; return (frames[list], balls[np])."""
    import numpy as np
    from gfootball.env import config as cfg, football_env
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
    """Accept only ONE unbroken passage of play — no goal, no reset, no dead ball.

    Three ways a clip breaks continuity, all rejected here:
      * a mid-play teleport (steps.max) — the ball snapping to the centre spot for a
        kickoff after a goal, or to a throw-in/free-kick spot after going out;
      * the ball crossing a goal line (|x| > 1.0) — i.e. a goal or a shot into the net,
        which is what the engine punishes with the "ball dropped into the centre" reset;
      * a degenerate path (too short = dead ball, too long = something teleported).
    Keeping only clean seeds is what guarantees continuous ball play with no resets.
    """
    import numpy as np
    if len(balls) < N_STEPS:
        return False
    steps = np.linalg.norm(np.diff(balls, axis=0), axis=1)
    path = float(steps.sum())
    if steps.max() >= 0.35:                       # teleport = kickoff / restart reset
        return False
    if float(np.abs(balls[:, 0]).max()) > 1.0:    # ball crossed a goal line = goal/shot in
        return False
    return 0.12 < path < 4.0


# ── PHASE vis: choose a clean seed per variant, render visible frames ─────────────
def phase_vis():
    import numpy as np
    from scenario_factory import write_scenario
    from lib import use_bundle
    use_bundle("noname")
    FRAMES.mkdir(parents=True, exist_ok=True)
    manifest = []
    skipped = []
    for drill_idx, var in all_variants():
        spec = variant_spec(drill_idx, var)
        level = write_scenario(spec, force=True)
        chosen = None
        for seed in SEEDS_TRY:
            frames, balls = render_scenario(level, seed)
            if _clean(balls):
                chosen = (seed, frames, balls)
                break
        if chosen is None:
            # No seed produced an unbroken passage of play — every one scored / reset.
            # We intentionally DO NOT fall back to a dirty seed: a reset/goal clip
            # (with the ball dropped to the centre) is exactly what we're avoiding.
            print(f"  SKIP {level}: no continuous seed in {len(SEEDS_TRY)} tries "
                  f"(all scored or reset) — not emitting a reset clip")
            skipped.append(level)
            continue
        seed, frames, balls = chosen
        np.savez_compressed(FRAMES / f"{level}_vis.npz", frames=np.array(frames))
        manifest.append({"drill": drill_idx, "variant": var, "level": level,
                         "seed": seed, "name": level})
        pathlen = float(np.linalg.norm(np.diff(balls, axis=0), axis=1).sum())
        print(f"  [vis] {level} seed={seed} path={pathlen:.2f}")
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"phase vis: wrote {len(manifest)} visible renders + manifest")
    if skipped:
        print(f"phase vis: SKIPPED {len(skipped)} variant(s) with no continuous seed: "
              + ", ".join(skipped))


# ── PHASE inv: replay each chosen (level, seed) with the ball invisible ───────────
def phase_inv():
    import numpy as np
    from lib import use_bundle
    use_bundle("noname_ball_invisible")
    manifest = json.loads(MANIFEST.read_text())
    for m in manifest:
        frames, _ = render_scenario(m["level"], m["seed"])
        np.savez_compressed(FRAMES / f"{m['level']}_inv.npz", frames=np.array(frames))
        print(f"  [inv] {m['level']} seed={m['seed']}")
    print(f"phase inv: wrote {len(manifest)} invisible renders")


# ── PHASE compose: diff -> ball, burn grid + circle, write both movs + GT ─────────
def _detect(vis, inv):
    """Ball (px,py) via visible-vs-invisible pixel diff (same method as the 30-clip GT)."""
    import numpy as np
    v = vis.astype(np.int16)
    i = inv.astype(np.int16)
    d = np.abs(v - i).sum(axis=2)                    # per-pixel change
    thr = max(40, d.max() * 0.35)
    mask = d > thr
    if mask.sum() == 0:
        mask = d > (d.max() * 0.5)
    bright = vis.astype(np.int32).sum(axis=2)        # ball is near-white
    ballmask = mask & (bright > 600)
    if ballmask.sum() < 3:
        ballmask = mask
    ys, xs = np.nonzero(ballmask)
    w = d[ys, xs].astype(float)
    return float(np.average(xs, weights=w)), float(np.average(ys, weights=w)), int(mask.sum())


def _circle(frame_rgb, px, py):
    from PIL import Image, ImageDraw
    import numpy as np
    img = Image.fromarray(frame_rgb)
    ImageDraw.Draw(img).ellipse(
        [px - CIRCLE_R, py - CIRCLE_R, px + CIRCLE_R, py + CIRCLE_R],
        outline=(255, 0, 0), width=3)
    return np.array(img)


def phase_compose():
    import numpy as np
    from lib import frames_to_mov
    manifest = json.loads(MANIFEST.read_text())
    overlay = build_grid()
    all_rows = [("drill", "variant", "clip_base", "seed",
                 "ball_start_cell", "start_px", "start_py",
                 "ball_final_cell", "final_px", "final_py")]
    per_drill = {}
    for m in manifest:
        level = m["level"]
        vis = np.load(FRAMES / f"{level}_vis.npz")["frames"]
        inv = np.load(FRAMES / f"{level}_inv.npz")["frames"]
        spx, spy, _ = _detect(vis[0], inv[0])
        fpx, fpy, _ = _detect(vis[-1], inv[-1])
        s_cell, _, _ = cell_of(spx, spy)
        f_cell, _, _ = cell_of(fpx, fpy)

        # visible clip: all frames visible, red circle on frame 0
        vis_out = []
        for i, f in enumerate(vis):
            g = burn_grid(f, overlay)
            if i == 0:
                g = _circle(g, spx, spy)
            vis_out.append(g)

        # split clip: 1 s visible + 4 s invisible, red circle on frame 0
        seq = list(vis[:VIS_FRAMES]) + list(inv[VIS_FRAMES:])
        split_out = []
        for i, f in enumerate(seq):
            g = burn_grid(f, overlay)
            if i == 0:
                g = _circle(g, spx, spy)
            split_out.append(g)

        d = drill_dir(m["drill"])
        frames_to_mov(vis_out, d / f"{level}_visible.mov", fps=FPS, crop_hud=False)
        frames_to_mov(split_out, d / f"{level}_split.mov", fps=FPS, crop_hud=False)

        row = (m["drill"], m["variant"] + 1, level, m["seed"],
               s_cell, round(spx, 1), round(spy, 1),
               f_cell, round(fpx, 1), round(fpy, 1))
        all_rows.append(row)
        per_drill.setdefault(m["drill"], []).append(row)
        print(f"  [compose] {level}: start={s_cell} final={f_cell}")

    # per-drill ground_truth.csv
    for drill_idx, rows in per_drill.items():
        csv = [",".join(map(str, all_rows[0]))]
        csv += [",".join(map(str, r)) for r in rows]
        (drill_dir(drill_idx) / "ground_truth.csv").write_text("\n".join(csv) + "\n")
    # global
    gcsv = [",".join(map(str, r)) for r in all_rows]
    (HERE / "ALL_GROUND_TRUTH.csv").write_text("\n".join(gcsv) + "\n")
    print(f"phase compose: wrote {len(manifest) * 2} movs + ground truth")


def phase_all():
    py = sys.executable
    for mode in ("vis", "inv", "compose"):
        print(f"\n===== phase {mode} =====")
        subprocess.run([py, __file__, mode], check=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    {"vis": phase_vis, "inv": phase_inv, "compose": phase_compose, "all": phase_all}[mode]()
