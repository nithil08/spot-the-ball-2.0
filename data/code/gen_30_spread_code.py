"""gen_30_spread_code.py — 30 natural 5s clips whose ball spreads across the 16x6 grid.

WHY THIS EXISTS
The stock engine camera centres on the ball, so the ball renders in the middle of
frame in every clip and the grid answer collapses: the 30-clip natural benchmark
puts 20/30 finals in row C and never leaves columns 5-12 (15 of 96 cells used).
That is camera geometry, not scenario design — moving play around the pitch does
not help, because the camera follows it.

The engine now reads GFOOTBALL_CAM_OFFSET_X / _Y (match.cpp, UpdateIngameCamera)
and shifts the camera's framing target by that many world units. The camera still
pans and tracks exactly as before; the shot is just framed off-centre, the way a
real operator frames play. Because the offset is a constant added to the target
every frame, and the camera's smoothing is linear, the ball's screen position
shifts by exactly K * offset — which makes the target cell directly solvable.

Calibration (measured, see calibrate() below):
    px shifts -26.58 px per world unit of offset X   (linear over the whole range)
    py shifts ~+13.6 px per unit of offset Y         (monotone, mildly nonlinear)
    offset Y also nudges px by ~-2.9 px/unit (perspective)

METHOD — the play is identical in all passes (the offset is purely a render-time
transform), so each window is measured once, then re-rendered reframed:
    probe      select goal-free moderate-movement windows, measure each one's
               natural (offset 0) final-frame ball pixel
    probe_inv  same windows, ball-invisible bundle, for the diff
    plan       assign each clip a target cell drawn from a broad, deliberately
               NON-uniform distribution; solve for the offset that lands it there
    vis        re-render each window with its offset -> the actual .mov clips
    inv        same, ball invisible -> final frames for ground truth
    gt         diff the reframed finals -> authoritative ball cell + heatmap data

Ground truth always comes from the real reframed render, never from the model, so
a calibration miss just means the ball lands a cell over — it is never wrong.

Run in order (bundle must be chosen before gfootball is imported, so each engine
phase is its own process):
    python3 gen_30_spread_code.py all
"""

import os
import sys
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = "/Users/nithilbalamurugan/gfootball_src"
EXP = "/Users/nithilbalamurugan/Desktop/Nithil Research/code/spot-the-ball-2.0/experiments"
for _p in (SRC, SRC + "/third_party", EXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = HERE.parent / "videos" / "spread_30_5s_grid_16x6"

# ── geometry: LOCKED 16x6 grid on the 1280x480 HUD-cropped render ───────────────
W, H = 1280, 480
COLS, ROWS = 16, 6
CELL_W, CELL_H = W / COLS, H / ROWS          # 80 x 80
STEPS = 50                                   # 5.0 s @ 10 fps
FPS = 10
HUD_TOP, HUD_BOT, FRAME_H = 60, 180, 720

LEVEL = "11_vs_11_stochastic"
WARMUP = 10
MAX_WINDOWS_SCAN = 10
CAP_PER_SEED = 4                             # spread clips over more matches
MOVE_LOW, MOVE_HIGH = 0.33, 0.95             # moderate-movement band
TARGET = 30
SEEDS = [42, 7, 11, 23, 3, 5, 17, 31, 55, 71, 13, 99, 101, 202]

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"

# ── calibration: camera offset (world units) -> ball pixel shift ────────────────
KX = -26.58                 # px per unit of offset X
CROSS_YX = -2.92            # px of X shift per unit of offset Y
# monotone offset-Y -> py-shift table, measured on the current build
OFFY_TABLE = [-20, -16, -14, -10, -6, -3, 0, 3, 6, 9, 12]
DPY_TABLE = [-228.3, -188.7, -168.0, -124.2, -77.3, -39.8, 0.0, 42.2, 87.4, 135.7, 187.0]
# How far the framing target may be pushed. These bound the OFFSET, but the real
# constraint is the resulting PIXEL (SAFE_PX/SAFE_PY below) — a clip whose ball
# already sits at px 900 can take a much larger positive X offset than one at
# px 400 before the ball leaves the frame, so the solve works from each clip's
# own measured position and these are only a backstop. Framing was eyeballed at
# both X extremes and stays entirely on the pitch.
OFFX_RANGE = (-26.0, 26.0)
OFFY_RANGE = (-12.0, 12.0)
# keep the ball clear of the frame edge so the diff always sees a whole ball
SAFE_PX = (55.0, 1225.0)
SAFE_PY = (35.0, 445.0)


# Engine-side bounds on the framing target (match.cpp), in world units, and the
# obs->world scales (defines.hpp). The linear calibration above only holds while
# the target is INSIDE these bounds: once the clamp bites, the camera stops
# panning and the engine's yaw term couples the two axes, so the ball lands
# nowhere near the solve. Keeping the target inside the envelope by construction
# is what makes the model trustworthy — clips whose ball sits near a pitch end
# simply get a smaller offset instead of a wrong one.
FRAMED_W = 55.0 * 0.95
FRAMED_H = 36.0 * 0.80
X_FIELD_SCALE, Y_FIELD_SCALE = 54.4, -83.6
# the camera aims at a player-biased, direction-led point rather than the ball
# itself, which can sit a few units off — stay clear of the bound by this much
CLAMP_MARGIN = 0.88


def solve_offsets(px0, py0, tgt_px, tgt_py, world=None):
    """Camera offset that moves the ball from its natural pixel to the target.

    `world` is the ball's (x, y) in engine world units at the final frame; when
    given, the offset is limited so the framing target stays inside the engine's
    clamp envelope and the linear calibration remains valid.
    """
    import numpy as np
    tgt_px = float(np.clip(tgt_px, *SAFE_PX))
    tgt_py = float(np.clip(tgt_py, *SAFE_PY))
    # invert the (monotone) y response by interpolation, then correct x for the
    # cross-term the y offset introduces
    offy = float(np.interp(tgt_py - py0, DPY_TABLE, OFFY_TABLE))
    offy = float(np.clip(offy, *OFFY_RANGE))
    offx = (tgt_px - px0 - CROSS_YX * offy) / KX
    offx = float(np.clip(offx, *OFFX_RANGE))
    if world is not None:
        wx, wy = world
        lim_x, lim_y = FRAMED_W * CLAMP_MARGIN, FRAMED_H * CLAMP_MARGIN
        offx = float(np.clip(offx, -lim_x - wx, lim_x - wx))
        offy = float(np.clip(offy, -lim_y - wy, lim_y - wy))
    return round(offx, 3), round(offy, 3)


# ── target distribution: broad, but deliberately NOT uniform ───────────────────
def target_cells(n):
    """n target cells drawn from a wide centre-weighted Gaussian over the grid.

    Uniform coverage would look synthetic and would not match how play actually
    distributes; a wide Gaussian keeps a natural centre-heavy shape while still
    reaching the corners. The per-cell cap stops the middle from clumping.
    """
    import numpy as np
    rng = np.random.default_rng(20260807)
    cap, counts, out = 2, {}, []
    # Row A is excluded on purpose. Putting the ball in the top row means aiming
    # the camera from beyond the near touchline, which fills the bottom of the
    # shot with hoardings and seating -- the framing stops looking like football
    # before the ball ever gets that high. Rows B-F are all reachable cleanly.
    MIN_ROW = 1
    while len(out) < n:
        c = int(round(rng.normal(7.5, 3.7)))
        r = int(round(rng.normal(2.7, 1.30)))
        if not (0 <= c < COLS and MIN_ROW <= r < ROWS):
            continue
        if counts.get((r, c), 0) >= cap:
            continue
        counts[(r, c)] = counts.get((r, c), 0) + 1
        out.append((r, c))
    return out


def cell_centre(r, c):
    return (c + 0.5) * CELL_W, (r + 0.5) * CELL_H


def cell_of(px, py):
    c = min(max(int(px // CELL_W), 0), COLS - 1)
    r = min(max(int(py // CELL_H), 0), ROWS - 1)
    return f"{chr(ord('A') + r)}{c + 1}", r, c


# ── grid overlay ───────────────────────────────────────────────────────────────
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
        dr.text((int(round(c * CELL_W)) + 2, 0), str(c + 1),
                fill=(255, 255, 0, 255), font=font)
    for r in range(ROWS):
        dr.text((1, int(round(r * CELL_H)) + 1), chr(ord("A") + r),
                fill=(255, 255, 0, 255), font=font)
    return ov


def burn_grid(frame_rgb, overlay):
    from PIL import Image
    import numpy as np
    base = Image.fromarray(frame_rgb).convert("RGBA")
    return np.array(Image.alpha_composite(base, overlay).convert("RGB"))


# ── ball detection (visible vs invisible diff) ─────────────────────────────────
def detect_ball(vis, inv):
    """Ball (px,py) from the visible/invisible diff, or None if it is not there.

    The None case matters: if the ball has been framed out of shot the diff is
    pure noise, and the old fallback branch happily returned the centroid of that
    noise as if it were a ball. Anything under MIN_SIGNAL changed pixels is a
    failure, not an answer.
    """
    import numpy as np
    MIN_SIGNAL = 8
    v = vis.astype(np.int16)
    i = inv.astype(np.int16)
    d = np.abs(v - i).sum(axis=2)
    if int((d > 40).sum()) < MIN_SIGNAL:
        # A part-occluded ball (behind a player, or sat on a white line) can leave
        # only a handful of strong pixels while still being plainly in shot. Drop
        # the threshold, but demand the blob be COMPACT — that is what separates a
        # real ball from scattered noise, and it is the check the original
        # fallback lacked when it returned the centroid of an empty frame.
        ys, xs = np.nonzero(d > 18)
        if len(xs) < 10:
            return None, None, 0
        if (xs.max() - xs.min()) > 20 or (ys.max() - ys.min()) > 20:
            return None, None, 0
        w = d[ys, xs].astype(float)
        return (float(np.average(xs, weights=w)),
                float(np.average(ys, weights=w)), int(len(xs)))
    thr = max(40, d.max() * 0.35)
    mask = d > thr
    bright = vis.astype(np.int32).sum(axis=2)
    bm = mask & (bright > 600)
    if bm.sum() < 3:
        bm = mask
    ys, xs = np.nonzero(bm)
    w = d[ys, xs].astype(float)
    if w.sum() == 0:
        return None, None, 0
    return float(np.average(xs, weights=w)), float(np.average(ys, weights=w)), int(mask.sum())


# ── engine ─────────────────────────────────────────────────────────────────────
def make_env(seed, work_dir):
    from gfootball.env import config as cfg, football_env
    bundle = os.environ.get("GFOOTBALL_DATA_DIR", "")
    blank = Path(bundle) / "media/fonts/alegreya/AlegreyaSansSC-ExtraBold.ttf"
    if blank.exists():
        os.environ["GFOOTBALL_FONT"] = str(blank)
    work_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "level": LEVEL, "players": [], "action_set": "full",
        "write_video": False, "dump_full_episodes": False, "dump_scores": False,
        "tracesdir": str(work_dir), "real_time": False,
        "game_engine_random_seed": seed, "video_quality_level": 2,
        "display_game_stats": False,
    }
    env = football_env.FootballEnv(cfg.Config(values))
    env.render("rgb_array")
    return env


def set_offset(offx, offy):
    """Set the framing offset for the next render.

    Note that setting these to 0 is NOT the same as leaving them unset: the unset
    path skips the offset branch entirely and is byte-identical to the stock
    camera, while "0" still applies the branch's pitch bound, which is a little
    wider than the stock follow clamp. Every phase here sets them (0 during the
    probe) precisely so the probe baseline and the reframed render go through the
    same camera bounds — otherwise the calibration would be measured against one
    camera and applied to another.
    """
    os.environ["GFOOTBALL_CAM_OFFSET_X"] = str(float(offx))
    os.environ["GFOOTBALL_CAM_OFFSET_Y"] = str(float(offy))


def crop(frame):
    import numpy as np
    return np.array(frame)[HUD_TOP:FRAME_H - HUD_BOT, :]


def run_seed(seed, work_dir, want_starts=None, offset=(0.0, 0.0), keep_all=False):
    """Play one match; return {window_start: (frames, balls, goal)}.

    frames is the full 50-frame window when keep_all, else just the final frame.
    """
    import numpy as np
    set_offset(*offset)
    env = make_env(seed, work_dir)
    env.reset()
    windows = {}
    n_windows = (MAX_WINDOWS_SCAN if want_starts is None
                 else max((s - WARMUP) // STEPS for s in want_starts) + 1)
    end = WARMUP + n_windows * STEPS
    buf, balls, goal = [], [], False
    cur = WARMUP
    for step in range(end):
        obs, r, done, _ = env.step([])
        o = obs[0] if isinstance(obs, list) else obs
        frame = env.render("rgb_array")      # render every step: keeps the camera
        if step < WARMUP:                    # smoothing history in sync
            continue
        keep = (want_starts is None) or (cur in want_starts)
        if keep:
            buf.append(crop(frame))
            if not keep_all and len(buf) > 1:
                buf.pop(0)                   # only the final frame is needed
        if float(np.sum(np.asarray(r))) != 0.0:
            goal = True
        balls.append(o["ball"][:2].copy())
        if len(balls) == STEPS:
            windows[cur] = (buf if keep else None, np.array(balls), goal)
            buf, balls, goal = [], [], False
            cur += STEPS
        if done:
            break
    env.close()
    return windows


def movement(balls):
    import numpy as np
    return float(np.sum(np.linalg.norm(np.diff(balls, axis=0), axis=1)))


# ── phase: probe ───────────────────────────────────────────────────────────────
def phase_probe():
    import numpy as np
    from lib import use_bundle
    use_bundle("noname")
    OUT.mkdir(parents=True, exist_ok=True)
    fdir = OUT / "_probe"
    fdir.mkdir(exist_ok=True)
    manifest, n = [], 0
    for seed in SEEDS:
        if n >= TARGET:
            break
        wins = run_seed(seed, OUT / "_work")
        kept = 0
        for start in sorted(wins):
            if n >= TARGET or kept >= CAP_PER_SEED:
                break
            frames, balls, goal = wins[start]
            mv = movement(balls)
            if goal or not (MOVE_LOW <= mv <= MOVE_HIGH):
                continue
            n += 1
            kept += 1
            np.save(fdir / f"clip_{n:02d}_vis.npy", frames[-1])
            manifest.append({"clip": n, "seed": seed, "window_start": start,
                             "movement": round(mv, 4)})
            print(f"  probe clip_{n:02d} seed={seed} start={start} mv={mv:.3f}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"probe: {n} windows selected")
    if n < TARGET:
        print(f"WARNING: only {n}/{TARGET} — add more SEEDS.")


def phase_probe_inv():
    import numpy as np
    from lib import use_bundle
    use_bundle("noname_ball_invisible")
    fdir = OUT / "_probe"
    manifest = json.loads((OUT / "manifest.json").read_text())
    by_seed = {}
    for m in manifest:
        by_seed.setdefault(m["seed"], []).append(m)
    for seed, items in by_seed.items():
        wins = run_seed(seed, OUT / "_work_inv",
                        want_starts={m["window_start"] for m in items})
        for m in items:
            np.save(fdir / f"clip_{m['clip']:02d}_inv.npy",
                    wins[m["window_start"]][0][-1])
            print(f"  probe_inv clip_{m['clip']:02d} seed={seed}")
    print("probe_inv: done")


# ── phase: world ───────────────────────────────────────────────────────────────
def phase_world():
    """Ball world position at each clip's final frame — no rendering.

    Rendering is ~50x the cost of stepping, and the camera does not affect the
    simulation, so the ball's world position can be replayed in seconds. Needed
    to keep the offset solve inside the engine's clamp envelope.
    """
    from lib import use_bundle
    use_bundle("noname")
    manifest = json.loads((OUT / "manifest.json").read_text())
    by_seed = {}
    for m in manifest:
        by_seed.setdefault(m["seed"], []).append(m)
    world = {}
    for seed, items in by_seed.items():
        finals = {m["window_start"] + STEPS - 1: m["clip"] for m in items}
        env = make_env(seed, OUT / "_work_world")
        obs = env.reset()
        for step in range(max(finals) + 1):
            obs, _, done, _ = env.step([])
            if step in finals:
                o = obs[0] if isinstance(obs, list) else obs
                b = o["ball"]
                world[str(finals[step])] = {
                    "obs": [float(b[0]), float(b[1]), float(b[2])],
                    "wx": float(b[0]) * X_FIELD_SCALE,
                    "wy": float(b[1]) * Y_FIELD_SCALE,
                }
            if done:
                break
        env.close()
        print(f"  world seed={seed}: {len(items)} clips")
    (OUT / "world.json").write_text(json.dumps(world, indent=2))
    hi = [c for c, v in world.items() if v["obs"][2] > 1.0]
    print(f"world: {len(world)} clips; {len(hi)} with an airborne ball at the "
          f"final frame{' -> ' + ','.join(sorted(hi, key=int)) if hi else ''}")


# ── phase: plan ────────────────────────────────────────────────────────────────
def phase_plan():
    import numpy as np
    fdir = OUT / "_probe"
    manifest = json.loads((OUT / "manifest.json").read_text())
    world = json.loads((OUT / "world.json").read_text())
    targets = target_cells(len(manifest))
    plan = []
    for m, (tr, tc) in zip(manifest, targets):
        vis = np.load(fdir / f"clip_{m['clip']:02d}_vis.npy")
        inv = np.load(fdir / f"clip_{m['clip']:02d}_inv.npy")
        px0, py0, _ = detect_ball(vis, inv)
        if px0 is None:
            raise RuntimeError(f"clip {m['clip']}: no ball in probe frame")
        tpx, tpy = cell_centre(tr, tc)
        w = world[str(m["clip"])]
        offx, offy = solve_offsets(px0, py0, tpx, tpy, world=(w["wx"], w["wy"]))
        nat_cell, _, _ = cell_of(px0, py0)
        plan.append({**m, "natural_px": round(px0, 1), "natural_py": round(py0, 1),
                     "natural_cell": nat_cell,
                     "target_cell": f"{chr(ord('A') + tr)}{tc + 1}",
                     "offset_x": offx, "offset_y": offy})
        print(f"  clip_{m['clip']:02d} natural={nat_cell} -> target="
              f"{chr(ord('A') + tr)}{tc + 1}  offset=({offx:+.2f},{offy:+.2f})")
    (OUT / "plan.json").write_text(json.dumps(plan, indent=2))
    print(f"plan: {len(plan)} clips")


# ── phase: render ──────────────────────────────────────────────────────────────
def phase_vis():
    import numpy as np
    from lib import use_bundle, frames_to_mov
    use_bundle("noname")
    plan = json.loads((OUT / "plan.json").read_text())
    overlay = build_grid()
    fdir = OUT / "_final"
    fdir.mkdir(exist_ok=True)
    for p in plan:                       # one match per clip: offset is per-clip
        wins = run_seed(p["seed"], OUT / "_work", want_starts={p["window_start"]},
                        offset=(p["offset_x"], p["offset_y"]), keep_all=True)
        frames = wins[p["window_start"]][0]
        name = f"clip_{p['clip']:02d}.mov"
        frames_to_mov([burn_grid(f, overlay) for f in frames], OUT / name,
                      fps=FPS, crop_hud=False)
        np.save(fdir / f"clip_{p['clip']:02d}_vis.npy", frames[-1])
        print(f"  vis clip_{p['clip']:02d} -> {name}")
    print(f"vis: {len(plan)} clips written")


def phase_inv():
    import numpy as np
    from lib import use_bundle
    use_bundle("noname_ball_invisible")
    plan = json.loads((OUT / "plan.json").read_text())
    fdir = OUT / "_final"
    for p in plan:
        wins = run_seed(p["seed"], OUT / "_work_inv", want_starts={p["window_start"]},
                        offset=(p["offset_x"], p["offset_y"]))
        np.save(fdir / f"clip_{p['clip']:02d}_inv.npy", wins[p["window_start"]][0][-1])
        print(f"  inv clip_{p['clip']:02d}")
    print(f"inv: {len(plan)} final frames")


# ── phase: ground truth ────────────────────────────────────────────────────────
def phase_gt():
    from PIL import Image, ImageDraw
    import numpy as np
    plan = json.loads((OUT / "plan.json").read_text())
    fdir = OUT / "_final"
    dbg = OUT / "_debug"
    dbg.mkdir(exist_ok=True)
    overlay = build_grid()
    rows, misses = [], 0
    for p in plan:
        cid = p["clip"]
        vis = np.load(fdir / f"clip_{cid:02d}_vis.npy")
        inv = np.load(fdir / f"clip_{cid:02d}_inv.npy")
        px, py, npix = detect_ball(vis, inv)
        if px is None:
            print(f"  clip_{cid:02d}  BALL NOT FOUND — framed out of shot")
            misses += 1
            continue
        cell, r, c = cell_of(px, py)
        rows.append({"clip": cid, "file": f"clip_{cid:02d}.mov", "seed": p["seed"],
                     "ball_cell": cell, "grid_row": chr(ord('A') + r), "grid_col": c + 1,
                     "px": round(px, 1), "py": round(py, 1),
                     "target_cell": p["target_cell"], "natural_cell": p["natural_cell"],
                     "offset_x": p["offset_x"], "offset_y": p["offset_y"],
                     "changed_pixels": npix})
        img = Image.fromarray(burn_grid(vis, overlay))
        dr = ImageDraw.Draw(img)
        dr.line([(px - 12, py), (px + 12, py)], fill=(255, 0, 0), width=2)
        dr.line([(px, py - 12), (px, py + 12)], fill=(255, 0, 0), width=2)
        dr.text((px + 8, py + 8), cell, fill=(255, 0, 0), font=_font(16))
        img.save(dbg / f"clip_{cid:02d}_gt.png")
        hit = "OK " if cell == p["target_cell"] else "   "
        print(f"  {hit}clip_{cid:02d}  {p['natural_cell']:>3} -> {cell:>3} "
              f"(target {p['target_cell']:>3})")
    (OUT / "ground_truth.json").write_text(json.dumps(rows, indent=2))
    hdr = ("clip,file,seed,ball_cell,grid_row,grid_col,px,py,"
           "target_cell,natural_cell,offset_x,offset_y")
    csv = [hdr] + [",".join(str(r[k]) for k in hdr.split(",")) for r in rows]
    (OUT / "ground_truth.csv").write_text("\n".join(csv) + "\n")
    exact = sum(1 for r in rows if r["ball_cell"] == r["target_cell"])
    print(f"\ngt: {len(rows)} clips ({misses} lost), {exact} hit their target cell exactly")


# ── phase: audit ───────────────────────────────────────────────────────────────
PITCH_FLOOR = 0.80


def phase_audit():
    """Flag clips whose framing stopped looking like football.

    Pushing the camera off the ball is only free while the shot stays on the
    pitch; past that the frame fills with seating and hoardings and the clip is
    unusable however well the ball landed. Green fraction is a blunt proxy but it
    caught every bad frame in the first batch (a third of the shot as empty
    stands scores ~53-72%, every clean shot scores >86%), so it runs every time
    rather than relying on somebody spot-checking thumbnails.
    """
    import numpy as np
    plan = {p["clip"]: p for p in json.loads((OUT / "plan.json").read_text())}
    fdir = OUT / "_final"
    rows = []
    for cid in sorted(plan):
        f = np.load(fdir / f"clip_{cid:02d}_vis.npy").astype(float)
        r, g, b = f[..., 0], f[..., 1], f[..., 2]
        # Relative, not absolute: half a pitch can sit in deep stand shadow, and
        # an absolute "g > r + 8" test reads that shadow as non-pitch and flags a
        # perfectly good frame. Comparing ratios is brightness-invariant, so lit
        # and shadowed turf both count as turf.
        green = float(((g > r * 1.04) & (g > b * 1.04) & (g > 12)).mean())
        rows.append((green, cid, plan[cid]))
    rows.sort()
    print(f"{'clip':>5} {'pitch%':>7} {'offx':>7} {'offy':>7}  target")
    for green, cid, p in rows:
        flag = "  <-- FRAMING" if green < PITCH_FLOOR else ""
        print(f"{cid:5d} {100 * green:7.1f} {p['offset_x']:+7.2f} "
              f"{p['offset_y']:+7.2f}  {p['target_cell']:>4}{flag}")
    bad = [c for g, c, _ in rows if g < PITCH_FLOOR]
    (OUT / "framing_audit.json").write_text(json.dumps(
        {"floor": PITCH_FLOOR, "flagged": bad,
         "pitch_fraction": {c: round(g, 4) for g, c, _ in rows}}, indent=2))
    print(f"\naudit: {len(bad)}/{len(rows)} below {PITCH_FLOOR:.0%} pitch"
          + (f" -> clips {bad}" if bad else " -- all clean"))


def phase_all():
    py = sys.executable
    for mode in ("probe", "probe_inv", "world", "plan", "vis", "inv", "gt", "audit"):
        print(f"\n===== phase {mode} =====")
        subprocess.run([py, __file__, mode], check=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    {"probe": phase_probe, "probe_inv": phase_probe_inv, "world": phase_world,
     "plan": phase_plan,
     "vis": phase_vis, "inv": phase_inv, "gt": phase_gt, "audit": phase_audit,
     "all": phase_all}[mode]()
