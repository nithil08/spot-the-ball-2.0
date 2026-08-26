"""Exact on-screen player counts for the 30_1s_visible_4s_invisible clips.

Method (the project's usual diff trick, applied to players instead of the ball):
replay each seed once per player slot with that one player rendered at ~0 scale
(GFOOTBALL_HIDE_SLOTS), and diff every frame against the normal render. Pixels that
change are exactly the pixels that player occupied, so "player s is on screen in frame
f" is a pixel fact, not a colour heuristic — it survives night lighting, dark kits and
crowds in the background.

A hidden player also loses his shadow, so a diff can fire for a player who is off-frame
with only his shadow showing. Shadow pixels keep the grass hue (they only darken), so
each diff is split into "body" pixels (hue moved) and shadow pixels, and a player counts
as on screen only when he has enough body pixels.

Usage:  python3 count_people_render.py <seed> [--verify]
        (seeds: 42 7 11 23 3 — the five seeds behind the 30 clips)
Writes: 30_1s_visible_4s_invisible/_people_counts/vis_seed<seed>.npz with per
        (clip, frame, slot) body/shadow pixel counts + bbox. --verify instead checks
        that this machine reproduces the cached clip frames bit for bit.
Then:   python3 count_people_report.py
"""
import os
import sys
import json
from pathlib import Path

import numpy as np

# the patched engine tree (GFOOTBALL_HIDE_SLOTS lives in its team.cpp)
SRC = Path(os.environ.get('GFOOTBALL_SRC', Path.home() / 'gfootball_src'))
REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / 'experiments/asset_bundles/noname/data'
BATCH = REPO / '30_1s_visible_4s_invisible'
FRAMES = BATCH / '_frames'
MANIFEST = Path(__file__).resolve().parent / 'results/natural_30_5s_grid/manifest.json'
OUTDIR = BATCH / '_people_counts'

# geometry / timing, copied from gen_natural_30_clips.py
HUD_TOP, HUD_BOTTOM, FRAME_H = 60, 180, 720
WARMUP, STEPS = 10, 50
LEVEL = '11_vs_11_stochastic'
SLOTS = [f'{side}{i}' for side in 'LR' for i in range(11)]

os.environ['GFOOTBALL_DATA_DIR'] = str(BUNDLE)
# the engine takes the name-caption font from its own env var, not from the bundle
os.environ['GFOOTBALL_FONT'] = str(BUNDLE / 'media/fonts/alegreya/AlegreyaSansSC-ExtraBold.ttf')
os.environ.pop('SDL_VIDEODRIVER', None)
os.environ.pop('DISPLAY', None)
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / 'third_party'))


def crop(frame):
    return np.array(frame)[HUD_TOP:FRAME_H - HUD_BOTTOM, :]


def make_env(seed, work_dir):
    from gfootball.env import config as cfg, football_env
    values = {
        'level': LEVEL, 'players': [], 'action_set': 'full',
        'write_video': False, 'dump_full_episodes': False, 'dump_scores': False,
        'tracesdir': str(work_dir), 'real_time': False,
        'game_engine_random_seed': seed, 'video_quality_level': 2,
        'display_game_stats': False,
    }
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    env = football_env.FootballEnv(cfg.Config(values))
    env.render('rgb_array')
    return env


def pass_frames(env, starts, on_frame):
    """Replay the match; call on_frame(window_start, idx, frame) inside wanted windows."""
    env.reset()
    n_windows = max((s - WARMUP) // STEPS for s in starts) + 1
    end = WARMUP + n_windows * STEPS
    cur_start, idx = WARMUP, 0
    for step in range(end):
        env.step([])
        f = env.render('rgb_array')
        if step < WARMUP:
            continue
        if cur_start in starts:
            on_frame(cur_start, idx, crop(f))
        idx += 1
        if idx == STEPS:
            cur_start += STEPS
            idx = 0


def main():
    seed = int(sys.argv[1])
    verify = '--verify' in sys.argv
    clips = [m for m in json.loads(MANIFEST.read_text()) if m['seed'] == seed]
    starts = {m['window_start'] for m in clips}
    start2clip = {m['window_start']: m['clip'] for m in clips}
    print(f'seed {seed}: clips {[m["clip"] for m in clips]}', flush=True)

    ref = {m['window_start']: np.load(FRAMES / f'clip{m["clip"]:02d}_vis.npz')['frames']
           for m in clips}

    if verify:                      # the cached render must be reproducible bit for bit
        bad = [0]

        def check(ws, i, f):
            if not np.array_equal(f, ref[ws][i]):
                bad[0] += 1
        env = make_env(seed, f'/tmp/gfhide_{seed}')
        pass_frames(env, starts, check)
        env.close()
        print(f'verify: {bad[0]} mismatching frames of {len(starts) * STEPS}', flush=True)
        return

    # body/shadow pixel counts and bbox for every (window, frame, slot)
    nclip, nslot = len(clips), len(SLOTS)
    body = np.zeros((nclip, STEPS, nslot), np.int32)
    shadow = np.zeros((nclip, STEPS, nslot), np.int32)
    bbox = np.zeros((nclip, STEPS, nslot, 4), np.int16)
    order = {m['window_start']: k for k, m in enumerate(clips)}

    import cv2
    for si, slot in enumerate(SLOTS):
        os.environ['GFOOTBALL_HIDE_SLOTS'] = slot

        def diff(ws, i, f, si=si):
            r = ref[ws][i]
            d = np.abs(f.astype(np.int16) - r.astype(np.int16)).max(axis=2) > 12
            if not d.any():
                return
            hr = cv2.cvtColor(r, cv2.COLOR_RGB2HSV)[..., 0].astype(np.int16)
            hf = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)[..., 0].astype(np.int16)
            dh = np.abs(hr - hf)
            dh = np.minimum(dh, 180 - dh)
            b = d & (dh > 12)          # kit/skin pixel: hue moved off grass
            k = order[ws]
            body[k, i, si] = b.sum()
            shadow[k, i, si] = d.sum() - b.sum()
            ys, xs = np.nonzero(b if b.any() else d)
            bbox[k, i, si] = (xs.min(), ys.min(), xs.max(), ys.max())

        # a fresh env per pass: the engine alternates the kick-off side on every
        # reset, so a second reset in the same process replays a different match
        env = make_env(seed, f'/tmp/gfhide_{seed}')
        pass_frames(env, starts, diff)
        env.close()
        tot = int((body[:, :, si] > 0).sum())
        print(f'  slot {slot}: on-screen in {tot} of {nclip * STEPS} frames', flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTDIR / f'vis_seed{seed}.npz', body=body, shadow=shadow,
                        bbox=bbox, clips=np.array([m['clip'] for m in clips]),
                        slots=np.array(SLOTS))
    print('wrote', OUTDIR / f'vis_seed{seed}.npz', flush=True)


if __name__ == '__main__':
    main()
