"""people-in-frame counts + figure for the 30_1s_visible_4s_invisible batch.

Reads the per-slot render diffs written by count_people_render.py and writes, into
30_1s_visible_4s_invisible/:
    people_per_frame.csv        clip, frame, t_sec, people, team_blue, team_red, goalkeepers
    people_per_second.csv       per clip, per second: mean / min / max
    people_in_frame_summary.json
    people_in_frame.png         histogram + time course + per-clip heatmap
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BATCH = REPO / '30_1s_visible_4s_invisible'
HERE = BATCH / '_people_counts'
SEEDS = [42, 7, 11, 23, 3]
MIN_BODY = 25       # kit/skin pixels a player must show to count as "in frame"


def load():
    """body[clip, frame, slot] pixel counts, clips ordered 1..30."""
    body = np.zeros((30, 50, 22), np.int32)
    slots = None
    for s in SEEDS:
        d = np.load(HERE / f'vis_seed{s}.npz', allow_pickle=True)
        for k, c in enumerate(d['clips']):
            body[c - 1] = d['body'][k]
        slots = [str(x) for x in d['slots']]
    return body, slots


def main():
    body, slots = load()
    on = body >= MIN_BODY                      # (clip, frame, slot)
    counts = on.sum(2)                         # people in frame
    left = on[:, :, :11].sum(2)
    right = on[:, :, 11:].sum(2)
    gk = on[:, :, 0].astype(int) + on[:, :, 11].astype(int)

    rows = ['clip,frame,t_sec,people,team_blue,team_red,goalkeepers']
    for c in range(30):
        for f in range(50):
            rows.append(f'{c + 1},{f},{f / 10:.1f},{counts[c, f]},'
                        f'{left[c, f]},{right[c, f]},{gk[c, f]}')
    (BATCH / 'people_per_frame.csv').write_text('\n'.join(rows) + '\n')

    # per-second (10 frames) means per clip
    per_sec = counts.reshape(30, 5, 10).mean(2)
    srows = ['clip,second,mean_people,min_people,max_people']
    mn = counts.reshape(30, 5, 10).min(2)
    mx = counts.reshape(30, 5, 10).max(2)
    for c in range(30):
        for s in range(5):
            srows.append(f'{c + 1},{s},{per_sec[c, s]:.1f},{mn[c, s]},{mx[c, s]}')
    (BATCH / 'people_per_second.csv').write_text('\n'.join(srows) + '\n')

    summary = {
        'frames': int(counts.size),
        'clips': 30,
        'min': int(counts.min()), 'max': int(counts.max()),
        'mean': round(float(counts.mean()), 2),
        'median': float(np.median(counts)),
        'p25': float(np.percentile(counts, 25)), 'p75': float(np.percentile(counts, 75)),
        'hist': {int(v): int((counts == v).sum()) for v in range(counts.min(), counts.max() + 1)},
        'per_second_mean': [round(float(x), 2) for x in counts.reshape(30, 5, 10).mean((0, 2))],
        'per_frame_mean': [round(float(x), 2) for x in counts.mean(0)],
        'per_frame_p25': [float(x) for x in np.percentile(counts, 25, axis=0)],
        'per_frame_p75': [float(x) for x in np.percentile(counts, 75, axis=0)],
        'per_frame_min': [int(x) for x in counts.min(0)],
        'per_frame_max': [int(x) for x in counts.max(0)],
        'clip_matrix': counts.tolist(),
        'clip_mean': [round(float(x), 2) for x in counts.mean(1)],
        'blue_mean': round(float(left.mean()), 2),
        'red_mean': round(float(right.mean()), 2),
        'gk_frames': int((gk > 0).sum()),
        'churn': round(float(np.abs(np.diff(counts, axis=1)).mean()), 3),
        'entries_exits': int(np.abs(np.diff(on.astype(int), axis=1)).sum()),
        'sensitivity': {str(t): round(float((body >= t).sum(2).mean()), 2)
                        for t in (1, 10, 25, 50, 100)},
    }
    (BATCH / 'people_in_frame_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ('clip_matrix', 'per_frame_mean', 'per_frame_p25',
                                   'per_frame_p75', 'per_frame_min', 'per_frame_max',
                                   'clip_mean')}, indent=2))


def _figure():
    import plot_people_in_frame          # noqa: F401  (draws on import)


if __name__ == '__main__':
    main()
    _figure()
