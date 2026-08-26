"""Sanity image: every counted player boxed, on the clips that broke colour detection."""
import json
from pathlib import Path

import numpy as np
import cv2

REPO = Path(__file__).resolve().parents[2]
BATCH = REPO / '30_1s_visible_4s_invisible'
HERE = BATCH / '_people_counts'
FRAMES = BATCH / '_frames'
MANIFEST = Path(__file__).resolve().parent / 'results/natural_30_5s_grid/manifest.json'
MIN_BODY = 25
PICKS = [(25, 25), (5, 0), (13, 30), (1, 49)]      # night clip, stands clip, dense, plain


def main():
    man = {m['clip']: m['seed'] for m in json.loads(MANIFEST.read_text())}
    tiles = []
    for clip, fi in PICKS:
        d = np.load(HERE / f'vis_seed{man[clip]}.npz', allow_pickle=True)
        k = list(d['clips']).index(clip)
        body, bbox, slots = d['body'][k, fi], d['bbox'][k, fi], d['slots']
        img = np.load(FRAMES / f'clip{clip:02d}_vis.npz')['frames'][fi].copy()
        n = 0
        for s, v, bb in zip(slots, body, bbox):
            if v < MIN_BODY:
                continue
            n += 1
            cv2.rectangle(img, (int(bb[0]) - 2, int(bb[1]) - 2), (int(bb[2]) + 2, int(bb[3]) + 2),
                          (255, 214, 0), 1)
            cv2.putText(img, str(s), (int(bb[0]) - 2, int(bb[1]) - 5), 0, 0.34, (255, 214, 0), 1)
        cv2.rectangle(img, (0, img.shape[0] - 40), (430, img.shape[0]), (10, 10, 10), -1)
        cv2.putText(img, f'clip{clip:02d}  t={fi / 10:.1f}s  ->  {n} people in frame',
                    (14, img.shape[0] - 14), 0, 0.7, (255, 255, 255), 2)
        tiles.append(img)
    out = np.concatenate(tiles, axis=0)
    cv2.imwrite(str(BATCH / 'people_in_frame_check.png'), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print('wrote', BATCH / 'people_in_frame_check.png')


if __name__ == '__main__':
    main()
