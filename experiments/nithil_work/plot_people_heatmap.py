"""Standalone heatmap: people in frame, every clip x every frame."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BATCH = Path(__file__).resolve().parents[2] / '30_1s_visible_4s_invisible'
S = json.loads((BATCH / 'people_in_frame_summary.json').read_text())
M = np.array(S['clip_matrix'])

SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
SEQ = LinearSegmentedColormap.from_list(
    'seq_blue', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])

plt.rcParams.update({'font.family': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE})

fig, ax = plt.subplots(figsize=(13, 7.4), dpi=200)
fig.subplots_adjust(left=0.075, right=0.9, top=0.845, bottom=0.075)

fig.text(0.075, 0.945, 'People in frame, every clip frame by frame',
         fontsize=19, weight='bold', color=INK)
fig.text(0.075, 0.905,
         f'30 clips x 5 s at 10 fps. Median {S["median"]:.0f} of the 22 players on the pitch  ·  '
         f'range {S["min"]}-{S["max"]}  ·  darker means more people on screen',
         fontsize=11, color=INK2)

im = ax.imshow(M, aspect='auto', cmap=SEQ, interpolation='nearest',
               extent=[0, 5, 30.5, 0.5], vmin=M.min(), vmax=M.max())
ax.set_yticks(np.arange(1, 31))
ax.set_yticklabels([f'clip {i:02d}' for i in range(1, 31)], fontsize=7.5)
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xticklabels(['0 s', '1 s', '2 s', '3 s', '4 s', '5 s'], fontsize=10)
ax.axvline(1.0, color=SURFACE, lw=1.4)
ax.text(1.05, 0.15, 'ball goes invisible', fontsize=9.5, color=MUTED, va='bottom')
for s in ('top', 'right', 'left', 'bottom'):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0, colors=MUTED)

cb = fig.colorbar(im, ax=ax, pad=0.014, fraction=0.022, shrink=0.85,
                  ticks=[M.min(), M.max()])
cb.ax.set_yticklabels([f'{M.min()} people', f'{M.max()} people'], fontsize=9, color=MUTED)
cb.outline.set_visible(False)
cb.ax.tick_params(length=0)

out = BATCH / 'people_in_frame_heatmap.png'
fig.savefig(out, facecolor=SURFACE)
print('wrote', out)
