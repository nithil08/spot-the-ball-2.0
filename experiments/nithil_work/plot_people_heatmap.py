"""People in frame, one number per second: 30 clips x 5 seconds, values in the cells."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

BATCH = Path(__file__).resolve().parents[2] / '30_1s_visible_4s_invisible'
S = json.loads((BATCH / 'people_in_frame_summary.json').read_text())

M = np.array(S['clip_matrix']).reshape(30, 5, 10).mean(2)   # mean people per second
clip_mean = M.mean(1)
sec_mean = M.mean(0)

SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
SEQ = LinearSegmentedColormap.from_list(
    'seq_blue', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
norm = Normalize(vmin=M.min(), vmax=M.max())

plt.rcParams.update({'font.family': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE})

fig, ax = plt.subplots(figsize=(9.6, 12.4), dpi=200)
fig.subplots_adjust(left=0.115, right=0.965, top=0.895, bottom=0.045)

fig.text(0.115, 0.955, 'People in frame, second by second', fontsize=21, weight='bold', color=INK)
fig.text(0.115, 0.928,
         'Average number of players on screen during each second of each clip.',
         fontsize=11.5, color=INK2)
fig.text(0.115, 0.909,
         f'22 are on the pitch. Across all 30 clips the average is {S["mean"]:.1f}  ·  '
         f'darker means more people.', fontsize=11.5, color=INK2)


def cell(x, y, v, w=1.0, h=1.0, bold=False, muted=False):
    face = SEQ(norm(v)) if not muted else '#f0efec'
    ax.add_patch(plt.Rectangle((x + 0.02, y + 0.02), w - 0.04, h - 0.04,
                               facecolor=face, linewidth=0))
    lum = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2] if not muted else 1
    ax.text(x + w / 2, y + h / 2, f'{v:.1f}', ha='center', va='center',
            fontsize=10.5, color=('#ffffff' if lum < 0.55 else INK),
            weight='bold' if bold else 'normal')


for r in range(30):
    for c in range(5):
        cell(c, r, M[r, c])
    cell(5.25, r, clip_mean[r], muted=True, bold=True)          # per-clip average
for c in range(5):                                              # bottom summary row
    cell(c, 30.35, sec_mean[c], muted=True, bold=True)
cell(5.25, 30.35, M.mean(), muted=True, bold=True)

for r in range(30):
    ax.text(-0.12, r + 0.5, f'clip {r + 1:02d}', ha='right', va='center',
            fontsize=10, color=INK2)
ax.text(-0.12, 30.85, 'all clips', ha='right', va='center', fontsize=10,
        color=INK, weight='bold')

for c, lab in enumerate(['0-1 s', '1-2 s', '2-3 s', '3-4 s', '4-5 s']):
    ax.text(c + 0.5, -0.28, lab, ha='center', va='bottom', fontsize=10.5, color=INK2)
ax.text(5.75, -0.28, 'clip avg', ha='center', va='bottom', fontsize=10.5, color=INK,
        weight='bold')
ax.text(0.5, -1.15, 'ball visible', ha='center', va='bottom', fontsize=10, color=MUTED)
ax.text(3.0, -1.15, 'ball invisible', ha='center', va='bottom', fontsize=10, color=MUTED)
ax.plot([0.02, 0.98], [-0.82, -0.82], color=MUTED, lw=1)
ax.plot([1.02, 4.98], [-0.82, -0.82], color=MUTED, lw=1)

ax.set_xlim(-1.35, 6.35)
ax.set_ylim(31.5, -1.45)
ax.axis('off')

out = BATCH / 'people_in_frame_by_second.png'
fig.savefig(out, facecolor=SURFACE)
print('wrote', out)
