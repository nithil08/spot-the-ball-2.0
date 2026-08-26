"""Figure: how many people are in frame in the 30_1s_visible_4s_invisible clips."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap

BATCH = Path(__file__).resolve().parents[2] / '30_1s_visible_4s_invisible'
S = json.loads((BATCH / 'people_in_frame_summary.json').read_text())

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASE = '#c3c2b7'
BLUE = '#2a78d6'
BLUE_D = '#184f95'
BLUE_L = '#9ec5f4'
SEQ = LinearSegmentedColormap.from_list(
    'seq_blue', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])

plt.rcParams.update({
    'font.family': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': BASE, 'font.size': 10,
})


def rounded_bar(ax, x, w, h, color, r=0.10, base=0):
    """Bar with a rounded data-end, square at the baseline."""
    ax.add_patch(FancyBboxPatch((x - w / 2, base), w, max(h, r * 2),
                                boxstyle=f'round,pad=0,rounding_size={r}',
                                mutation_aspect=1, linewidth=0, facecolor=color,
                                clip_on=False, zorder=3))
    ax.add_patch(Rectangle((x - w / 2, base), w, min(h * 0.5, h - r),
                           linewidth=0, facecolor=color, zorder=3))


def clean(ax, bottom=True):
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['bottom'].set_color(BASE)
    ax.tick_params(length=0, labelsize=9)


fig = plt.figure(figsize=(13, 9.2), dpi=200)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], width_ratios=[1, 1.15],
                      hspace=0.42, wspace=0.2, left=0.06, right=0.935, top=0.85, bottom=0.075)

fig.text(0.06, 0.955, 'How many people are in frame', fontsize=21, weight='bold', color=INK)
fig.text(0.06, 0.918,
         f'30 clips x 5 s at 10 fps = 1,500 frames. A player counts as in frame when the render diff '
         f'shows at least {25} of his pixels.',
         fontsize=11, color=INK2)
fig.text(0.06, 0.893,
         f'Median {S["median"]:.0f} people   ·   middle half {S["p25"]:.0f}-{S["p75"]:.0f}   ·   '
         f'full range {S["min"]}-{S["max"]} of 22 on the pitch',
         fontsize=11, color=INK2)

# ── panel 1: the histogram ────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
hist = {int(k): v for k, v in S['hist'].items()}
ks = sorted(hist)
vals = [hist[k] for k in ks]
top = max(vals)
for k, v in zip(ks, vals):
    if v:
        rounded_bar(ax, k, 0.72, v / top * 100, BLUE, r=1.2, base=0)
ax.set_xlim(min(ks) - 1, max(ks) + 1)
ax.set_ylim(0, 112)
med = S['median']
ax.axvline(med, color=INK, lw=1, ls=(0, (4, 3)), zorder=4)
ax.text(med - 0.5, 106, f'median {med:.0f}', fontsize=9.5, color=INK, va='center', ha='right')
kmax = ks[len(vals) - 1 - int(np.argmax(vals[::-1]))]      # rightmost of tied peaks
ax.text(kmax + 0.55, max(vals) / top * 100 + 3, f'{max(vals)} frames',
        ha='left', fontsize=9.5, color=INK)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(['0', '', '', '', f'{top}'])
ax.set_xticks([k for k in ks if k % 2 == 0])
ax.set_xlabel('people in frame', fontsize=10, labelpad=6)
ax.set_title('Distribution across all 1,500 frames', fontsize=12.5, color=INK,
             loc='left', pad=10, weight='bold')
for y in (25, 50, 75, 100):
    ax.axhline(y, color=GRID, lw=0.8, zorder=1)
clean(ax)

# ── panel 2: over the 5 seconds ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
t = np.arange(50) / 10
mean = np.array(S['per_frame_mean'])
p25 = np.array(S['per_frame_p25'])
p75 = np.array(S['per_frame_p75'])
lo = np.array(S['per_frame_min'])
hi = np.array(S['per_frame_max'])
ax2.fill_between(t, lo, hi, color=BLUE, alpha=0.10, lw=0, zorder=2)
ax2.fill_between(t, p25, p75, color=BLUE, alpha=0.22, lw=0, zorder=3)
ax2.plot(t, mean, color=BLUE_D, lw=2, solid_capstyle='round', zorder=4)
ax2.axvline(1.0, color=BASE, lw=1, zorder=1)
ax2.text(1.06, hi.max() - 0.3, 'ball goes invisible', fontsize=9.5, color=MUTED)
ax2.text(t[-1], mean[-1] + 0.45, f'mean {mean[-1]:.1f}', fontsize=9.5, color=INK, ha='right')
ax2.plot([t[-1]], [mean[-1]], 'o', ms=7, color=BLUE_D, mec=SURFACE, mew=2, zorder=5)
ax2.set_xlim(-0.05, 5.0)
ax2.set_xticks([0, 1, 2, 3, 4, 4.9])
ax2.set_xticklabels(['0 s', '1 s', '2 s', '3 s', '4 s', '5 s'])
ax2.set_ylim(min(lo) - 1, max(hi) + 1.5)
ax2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=6))
ax2.set_title('Over the clip: mean, middle half, and full spread of the 30 clips',
              fontsize=12.5, color=INK, loc='left', pad=10, weight='bold')
for y in ax2.get_yticks():
    ax2.axhline(y, color=GRID, lw=0.8, zorder=1)
clean(ax2)

# ── panel 3: every clip ───────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :])
M = np.array(S['clip_matrix'])
im = ax3.imshow(M, aspect='auto', cmap=SEQ, interpolation='nearest',
                extent=[0, 5, 30.5, 0.5], vmin=M.min(), vmax=M.max())
ax3.set_yticks(np.arange(1, 31, 2))
ax3.set_yticklabels([f'clip {i:02d}' for i in range(1, 31, 2)], fontsize=8)
ax3.set_xticks([0, 1, 2, 3, 4, 5])
ax3.set_xticklabels(['0 s', '1 s', '2 s', '3 s', '4 s', '5 s'])
ax3.axvline(1.0, color=SURFACE, lw=1.4)
ax3.set_title('Every clip, frame by frame — darker means more people on screen',
              fontsize=12.5, color=INK, loc='left', pad=10, weight='bold')
for s in ('top', 'right', 'left', 'bottom'):
    ax3.spines[s].set_visible(False)
ax3.tick_params(length=0)
cb = fig.colorbar(im, ax=ax3, pad=0.012, fraction=0.022, shrink=0.9,
                  ticks=[M.min(), M.max()])
cb.ax.set_yticklabels([f'{M.min()} people', f'{M.max()} people'], fontsize=9, color=MUTED)
cb.outline.set_visible(False)
cb.ax.tick_params(length=0)

out = BATCH / 'people_in_frame.png'
fig.savefig(out, facecolor=SURFACE)
print('wrote', out)
