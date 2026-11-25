# Deck figures (transparent, Carlito = Calibri metrics). usage: python3 -I figs_deck.py <out_dir> <figs_dir>
import sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib import font_manager as fm

OUT, FIG = sys.argv[1], sys.argv[2]
for f in fm.findSystemFonts():
    if 'Carlito' in f or 'Caladea' in f: fm.fontManager.addfont(f)
NAVY, TEAL, MINT, CORAL, GREY, RULE, MARK = '#0E2233', '#13776B', '#5FD3C1', '#B8472A', '#46535F', '#CFC9BB', '#C3CED5'
plt.rcParams.update({'font.family': 'Carlito', 'font.size': 15, 'axes.edgecolor': GREY, 'text.color': NAVY,
                     'axes.labelcolor': GREY, 'xtick.color': GREY, 'ytick.color': GREY})
G = json.load(open(f'{OUT}/uci_geometry.json'))
TH = np.deg2rad(55.3); C, S = np.cos(TH), np.sin(TH)

def frame(ax):
    ax.add_patch(plt.Rectangle((0, 0), 1000, 1010, fill=False, ls=(0, (4, 3)), lw=1.2, ec=RULE))
    ax.set_xlim(-40, 1150); ax.set_ylim(-40, 1150); ax.set_aspect('equal'); ax.axis('off')
    a0 = np.array([1000 + C * 135, 1010 + S * 135]); a1 = np.array([1000 + C * 25, 1010 + S * 25])
    ax.annotate('', xy=a1, xytext=a0, arrowprops=dict(arrowstyle='-|>', lw=3.2, color=NAVY, mutation_scale=24))
    ax.text(a0[0] - 22, a0[1] + 4, 'waves', ha='right', va='center', fontsize=16, fontweight='bold', color=NAVY)
    for v in [0, 500, 1000]:
        ax.text(v, -28, f'{v:,} m' if v else '0', ha='center', va='top', fontsize=12, color=GREY)

# --- best layout, diverging around a lone buoy (96.4 kW)
b = G['best']; x, y, p = np.array(b['x']), np.array(b['y']), np.array(b['p']) / 1e3
cmap = LinearSegmentedColormap.from_list('pw', [CORAL, MARK, TEAL])
norm = TwoSlopeNorm(vmin=55, vcenter=96.374, vmax=110)
fig, ax = plt.subplots(figsize=(6.2, 6.6))
frame(ax)
sc = ax.scatter(x, y, c=p, cmap=cmap, norm=norm, s=190, edgecolors='white', linewidths=1.4, zorder=3)
cax = fig.add_axes([0.16, 0.085, 0.68, 0.025])
cb = fig.colorbar(sc, cax=cax, orientation='horizontal'); cb.outline.set_visible(False)
cb.set_ticks([55, 70, 85, 96.374, 110]); cb.set_ticklabels(['55', '70', '85', '96.4', '110'])
cb.ax.tick_params(labelsize=12, length=0)
cb.set_label('power per buoy, kW (grey = same as a lone buoy)', fontsize=12.5, color=GREY)
fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.17)
fig.savefig(f'{FIG}/deck_best_layout.png', dpi=220, transparent=True); plt.close(fig)

# --- near-twin pair
A, B = G['pair']['a'], G['pair']['b']
pa = {(round(u, 2), round(v, 2)) for u, v in zip(A['x'], A['y'])}
pb = {(round(u, 2), round(v, 2)) for u, v in zip(B['x'], B['y'])}
(ax_, ay_), (bx_, by_) = list(pa - pb)[0], list(pb - pa)[0]
sh = np.array(list(pa & pb))
fig, ax = plt.subplots(figsize=(6.2, 6.6))
frame(ax)
ax.scatter(sh[:, 0], sh[:, 1], s=150, c=TEAL, edgecolors='white', linewidths=1.2, zorder=3, label='48 buoys in identical positions')
ax.scatter([ax_], [ay_], s=190, facecolors='none', edgecolors=CORAL, linewidths=2.4, zorder=4, label='Layout A: one buoy here')
ax.scatter([bx_], [by_], s=190, c=CORAL, edgecolors='white', linewidths=1.2, zorder=4, label='Layout B: moved 75 m')
ax.annotate('', xy=(bx_, by_), xytext=(ax_, ay_), arrowprops=dict(arrowstyle='-|>', lw=2, color=CORAL, mutation_scale=16, shrinkA=7, shrinkB=7), zorder=5)
leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=14, ncol=1, handletextpad=0.4)
fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.2)
fig.savefig(f'{FIG}/deck_near_twin.png', dpi=220, transparent=True); plt.close(fig)

# --- site map
lm = np.load(f'{OUT}/landmask.npz'); band = pd.read_csv(f'{OUT}/band_cells.csv'); band = band[band.wpd.notna()]
fig, ax = plt.subplots(figsize=(7.6, 6.6))
ax.set_facecolor('#E3ECF1')
ax.imshow(np.where(lm['land'], 1.0, np.nan), extent=[lm['lon'].min(), lm['lon'].max(), lm['lat'].min(), lm['lat'].max()],
          origin='lower', cmap=LinearSegmentedColormap.from_list('l', ['#D8D1C1', '#D8D1C1']), vmin=0, vmax=1, zorder=1, interpolation='nearest')
wcm = LinearSegmentedColormap.from_list('w', ['#F0C36D', '#239C8C', NAVY])
sc = ax.scatter(band.lon, band.lat, s=3.2, c=band.wpd, cmap=wcm, vmin=2, vmax=18, zorder=3, linewidths=0)
sites = [(7.6083, 77.9583, '1', (0.25, -1.25)), (8.0083, 78.2583, '2', (1.25, 0.9)), (7.5583, 77.1583, '3', (-1.3, -0.35)),
         (6.6083, 81.9583, '4', (1.2, 0.5)), (6.0083, 81.4083, '5', (0.25, -1.25))]
for la, lo, t, (dx, dy) in sites:
    ax.plot([lo, lo + dx], [la, la + dy], color=NAVY, lw=1, zorder=5)
    ax.scatter([lo], [la], s=14, c=NAVY, zorder=6)
    ax.scatter([lo + dx], [la + dy], s=330, c=CORAL, edgecolors='white', linewidths=1.6, zorder=6)
    ax.text(lo + dx, la + dy, t, ha='center', va='center', color='white', fontsize=13, fontweight='bold', zorder=7)
ax.scatter([93.8083], [6.5083], s=170, marker='D', c=NAVY, edgecolors='white', linewidths=1.4, zorder=6)
ax.text(93.2, 5.3, 'most wave power\n(17.9 kW/m)\nranks 15th', ha='right', va='top', fontsize=13, color=NAVY)
ax.set_xlim(70, 95); ax.set_ylim(-1, 20); ax.set_aspect('equal')
ax.set_xticks([75, 80, 85, 90]); ax.set_xticklabels([f'{v}°E' for v in [75, 80, 85, 90]])
ax.set_yticks([0, 5, 10, 15]); ax.set_yticklabels([f'{v}°N' for v in [0, 5, 10, 15]])
ax.tick_params(length=0, labelsize=12)
for s_ in ax.spines.values(): s_.set_visible(False)
cax = fig.add_axes([0.14, 0.07, 0.72, 0.022])
cb = fig.colorbar(sc, cax=cax, orientation='horizontal'); cb.outline.set_visible(False)
cb.set_ticks([2, 6, 10, 14, 18]); cb.ax.tick_params(labelsize=12, length=0)
cb.set_label('mean wave power, kW per metre of wave front (ERA5 2016–17)', fontsize=12.5, color=GREY)
fig.subplots_adjust(left=0.07, right=0.99, top=0.99, bottom=0.17)
fig.savefig(f'{FIG}/deck_sites_map.png', dpi=220, transparent=True); plt.close(fig)
print('deck figures saved')
