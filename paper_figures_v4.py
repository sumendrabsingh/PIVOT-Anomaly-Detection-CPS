"""Figures for the evaluation paper (PaperV4/figures/), IEEE column width, text >= 7 pt.

  baselines.pdf          development plants, label-free: false-alarm rate against attacks caught beyond chance for
                         PIVOT's OR vote, the monolithic AE (both per-sensor scoring), range checks, PCA-SPE and
                         (HAI) the forecaster
  heldout_or_vs_mono.pdf pre-registered H3 (from paper_figures_v3.heldout)
  drift_isolation.pdf    HAI 22.04 drift isolation (from paper_figures_v3.drift_isolation)
  deployed.pdf           pre-registered H5 (HAI) and H6 (SWaT cross-year): shift-calibrated operating points

Run from the repo root:  python paper_figures_v4.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
import paper_figures_v3 as v3
import label_free as lfp
from evaluate_events import evaluate

OUT = 'PaperV4/figures'
INK, MUTED, GRID, SERIES = v3.INK, v3.MUTED, v3.GRID, v3.SERIES


def logx(ax, lo, hi, ticks):
    ax.set_xscale('log'); ax.set_xlim(lo, hi)
    ax.set_xticks(ticks); ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(color=GRID, lw=0.5); ax.spines[['top', 'right']].set_visible(False); ax.tick_params(labelsize=7)


def baselines():
    swat = pd.read_csv('baselines_simple_label_free_swat.csv').groupby('detector').mean(numeric_only=True)
    swat = swat.rename(columns={'excess': 'beyond'})
    gen = pd.read_csv('baselines_simple_generalization.csv')
    fc = []
    for k in range(1, 6):
        d = pd.read_csv(f'preds_hai_temporal_seed{k}.csv')
        y, p = d.y_true.values.astype(int), d.y_pred.values.astype(int)
        s, _ = evaluate(y, p); ch, _ = lfp.chance_events(p, y)
        fc.append((s['false_alarm'], s['n_detected'] - ch))
    fc = np.mean(fc, axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for ax, (plant, name) in zip(axes, (('swat', 'SWaT'), ('wadi', 'WADI A2'), ('hai', 'HAI 23.05'))):
        o = pd.read_csv(f'orvote_check_{plant}_labelfree.csv').groupby('detector').mean(numeric_only=True)
        if plant == 'swat':
            rng = [(swat.loc[f'MinMax (tol {t}% of range)'].FAR, swat.loc[f'MinMax (tol {t}% of range)'].beyond) for t in (0, 5, 10)]
            pca = [(swat.loc['PCA-SPE (14 components)'].FAR, swat.loc['PCA-SPE (14 components)'].beyond)]
        else:
            g = gen[gen.plant == name.split()[0]].set_index('detector')
            rng = [(g.loc[f'MinMax (tol {t}%)'].FAR, g.loc[f'MinMax (tol {t}%)'].beyond) for t in (0, 5, 10)]
            pca = [(g.loc[d].FAR, g.loc[d].beyond) for d in g.index if d.startswith('PCA')]
        pts = [('Range check', rng, SERIES[1], '^'), ('PCA-SPE', pca, SERIES[2], 'D'),
               ('Monolithic AE', [(o.loc['monolithic, per-sensor scoring'].FAR, o.loc['monolithic, per-sensor scoring'].beyond)], MUTED, 's'),
               ('PIVOT', [(o.loc['OR vote, per-sensor scoring'].FAR, o.loc['OR vote, per-sensor scoring'].beyond)], SERIES[0], 'o')]
        if plant == 'hai':
            pts.append(('Forecaster', [tuple(fc)], INK, '*'))
        for label, xy, c, m in pts:
            xy = np.array(xy, dtype=float)
            ax.scatter(np.maximum(xy[:, 0] * 100, 0.012), xy[:, 1], s=30 if m != '*' else 55, color=c, marker=m,
                       label=label, zorder=3, edgecolor='white', lw=0.6)
        ax.set_title(name, fontsize=8, color=INK)
        logx(ax, 0.01, 60, [0.01, 0.1, 1, 10])
    axes[0].set_ylabel('Attacks beyond chance')
    axes[1].set_xlabel('False-alarm rate (%), log scale')
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc='upper center', ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.01),
               handletextpad=0.2, columnspacing=1.2)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
    fig.savefig(f'{OUT}/baselines.pdf'); plt.close(fig)
    print(f'[SAVED] {OUT}/baselines.pdf')


def deployed():
    rels = [('hai2007', 'HAI 20.07'), ('hai2204', 'HAI 22.04'), ('hai2103', 'HAI 21.03'),
            ('swat2019jul', 'SWaT Jul 2019'), ('swat2019dec', 'SWaT Dec 2019'), ('swat2022nov', 'SWaT Nov 2022')]
    dets = [('pivot', 'PIVOT', SERIES[0], 'o'), ('mono', 'Monolithic AE', MUTED, 's')]
    fig, axes = plt.subplots(2, 3, figsize=(3.45, 3.1))
    for ax, (rel, name) in zip(axes.ravel(), rels):
        d = pd.read_csv(f'orvote_deploy_{rel}_shift_seeds.csv').query('window == "primary"')
        for det, label, c, m in dets:
            g = d[d.detector == det]
            ax.scatter(g.FAR * 100, g.beyond, s=7, color=c, alpha=0.4, marker=m, lw=0)
            ax.scatter(g.FAR.mean() * 100, g.beyond.mean(), s=34, marker=m, color=c, label=label, zorder=3,
                       edgecolor='white', lw=0.6)
        ax.set_title(name, fontsize=8, color=INK)
        logx(ax, 0.1, 100, [0.1, 1, 10, 100])
    for ax in axes[:, 0]:
        ax.set_ylabel('Attacks beyond chance')
    axes[1, 1].set_xlabel('False-alarm rate (%), log scale')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.94))
    fig.savefig(f'{OUT}/deployed.pdf'); plt.close(fig)
    print(f'[SAVED] {OUT}/deployed.pdf')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    v3.OUT = OUT
    baselines()
    deployed()
    v3.heldout()
    v3.drift_isolation()
