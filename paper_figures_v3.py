"""Figures for the OR-vote / drift-isolation paper (PaperV3/figures/), IEEE column width, text >= 8 pt.

  drift_isolation.pdf   HAI 22.04, seed 4: the monolithic score and the OR vote over the test stream, each divided by
                        its own threshold at 0.1% FAR (so both thresholds sit at 1), attack windows shaded
  tradeoff.pdf          pre-registered H5: deployed operating points (shift calibration) per held-out release
  heldout_or_vs_mono.pdf  pre-registered H3: OR vote minus monolithic in attacks beyond chance, 95% CI, per held-out
                        release and FAR (orvote_confirm_<rel>_summary.csv)
Also writes drift_isolation_hai2204.csv (per-stage drift statistics and attacks caught) for paper_numbers_v3.py.

Run from the repo root:  python paper_figures_v3.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

OUT = 'PaperV3/figures'
INK, MUTED, GRID = '#1f1f1e', '#6b6a65', '#d9d8d2'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']  # reference palette, categorical slots 1-3 (validated set)
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 8, 'font.family': 'serif', 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'pdf.fonttype': 42})
REL, SEED, TARGET = 'hai2204', 4, 0.001


def drift_isolation():
    import decomposition_mechanism as dm
    import label_free as lfp
    stages = json.load(open(f'Datasets/{REL}_subsystems.json'))
    tr = pd.read_csv(f'Datasets/{REL}_train_preprocessed.csv'); te = pd.read_csv(f'Datasets/{REL}_test_preprocessed.csv')
    y = te['Label'].values.astype(int); te, _ = lfp.fault_rule(tr, te)
    fill = tr.select_dtypes('number').mean(); normal = y == 0
    starts, lens = lfp.runs(y == 1)

    def score(a, b):
        mu, sd = a.mean(0), a.std(0) + 1e-6
        return lfp.smooth(((a - mu) / sd).max(1), 25), lfp.smooth(((b - mu) / sd).max(1), 25)

    st = {k: score(dm.sensor_errors(f'Oracles/{REL}_none_seed{SEED}', k, tr, fill)[1],
                   dm.sensor_errors(f'Oracles/{REL}_none_seed{SEED}', k, te, fill)[1]) for k in stages}
    m_tr, m_te = score(dm.sensor_errors(f'Oracles/{REL}_mono_none_seed{SEED}', 'Global', tr, fill)[1],
                       dm.sensor_errors(f'Oracles/{REL}_mono_none_seed{SEED}', 'Global', te, fill)[1])
    # thresholds at 0.1% total FAR: one for the monolithic score; an equal share per stage for the OR vote
    t_mono = np.quantile(m_te[normal], 1 - TARGET)
    qs = {k: np.sort(b[normal]) for k, (_, b) in st.items()}
    lo, hi = 0.0, TARGET
    for _ in range(30):
        mid = (lo + hi) / 2
        far = np.max([st[k][1] > np.quantile(qs[k], 1 - mid) for k in st], axis=0)[normal].mean()
        lo, hi = (mid, hi) if far <= TARGET else (lo, mid)
    t_st = {k: np.quantile(qs[k], 1 - lo) for k in st}
    mono_margin = m_te / t_mono
    or_margin = np.max([st[k][1] / t_st[k] for k in st], axis=0)
    caught = lambda margin: np.array([(margin[a:a + n] > 1).any() for a, n in zip(starts, lens)])
    c_mono, c_or = caught(mono_margin), caught(or_margin)
    rows = [{'stage': k, 'train_p99': np.percentile(a, 99), 'test_normal_median': np.median(b[normal]),
             'test_normal_p99': np.percentile(b[normal], 99), 'threshold_at_target': t_st[k],
             'attacks_crossing': int(caught(b / t_st[k]).sum())} for k, (a, b) in st.items()]
    rows.append({'stage': 'monolithic', 'train_p99': np.percentile(m_tr, 99), 'test_normal_median': np.median(m_te[normal]),
                 'test_normal_p99': np.percentile(m_te[normal], 99), 'threshold_at_target': t_mono,
                 'attacks_crossing': int(c_mono.sum())})
    rows.append({'stage': 'OR vote', 'attacks_crossing': int(c_or.sum()), 'n_attacks': len(starts)})
    pd.DataFrame(rows).to_csv('drift_isolation_hai2204.csv', index=False)

    # plot: 60-sample (1 min) maxima so 361k samples fit a column
    b = 60
    n = len(y) // b * b
    t = np.arange(n // b) / 60.0  # hours
    agg = lambda v: v[:n].reshape(-1, b).max(1)
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.1), sharex=True)
    for ax, margin, c, title in ((axes[0], mono_margin, MUTED, f'Monolithic: {int(c_mono.sum())} of {len(starts)} attacks cross'),
                                 (axes[1], or_margin, SERIES[0], f'Per-stage OR vote: {int(c_or.sum())} of {len(starts)} cross')):
        ax.plot(t, agg(margin), color=c, lw=0.6)
        ax.set_yscale('log'); ax.set_ylim(0.05, max(30, agg(margin).max() * 1.2))
        for a, l in zip(starts, lens):   # attack windows as a rug along the bottom (minutes long on a 100 h axis)
            ax.axvline(a / 3600, ymin=0, ymax=0.07, color=SERIES[1], lw=1.2)
        ax.axhline(1, color=INK, lw=0.8, ls='--')
        ax.set_title(title, fontsize=8, loc='left', color=INK)
        ax.set_ylabel('Score / threshold')
        ax.grid(axis='y', color=GRID, lw=0.5); ax.spines[['top', 'right']].set_visible(False)
    axes[1].set_xlabel('Test stream (hours)')
    fig.tight_layout(pad=0.3)
    fig.savefig(f'{OUT}/drift_isolation.pdf'); plt.close(fig)
    print(f'[SAVED] {OUT}/drift_isolation.pdf, drift_isolation_hai2204.csv (mono {int(c_mono.sum())}, OR {int(c_or.sum())} of {len(starts)})')


def heldout():
    rels = [('hai2007', 'HAI 20.07'), ('hai2204', 'HAI 22.04'), ('hai2103', 'HAI 21.03')]
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    fars = None
    for j, ((rel, label), c, m) in enumerate(zip(rels, SERIES, ['o', 's', 'D'])):
        d = pd.read_csv(f'orvote_confirm_{rel}_summary.csv')
        fars = d.target_FAR.values
        x = np.arange(len(d)) + (j - 1) * 0.18
        ax.errorbar(x, d['diff'], yerr=[d['diff'] - d.CI_low, d.CI_high - d['diff']], fmt=m, color=c, ms=4.5,
                    lw=1.2, capsize=2, label=label)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(fars))); ax.set_xticklabels([f'{f * 100:g}%' for f in fars])
    ax.set_xlabel('Total false-alarm rate (matched)')
    ax.set_ylabel('OR vote minus monolithic\n(attacks beyond chance)')
    ax.grid(axis='y', color=GRID, lw=0.5); ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(f'{OUT}/heldout_or_vs_mono.pdf'); plt.close(fig)
    print(f'[SAVED] {OUT}/heldout_or_vs_mono.pdf')


def tradeoff():
    """Deployed (shift-calibrated, 30-minute window) operating points on the held-out releases: false-alarm rate
    against attacks beyond chance; seed means (large) and single seeds (small)."""
    rels = [('hai2007', 'HAI 20.07'), ('hai2204', 'HAI 22.04'), ('hai2103', 'HAI 21.03')]
    dets = [('pivot', 'PIVOT (shift)', SERIES[0], 'o', True), ('mono', 'Monolithic (shift)', MUTED, 's', True),
            ('mono_window', 'Monolithic (window)', MUTED, 's', False)]
    fig, axes = plt.subplots(1, 3, figsize=(3.45, 1.95))
    for ax, (rel, name) in zip(axes, rels):
        d = pd.read_csv(f'orvote_deploy_{rel}_shift_seeds.csv').query('window == "primary"')
        for det, label, c, m, filled in dets:
            g = d[d.detector == det]
            ax.scatter(g.FAR * 100, g.beyond, s=6, color=c, alpha=0.35, marker=m, lw=0)
            ax.scatter(g.FAR.mean() * 100, g.beyond.mean(), s=34, marker=m, label=label, zorder=3,
                       facecolor=c if filled else 'white', edgecolor=c, lw=1.1)
        ax.set_title(name, fontsize=8, color=INK)
        ax.set_xscale('log'); ax.set_xlim(0.15, 40)
        ax.set_xticks([0.3, 1, 3, 10, 30]); ax.set_xticklabels(['0.3', '1', '3', '10', '30'])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(color=GRID, lw=0.5); ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel('Attacks beyond chance')
    axes[1].set_xlabel('False-alarm rate (%), log scale')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.02),
               handletextpad=0.2, columnspacing=0.6)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.88))
    fig.savefig(f'{OUT}/tradeoff.pdf'); plt.close(fig)
    print(f'[SAVED] {OUT}/tradeoff.pdf')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    heldout()
    tradeoff()
    drift_isolation()
