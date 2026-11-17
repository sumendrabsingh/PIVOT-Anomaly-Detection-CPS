"""Physically valid adaptive attack on the label-free SWaT detector (reviewer points R2.6, R2.11).

adaptive_attack_label_free.py perturbs every input column continuously, valve one-hots and pump states
included, and never clips sensors to a physical range. Its samples are not guaranteed to be valid plant
states, and it minimizes reconstruction error instead of the decision function it is scored on. This
script adds feasible threat models and an objective aimed at the deployed dual-threshold rule.

Target: subsystem array Oracles/swat_phys_none_seed4 (lambda = 0; --models to change), errors standardized on training data,
25-sample smoothing, max over subsystems, dual threshold P99.9 / P99 of training, W_dur = 50, causal;
OR-ed with the analyzer abrupt-change arm (analyzer_arm.py). Attack rows only, full knowledge.

Threat models (--threats); eps is an L-inf budget in training-SD units on sensors
  unconstrained  every column moves by up to eps in the model's normalized units, as in R1.2
  sensors        actuator columns (pumps, UV, valve one-hots) keep their true state; each sensor moves
                 by up to eps and stays inside its physical range, taken as the training [min, max]
                 widened to include the true reading (the attacker can always report the truth)
  actuators      as 'sensors', plus the attacker may report ANY valid actuator state (a status spoof
                 is all-or-nothing, so it is not charged against eps): pumps relaxed to the interval
                 between their two states and valves to the [0, 1] box, rounded to the nearest valid
                 state (pump level / one-hot argmax), then the sensors are re-optimized with the
                 rounded states fixed; also a replay spoof (each actuator reports its state from just
                 before the attack began) with re-optimized sensors. Per attack event the attacker
                 keeps whichever of no-spoof / relaxed spoof / replay leaves the fewest alarms
Every attack keeps, per attack event, its best iterate scored on the real alarms, with no perturbation
as the first candidate: a full-knowledge attacker never applies a perturbation that raises detection.
Objectives (--objectives)
  recon     minimize each subsystem's standardized max reconstruction error (as R1.2)
  decision  hinge on the deployed rule itself: the smoothed aggregate below tau_H - m everywhere and
            the 50-sample rolling minimum below tau_L - m (a dual-threshold low-tier alarm needs 50
            consecutive samples above tau_L, so breaking every such run is exactly what evasion needs)
Analyzer arm: each analyzer is moved toward its clean rolling baseline by up to eps (as R1.2). The
baseline is a reading the sensor itself reported earlier, so these values are physically valid.

Run from the repo root:  python feasible_attack_label_free.py
"""
import argparse
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
import analyzer_arm as aa
import label_free as lfp
import label_free_dual_threshold as lf
from evaluate_events import evaluate

MODELS = 'Oracles/swat_phys_none_seed4'  # lambda = 0 headline array (legacy: Oracles/swat_seed4_noAIT)
EPSILONS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
W_DUR = 50
DEV = lf.DEV


def smooth_t(v, w=lf.W):
    """Rolling mean with min_periods=1 (pandas' rolling(w, min_periods=1).mean()), differentiable."""
    c = torch.cumsum(torch.cat([v.new_zeros(1), v.double()]), 0)
    n = torch.arange(1, len(v) + 1, device=v.device)
    lo = torch.clamp(n - w, min=0)
    return ((c[n] - c[lo]) / (n - lo)).float()


def actuator_groups(cols, tr):
    """Valve one-hot groups (MVxxx_k) and two-state columns (pumps, UV) by position in cols."""
    valves = {}
    for i, c in enumerate(cols):
        m = re.fullmatch(r'(MV\d+)_\d', c)
        if m:
            valves.setdefault(m.group(1), []).append(i)
    two_state = {i: np.sort(tr[c].unique()) for i, c in enumerate(cols)
                 if not re.fullmatch(r'MV\d+_\d', c) and tr[c].nunique() == 2}
    return list(valves.values()), two_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=MODELS)
    ap.add_argument('--epsilons', default=','.join(map(str, EPSILONS)))
    ap.add_argument('--threats', default='unconstrained,sensors,actuators')
    ap.add_argument('--objectives', default='recon,decision')
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--margin', type=float, default=0.1, help='hinge margin m, smoothed-score units')
    ap.add_argument('--out', default='feasible_attack_label_free_swat.csv')
    ap.add_argument('--subs', default=None,
                    help="comma-separated model names in --models (default: the six SWaT stages; 'Global' for the "
                         "monolithic control in Oracles/swat_mono_noAIT_none_seed<k>)")
    ap.add_argument('--far_target', type=float, default=None,
                    help='set tau_L / tau_H so the CLEAN dual-threshold FAR on test normal rows is this fraction '
                         '(tau_H at a tenth of tau_L\'s exceedance). Reads the test labels: a matched-FAR '
                         'comparison (matched_far_attack.py), not an operating point')
    args = ap.parse_args()
    epsilons = [float(e) for e in args.epsilons.split(',')]
    if args.subs:
        lf.SUBS = {n: None for n in args.subs.split(',')}

    te = pd.read_csv(lf.TEST); te.columns = te.columns.str.strip()
    tr = pd.read_csv(lf.TRAIN); tr.columns = tr.columns.str.strip()
    y = lfp.swat_test_labels()   # official labels (same as y_original of swat_test_labels_recovery.csv)
    atk_np = y == 1
    atk = torch.as_tensor(atk_np, device=DEV)

    # Array: one column space for all subsystems (their features are disjoint)
    lf.MODELS = args.models
    subs, cols = [], []
    for name in lf.SUBS:
        model, ck = lf.load_model(name)
        for p in model.parameters():
            p.requires_grad_(False)
        cols += ck['features']
        subs.append((model, ck))
    X0 = torch.tensor(te[cols].fillna(te[cols].mean()).values, dtype=torch.float32, device=DEV)
    Xtr = torch.tensor(tr[cols].fillna(tr[cols].mean()).values, dtype=torch.float32, device=DEV)
    bank, start = [], 0
    for model, ck in subs:
        idx = torch.arange(start, start + len(ck['features']), device=DEV); start += len(idx)
        mean = torch.as_tensor(ck['mean'], dtype=torch.float32, device=DEV)
        std = torch.as_tensor(ck['std'], dtype=torch.float32, device=DEV)
        with torch.no_grad():
            Xn = (Xtr[:, idx] - mean) / std
            e_tr = torch.cat([((Xn[i:i + 65536] - model(Xn[i:i + 65536])) ** 2).max(1)[0]
                              for i in range(0, len(Xn), 65536)]).cpu().numpy()
        bank.append((model, idx, mean, std, float(e_tr.mean()), float(e_tr.std() + 1e-6), e_tr))
    sm = lambda v: pd.Series(v).rolling(lf.W, min_periods=1).mean().values
    cal = np.max([sm((e - m) / s) for *_, m, s, e in bank], axis=0)
    th, tl = np.percentile(cal, 99.9), np.percentile(cal, 99.0)
    print(f'array thresholds: tau_H {th:.3f}  tau_L {tl:.3f}  (margin {args.margin})')

    def std_err(X, model, idx, mean, std, m, s):
        Xn = (X[:, idx] - mean) / std
        return (((Xn - model(Xn)) ** 2).max(1)[0] - m) / s

    def array_score(X):
        with torch.no_grad():
            sc = [sm(torch.cat([std_err(X[i:i + 65536], *b[:6]) for i in range(0, len(X), 65536)]).cpu().numpy())
                  for b in bank]
        return np.max(sc, axis=0)

    def array_pred(X):
        return lf.dual(array_score(X), th, tl, W_DUR, causal=True)

    with torch.no_grad():
        e_clean = [torch.cat([std_err(X0[i:i + 65536], *b[:6]) for i in range(0, len(X0), 65536)]) for b in bank]

    if args.far_target is not None:
        # One knob q: tau_L / tau_H are the clean score's (1 - q) / (1 - q/10) quantiles on test normal rows;
        # bisect log q until the clean dual-threshold FAR meets the target.
        agg0, normal = array_score(X0), ~atk_np
        def thresholds(q):
            return np.quantile(agg0[normal], 1 - q / 10), np.quantile(agg0[normal], 1 - q)
        lo_q, hi_q = 1e-6, 0.5
        for _ in range(40):
            q = np.sqrt(lo_q * hi_q)
            far = lf.dual(agg0, *thresholds(q), W_DUR, causal=True)[normal].mean()
            lo_q, hi_q = (q, hi_q) if far <= args.far_target else (lo_q, q)
        th, tl = thresholds(lo_q)
        far = lf.dual(agg0, th, tl, W_DUR, causal=True)[normal].mean()
        print(f'matched FAR {args.far_target:.2%}: tau_H {th:.3f}  tau_L {tl:.3f}  clean FAR {far:.3%}')

    # Per-column bounds: normalized scale (eps budget), physical range, actuator structure
    scale = torch.zeros(len(cols), device=DEV)
    for model, idx, mean, std, *_ in bank:
        scale[idx] = std
    lo_tr, hi_tr = Xtr.min(0)[0], Xtr.max(0)[0]
    valves, two_state = actuator_groups(cols, tr)
    act = torch.zeros(len(cols), dtype=torch.bool, device=DEV)
    for g in valves:
        act[g] = True
    act[list(two_state)] = True
    print(f'{len(cols)} array columns: {int(act.sum())} actuator columns '
          f'({len(valves)} valves, {len(two_state)} two-state), {int((~act).sum())} sensors')
    act_step = torch.full((len(cols),), 0.1, device=DEV)  # relaxed actuators reach any state in 10 steps
    for i, (a, b) in two_state.items():
        act_step[i] = 0.1 * float(b - a)
    x0 = X0[atk]
    lo_phys, hi_phys = torch.minimum(lo_tr, x0), torch.maximum(hi_tr, x0)

    def loss_fn(X_atk, objective):
        if objective == 'recon':
            return sum(std_err(X_atk, *b[:6]).mean() for b in bank)
        aggs = []
        for b, e0 in zip(bank, e_clean):
            e = e0.clone()
            e[atk] = std_err(X_atk, *b[:6])
            aggs.append(smooth_t(e))
        agg = torch.stack(aggs).max(0)[0]
        run_min = -F.max_pool1d(-agg[None, None], W_DUR, stride=1)[0, 0]
        return (F.relu(agg - (th - args.margin)).sum() / W_DUR + F.relu(run_min - (tl - args.margin)).sum())

    ev_start, ev_len = lfp.runs(atk_np)
    ev_rows = torch.as_tensor(np.concatenate([np.full(n, i) for i, n in enumerate(ev_len)]), device=DEV)
    ev_off = np.r_[0, np.cumsum(ev_len)[:-1]]

    # replay spoof: every actuator reports the state it had just before its attack event began
    x_replay = torch.where(act, X0[torch.as_tensor(ev_start - 1, device=DEV)][ev_rows], x0)

    def event_alarms(x):
        """Array alarms inside each attack event when the attack rows read x (the attacker's oracle)."""
        X = X0.clone(); X[atk] = x
        return np.add.reduceat(array_pred(X)[atk_np], ev_off)

    def pick_per_event(cands):
        """Per attack event, the candidate with the fewest array alarms on that event (ties: the earlier
        candidate). Events are far apart relative to the 25-sample smoothing and 50-sample persistence
        windows, so per-event choices do not interact. A full-knowledge attacker never applies a
        perturbation that raises detection, so x0 (no perturbation) is always the first candidate."""
        al = np.stack([event_alarms(c) for c in cands])
        win = torch.as_tensor(al.argmin(0), device=DEV)[ev_rows]
        out = cands[0].clone()
        for k, c in enumerate(cands[1:], 1):
            out = torch.where((win == k)[:, None], c, out)
        return out

    def pgd(eps, threat, objective, x_start=None, free_act=False):
        """Sign-gradient PGD on the attack rows; returns the adversarial attack-row block. Fixed
        actuator columns keep their values in x_start (the true states unless a spoof was rounded in)."""
        x_start = x0 if x_start is None else x_start
        x = x_start.clone().requires_grad_(True)
        best, best_al = x_start.clone(), event_alarms(x_start)
        for it in range(args.iters):
            step = eps / 10 * (1 - it / args.iters) + eps / 100  # linear decay eps/10 -> eps/100
            g, = torch.autograd.grad(loss_fn(x, objective), x)
            with torch.no_grad():
                if threat == 'unconstrained':
                    x = x0 + torch.clamp(x - step * scale * g.sign() - x0, -eps * scale, eps * scale)
                else:
                    xs = x - step * scale * g.sign()
                    xs = torch.max(torch.min(xs, x0 + eps * scale), x0 - eps * scale)
                    xs = torch.max(torch.min(xs, hi_phys), lo_phys)
                    if free_act:  # relaxed actuators: fixed step, kept inside their valid hull
                        xa = torch.max(torch.min(x - act_step * g.sign(), hi_tr), lo_tr)
                        x = torch.where(act, xa, xs)
                    else:
                        x = torch.where(act, x_start, xs)
            x.requires_grad_(True)
            if (it + 1) % 5 == 0 or it + 1 == args.iters:  # keep each event's best iterate
                al = event_alarms(x.detach())
                better = torch.as_tensor(al < best_al, device=DEV)[ev_rows]
                best = torch.where(better[:, None], x.detach(), best)
                best_al = np.minimum(al, best_al)
        return best

    def round_actuators(x):
        x = x.clone()
        for i, (a, b) in two_state.items():
            x[:, i] = torch.where((x[:, i] - a).abs() <= (x[:, i] - b).abs(), float(a), float(b))
        for g in valves:
            x[:, g] = F.one_hot(x[:, g].argmax(1), len(g)).float()
        return x

    # Analyzer arm (as R1.2)
    ait = sorted(c for c in te.columns if c.startswith('AIT'))
    _, ref = aa.change_scores(tr, ait)
    s_tr = aa.score(aa.scale_adapted(tr, ait, ref))
    ath, atl = np.percentile(s_tr, 99.9), np.percentile(s_tr, 99.0)
    base = te[ait].shift(aa.GAP).rolling(aa.H, min_periods=aa.H // 4).median().fillna(te[ait])
    arm_pred = lambda df: lf.dual(aa.score(aa.scale_adapted(df, ait, ref)), ath, atl, W_DUR, causal=True)
    arm_cache = {}

    def arm_at(eps):
        if eps not in arm_cache:
            df = te
            if eps > 0:
                df = te.copy()
                df.loc[atk_np, ait] = (te[ait] - (te[ait] - base).clip(-eps, eps)).loc[atk_np]
            arm_cache[eps] = arm_pred(df)
        return arm_cache[eps]

    rows = []

    def record(threat, objective, eps, X, x_adv):
        pa, pr = array_pred(X), arm_at(eps)
        d = (x_adv - x0) / scale
        diag = {'max_sensor_shift_sd': float(d[:, ~act].abs().max()),
                'mean_sensor_shift_sd': float(d[:, ~act].abs().mean()),
                'rows_actuator_spoofed': float((x_adv[:, act] != x0[:, act]).any(1).float().mean()),
                'out_of_range_cells': int(((x_adv < lo_phys - 1e-5) | (x_adv > hi_phys + 1e-5)).sum())}
        for tag, p in (('array', pa), ('analyzer arm', pr), ('union', np.maximum(pa, pr))):
            s, _ = evaluate(y, p)
            ch, ch95 = lfp.chance_events(p, y, n=100)
            rows.append({'threat': threat, 'objective': objective, 'epsilon': eps, 'detector': tag,
                         'events': s['n_detected'], 'n_events': s['n_events'], 'chance': ch, 'chance_p95': ch95,
                         'recall': s['recall'], 'F1': s['f1'], 'FAR': s['false_alarm'], **diag})
        r = rows[-3:]
        print(f'  {threat:>13} {objective:>8} eps={eps:>5}: array {r[0]["events"]:>2} | arm {r[1]["events"]:>2} | '
              f'union {r[2]["events"]:>2} of {r[2]["n_events"]} (chance {r[2]["chance"]:.1f}) | '
              f'max sensor shift {diag["max_sensor_shift_sd"]:.2f} SD, actuator-spoofed rows '
              f'{diag["rows_actuator_spoofed"]:.1%}, out-of-range cells {diag["out_of_range_cells"]}', flush=True)

    record('clean', '-', 0.0, X0, x0)
    for threat in args.threats.split(','):
        for eps in epsilons:
            results = []
            for objective in args.objectives.split(','):
                if threat == 'actuators':
                    xr = round_actuators(pgd(eps, threat, objective, free_act=True))
                    spoof = pgd(eps, 'sensors', objective, x_start=xr)  # sensors only, rounded states fixed
                    replay = pgd(eps, 'sensors', objective, x_start=x_replay)
                    x_adv = pick_per_event([x0, pgd(eps, 'sensors', objective), spoof, replay])
                else:
                    x_adv = pgd(eps, threat, objective)
                results.append(x_adv)
                X = X0.clone(); X[atk] = x_adv
                record(threat, objective, eps, X, x_adv)
            if len(results) > 1:
                x_adv = pick_per_event([x0] + results)
                X = X0.clone(); X[atk] = x_adv
                record(threat, 'best', eps, X, x_adv)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f'[SAVED] {args.out}')


if __name__ == '__main__':
    main()
