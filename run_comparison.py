"""Complete comparative analysis of single FBCCA, CCA, and fusion algorithms for 3 subjects.

Constraints strictly followed:
1. Data: E:\\BUAA\\ssvep_mi\\data\\20261027 (3 subjects: YDL, YZL, FHY; 160 trials each, 8 blocks x 20 trials).
2. Experiment & Paradigms: single_finger_MI_SSVEP_DualHost (frequencies: 12, 10, 8.5714, 7.5 Hz; 4 classes).
3. Methods & Algorithms: referenced directly from MI_SSVEP_data_analysis.
4. Algorithms compared: Single FBCCA, Single CCA, MI (FBCSP+LDA), and multiple Fusion algorithms.
5. Mode: central_raw only (no projecting out SSVEP frequencies, no greedy lead selection).
6. Time windows: start at 0.0s and start at 0.5s.
7. SSVEP leads: 3 leads [O1, Oz, O2] and 2 leads [O1, O2] (avoiding 9-lead ceiling).
"""
from runtime import arguments, windows
from windowing import (
    accumulate_log_probs,
    cca_scores,
    covariance,
    extract_dual_features,
    fit_temperature,
    format_window_span,
    logp,
    reference,
    resolve_window_configs,
    softmax,
)
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import linalg, optimize, special
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Configuration & Constants
FS = 250
HARMONICS = 3
NOMINAL_FREQUENCIES = [12.0, 10.0, 60.0 / 7.0, 7.5]  # thumb, index, middle, little
CLASS_NAMES = ['thumb', 'index', 'middle', 'little']

CENTRAL = ['C3', 'C4', 'FC3', 'FC4', 'CP3', 'CP4', 'C1', 'C2', 'C5', 'C6',
           'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6']

FBCCA_BANDS = [(6, 45), (10, 45), (14, 45), (18, 45), (22, 45)]
FB_WEIGHTS = np.arange(1, len(FBCCA_BANDS) + 1)**(-1.25) + 0.25

MI_BROAD_BAND = (8, 30)
MI_SUB_BANDS = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
ALL_MI_BANDS = [MI_BROAD_BAND] + MI_SUB_BANDS


def shrink(c, amount=0.1):
    c = np.asarray(c)
    n = c.shape[-1]
    return (1 - amount) * c + amount * np.trace(c, axis1=-2, axis2=-1)[..., None, None] * np.eye(n) / n


def normtrace(c):
    return c / np.maximum(np.trace(c, axis1=-2, axis2=-1)[..., None, None], 1e-12)

# Models
class FBCSPLDAModel:
    def __init__(self, k=8, reg=0.1):
        self.k = k
        self.reg = reg

    def fit(self, covs, y):
        self.filters_ = []
        for band in range(1, len(ALL_MI_BANDS)):
            z = normtrace(covs[:, band])
            sub_filters = []
            for label in range(4):
                a = shrink(z[y == label].mean(0), self.reg)
                b = shrink(z[y != label].mean(0), self.reg)
                _, w = linalg.eigh(a, a + b)
                sub_filters.extend([w[:, 0], w[:, -1]])
            self.filters_.append(np.array(sub_filters))

        x = self._features(covs)
        self.sel = SelectKBest(f_classif, k=min(self.k, x.shape[1])).fit(x, y)
        x_sel = self.sel.transform(x)
        self.scaler = StandardScaler().fit(x_sel)
        self.clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5).fit(self.scaler.transform(x_sel), y)
        return self

    def _features(self, covs):
        feats = []
        for band in range(1, len(ALL_MI_BANDS)):
            w = self.filters_[band - 1]
            z = normtrace(covs[:, band])
            f = np.log(np.maximum(np.einsum('kc,ncd,kd->nk', w, z, w), 1e-12))
            feats.append(f)
        return np.concatenate(feats, axis=1)

    def transform(self, covs):
        return self.sel.transform(self._features(covs))

    def predict_proba(self, covs):
        x = self.scaler.transform(self.transform(covs))
        classes = self.clf.classes_
        p = self.clf.predict_proba(x)
        return p[:, np.argsort(classes)]

class FixedFusion:
    def __init__(self, weight=0.5):
        self.weight = weight

    def predict_proba(self, ps, pm):
        return (1 - self.weight) * ps + self.weight * pm

class EvidenceFusion:
    def __init__(self):
        self.coef = np.r_[1.0, 0.0, np.zeros(4)]

    def fit(self, ps, pm, y):
        ls = logp(ps)
        lm = logp(pm)
        n = len(y)
        def value_grad(v):
            a, b = v[:2]
            bias = v[2:]
            z = a * ls + b * lm + bias
            p = softmax(z)
            loss = -np.mean(np.log(np.clip(p[np.arange(n), y], 1e-12, 1)))
            loss += 0.02 * ((a - 1)**2 + b**2) + 0.05 * np.sum(bias**2)
            e = p.copy()
            e[np.arange(n), y] -= 1
            e /= n
            grad = np.r_[np.sum(e * ls) + 0.04 * (a - 1), np.sum(e * lm) + 0.04 * b, e.sum(0) + 0.1 * bias]
            return loss, grad

        bounds = [(0, 5), (0, 5)] + [(-2, 2)] * 4
        res = optimize.minimize(value_grad, np.r_[1.0, 0.0, np.zeros(4)], jac=True, bounds=bounds, method='L-BFGS-B')
        if res.success:
            self.coef = res.x
        return self

    def predict_proba(self, ps, pm):
        z = self.coef[0] * logp(ps) + self.coef[1] * logp(pm) + self.coef[2:]
        return softmax(z)

class AnchoredEvidenceFusion:
    def __init__(self, reg=0.1):
        self.reg = reg
        self.b = 0.0

    def fit(self, ps, pm, y):
        ls = logp(ps)
        lm = logp(pm)
        ix = np.arange(len(y))
        def loss(b):
            p = softmax(ls + b * lm)
            return -np.mean(np.log(np.clip(p[ix, y], 1e-12, 1))) + self.reg * b * b
        fit = optimize.minimize_scalar(loss, bounds=(0, 3), method='bounded')
        self.b = float(fit.x) if (fit.success and loss(fit.x) < loss(0) - 1e-10) else 0.0
        return self

    def predict_proba(self, ps, pm):
        if self.b == 0.0:
            return np.asarray(ps).copy()
        return softmax(logp(ps) + self.b * logp(pm))

class ReliabilityGate:
    def _entropy(self, p):
        return -np.sum(p * logp(p), axis=1) / np.log(4)

    def _design(self, ps, pm):
        return np.column_stack([np.ones(len(ps)), self._entropy(ps), self._entropy(pm)])

    def fit(self, ps, pm, y):
        x = self._design(ps, pm)
        n = len(y)
        ix = np.arange(n)
        s = ps[ix, y]
        m = pm[ix, y]
        prior = np.array([-2.0, 0.0, 0.0])

        def fun(v):
            w = special.expit(x @ v)
            p = (1 - w) * s + w * m
            loss = -np.mean(np.log(np.clip(p, 1e-12, 1))) + 0.01 * np.sum((v - prior)**2)
            g = x.T @ ((s - m) / np.clip(p, 1e-12, 1) * w * (1 - w)) / n + 0.02 * (v - prior)
            return loss, g

        res = optimize.minimize(fun, prior, jac=True, bounds=[(-8, 4), (0, 8), (-8, 0)], method='L-BFGS-B')
        self.coef = res.x if res.success else prior
        return self

    def predict_proba(self, ps, pm):
        w = special.expit(self._design(ps, pm) @ self.coef)[:, None]
        return (1 - w) * ps + w * pm

class StackFusion:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=0.1, max_iter=2000)

    def fit(self, ps, pm, y):
        x = np.column_stack([logp(ps), logp(pm)])
        self.scaler.fit(x)
        self.clf.fit(self.scaler.transform(x), y)
        return self

    def predict_proba(self, ps, pm):
        x = np.column_stack([logp(ps), logp(pm)])
        p = self.clf.predict_proba(self.scaler.transform(x))
        return p[:, np.argsort(self.clf.classes_)]

class FeatureFusionModel:
    def __init__(self, k=8, C=0.1):
        self.k = k
        self.C = C

    def fit(self, mi_feats, ss_scores, y):
        self.mi_scaler = StandardScaler().fit(mi_feats)
        self.ss_scaler = StandardScaler().fit(ss_scores)
        x = np.column_stack([self.mi_scaler.transform(mi_feats), self.ss_scaler.transform(ss_scores)])
        self.clf = LogisticRegression(C=self.C, max_iter=2000).fit(x, y)
        return self

    def predict_proba(self, mi_feats, ss_scores):
        x = np.column_stack([self.mi_scaler.transform(mi_feats), self.ss_scaler.transform(ss_scores)])
        p = self.clf.predict_proba(x)
        return p[:, np.argsort(self.clf.classes_)]

class SelectiveCorrection:
    @staticmethod
    def _evidence(ps, pm):
        s = ps.argmax(1)
        m = pm.argmax(1)
        ix = np.arange(len(ps))
        odds = np.log(np.clip(pm[ix, m], 1e-12, 1) / np.clip(pm[ix, s], 1e-12, 1))
        return s, m, odds

    def _mask(self, ps, pm, margin, rule):
        s, m, odds = self._evidence(ps, pm)
        return ((s != m) & (ps.max(1) <= rule['ss_confidence_max']) &
                (np.asarray(margin) <= rule['ss_margin_max']) & (pm.max(1) >= 0.4) &
                (odds >= rule['mi_log_odds_min']))

    def fit(self, ps, pm, y, margin, groups):
        self.rule_ = None
        s, m, _ = self._evidence(ps, pm)
        best = (0, 0, 0)
        for conf in [0.5, 0.65, 0.8, 0.95]:
            for gap in np.unique(np.quantile(margin, [0.25, 0.5, 1.0])):
                for odds in [0.5, 1.0, 1.5]:
                    rule = {'ss_confidence_max': conf, 'ss_margin_max': float(gap), 'mi_log_odds_min': odds}
                    change = self._mask(ps, pm, margin, rule)
                    gain = (m == y).astype(int) - (s == y).astype(int)
                    perblock = {str(b): int(gain[change & (groups == b)].sum()) for b in np.unique(groups)}
                    net = int(gain[change].sum())
                    harm = int(np.sum(change & (s == y) & (m != y)))
                    valid = int(change.sum()) >= 3 and net >= 2 and all(v >= 0 for v in perblock.values())
                    rank = (net, -harm, -int(change.sum()))
                    if valid and rank > best:
                        self.rule_ = rule
                        best = rank
        return self

    def predict_proba(self, ps, pm, margin):
        if self.rule_ is None:
            return np.asarray(ps).copy()
        p = np.asarray(ps).copy()
        switch = self._mask(ps, pm, margin, self.rule_)
        p[switch] = pm[switch]
        return p

def _init_comparison_preds(n_trials):
    methods = (
        'ss_cca', 'ss_fbcca', 'mi_fbcsp',
        'fusion_fixed_cca', 'fusion_fixed_fbcca',
        'fusion_evidence_cca', 'fusion_evidence_fbcca',
        'fusion_evidence_anchored',
        'fusion_gate_cca', 'fusion_gate_fbcca',
        'fusion_stack_cca', 'fusion_stack_fbcca',
        'fusion_feature_fbcsp_fbcca',
        'fusion_selective_correction',
    )
    return {m: np.zeros((n_trials, 4)) for m in methods}


def _calibrate_comparison_fold(covs, raw_ss_cca, raw_ss_fbcca, y, blocks, train_idx, test_idx):
    mi_model = FBCSPLDAModel(k=8, reg=0.1).fit(covs[train_idx], y[train_idx])
    test_mi_p = mi_model.predict_proba(covs[test_idx])

    oof_mi_p = np.zeros((len(train_idx), 4))
    train_blocks = blocks[train_idx]
    for inner_b in np.unique(train_blocks):
        inner_val_sub = np.where(train_blocks == inner_b)[0]
        inner_tr_sub = np.where(train_blocks != inner_b)[0]
        sub_mi_model = FBCSPLDAModel(k=8, reg=0.1).fit(covs[train_idx[inner_tr_sub]], y[train_idx[inner_tr_sub]])
        oof_mi_p[inner_val_sub] = sub_mi_model.predict_proba(covs[train_idx[inner_val_sub]])

    tm = fit_temperature(logp(oof_mi_p), y[train_idx], bounds=(0.25, 8.0))
    ts_cca = fit_temperature(raw_ss_cca[train_idx], y[train_idx], bounds=(0.005, 2.0))
    ts_fbcca = fit_temperature(raw_ss_fbcca[train_idx], y[train_idx], bounds=(0.005, 2.0))

    s_sort_tr = np.sort(raw_ss_fbcca[train_idx], axis=1)
    s_sort_te = np.sort(raw_ss_fbcca[test_idx], axis=1)
    return {
        'pm_tr': softmax(logp(oof_mi_p) / tm),
        'pm_te': softmax(logp(test_mi_p) / tm),
        'ps_cca_tr': softmax(raw_ss_cca[train_idx] / ts_cca),
        'ps_cca_te': softmax(raw_ss_cca[test_idx] / ts_cca),
        'ps_fbcca_tr': softmax(raw_ss_fbcca[train_idx] / ts_fbcca),
        'ps_fbcca_te': softmax(raw_ss_fbcca[test_idx] / ts_fbcca),
        'mi_feats_tr': mi_model.transform(covs[train_idx]),
        'mi_feats_te': mi_model.transform(covs[test_idx]),
        'ss_raw_tr': raw_ss_fbcca[train_idx],
        'ss_raw_te': raw_ss_fbcca[test_idx],
        'margin_tr': s_sort_tr[:, -1] - s_sort_tr[:, -2],
        'margin_te': s_sort_te[:, -1] - s_sort_te[:, -2],
    }


def _accumulate_comparison_fold(seg_folds_prefix, y_tr):
    if len(seg_folds_prefix) == 1:
        return seg_folds_prefix[0]
    pm_tr, pm_te, _ = accumulate_log_probs(
        [f['pm_tr'] for f in seg_folds_prefix], [f['pm_te'] for f in seg_folds_prefix], y_tr)
    ps_cca_tr, ps_cca_te, _ = accumulate_log_probs(
        [f['ps_cca_tr'] for f in seg_folds_prefix], [f['ps_cca_te'] for f in seg_folds_prefix], y_tr)
    ps_fbcca_tr, ps_fbcca_te, _ = accumulate_log_probs(
        [f['ps_fbcca_tr'] for f in seg_folds_prefix], [f['ps_fbcca_te'] for f in seg_folds_prefix], y_tr)

    ss_raw_tr = np.mean([f['ss_raw_tr'] for f in seg_folds_prefix], axis=0)
    ss_raw_te = np.mean([f['ss_raw_te'] for f in seg_folds_prefix], axis=0)
    s_sort_tr = np.sort(ss_raw_tr, axis=1)
    s_sort_te = np.sort(ss_raw_te, axis=1)
    return {
        'pm_tr': pm_tr, 'pm_te': pm_te,
        'ps_cca_tr': ps_cca_tr, 'ps_cca_te': ps_cca_te,
        'ps_fbcca_tr': ps_fbcca_tr, 'ps_fbcca_te': ps_fbcca_te,
        'mi_feats_tr': np.mean([f['mi_feats_tr'] for f in seg_folds_prefix], axis=0),
        'mi_feats_te': np.mean([f['mi_feats_te'] for f in seg_folds_prefix], axis=0),
        'ss_raw_tr': ss_raw_tr, 'ss_raw_te': ss_raw_te,
        'margin_tr': s_sort_tr[:, -1] - s_sort_tr[:, -2],
        'margin_te': s_sort_te[:, -1] - s_sort_te[:, -2],
    }


def _eval_comparison_fold(fd, y_tr, groups_tr, fold_preds, test_idx):
    pm_tr, pm_te = fd['pm_tr'], fd['pm_te']
    ps_cca_tr, ps_cca_te = fd['ps_cca_tr'], fd['ps_cca_te']
    ps_fbcca_tr, ps_fbcca_te = fd['ps_fbcca_tr'], fd['ps_fbcca_te']

    fold_preds['ss_cca'][test_idx] = ps_cca_te
    fold_preds['ss_fbcca'][test_idx] = ps_fbcca_te
    fold_preds['mi_fbcsp'][test_idx] = pm_te

    fold_preds['fusion_fixed_cca'][test_idx] = FixedFusion(0.5).predict_proba(ps_cca_te, pm_te)
    fold_preds['fusion_fixed_fbcca'][test_idx] = FixedFusion(0.5).predict_proba(ps_fbcca_te, pm_te)

    fold_preds['fusion_evidence_cca'][test_idx] = EvidenceFusion().fit(ps_cca_tr, pm_tr, y_tr).predict_proba(ps_cca_te, pm_te)
    fold_preds['fusion_evidence_fbcca'][test_idx] = EvidenceFusion().fit(ps_fbcca_tr, pm_tr, y_tr).predict_proba(ps_fbcca_te, pm_te)
    fold_preds['fusion_evidence_anchored'][test_idx] = AnchoredEvidenceFusion(0.1).fit(ps_fbcca_tr, pm_tr, y_tr).predict_proba(ps_fbcca_te, pm_te)

    fold_preds['fusion_gate_cca'][test_idx] = ReliabilityGate().fit(ps_cca_tr, pm_tr, y_tr).predict_proba(ps_cca_te, pm_te)
    fold_preds['fusion_gate_fbcca'][test_idx] = ReliabilityGate().fit(ps_fbcca_tr, pm_tr, y_tr).predict_proba(ps_fbcca_te, pm_te)

    fold_preds['fusion_stack_cca'][test_idx] = StackFusion().fit(ps_cca_tr, pm_tr, y_tr).predict_proba(ps_cca_te, pm_te)
    fold_preds['fusion_stack_fbcca'][test_idx] = StackFusion().fit(ps_fbcca_tr, pm_tr, y_tr).predict_proba(ps_fbcca_te, pm_te)

    ff_model = FeatureFusionModel(k=8, C=0.1).fit(fd['mi_feats_tr'], fd['ss_raw_tr'], y_tr)
    fold_preds['fusion_feature_fbcsp_fbcca'][test_idx] = ff_model.predict_proba(fd['mi_feats_te'], fd['ss_raw_te'])

    sc_model = SelectiveCorrection().fit(ps_fbcca_tr, pm_tr, y_tr, fd['margin_tr'], groups_tr)
    fold_preds['fusion_selective_correction'][test_idx] = sc_model.predict_proba(ps_fbcca_te, pm_te, fd['margin_te'])


def _record_comparison_outputs(fold_preds, l_key, occ_leads, meta, sub, y, blocks, trial_ids,
                               all_results, trial_predictions, extended_meta):
    n_trials = len(y)
    for m_name, preds_arr in fold_preds.items():
        pred_labels = preds_arr.argmax(axis=1)
        acc = float(np.mean(pred_labels == y))
        res_row = {'lead_config': l_key, 'leads': '+'.join(occ_leads)}
        if extended_meta:
            res_row.update({'window_mode': meta['window_mode'], 'scheme': meta['scheme']})
        res_row.update({'window_start_s': meta['window_start_s'], 'duration_s': meta['duration_s']})
        if extended_meta:
            res_row.update({
                'decision_time_s': meta['decision_time_s'],
                'ssvep_window': meta['ssvep_window'],
                'mi_window': meta['mi_window'],
                'segment_index': meta['segment_index'],
            })
        res_row.update({
            'subject': sub, 'method': m_name, 'accuracy': acc,
            'correct_count': int(np.sum(pred_labels == y)), 'total_trials': n_trials,
        })
        all_results.append(res_row)

        for ti in range(n_trials):
            pred_row = {'lead_config': l_key}
            if extended_meta:
                pred_row.update({
                    'window_mode': meta['window_mode'], 'scheme': meta['scheme'],
                    'segment_index': meta['segment_index'], 'decision_time_s': meta['decision_time_s'],
                })
            pred_row.update({
                'window_start_s': meta['window_start_s'], 'duration_s': meta['duration_s'],
                'subject': sub, 'trial': int(trial_ids[ti]), 'block': int(blocks[ti]),
                'target': int(y[ti]), 'target_name': CLASS_NAMES[y[ti]],
                'method': m_name, 'prediction': int(pred_labels[ti]),
                'prediction_name': CLASS_NAMES[pred_labels[ti]],
                'correct': bool(pred_labels[ti] == y[ti]),
                'p0': float(preds_arr[ti, 0]), 'p1': float(preds_arr[ti, 1]),
                'p2': float(preds_arr[ti, 2]), 'p3': float(preds_arr[ti, 3]),
            })
            trial_predictions.append(pred_row)


def run_experiment(cache_dir, out_dir, subjects=None, window_configs=None):
    cache_dir = Path(cache_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = subjects or ['YDL', 'YZL', 'FHY']
    loaded_data = {}
    for sub in subjects:
        data = np.load(cache_dir / f'{sub}_preprocessed.npz', allow_pickle=True)
        loaded_data[sub] = {
            'epochs': data['ds_epochs'],
            'y': data['y'],
            'blocks': data['blocks'],
            'trial_ids': data['trial_ids'],
            'ons': data['ons'],
            'ch_labels': list(data['ch_labels']),
        }

    lead_configs = {'3leads': ['O1', 'Oz', 'O2'], '2leads': ['O1', 'O2']}
    window_configs = window_configs or resolve_window_configs([0.0, 0.5], None, ('sync',))
    extended_meta = any(wc.get('mode', 'sync') != 'sync' for wc in window_configs)

    all_results = []
    trial_predictions = []

    print("Starting full evaluation...")
    t_start = time.time()

    for l_key, occ_leads in lead_configs.items():
        print(f"\n==========================================")
        print(f"EVALUATING SSVEP LEAD SET: {l_key} ({occ_leads})")
        print(f"==========================================")

        for w_conf in window_configs:
            w_mode = w_conf.get('mode', 'sync')
            w_name = w_conf['name']
            specs = w_conf.get('specs') or [
                {
                    'mode': 'sync', 'scheme': w_name,
                    'window_start_s': w_conf['start_s'], 'duration_s': d, 'cum_duration_s': d,
                    'decision_time_s': round(w_conf['start_s'] + d, 6),
                    'ssvep_window': (w_conf['start_s'], round(w_conf['start_s'] + d, 6)),
                    'mi_window': (w_conf['start_s'], round(w_conf['start_s'] + d, 6)),
                    'segment_index': None,
                }
                for d in w_conf['durations']
            ]
            print(f"\n--- Window Mode: {w_mode} ({w_name}) ---")

            if w_mode == 'equal_segmented':
                n_segs = len(specs)
                ss_lat = specs[0]['ssvep_latency_s']
                mi_lat = specs[0]['mi_latency_s']
                step_metas = [
                    {
                        'window_mode': 'equal_segmented_step', 'scheme': 'equal_segmented_step',
                        'window_start_s': sp['window_start_s'], 'duration_s': sp['duration_s'],
                        'decision_time_s': sp['decision_time_s'],
                        'ssvep_window': format_window_span(sp['ssvep_window']),
                        'mi_window': format_window_span(sp['mi_window']),
                        'segment_index': sp['segment_index'],
                    }
                    for sp in specs
                ]
                accum_metas = [
                    {
                        'window_mode': 'equal_segmented_accum', 'scheme': 'equal_segmented_accum',
                        'window_start_s': mi_lat, 'duration_s': sp['cum_duration_s'],
                        'decision_time_s': sp['decision_time_s'],
                        'ssvep_window': f"[{ss_lat:.2f}, {sp['ssvep_window'][1]:.2f}]",
                        'mi_window': f"[{mi_lat:.2f}, {sp['mi_window'][1]:.2f}]",
                        'segment_index': sp['segment_index'],
                    }
                    for sp in specs
                ]

                for sub in subjects:
                    sub_t0 = time.time()
                    print(f"  Evaluating {sub} ({n_segs} segments) ...", end='', flush=True)
                    s_data = loaded_data[sub]
                    y, blocks, ons = s_data['y'], s_data['blocks'], s_data['ons']
                    n_trials = len(y)
                    seg_feats = [
                        extract_dual_features(
                            s_data['epochs'], ons, s_data['ch_labels'], occ_leads, CENTRAL,
                            sp['ssvep_window'], sp['mi_window'], ALL_MI_BANDS, FBCCA_BANDS,
                            FB_WEIGHTS, NOMINAL_FREQUENCIES, FS, HARMONICS,
                            segment_index=sp['segment_index'], extract_cca=True,
                        )
                        for sp in specs
                    ]
                    step_preds = [_init_comparison_preds(n_trials) for _ in range(n_segs)]
                    accum_preds = [_init_comparison_preds(n_trials) for _ in range(n_segs)]

                    for test_b in np.unique(blocks):
                        test_idx = np.where(blocks == test_b)[0]
                        train_idx = np.where(blocks != test_b)[0]
                        y_tr, groups_tr = y[train_idx], blocks[train_idx]
                        fold_segs = []
                        for k in range(n_segs):
                            covs_k, raw_fb_k, raw_cca_k = seg_feats[k]
                            fd_k = _calibrate_comparison_fold(
                                covs_k, raw_cca_k, raw_fb_k, y, blocks, train_idx, test_idx,
                            )
                            fold_segs.append(fd_k)
                            _eval_comparison_fold(fd_k, y_tr, groups_tr, step_preds[k], test_idx)
                            fd_acc = _accumulate_comparison_fold(fold_segs, y_tr)
                            _eval_comparison_fold(fd_acc, y_tr, groups_tr, accum_preds[k], test_idx)

                    for k in range(n_segs):
                        _record_comparison_outputs(
                            step_preds[k], l_key, occ_leads, step_metas[k], sub, y,
                            blocks, s_data['trial_ids'], all_results, trial_predictions, extended_meta,
                        )
                        _record_comparison_outputs(
                            accum_preds[k], l_key, occ_leads, accum_metas[k], sub, y,
                            blocks, s_data['trial_ids'], all_results, trial_predictions, extended_meta,
                        )
                    print(f" done ({time.time() - sub_t0:.1f}s)")
                continue

            for sp in specs:
                dur = sp['duration_s']
                print(f"  Evaluating duration: {dur:.1f}s ...", end='', flush=True)
                dur_t0 = time.time()
                meta = {
                    'window_mode': sp['mode'], 'scheme': sp['scheme'],
                    'window_start_s': sp['window_start_s'], 'duration_s': dur,
                    'decision_time_s': sp['decision_time_s'],
                    'ssvep_window': format_window_span(sp['ssvep_window']),
                    'mi_window': format_window_span(sp['mi_window']),
                    'segment_index': '' if sp['segment_index'] is None else sp['segment_index'],
                }
                for sub in subjects:
                    s_data = loaded_data[sub]
                    y, blocks, ons = s_data['y'], s_data['blocks'], s_data['ons']
                    covs, raw_ss_fbcca, raw_ss_cca = extract_dual_features(
                        s_data['epochs'], ons, s_data['ch_labels'], occ_leads, CENTRAL,
                        sp['ssvep_window'], sp['mi_window'], ALL_MI_BANDS, FBCCA_BANDS,
                        FB_WEIGHTS, NOMINAL_FREQUENCIES, FS, HARMONICS, extract_cca=True,
                    )
                    fold_preds = _init_comparison_preds(len(y))
                    for test_b in np.unique(blocks):
                        test_idx = np.where(blocks == test_b)[0]
                        train_idx = np.where(blocks != test_b)[0]
                        fd = _calibrate_comparison_fold(
                            covs, raw_ss_cca, raw_ss_fbcca, y, blocks, train_idx, test_idx,
                        )
                        _eval_comparison_fold(fd, y[train_idx], blocks[train_idx], fold_preds, test_idx)
                    _record_comparison_outputs(
                        fold_preds, l_key, occ_leads, meta, sub, y,
                        blocks, s_data['trial_ids'], all_results, trial_predictions, extended_meta,
                    )
                print(f" done ({time.time() - dur_t0:.1f}s)")

    df_results = pd.DataFrame(all_results)
    df_preds = pd.DataFrame(trial_predictions)

    df_results.to_csv(out_dir / 'all_metrics.csv', index=False, encoding='utf-8-sig')
    df_preds.to_csv(out_dir / 'all_predictions.csv', index=False, encoding='utf-8-sig')

    for l_key in lead_configs:
        if not extended_meta:
            for w_conf in window_configs:
                w_start = w_conf['start_s']
                sub_df = df_results[(df_results['lead_config'] == l_key) & (df_results['window_start_s'] == w_start)]
                tag = w_conf['name']
                sub_df.pivot_table(index=['subject', 'duration_s'], columns='method', values='accuracy').to_csv(
                    out_dir / f'summary_{l_key}_{tag}.csv', encoding='utf-8-sig')
                sub_df.pivot_table(index='duration_s', columns='method', values='accuracy', aggfunc='mean').to_csv(
                    out_dir / f'mean_summary_{l_key}_{tag}.csv', encoding='utf-8-sig')
        else:
            for scheme_key, sub_df in df_results[df_results['lead_config'] == l_key].groupby('scheme'):
                idx_cols = ['subject', 'segment_index', 'duration_s'] if scheme_key == 'equal_segmented_step' else ['subject', 'duration_s']
                mean_idx = ['segment_index', 'duration_s'] if scheme_key == 'equal_segmented_step' else 'duration_s'
                sub_df.pivot_table(index=idx_cols, columns='method', values='accuracy').to_csv(
                    out_dir / f'summary_{l_key}_{scheme_key}.csv', encoding='utf-8-sig')
                sub_df.pivot_table(index=mean_idx, columns='method', values='accuracy', aggfunc='mean').to_csv(
                    out_dir / f'mean_summary_{l_key}_{scheme_key}.csv', encoding='utf-8-sig')

    print(f"\nAll evaluations complete in {time.time() - t_start:.1f}s!")
    print(f"Results saved to: {out_dir}")


if __name__ == '__main__':
    args = arguments('comparison')
    run_experiment(args.cache_dir, args.out_dir, args.subjects, args.resolved_windows)
