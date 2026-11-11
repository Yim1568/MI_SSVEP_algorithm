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
import sys
from runtime import arguments, windows
from diagnostics import Diagnostics
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import linalg, optimize, signal, special
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

# Helper Functions
def reference(n, fs, freq, harmonics=3):
    t = np.arange(n) / fs
    hs = [h for h in range(1, int(harmonics) + 1) if freq * h < fs / 2]
    return np.column_stack([fun(2 * np.pi * freq * h * t) for h in hs for fun in [np.sin, np.cos]])

def cca_scores(x, freqs, fs=250, harmonics=3):
    x = np.asarray(x, float)
    if x.ndim != 2 or len(x) < 3 or x.shape[1] < 1:
        return np.zeros(len(freqs))
    x = x - x.mean(0)
    u, s, _ = linalg.svd(x, full_matrices=False)
    u = u[:, s > max(float(s.max(initial=0)) * 1e-8, 1e-12)]
    values = []
    for f in freqs:
        r = reference(len(x), fs, f, harmonics)
        r -= r.mean(0)
        q, sv, _ = linalg.svd(r, full_matrices=False)
        q = q[:, sv > max(float(sv.max(initial=0)) * 1e-8, 1e-12)]
        values.append(float(np.clip(linalg.svdvals(u.T @ q)[0], 0, 1)) if u.shape[1] and q.shape[1] else 0.0)
    return np.array(values)

def softmax(z):
    return special.softmax(z, axis=1)

def logp(p):
    return np.log(np.clip(p, 1e-8, 1))

def shrink(c, amount=0.1):
    c = np.asarray(c)
    n = c.shape[-1]
    return (1 - amount) * c + amount * np.trace(c, axis1=-2, axis2=-1)[..., None, None] * np.eye(n) / n

def normtrace(c):
    return c / np.maximum(np.trace(c, axis1=-2, axis2=-1)[..., None, None], 1e-12)

def covariance(x):
    x = x - x.mean(0)
    return x.T @ x / (len(x) - 1)

def fit_temperature(z, y, bounds=(0.005, 8.0)):
    def fun(logt):
        return -np.mean(np.log(np.clip(softmax(z / np.exp(logt))[np.arange(len(y)), y], 1e-12, 1)))
    opt = optimize.minimize_scalar(fun, bounds=np.log(bounds), method='bounded')
    return float(np.exp(opt.x)) if opt.success else 1.0

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
            'ch_labels': list(data['ch_labels'])
        }

    # SSVEP Lead sets to evaluate:
    lead_configs = {
        '3leads': ['O1', 'Oz', 'O2'],
        '2leads': ['O1', 'O2']
    }

    # Window configurations
    window_configs = window_configs or [
        {'name': 'start00', 'start_s': 0.0, 'durations': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]},
        {'name': 'start05', 'start_s': 0.5, 'durations': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]}
    ]

    all_results = []
    trial_predictions = []

    print("Starting full evaluation...")
    t_start = time.time()

    for l_key, occ_leads in lead_configs.items():
        print(f"\n==========================================")
        print(f"EVALUATING SSVEP LEAD SET: {l_key} ({occ_leads})")
        print(f"==========================================")

        for w_conf in window_configs:
            start_s = w_conf['start_s']
            durations = w_conf['durations']
            w_name = w_conf['name']
            print(f"\n--- Window Start: {start_s}s ({w_name}) ---")

            for dur in durations:
                print(f"  Evaluating duration: {dur:.1f}s ...", end='', flush=True)
                dur_t0 = time.time()

                for sub in subjects:
                    s_data = loaded_data[sub]
                    epochs = s_data['epochs']
                    y = s_data['y']
                    blocks = s_data['blocks']
                    trial_ids = s_data['trial_ids']
                    ons = s_data['ons']
                    ch_labels = s_data['ch_labels']

                    ci = [ch_labels.index(c) for c in CENTRAL]
                    oi = [ch_labels.index(c) for c in occ_leads]

                    n_trials = len(y)
                    # 1. Extract trial features for this window
                    covs = np.zeros((n_trials, len(ALL_MI_BANDS), len(CENTRAL), len(CENTRAL)))
                    raw_ss_cca = np.zeros((n_trials, 4))
                    raw_ss_fbcca = np.zeros((n_trials, 4))

                    for i in range(n_trials):
                        ds = epochs[i]
                        on = ons[i]
                        w_start = int(round(on + start_s * FS))
                        w_end = w_start + int(round(dur * FS))

                        x_central = ds[w_start:w_end, ci]
                        x_occ = ds[w_start:w_end, oi]

                        # MI covariances for all bands
                        for b_idx, band in enumerate(ALL_MI_BANDS):
                            xf = signal.sosfiltfilt(signal.butter(4, band, fs=FS, btype='bandpass', output='sos'), x_central, axis=0)
                            covs[i, b_idx] = covariance(xf)

                        # SSVEP CCA
                        raw_ss_cca[i] = cca_scores(x_occ, NOMINAL_FREQUENCIES, FS, HARMONICS)

                        # SSVEP FBCCA
                        fb_score = np.zeros(4)
                        for b_idx, (lo, hi) in enumerate(FBCCA_BANDS):
                            x_sub = signal.sosfiltfilt(signal.butter(4, [lo, hi], fs=FS, btype='bandpass', output='sos'), x_occ, axis=0)
                            sub_scores = cca_scores(x_sub, NOMINAL_FREQUENCIES, FS, HARMONICS)
                            fb_score += FB_WEIGHTS[b_idx] * (sub_scores ** 2)
                        raw_ss_fbcca[i] = fb_score

                    # 2. 8-Fold Cross-Validation (Leave-One-Block-Out)
                    unique_blocks = np.unique(blocks)
                    fold_preds = {m: np.zeros((n_trials, 4)) for m in [
                        'ss_cca', 'ss_fbcca', 'mi_fbcsp',
                        'fusion_fixed_cca', 'fusion_fixed_fbcca',
                        'fusion_evidence_cca', 'fusion_evidence_fbcca',
                        'fusion_evidence_anchored',
                        'fusion_gate_cca', 'fusion_gate_fbcca',
                        'fusion_stack_cca', 'fusion_stack_fbcca',
                        'fusion_feature_fbcsp_fbcca',
                        'fusion_selective_correction'
                    ]}

                    for test_b in unique_blocks:
                        test_idx = np.where(blocks == test_b)[0]
                        train_idx = np.where(blocks != test_b)[0]

                        # Fit MI model on full train_idx
                        mi_model = FBCSPLDAModel(k=8, reg=0.1).fit(covs[train_idx], y[train_idx])
                        test_mi_p = mi_model.predict_proba(covs[test_idx])

                        # Generate nested OOF for calibration and meta-model training
                        oof_mi_p = np.zeros((len(train_idx), 4))
                        train_blocks = blocks[train_idx]
                        for inner_b in np.unique(train_blocks):
                            inner_val_sub = np.where(train_blocks == inner_b)[0]
                            inner_tr_sub = np.where(train_blocks != inner_b)[0]
                            inner_val = train_idx[inner_val_sub]
                            inner_tr = train_idx[inner_tr_sub]

                            sub_mi_model = FBCSPLDAModel(k=8, reg=0.1).fit(covs[inner_tr], y[inner_tr])
                            oof_mi_p[inner_val_sub] = sub_mi_model.predict_proba(covs[inner_val])

                        # Temperature scaling for train OOF
                        tm = fit_temperature(logp(oof_mi_p), y[train_idx], bounds=(0.25, 8.0))
                        ts_cca = fit_temperature(raw_ss_cca[train_idx], y[train_idx], bounds=(0.005, 2.0))
                        ts_fbcca = fit_temperature(raw_ss_fbcca[train_idx], y[train_idx], bounds=(0.005, 2.0))

                        # Calibrated probabilities on train
                        pm_tr = softmax(logp(oof_mi_p) / tm)
                        ps_cca_tr = softmax(raw_ss_cca[train_idx] / ts_cca)
                        ps_fbcca_tr = softmax(raw_ss_fbcca[train_idx] / ts_fbcca)

                        # Margins for selective correction
                        s_sort = np.sort(raw_ss_fbcca[train_idx], axis=1)
                        margin_tr = s_sort[:, -1] - s_sort[:, -2]

                        # Calibrated test probabilities
                        pm_te = softmax(logp(test_mi_p) / tm)
                        ps_cca_te = softmax(raw_ss_cca[test_idx] / ts_cca)
                        ps_fbcca_te = softmax(raw_ss_fbcca[test_idx] / ts_fbcca)

                        s_sort_te = np.sort(raw_ss_fbcca[test_idx], axis=1)
                        margin_te = s_sort_te[:, -1] - s_sort_te[:, -2]

                        # Record standalone predictions
                        fold_preds['ss_cca'][test_idx] = ps_cca_te
                        fold_preds['ss_fbcca'][test_idx] = ps_fbcca_te
                        fold_preds['mi_fbcsp'][test_idx] = pm_te

                        # Fixed Fusion (w=0.5)
                        fold_preds['fusion_fixed_cca'][test_idx] = FixedFusion(0.5).predict_proba(ps_cca_te, pm_te)
                        fold_preds['fusion_fixed_fbcca'][test_idx] = FixedFusion(0.5).predict_proba(ps_fbcca_te, pm_te)

                        # Evidence Fusion
                        ev_cca = EvidenceFusion().fit(ps_cca_tr, pm_tr, y[train_idx])
                        fold_preds['fusion_evidence_cca'][test_idx] = ev_cca.predict_proba(ps_cca_te, pm_te)

                        ev_fbcca = EvidenceFusion().fit(ps_fbcca_tr, pm_tr, y[train_idx])
                        fold_preds['fusion_evidence_fbcca'][test_idx] = ev_fbcca.predict_proba(ps_fbcca_te, pm_te)

                        # Anchored Evidence Fusion (with FBCCA)
                        anch_fbcca = AnchoredEvidenceFusion(0.1).fit(ps_fbcca_tr, pm_tr, y[train_idx])
                        fold_preds['fusion_evidence_anchored'][test_idx] = anch_fbcca.predict_proba(ps_fbcca_te, pm_te)

                        # Reliability Gate
                        gate_cca = ReliabilityGate().fit(ps_cca_tr, pm_tr, y[train_idx])
                        fold_preds['fusion_gate_cca'][test_idx] = gate_cca.predict_proba(ps_cca_te, pm_te)

                        gate_fbcca = ReliabilityGate().fit(ps_fbcca_tr, pm_tr, y[train_idx])
                        fold_preds['fusion_gate_fbcca'][test_idx] = gate_fbcca.predict_proba(ps_fbcca_te, pm_te)

                        # Stacking
                        stack_cca = StackFusion().fit(ps_cca_tr, pm_tr, y[train_idx])
                        fold_preds['fusion_stack_cca'][test_idx] = stack_cca.predict_proba(ps_cca_te, pm_te)

                        stack_fbcca = StackFusion().fit(ps_fbcca_tr, pm_tr, y[train_idx])
                        fold_preds['fusion_stack_fbcca'][test_idx] = stack_fbcca.predict_proba(ps_fbcca_te, pm_te)

                        # Feature Fusion (FBCSP + FBCCA)
                        mi_feats_tr = mi_model.transform(covs[train_idx])
                        mi_feats_te = mi_model.transform(covs[test_idx])
                        ff_model = FeatureFusionModel(k=8, C=0.1).fit(mi_feats_tr, raw_ss_fbcca[train_idx], y[train_idx])
                        fold_preds['fusion_feature_fbcsp_fbcca'][test_idx] = ff_model.predict_proba(mi_feats_te, raw_ss_fbcca[test_idx])

                        # Selective Correction
                        sc_model = SelectiveCorrection().fit(ps_fbcca_tr, pm_tr, y[train_idx], margin_tr, blocks[train_idx])
                        fold_preds['fusion_selective_correction'][test_idx] = sc_model.predict_proba(ps_fbcca_te, pm_te, margin_te)

                    # Compute and record accuracies
                    for m_name, preds_arr in fold_preds.items():
                        pred_labels = preds_arr.argmax(axis=1)
                        acc = float(np.mean(pred_labels == y))

                        all_results.append({
                            'lead_config': l_key,
                            'leads': '+'.join(occ_leads),
                            'window_start_s': start_s,
                            'duration_s': dur,
                            'subject': sub,
                            'method': m_name,
                            'accuracy': acc,
                            'correct_count': int(np.sum(pred_labels == y)),
                            'total_trials': n_trials
                        })

                        # Save individual trial predictions for audit
                        for ti in range(n_trials):
                            trial_predictions.append({
                                'lead_config': l_key,
                                'window_start_s': start_s,
                                'duration_s': dur,
                                'subject': sub,
                                'trial': int(trial_ids[ti]),
                                'block': int(blocks[ti]),
                                'target': int(y[ti]),
                                'target_name': CLASS_NAMES[y[ti]],
                                'method': m_name,
                                'prediction': int(pred_labels[ti]),
                                'prediction_name': CLASS_NAMES[pred_labels[ti]],
                                'correct': bool(pred_labels[ti] == y[ti]),
                                'p0': float(preds_arr[ti, 0]),
                                'p1': float(preds_arr[ti, 1]),
                                'p2': float(preds_arr[ti, 2]),
                                'p3': float(preds_arr[ti, 3])
                            })

                print(f" done ({time.time() - dur_t0:.1f}s)")

    df_results = pd.DataFrame(all_results)
    df_preds = pd.DataFrame(trial_predictions)

    # Save full tables
    df_results.to_csv(out_dir / 'all_metrics.csv', index=False, encoding='utf-8-sig')
    df_preds.to_csv(out_dir / 'all_predictions.csv', index=False, encoding='utf-8-sig')

    # Generate Pivot Summary Tables
    for l_key in lead_configs:
        for w_conf in window_configs:
            w_start = w_conf['start_s']
            sub_df = df_results[(df_results['lead_config'] == l_key) & (df_results['window_start_s'] == w_start)]

            pivot = sub_df.pivot_table(
                index=['subject', 'duration_s'],
                columns='method',
                values='accuracy'
            )
            pivot.to_csv(out_dir / f'summary_{l_key}_start{int(w_start*10):02d}.csv', encoding='utf-8-sig')

            # Also create mean across subjects
            mean_pivot = sub_df.pivot_table(
                index='duration_s',
                columns='method',
                values='accuracy',
                aggfunc='mean'
            )
            mean_pivot.to_csv(out_dir / f'mean_summary_{l_key}_start{int(w_start*10):02d}.csv', encoding='utf-8-sig')

    print(f"\nAll evaluations complete in {time.time() - t_start:.1f}s!")
    print(f"Results saved to: {out_dir}")

if __name__ == '__main__':
    args = arguments('comparison')
    run_experiment(args.cache_dir, args.out_dir, args.subjects, windows(args))
