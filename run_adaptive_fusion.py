"""Subject-Adaptive MI-SSVEP Fusion under 2 Occipital Leads (O1, O2).

Key Features:
1. Implements multiple MI expert models:
   - Filter-bank power (fbpower_lda)
   - FBCSP + LDA (fbcsp)
   - Regularized CSP (rcsp)
   - Riemannian Tangent Space (tangent_logreg)
2. Implements two adaptive pipelines:
   - Subject-Specific Profiling (e.g., FHY uses Filter-bank power; YDL uses FBCSP-LDA)
   - Fold-internal Auto-Adaptive Selection (strictly out-of-fold cross-validation on training blocks to pick the best MI expert per window and fold)
3. Evaluates 2-lead SSVEP (O1, O2) with FBCCA under:
   - Window start 0.5s (durations: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5s)
   - Window start 0.0s (durations: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0s)
4. Evaluates fusion methods: Evidence, Stacking, Fixed (w=0.5).
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
from pyriemann.tangentspace import TangentSpace

# Constants
FS = 250
HARMONICS = 3
NOMINAL_FREQUENCIES = [12.0, 10.0, 60.0 / 7.0, 7.5]  # thumb, index, middle, little
CLASS_NAMES = ['thumb', 'index', 'middle', 'little']

CENTRAL = ['C3', 'C4', 'FC3', 'FC4', 'CP3', 'CP4', 'C1', 'C2', 'C5', 'C6',
           'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6']
OCC2 = ['O1', 'O2']

FBCCA_BANDS = [(6, 45), (10, 45), (14, 45), (18, 45), (22, 45)]
FB_WEIGHTS = np.arange(1, len(FBCCA_BANDS) + 1)**(-1.25) + 0.25

MI_BROAD_BAND = (8, 30)
MI_SUB_BANDS = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
ALL_MI_BANDS = [MI_BROAD_BAND] + MI_SUB_BANDS

# Math & Preprocessing helpers
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

def softmax(z): return special.softmax(z, axis=1)
def logp(p): return np.log(np.clip(p, 1e-8, 1))
def covariance(x): return (x - x.mean(0)).T @ (x - x.mean(0)) / (len(x) - 1)

def shrink(c, amount=0.1):
    c = np.asarray(c)
    n = c.shape[-1]
    return (1 - amount) * c + amount * np.trace(c, axis1=-2, axis2=-1)[..., None, None] * np.eye(n) / n

def normtrace(c):
    return c / np.maximum(np.trace(c, axis1=-2, axis2=-1)[..., None, None], 1e-12)

def fit_temperature(z, y, bounds=(0.005, 8.0)):
    def fun(logt):
        return -np.mean(np.log(np.clip(softmax(z / np.exp(logt))[np.arange(len(y)), y], 1e-12, 1)))
    opt = optimize.minimize_scalar(fun, bounds=np.log(bounds), method='bounded')
    return float(np.exp(opt.x)) if opt.success else 1.0

# MI Models
class FBPowerLDAModel:
    def fit(self, covs, y):
        x = np.log(np.maximum(np.diagonal(covs[:, 1:], axis1=-2, axis2=-1), 1e-12)).reshape(len(covs), -1)
        self.scaler = StandardScaler().fit(x)
        self.clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5).fit(self.scaler.transform(x), y)
        return self

    def predict_proba(self, covs):
        x = np.log(np.maximum(np.diagonal(covs[:, 1:], axis1=-2, axis2=-1), 1e-12)).reshape(len(covs), -1)
        p = self.clf.predict_proba(self.scaler.transform(x))
        return p[:, np.argsort(self.clf.classes_)]

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

    def predict_proba(self, covs):
        x_sel = self.sel.transform(self._features(covs))
        p = self.clf.predict_proba(self.scaler.transform(x_sel))
        return p[:, np.argsort(self.clf.classes_)]

class RCSPModel:
    def fit(self, covs, y):
        c_broad = normtrace(covs[:, 0])
        filters = []
        for lab in range(4):
            a = shrink(c_broad[y == lab].mean(0), 0.1)
            b = shrink(c_broad[y != lab].mean(0), 0.1)
            _, w = linalg.eigh(a, a + b)
            filters.extend([w[:, 0], w[:, 1], w[:, -2], w[:, -1]])
        self.w = np.array(filters)
        x = np.log(np.maximum(np.einsum('kc,ncd,kd->nk', self.w, c_broad, self.w), 1e-12))
        self.scaler = StandardScaler().fit(x)
        self.clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5).fit(self.scaler.transform(x), y)
        return self

    def predict_proba(self, covs):
        c_broad = normtrace(covs[:, 0])
        x = np.log(np.maximum(np.einsum('kc,ncd,kd->nk', self.w, c_broad, self.w), 1e-12))
        p = self.clf.predict_proba(self.scaler.transform(x))
        return p[:, np.argsort(self.clf.classes_)]

class TangentLogRegModel:
    def fit(self, covs, y):
        self.ts = TangentSpace(metric='riemann', tsupdate=False)
        x = self.ts.fit_transform(shrink(covs[:, 0], 0.1))
        self.scaler = StandardScaler().fit(x)
        self.clf = LogisticRegression(C=0.1, max_iter=2000).fit(self.scaler.transform(x), y)
        return self

    def predict_proba(self, covs):
        x = self.ts.transform(shrink(covs[:, 0], 0.1))
        p = self.clf.predict_proba(self.scaler.transform(x))
        return p[:, np.argsort(self.clf.classes_)]

# Factory of candidate MI models
MI_CANDIDATE_FACTORIES = {
    'fbpower_lda': FBPowerLDAModel,
    'fbcsp': FBCSPLDAModel,
    'rcsp': RCSPModel,
    'tangent_logreg': TangentLogRegModel
}

# Fusion models
def fit_evidence(ps, pm, y_t):
    ls = logp(ps); lm = logp(pm); n = len(y_t)
    def vg(v):
        a, b = v[:2]; bias = v[2:]; z = a*ls + b*lm + bias; p = softmax(z)
        loss = -np.mean(np.log(np.clip(p[np.arange(n), y_t], 1e-12, 1))) + 0.02*((a-1)**2 + b**2) + 0.05*np.sum(bias**2)
        e = p.copy(); e[np.arange(n), y_t] -= 1; e /= n
        grad = np.r_[np.sum(e*ls)+0.04*(a-1), np.sum(e*lm)+0.04*b, e.sum(0)+0.1*bias]
        return loss, grad
    res = optimize.minimize(vg, np.r_[1.0, 0.0, np.zeros(4)], jac=True, bounds=[(0,5),(0,5)]+[(-2,2)]*4, method='L-BFGS-B')
    return res.x if res.success else np.r_[1.0, 0.0, np.zeros(4)]

def predict_evidence(ps, pm, coef):
    return softmax(coef[0]*logp(ps) + coef[1]*logp(pm) + coef[2:])

class StackFusion:
    def fit(self, ps, pm, y):
        x = np.column_stack([logp(ps), logp(pm)])
        self.scaler = StandardScaler().fit(x)
        self.clf = LogisticRegression(C=0.1, max_iter=2000).fit(self.scaler.transform(x), y)
        return self

    def predict_proba(self, ps, pm):
        x = np.column_stack([logp(ps), logp(pm)])
        p = self.clf.predict_proba(self.scaler.transform(x))
        return p[:, np.argsort(self.clf.classes_)]

def main():
    args = arguments('adaptive')
    cache_dir, out_dir = args.cache_dir, args.out_dir
    subjects = args.subjects
    # Subject-specific empirical best MI model mapping:
    # FHY: Filter-bank power (65%)
    # YDL: FBCSP-LDA (43.1%)
    # YZL: Filter-bank power (49.7%)
    subject_prior_best = {
        'FHY': 'fbpower_lda',
        'YDL': 'fbcsp',
        'YZL': 'fbpower_lda'
    }

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

    window_configs = windows(args)

    all_records = []
    diagnostic = Diagnostics(out_dir)
    print("Starting Subject-Adaptive MI-SSVEP Fusion (2 Leads: O1, O2)...")
    t0_all = time.time()

    for w_conf in window_configs:
        start_s = w_conf['start_s']
        durations = w_conf['durations']
        w_name = w_conf['name']
        print(f"\n==========================================")
        print(f"WINDOW START: {start_s}s ({w_name}) - 2 LEADS")
        print(f"==========================================")

        for dur in durations:
            t_dur = time.time()
            print(f"  Duration: {dur:.1f}s ...", end='', flush=True)

            for sub in subjects:
                s_data = loaded_data[sub]
                epochs = s_data['epochs']
                y = s_data['y']
                blocks = s_data['blocks']
                ons = s_data['ons']
                ch_labels = s_data['ch_labels']
                n_trials = len(y)

                ci = [ch_labels.index(c) for c in CENTRAL]
                oi = [ch_labels.index(c) for c in OCC2]

                # 1. Feature extraction for this window
                covs = np.zeros((n_trials, len(ALL_MI_BANDS), len(ci), len(ci)))
                raw_ss_fbcca = np.zeros((n_trials, 4))
                raw_ss_cca = np.zeros((n_trials, 4))

                for i in range(n_trials):
                    ds = epochs[i]
                    on = ons[i]
                    w_start = int(round(on + start_s * FS))
                    w_end = w_start + int(round(dur * FS))
                    x_central = ds[w_start:w_end, ci]
                    x_occ = ds[w_start:w_end, oi]

                    for b_idx, band in enumerate(ALL_MI_BANDS):
                        xf = signal.sosfiltfilt(signal.butter(4, band, fs=FS, btype='bandpass', output='sos'), x_central, axis=0)
                        covs[i, b_idx] = covariance(xf)

                    raw_ss_cca[i] = cca_scores(x_occ, NOMINAL_FREQUENCIES, FS, HARMONICS)
                    fb_score = np.zeros(4)
                    for b_idx, (lo, hi) in enumerate(FBCCA_BANDS):
                        x_sub = signal.sosfiltfilt(signal.butter(4, [lo, hi], fs=FS, btype='bandpass', output='sos'), x_occ, axis=0)
                        fb_score += FB_WEIGHTS[b_idx] * (cca_scores(x_sub, NOMINAL_FREQUENCIES, FS, HARMONICS) ** 2)
                    raw_ss_fbcca[i] = fb_score

                # 2. Block LOBO Cross Validation
                unique_blocks = np.unique(blocks)

                # Predictions dictionary
                preds = {
                    'ss_fbcca': np.zeros(n_trials),
                    'ss_cca': np.zeros(n_trials),
                    'mi_fbcsp': np.zeros(n_trials),
                    'mi_fbpower': np.zeros(n_trials),
                    'mi_prior_best': np.zeros(n_trials),
                    'mi_auto_selected': np.zeros(n_trials),
                    # Generic FBCSP fusions (baseline)
                    'fusion_evidence_fbcsp': np.zeros(n_trials),
                    'fusion_stack_fbcsp': np.zeros(n_trials),
                    'fusion_fixed_fbcsp': np.zeros(n_trials),
                    # Subject-specific prior best MI fusions
                    'fusion_evidence_prior_best': np.zeros(n_trials),
                    'fusion_stack_prior_best': np.zeros(n_trials),
                    'fusion_fixed_prior_best': np.zeros(n_trials),
                    # Fold-internal Auto-adaptive MI fusions
                    'fusion_evidence_auto_adaptive': np.zeros(n_trials),
                    'fusion_stack_auto_adaptive': np.zeros(n_trials),
                    'fusion_fixed_auto_adaptive': np.zeros(n_trials)
                }

                auto_selected_model_names = []

                for test_b in unique_blocks:
                    te = np.where(blocks == test_b)[0]
                    tr = np.where(blocks != test_b)[0]
                    y_tr, y_te = y[tr], y[te]

                    # SSVEP calibrated
                    ts_fb = fit_temperature(raw_ss_fbcca[tr], y_tr, (0.005, 2.0))
                    ps_tr = softmax(raw_ss_fbcca[tr] / ts_fb)
                    ps_te = softmax(raw_ss_fbcca[te] / ts_fb)
                    preds['ss_fbcca'][te] = ps_te.argmax(axis=1)

                    ts_c = fit_temperature(raw_ss_cca[tr], y_tr, (0.005, 2.0))
                    preds['ss_cca'][te] = softmax(raw_ss_cca[te] / ts_c).argmax(axis=1)

                    # Generate inner OOF for all candidate MI models on training blocks
                    candidate_oof = {}
                    candidate_test = {}
                    candidate_scores = {}
                    candidate_temperatures = {}

                    train_blocks = blocks[tr]
                    for m_key, m_cls in MI_CANDIDATE_FACTORIES.items():
                        oof_m = np.zeros((len(tr), 4))
                        for in_b in np.unique(train_blocks):
                            in_val = np.where(train_blocks == in_b)[0]
                            in_tr = np.where(train_blocks != in_b)[0]
                            sub_m = m_cls().fit(covs[tr[in_tr]], y_tr[in_tr])
                            oof_m[in_val] = sub_m.predict_proba(covs[tr[in_val]])

                        tm_m = fit_temperature(logp(oof_m), y_tr, (0.25, 8.0))
                        pm_tr_m = softmax(logp(oof_m) / tm_m)
                        candidate_oof[m_key] = pm_tr_m
                        candidate_temperatures[m_key] = tm_m
                        candidate_scores[m_key] = np.mean(pm_tr_m.argmax(axis=1) == y_tr)

                        # Full fit on training set
                        full_m = m_cls().fit(covs[tr], y_tr)
                        pm_te_m = softmax(logp(full_m.predict_proba(covs[te])) / tm_m)
                        candidate_test[m_key] = pm_te_m

                    # Record standalone MI
                    preds['mi_fbcsp'][te] = candidate_test['fbcsp'].argmax(axis=1)
                    preds['mi_fbpower'][te] = candidate_test['fbpower_lda'].argmax(axis=1)

                    # Track 1: Prior Best
                    prior_m_name = subject_prior_best[sub]
                    pm_prior_tr = candidate_oof[prior_m_name]
                    pm_prior_te = candidate_test[prior_m_name]
                    preds['mi_prior_best'][te] = pm_prior_te.argmax(axis=1)

                    # Track 2: Inner-CV Auto-Adaptive Selection
                    best_m_name = max(candidate_scores, key=candidate_scores.get)
                    auto_selected_model_names.append(best_m_name)
                    pm_auto_tr = candidate_oof[best_m_name]
                    pm_auto_te = candidate_test[best_m_name]
                    preds['mi_auto_selected'][te] = pm_auto_te.argmax(axis=1)

                    # Diagnostic-only candidate fusions: no test label affects fitting/selection.
                    context = dict(subject=sub, window_start_s=start_s, duration_s=dur,
                                   mi_channels='18ch_central_raw', test_block=int(test_b))
                    coefficients = {name: fit_evidence(ps_tr, p, y_tr)
                                    for name, p in candidate_oof.items()}
                    methods = {'ss_fbcca': ps_te}
                    methods.update({f'mi_{name}': p for name, p in candidate_test.items()})
                    methods.update({f'evidence_{name}': predict_evidence(ps_te, candidate_test[name], coef)
                                    for name, coef in coefficients.items()})
                    methods['fusion_evidence_auto_adaptive'] = methods[f'evidence_{best_m_name}']
                    methods['fusion_evidence_prior_best'] = methods[f'evidence_{prior_m_name}']
                    methods['fusion_evidence_fbcsp'] = methods['evidence_fbcsp']
                    diagnostic.fold(context, candidate_scores, candidate_temperatures, ts_fb,
                                    best_m_name, coefficients)
                    diagnostic.predictions(context, s_data['trial_ids'][te], y_te, ps_te, methods)

                    # 1) Generic FBCSP Fusions
                    pm_fbcsp_tr = candidate_oof['fbcsp']
                    pm_fbcsp_te = candidate_test['fbcsp']
                    coef_fbcsp = fit_evidence(ps_tr, pm_fbcsp_tr, y_tr)
                    preds['fusion_evidence_fbcsp'][te] = predict_evidence(ps_te, pm_fbcsp_te, coef_fbcsp).argmax(axis=1)
                    st_fbcsp = StackFusion().fit(ps_tr, pm_fbcsp_tr, y_tr)
                    preds['fusion_stack_fbcsp'][te] = st_fbcsp.predict_proba(ps_te, pm_fbcsp_te).argmax(axis=1)
                    preds['fusion_fixed_fbcsp'][te] = (0.5 * ps_te + 0.5 * pm_fbcsp_te).argmax(axis=1)

                    # 2) Subject-Specific Prior Best Fusions
                    coef_prior = fit_evidence(ps_tr, pm_prior_tr, y_tr)
                    preds['fusion_evidence_prior_best'][te] = predict_evidence(ps_te, pm_prior_te, coef_prior).argmax(axis=1)
                    st_prior = StackFusion().fit(ps_tr, pm_prior_tr, y_tr)
                    preds['fusion_stack_prior_best'][te] = st_prior.predict_proba(ps_te, pm_prior_te).argmax(axis=1)
                    preds['fusion_fixed_prior_best'][te] = (0.5 * ps_te + 0.5 * pm_prior_te).argmax(axis=1)

                    # 3) Fold-internal Auto-Adaptive Fusions
                    coef_auto = fit_evidence(ps_tr, pm_auto_tr, y_tr)
                    preds['fusion_evidence_auto_adaptive'][te] = predict_evidence(ps_te, pm_auto_te, coef_auto).argmax(axis=1)
                    st_auto = StackFusion().fit(ps_tr, pm_auto_tr, y_tr)
                    preds['fusion_stack_auto_adaptive'][te] = st_auto.predict_proba(ps_te, pm_auto_te).argmax(axis=1)
                    preds['fusion_fixed_auto_adaptive'][te] = (0.5 * ps_te + 0.5 * pm_auto_te).argmax(axis=1)

                # Compute accuracies
                for m_key, p_arr in preds.items():
                    acc = float(np.mean(p_arr == y))
                    all_records.append({
                        'window_start_s': start_s,
                        'duration_s': dur,
                        'subject': sub,
                        'method': m_key,
                        'accuracy': acc
                    })

            print(f" done ({time.time() - t_dur:.1f}s)")

    diagnostic.save()
    df_out = pd.DataFrame(all_records)
    df_out.to_csv(out_dir / 'adaptive_fusion_metrics_2leads.csv', index=False, encoding='utf-8-sig')
    print(f"\nAll evaluations complete in {time.time() - t0_all:.1f}s!")
    print(f"Saved metrics to: {out_dir / 'adaptive_fusion_metrics_2leads.csv'}")

if __name__ == '__main__':
    main()
