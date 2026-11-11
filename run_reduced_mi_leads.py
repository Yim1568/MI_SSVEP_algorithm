"""Parallel High-performance Evaluation of Reduced MI Leads (2-4 channels) with 2-lead SSVEP.

Evaluates:
- 2-lead sets: ['C3', 'C4'], ['C3', 'CP3']
- 3-lead sets: ['FC3', 'C3', 'CP3'], ['C3', 'Cz', 'C4']
- 4-lead sets: ['FC3', 'FC4', 'C3', 'C4'], ['FC3', 'C3', 'CP3', 'C1'], ['C1', 'C2', 'C3', 'C4']
- Benchmark: 18-lead central_raw
Across:
- 4 MI models: fbpower_lda, fbcsp, rcsp, tangent_logreg
- Auto Evidence Fusion with 2-lead FBCCA (O1, O2)
- Window starts: 0.5s and 0.0s
"""
import sys
from runtime import arguments, windows
from diagnostics import Diagnostics
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
from joblib import Parallel, delayed

FS = 250
HARMONICS = 3
NOMINAL_FREQUENCIES = [12.0, 10.0, 60.0 / 7.0, 7.5]
CLASS_NAMES = ['thumb', 'index', 'middle', 'little']

FBCCA_BANDS = [(6, 45), (10, 45), (14, 45), (18, 45), (22, 45)]
FB_WEIGHTS = np.arange(1, len(FBCCA_BANDS) + 1)**(-1.25) + 0.25

MI_BROAD_BAND = (8, 30)
MI_SUB_BANDS = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
ALL_MI_BANDS = [MI_BROAD_BAND] + MI_SUB_BANDS

OCC2 = ['O1', 'O2']

ALL_MI_CHANNELS = ['C3', 'C4', 'FC3', 'FC4', 'CP3', 'CP4', 'C1', 'C2', 'Cz', 'C5', 'C6',
                    'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6']

MI_CHANNEL_CONFIGS = {
    '2ch_C3_C4': ['C3', 'C4'],
    '2ch_C3_CP3': ['C3', 'CP3'],
    '3ch_FC3_C3_CP3': ['FC3', 'C3', 'CP3'],
    '3ch_C3_Cz_C4': ['C3', 'Cz', 'C4'],
    '4ch_bilateral': ['FC3', 'FC4', 'C3', 'C4'],
    '4ch_contra_dense': ['FC3', 'C3', 'CP3', 'C1'],
    '4ch_central4': ['C1', 'C2', 'C3', 'C4'],
    '18ch_central_raw': ['C3', 'C4', 'FC3', 'FC4', 'CP3', 'CP4', 'C1', 'C2', 'C5', 'C6',
                         'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6']
}

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
        c_dim = covs.shape[-1]
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
        c_dim = covs.shape[-1]
        filters = []
        for lab in range(4):
            a = shrink(c_broad[y == lab].mean(0), 0.1)
            b = shrink(c_broad[y != lab].mean(0), 0.1)
            _, w = linalg.eigh(a, a + b)
            if c_dim >= 4:
                filters.extend([w[:, 0], w[:, 1], w[:, -2], w[:, -1]])
            else:
                filters.extend([w[:, 0], w[:, -1]])
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
        self.clf = LogisticRegression(C=0.1, max_iter=200).fit(self.scaler.transform(x), y)
        return self

    def predict_proba(self, covs):
        x = self.ts.transform(shrink(covs[:, 0], 0.1))
        p = self.clf.predict_proba(self.scaler.transform(x))
        return p[:, np.argsort(self.clf.classes_)]

MI_CANDIDATES = {
    'fbpower_lda': FBPowerLDAModel,
    'fbcsp': FBCSPLDAModel,
    'rcsp': RCSPModel,
    'tangent_logreg': TangentLogRegModel
}

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

def process_subject_window(sub, start_s, dur, cache_path, diagnostics_dir=None):
    data = np.load(cache_path, allow_pickle=True)
    epochs = data['ds_epochs']
    y = data['y']
    diagnostic = Diagnostics(diagnostics_dir) if diagnostics_dir is not None else None
    trial_ids = data['trial_ids']
    blocks = data['blocks']
    ons = data['ons']
    ch_labels = list(data['ch_labels'])
    n_trials = len(y)
    unique_blocks = np.unique(blocks)

    oi = [ch_labels.index(c) for c in OCC2]
    all_mi_i = [ch_labels.index(c) for c in ALL_MI_CHANNELS]

    # Precompute SSVEP scores and ALL MI covariances
    raw_ss_fbcca = np.zeros((n_trials, 4))
    covs_all = np.zeros((n_trials, len(ALL_MI_BANDS), len(all_mi_i), len(all_mi_i)))

    for i in range(n_trials):
        ds = epochs[i]
        on = ons[i]
        w_start = int(round(on + start_s * FS))
        w_end = w_start + int(round(dur * FS))

        # SSVEP
        x_occ = ds[w_start:w_end, oi]
        fb_score = np.zeros(4)
        for b_idx, (lo, hi) in enumerate(FBCCA_BANDS):
            x_sub = signal.sosfiltfilt(signal.butter(4, [lo, hi], fs=FS, btype='bandpass', output='sos'), x_occ, axis=0)
            fb_score += FB_WEIGHTS[b_idx] * (cca_scores(x_sub, NOMINAL_FREQUENCIES, FS, HARMONICS) ** 2)
        raw_ss_fbcca[i] = fb_score

        # ALL MI
        x_mi_full = ds[w_start:w_end, all_mi_i]
        for b_idx, band in enumerate(ALL_MI_BANDS):
            xf = signal.sosfiltfilt(signal.butter(4, band, fs=FS, btype='bandpass', output='sos'), x_mi_full, axis=0)
            covs_all[i, b_idx] = covariance(xf)

    # SSVEP LOBO predictions
    preds_fbcca = np.zeros(n_trials)
    for test_b in unique_blocks:
        te = np.where(blocks == test_b)[0]
        tr = np.where(blocks != test_b)[0]
        ts_fb = fit_temperature(raw_ss_fbcca[tr], y[tr], (0.005, 2.0))
        preds_fbcca[te] = softmax(raw_ss_fbcca[te] / ts_fb).argmax(axis=1)

    acc_fbcca = float(np.mean(preds_fbcca == y))

    records = []
    # Iterate over MI channel configurations by submatrix slicing
    for ch_name, ch_list in MI_CHANNEL_CONFIGS.items():
        sub_idx = [ALL_MI_CHANNELS.index(c) for c in ch_list]
        covs = covs_all[:, :, sub_idx, :][:, :, :, sub_idx]

        preds_mi = {m: np.zeros(n_trials) for m in MI_CANDIDATES}
        preds_auto_mi = np.zeros(n_trials)
        preds_auto_fusion = np.zeros(n_trials)

        for test_b in unique_blocks:
            te = np.where(blocks == test_b)[0]
            tr = np.where(blocks != test_b)[0]
            y_tr, y_te = y[tr], y[te]
            train_blocks = blocks[tr]

            candidate_oof = {}
            candidate_test = {}
            candidate_scores = {}
            candidate_temperatures = {}

            for m_key, m_cls in MI_CANDIDATES.items():
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

                full_m = m_cls().fit(covs[tr], y_tr)
                pm_te_m = softmax(logp(full_m.predict_proba(covs[te])) / tm_m)
                candidate_test[m_key] = pm_te_m
                preds_mi[m_key][te] = pm_te_m.argmax(axis=1)

            # Auto-selected best MI model on inner-CV
            best_m_name = max(candidate_scores, key=candidate_scores.get)
            pm_auto_tr = candidate_oof[best_m_name]
            pm_auto_te = candidate_test[best_m_name]
            preds_auto_mi[te] = pm_auto_te.argmax(axis=1)

            # Auto Evidence Fusion
            ts_fb = fit_temperature(raw_ss_fbcca[tr], y_tr, (0.005, 2.0))
            ps_tr = softmax(raw_ss_fbcca[tr] / ts_fb)
            ps_te = softmax(raw_ss_fbcca[te] / ts_fb)

            coef = fit_evidence(ps_tr, pm_auto_tr, y_tr)
            preds_auto_fusion[te] = predict_evidence(ps_te, pm_auto_te, coef).argmax(axis=1)
            if diagnostic is not None:
                context = dict(subject=sub, window_start_s=start_s, duration_s=dur,
                               mi_channels=ch_name, test_block=int(test_b))
                coefficients = {name: fit_evidence(ps_tr, p, y_tr)
                                for name, p in candidate_oof.items()}
                methods = {'ss_fbcca': ps_te}
                methods.update({f'mi_{name}': p for name, p in candidate_test.items()})
                methods.update({f'evidence_{name}': predict_evidence(ps_te, candidate_test[name], c)
                                for name, c in coefficients.items()})
                methods['fusion_evidence_auto_adaptive'] = methods[f'evidence_{best_m_name}']
                diagnostic.fold(context, candidate_scores, candidate_temperatures, ts_fb,
                                best_m_name, coefficients)
                diagnostic.predictions(context, trial_ids[te], y_te, ps_te, methods)


        rec = {
            'window_start_s': start_s,
            'duration_s': dur,
            'subject': sub,
            'mi_channels': ch_name,
            'n_mi_channels': len(ch_list),
            'acc_fbcca_2leads': acc_fbcca,
            'acc_mi_fbpower': float(np.mean(preds_mi['fbpower_lda'] == y)),
            'acc_mi_fbcsp': float(np.mean(preds_mi['fbcsp'] == y)),
            'acc_mi_rcsp': float(np.mean(preds_mi['rcsp'] == y)),
            'acc_mi_tangent': float(np.mean(preds_mi['tangent_logreg'] == y)),
            'acc_mi_auto_selected': float(np.mean(preds_auto_mi == y)),
            'acc_auto_evidence_fusion': float(np.mean(preds_auto_fusion == y)),
        }
        rec['gain_vs_fbcca'] = rec['acc_auto_evidence_fusion'] - rec['acc_fbcca_2leads']
        records.append(rec)

    if diagnostic is not None:
        diagnostic.save()
    return records

def main():
    args = arguments('reduced')
    cache_dir, out_dir = args.cache_dir, args.out_dir
    subjects = args.subjects
    cache_paths = {sub: cache_dir / f'{sub}_preprocessed.npz' for sub in subjects}

    window_configs = windows(args)

    tasks = []
    for w in window_configs:
        for dur in w['durations']:
            for sub in subjects:
                tasks.append((sub, w['start_s'], dur, cache_paths[sub]))

    print(f"Launching {len(tasks)} tasks on {args.n_jobs} parallel CPU workers...")
    t_start = time.time()

    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(process_subject_window)(sub, start_s, dur, p,
            out_dir / 'diagnostics' / f'{sub}_start{start_s}_dur{dur}')
        for sub, start_s, dur, p in tasks
    )

    all_results = [r for sublist in results for r in sublist]
    df = pd.DataFrame(all_results)
    out_path = out_dir / 'reduced_mi_leads_metrics.csv'
    df.to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f"\nAll evaluations complete in {time.time() - t_start:.1f}s!")
    print(f"Saved {len(df)} records to: {out_path}")

if __name__ == '__main__':
    main()
