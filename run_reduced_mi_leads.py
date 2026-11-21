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
from windowing import (
    accumulate_log_probs,
    cca_scores,
    covariance,
    extract_dual_features,
    fit_temperature,
    format_window_span,
    logp,
    reference,
    softmax,
)
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import linalg, optimize
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


def shrink(c, amount=0.1):
    c = np.asarray(c)
    n = c.shape[-1]
    return (1 - amount) * c + amount * np.trace(c, axis1=-2, axis2=-1)[..., None, None] * np.eye(n) / n


def normtrace(c):
    return c / np.maximum(np.trace(c, axis1=-2, axis2=-1)[..., None, None], 1e-12)

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

def _calibrate_ssvep_fold(raw_ss_fbcca, y, tr, te):
    ts_fb = fit_temperature(raw_ss_fbcca[tr], y[tr], (0.005, 2.0))
    return ts_fb, softmax(raw_ss_fbcca[tr] / ts_fb), softmax(raw_ss_fbcca[te] / ts_fb)


def _calibrate_reduced_mi_fold(covs, y, blocks, tr, te):
    y_tr = y[tr]
    train_blocks = blocks[tr]
    candidate_oof, candidate_test, candidate_scores, candidate_temperatures = {}, {}, {}, {}
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
        candidate_scores[m_key] = float(np.mean(pm_tr_m.argmax(axis=1) == y_tr))

        full_m = m_cls().fit(covs[tr], y_tr)
        candidate_test[m_key] = softmax(logp(full_m.predict_proba(covs[te])) / tm_m)
    return candidate_oof, candidate_test, candidate_scores, candidate_temperatures


def _init_reduced_preds(n_trials):
    return {
        'fbcca': np.zeros(n_trials),
        'mi': {m: np.zeros(n_trials) for m in MI_CANDIDATES},
        'auto_mi': np.zeros(n_trials),
        'auto_fusion': np.zeros(n_trials),
    }


def _eval_reduced_fold(ps_tr, ps_te, ts_fb, candidate_oof, candidate_test,
                       candidate_scores, candidate_temperatures, y_tr, y_te,
                       trial_ids_te, context, diagnostic, pred_buf, te):
    pred_buf['fbcca'][te] = ps_te.argmax(axis=1)
    for m_key, pm_te_m in candidate_test.items():
        pred_buf['mi'][m_key][te] = pm_te_m.argmax(axis=1)

    best_m_name = max(candidate_scores, key=candidate_scores.get)
    pm_auto_tr = candidate_oof[best_m_name]
    pm_auto_te = candidate_test[best_m_name]
    pred_buf['auto_mi'][te] = pm_auto_te.argmax(axis=1)

    coef = fit_evidence(ps_tr, pm_auto_tr, y_tr)
    pred_buf['auto_fusion'][te] = predict_evidence(ps_te, pm_auto_te, coef).argmax(axis=1)

    if diagnostic is not None:
        coefficients = {name: fit_evidence(ps_tr, p, y_tr) for name, p in candidate_oof.items()}
        methods = {'ss_fbcca': ps_te}
        methods.update({f'mi_{name}': p for name, p in candidate_test.items()})
        methods.update({f'evidence_{name}': predict_evidence(ps_te, candidate_test[name], c)
                        for name, c in coefficients.items()})
        methods['fusion_evidence_auto_adaptive'] = methods[f'evidence_{best_m_name}']
        diagnostic.fold(context, candidate_scores, candidate_temperatures, ts_fb, best_m_name, coefficients)
        diagnostic.predictions(context, trial_ids_te, y_te, ps_te, methods)


def _build_reduced_record(meta, sub, ch_name, ch_list, pred_buf, y, extended_meta):
    acc_fbcca = float(np.mean(pred_buf['fbcca'] == y))
    acc_fusion = float(np.mean(pred_buf['auto_fusion'] == y))
    rec = {}
    if extended_meta:
        rec.update({'window_mode': meta['window_mode'], 'scheme': meta['scheme']})
    rec.update({'window_start_s': meta['window_start_s'], 'duration_s': meta['duration_s']})
    if extended_meta:
        rec.update({
            'decision_time_s': meta['decision_time_s'],
            'ssvep_window': meta['ssvep_window'],
            'mi_window': meta['mi_window'],
            'segment_index': meta['segment_index'],
        })
    rec.update({
        'subject': sub,
        'mi_channels': ch_name,
        'n_mi_channels': len(ch_list),
        'acc_fbcca_2leads': acc_fbcca,
        'acc_mi_fbpower': float(np.mean(pred_buf['mi']['fbpower_lda'] == y)),
        'acc_mi_fbcsp': float(np.mean(pred_buf['mi']['fbcsp'] == y)),
        'acc_mi_rcsp': float(np.mean(pred_buf['mi']['rcsp'] == y)),
        'acc_mi_tangent': float(np.mean(pred_buf['mi']['tangent_logreg'] == y)),
        'acc_mi_auto_selected': float(np.mean(pred_buf['auto_mi'] == y)),
        'acc_auto_evidence_fusion': acc_fusion,
        'gain_vs_fbcca': acc_fusion - acc_fbcca,
    })
    return rec


def _make_reduced_context(sub, meta, ch_name, test_b, extended_meta):
    ctx = dict(subject=sub, window_start_s=meta['window_start_s'], duration_s=meta['duration_s'],
               mi_channels=ch_name, test_block=int(test_b))
    if extended_meta:
        ctx.update(window_mode=meta['window_mode'], decision_time_s=meta['decision_time_s'],
                   ssvep_window=meta['ssvep_window'], mi_window=meta['mi_window'],
                   segment_index=meta['segment_index'])
    return ctx


def process_subject_spec(sub, spec, cache_path, diagnostics_dir=None, extended_meta=False):
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

    ss_win = spec['ssvep_window']
    mi_win = spec['mi_window']
    covs_all, raw_ss_fbcca, _ = extract_dual_features(
        epochs, ons, ch_labels, OCC2, ALL_MI_CHANNELS,
        ss_win, mi_win, ALL_MI_BANDS, FBCCA_BANDS, FB_WEIGHTS,
        NOMINAL_FREQUENCIES, FS, HARMONICS, extract_cca=False,
    )

    ssvep_folds = {}
    for test_b in unique_blocks:
        te = np.where(blocks == test_b)[0]
        tr = np.where(blocks != test_b)[0]
        ssvep_folds[test_b] = _calibrate_ssvep_fold(raw_ss_fbcca, y, tr, te)

    meta = {
        'window_mode': spec['mode'],
        'scheme': spec['scheme'],
        'window_start_s': spec['window_start_s'],
        'duration_s': spec['duration_s'],
        'decision_time_s': spec['decision_time_s'],
        'ssvep_window': format_window_span(ss_win),
        'mi_window': format_window_span(mi_win),
        'segment_index': '' if spec['segment_index'] is None else spec['segment_index'],
    }

    records = []
    for ch_name, ch_list in MI_CHANNEL_CONFIGS.items():
        sub_idx = [ALL_MI_CHANNELS.index(c) for c in ch_list]
        covs = covs_all[:, :, sub_idx, :][:, :, :, sub_idx]
        pred_buf = _init_reduced_preds(n_trials)

        for test_b in unique_blocks:
            te = np.where(blocks == test_b)[0]
            tr = np.where(blocks != test_b)[0]
            ts_fb, ps_tr, ps_te = ssvep_folds[test_b]
            cand_oof, cand_test, cand_scores, cand_temps = _calibrate_reduced_mi_fold(covs, y, blocks, tr, te)
            ctx = _make_reduced_context(sub, meta, ch_name, test_b, extended_meta)
            _eval_reduced_fold(
                ps_tr, ps_te, ts_fb, cand_oof, cand_test, cand_scores, cand_temps,
                y[tr], y[te], trial_ids[te], ctx, diagnostic, pred_buf, te,
            )

        records.append(_build_reduced_record(meta, sub, ch_name, ch_list, pred_buf, y, extended_meta))

    if diagnostic is not None:
        diagnostic.save()
    return records


def process_subject_segmented(sub, w_conf, cache_path, diagnostics_dir=None, extended_meta=True):
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

    specs = w_conf['specs']
    n_segs = len(specs)
    ss_lat = specs[0]['ssvep_latency_s']
    mi_lat = specs[0]['mi_latency_s']

    seg_features = [
        extract_dual_features(
            epochs, ons, ch_labels, OCC2, ALL_MI_CHANNELS,
            sp['ssvep_window'], sp['mi_window'], ALL_MI_BANDS, FBCCA_BANDS, FB_WEIGHTS,
            NOMINAL_FREQUENCIES, FS, HARMONICS, segment_index=sp['segment_index'], extract_cca=False,
        )
        for sp in specs
    ]

    ssvep_step = {test_b: [] for test_b in unique_blocks}
    ssvep_accum = {test_b: [] for test_b in unique_blocks}
    for test_b in unique_blocks:
        te = np.where(blocks == test_b)[0]
        tr = np.where(blocks != test_b)[0]
        tr_ps_list, te_ps_list = [], []
        for k in range(n_segs):
            _, raw_fb_k, _ = seg_features[k]
            ts_k, ps_tr_k, ps_te_k = _calibrate_ssvep_fold(raw_fb_k, y, tr, te)
            ssvep_step[test_b].append((ts_k, ps_tr_k, ps_te_k))
            tr_ps_list.append(ps_tr_k)
            te_ps_list.append(ps_te_k)
            if k == 0:
                ssvep_accum[test_b].append((ts_k, ps_tr_k, ps_te_k))
            else:
                ps_acc_tr, ps_acc_te, ts_acc = accumulate_log_probs(tr_ps_list, te_ps_list, y[tr])
                ssvep_accum[test_b].append((ts_acc, ps_acc_tr, ps_acc_te))

    step_metas = [
        {
            'window_mode': 'equal_segmented_step',
            'scheme': 'equal_segmented_step',
            'window_start_s': sp['window_start_s'],
            'duration_s': sp['duration_s'],
            'decision_time_s': sp['decision_time_s'],
            'ssvep_window': format_window_span(sp['ssvep_window']),
            'mi_window': format_window_span(sp['mi_window']),
            'segment_index': sp['segment_index'],
        }
        for sp in specs
    ]
    accum_metas = [
        {
            'window_mode': 'equal_segmented_accum',
            'scheme': 'equal_segmented_accum',
            'window_start_s': mi_lat,
            'duration_s': sp['cum_duration_s'],
            'decision_time_s': sp['decision_time_s'],
            'ssvep_window': f"[{ss_lat:.2f}, {sp['ssvep_window'][1]:.2f}]",
            'mi_window': f"[{mi_lat:.2f}, {sp['mi_window'][1]:.2f}]",
            'segment_index': sp['segment_index'],
        }
        for sp in specs
    ]

    records = []
    for ch_name, ch_list in MI_CHANNEL_CONFIGS.items():
        sub_idx = [ALL_MI_CHANNELS.index(c) for c in ch_list]
        step_bufs = [_init_reduced_preds(n_trials) for _ in range(n_segs)]
        accum_bufs = [_init_reduced_preds(n_trials) for _ in range(n_segs)]

        for test_b in unique_blocks:
            te = np.where(blocks == test_b)[0]
            tr = np.where(blocks != test_b)[0]
            y_tr, y_te = y[tr], y[te]
            mi_oof_lists = {m: [] for m in MI_CANDIDATES}
            mi_test_lists = {m: [] for m in MI_CANDIDATES}

            for k in range(n_segs):
                covs_all_k = seg_features[k][0]
                covs_k = covs_all_k[:, :, sub_idx, :][:, :, :, sub_idx]
                cand_oof_k, cand_test_k, cand_scores_k, cand_temps_k = _calibrate_reduced_mi_fold(
                    covs_k, y, blocks, tr, te,
                )

                ts_k, ps_tr_k, ps_te_k = ssvep_step[test_b][k]
                ctx_step = _make_reduced_context(sub, step_metas[k], ch_name, test_b, extended_meta)
                _eval_reduced_fold(
                    ps_tr_k, ps_te_k, ts_k, cand_oof_k, cand_test_k, cand_scores_k, cand_temps_k,
                    y_tr, y_te, trial_ids[te], ctx_step, diagnostic, step_bufs[k], te,
                )

                for m_key in MI_CANDIDATES:
                    mi_oof_lists[m_key].append(cand_oof_k[m_key])
                    mi_test_lists[m_key].append(cand_test_k[m_key])

                if k == 0:
                    acc_oof, acc_test, acc_scores, acc_temps = cand_oof_k, cand_test_k, cand_scores_k, cand_temps_k
                else:
                    acc_oof, acc_test, acc_scores, acc_temps = {}, {}, {}, {}
                    for m_key in MI_CANDIDATES:
                        pm_acc_tr, pm_acc_te, tm_acc = accumulate_log_probs(
                            mi_oof_lists[m_key], mi_test_lists[m_key], y_tr,
                        )
                        acc_oof[m_key] = pm_acc_tr
                        acc_test[m_key] = pm_acc_te
                        acc_temps[m_key] = tm_acc
                        acc_scores[m_key] = float(np.mean(pm_acc_tr.argmax(axis=1) == y_tr))

                ts_acc, ps_acc_tr, ps_acc_te = ssvep_accum[test_b][k]
                ctx_acc = _make_reduced_context(sub, accum_metas[k], ch_name, test_b, extended_meta)
                _eval_reduced_fold(
                    ps_acc_tr, ps_acc_te, ts_acc, acc_oof, acc_test, acc_scores, acc_temps,
                    y_tr, y_te, trial_ids[te], ctx_acc, diagnostic, accum_bufs[k], te,
                )

        for k in range(n_segs):
            records.append(_build_reduced_record(step_metas[k], sub, ch_name, ch_list, step_bufs[k], y, extended_meta))
            records.append(_build_reduced_record(accum_metas[k], sub, ch_name, ch_list, accum_bufs[k], y, extended_meta))

    if diagnostic is not None:
        diagnostic.save()
    return records


def process_subject_window(sub, start_s, dur, cache_path, diagnostics_dir=None):
    spec = {
        'mode': 'sync',
        'scheme': f'start{int(round(start_s * 10)):02d}',
        'window_start_s': start_s,
        'duration_s': dur,
        'cum_duration_s': dur,
        'decision_time_s': round(start_s + dur, 6),
        'ssvep_window': (start_s, round(start_s + dur, 6)),
        'mi_window': (start_s, round(start_s + dur, 6)),
        'segment_index': None,
        'segment_step_s': None,
    }
    return process_subject_spec(sub, spec, cache_path, diagnostics_dir, extended_meta=False)


def main():
    args = arguments('reduced')
    cache_dir, out_dir = args.cache_dir, args.out_dir
    subjects = args.subjects
    cache_paths = {sub: cache_dir / f'{sub}_preprocessed.npz' for sub in subjects}

    window_configs = windows(args)
    extended_meta = any(wc['mode'] != 'sync' for wc in window_configs)

    jobs = []
    for w in window_configs:
        if w['mode'] == 'equal_segmented':
            for sub in subjects:
                diag_dir = out_dir / 'diagnostics' / f"{sub}_{w['name']}"
                jobs.append(delayed(process_subject_segmented)(
                    sub, w, cache_paths[sub], diag_dir, extended_meta,
                ))
        else:
            for sp in w['specs']:
                for sub in subjects:
                    if extended_meta:
                        tag = f"{sub}_{sp['scheme']}_dur{sp['duration_s']}"
                    else:
                        tag = f"{sub}_start{sp['window_start_s']}_dur{sp['duration_s']}"
                    diag_dir = out_dir / 'diagnostics' / tag
                    jobs.append(delayed(process_subject_spec)(
                        sub, sp, cache_paths[sub], diag_dir, extended_meta,
                    ))

    print(f"Launching {len(jobs)} tasks on {args.n_jobs} parallel CPU workers...")
    t_start = time.time()

    results = Parallel(n_jobs=args.n_jobs, verbose=10)(jobs)

    all_results = [r for sublist in results for r in sublist]
    df = pd.DataFrame(all_results)
    out_path = out_dir / 'reduced_mi_leads_metrics.csv'
    df.to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f"\nAll evaluations complete in {time.time() - t_start:.1f}s!")
    print(f"Saved {len(df)} records to: {out_path}")


if __name__ == '__main__':
    main()
