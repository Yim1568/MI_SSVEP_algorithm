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
   - Synchronized windows (--window-mode sync, default: start 0.5s and 0.0s)
   - Physiological latency decoupled alignment (--window-mode decoupled)
   - Equal-duration latency-shifted cumulative & segmented schemes
     (--window-mode equal_cumulative equal_segmented / all_equal)
4. Evaluates fusion methods: Evidence, Stacking, Fixed (w=0.5).
"""
from runtime import arguments, windows
from diagnostics import Diagnostics
from channel_selection import MI_POOLS, select_channels
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
import numpy as np
import pandas as pd
from scipy import linalg, optimize
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

def true_oof_fusion_gain(ps_tr_base, cand_oof_m, y_tr, tr_blocks, ps_tr_fb):
    """Evaluate fusion gain strictly on inner training blocks out-of-fold."""
    n_tr = len(y_tr)
    p_ev_oof = np.zeros((n_tr, 4))
    unique_in_blocks = np.unique(tr_blocks)
    if len(unique_in_blocks) < 2:
        full_coef = fit_evidence(ps_tr_base, cand_oof_m, y_tr)
        p_ev_oof = predict_evidence(ps_tr_base, cand_oof_m, full_coef)
        oof_gain = float(np.mean(p_ev_oof.argmax(1) == y_tr) - np.mean(ps_tr_fb.argmax(1) == y_tr))
        return oof_gain, full_coef

    for inner_b in unique_in_blocks:
        in_te_idx = np.where(tr_blocks == inner_b)[0]
        in_tr_idx = np.where(tr_blocks != inner_b)[0]
        if len(in_tr_idx) < 4 or len(np.unique(y_tr[in_tr_idx])) < 4:
            p_ev_oof[in_te_idx] = ps_tr_base[in_te_idx]
            continue
        coef_inner = fit_evidence(ps_tr_base[in_tr_idx], cand_oof_m[in_tr_idx], y_tr[in_tr_idx])
        p_ev_oof[in_te_idx] = predict_evidence(ps_tr_base[in_te_idx], cand_oof_m[in_te_idx], coef_inner)
    oof_fusion_acc = float(np.mean(p_ev_oof.argmax(1) == y_tr))
    fbcca_acc = float(np.mean(ps_tr_fb.argmax(1) == y_tr))
    full_coef = fit_evidence(ps_tr_base, cand_oof_m, y_tr)
    return oof_fusion_acc - fbcca_acc, full_coef

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

def _safe_floor_params(n_mi, n_ss, dur):
    if n_mi <= 2:
        return 0.020, 0.33, 0.050
    if n_ss <= 2:
        if dur >= 3.0:
            min_gain, min_mi_acc = 0.025, 0.35
        elif dur <= 1.0:
            min_gain, min_mi_acc = 0.015, 0.30
        else:
            min_gain, min_mi_acc = 0.010, 0.30
        return min_gain, min_mi_acc, 0.050
    return 0.005, 0.27, 0.035


def _init_pred_buffers(n_trials, extended_meta=False):
    keys = [
        'ss_fbcca', 'ss_cca', 'mi_fbcsp', 'mi_fbpower', 'mi_prior_best', 'mi_auto_selected',
        'fusion_evidence_fbcsp', 'fusion_stack_fbcsp', 'fusion_fixed_fbcsp',
        'fusion_evidence_prior_best', 'fusion_stack_prior_best', 'fusion_fixed_prior_best',
        'fusion_evidence_auto_adaptive', 'fusion_stack_auto_adaptive', 'fusion_fixed_auto_adaptive',
    ]
    if extended_meta:
        keys.extend(['fusion_evidence_fbcsp_raw', 'fusion_evidence_auto_raw'])
    return {k: np.zeros(n_trials) for k in keys}


def _apply_safe_evidence(ps_tr_base, ps_te_base, ps_tr, ps_te, pm_tr, pm_te,
                         y_tr, train_blocks, mi_score, min_gain, min_mi_acc, margin_te):
    oof_gain, coef = true_oof_fusion_gain(ps_tr_base, pm_tr, y_tr, train_blocks, ps_tr)
    p_ev_raw = predict_evidence(ps_te_base, pm_te, coef)
    if oof_gain < min_gain or mi_score < min_mi_acc:
        return ps_te.argmax(axis=1), ps_te.copy(), p_ev_raw, oof_gain, coef
    p_ev = p_ev_raw.copy()
    final = p_ev.argmax(axis=1)
    mask = margin_te >= 0.6
    final[mask] = ps_te_base.argmax(axis=1)[mask]
    p_ev[mask] = ps_te_base[mask]
    return final, p_ev, p_ev_raw, oof_gain, coef


def _calibrate_ssvep_part(raw_ss_fbcca, raw_ss_cca, y_tr, tr, te, ssvep_start_s, ss_dur):
    ts_fb = fit_temperature(raw_ss_fbcca[tr], y_tr, (0.005, 2.0))
    ps_tr = softmax(raw_ss_fbcca[tr] / ts_fb)
    ps_te = softmax(raw_ss_fbcca[te] / ts_fb)

    ts_c = fit_temperature(raw_ss_cca[tr], y_tr, (0.005, 2.0))
    ps_cca_tr = softmax(raw_ss_cca[tr] / ts_c)
    ps_cca_te = softmax(raw_ss_cca[te] / ts_c)

    # Dual-SSVEP synergy: broadband CCA stabilizes few-lead FBCCA in short windows.
    cca_max_dur = 0.5 if ssvep_start_s >= 0.5 else 1.0
    w_cca = 0.5 if (len(OCC2) <= 2 and ss_dur <= cca_max_dur) else 0.0
    if w_cca > 0:
        ps_tr_base = softmax(logp(ps_tr) + w_cca * logp(ps_cca_tr))
        ps_te_base = softmax(logp(ps_te) + w_cca * logp(ps_cca_te))
    else:
        ps_tr_base, ps_te_base = ps_tr, ps_te

    return {
        'ts_fb': ts_fb,
        'ps_tr': ps_tr, 'ps_te': ps_te,
        'ps_cca_tr': ps_cca_tr, 'ps_cca_te': ps_cca_te,
        'ps_tr_base': ps_tr_base, 'ps_te_base': ps_te_base,
    }


def _calibrate_fold(raw_ss_fbcca, raw_ss_cca, covs_fold, y, blocks, tr, te,
                    ssvep_start_s, ss_dur, ss_part=None):
    y_tr = y[tr]
    train_blocks = blocks[tr]
    out = ss_part if ss_part is not None else _calibrate_ssvep_part(
        raw_ss_fbcca, raw_ss_cca, y_tr, tr, te, ssvep_start_s, ss_dur,
    )

    candidate_oof, candidate_test, candidate_scores, candidate_temperatures = {}, {}, {}, {}
    for m_key, m_cls in MI_CANDIDATE_FACTORIES.items():
        oof_m = np.zeros((len(tr), 4))
        for in_b in np.unique(train_blocks):
            in_val = np.where(train_blocks == in_b)[0]
            in_tr = np.where(train_blocks != in_b)[0]
            sub_m = m_cls().fit(covs_fold[tr[in_tr]], y_tr[in_tr])
            oof_m[in_val] = sub_m.predict_proba(covs_fold[tr[in_val]])

        tm_m = fit_temperature(logp(oof_m), y_tr, (0.25, 8.0))
        pm_tr_m = softmax(logp(oof_m) / tm_m)
        candidate_oof[m_key] = pm_tr_m
        candidate_temperatures[m_key] = tm_m
        candidate_scores[m_key] = float(np.mean(pm_tr_m.argmax(axis=1) == y_tr))

        full_m = m_cls().fit(covs_fold[tr], y_tr)
        candidate_test[m_key] = softmax(logp(full_m.predict_proba(covs_fold[te])) / tm_m)

    return dict(
        out,
        candidate_oof=candidate_oof,
        candidate_test=candidate_test,
        candidate_scores=candidate_scores,
        candidate_temperatures=candidate_temperatures,
    )


def _run_fold_fusion(fold_data, y_tr, y_te, train_blocks, trial_ids_te,
                     prior_m_name, floor_dur_s, n_mi_channels, context,
                     diagnostic, preds, te, extended_meta=False):
    ps_tr, ps_te = fold_data['ps_tr'], fold_data['ps_te']
    ps_tr_base, ps_te_base = fold_data['ps_tr_base'], fold_data['ps_te_base']
    ps_cca_te = fold_data['ps_cca_te']
    cand_oof, cand_test = fold_data['candidate_oof'], fold_data['candidate_test']
    cand_scores, cand_temps = fold_data['candidate_scores'], fold_data['candidate_temperatures']

    preds['ss_fbcca'][te] = ps_te.argmax(axis=1)
    preds['ss_cca'][te] = ps_cca_te.argmax(axis=1)
    preds['mi_fbcsp'][te] = cand_test['fbcsp'].argmax(axis=1)
    preds['mi_fbpower'][te] = cand_test['fbpower_lda'].argmax(axis=1)

    min_gain, min_mi_acc, sig_thresh = _safe_floor_params(n_mi_channels, len(OCC2), floor_dur_s)

    pm_prior_tr, pm_prior_te = cand_oof[prior_m_name], cand_test[prior_m_name]
    preds['mi_prior_best'][te] = pm_prior_te.argmax(axis=1)

    fbcsp_score = cand_scores['fbcsp']
    best_cand_name = max(cand_scores, key=cand_scores.get)
    best_m_name = best_cand_name if cand_scores[best_cand_name] >= fbcsp_score + sig_thresh else 'fbcsp'
    pm_auto_tr, pm_auto_te = cand_oof[best_m_name], cand_test[best_m_name]
    preds['mi_auto_selected'][te] = pm_auto_te.argmax(axis=1)

    margin_te = np.sort(ps_te_base, axis=1)[:, -1] - np.sort(ps_te_base, axis=1)[:, -2]

    # Track 1: Generic FBCSP
    pm_fbcsp_tr, pm_fbcsp_te = cand_oof['fbcsp'], cand_test['fbcsp']
    final_fbcsp, p_ev_fbcsp, p_ev_fbcsp_raw, oof_gain_fixed, _ = _apply_safe_evidence(
        ps_tr_base, ps_te_base, ps_tr, ps_te, pm_fbcsp_tr, pm_fbcsp_te,
        y_tr, train_blocks, cand_scores['fbcsp'], min_gain, min_mi_acc, margin_te,
    )
    preds['fusion_evidence_fbcsp'][te] = final_fbcsp
    if extended_meta:
        preds['fusion_evidence_fbcsp_raw'][te] = p_ev_fbcsp_raw.argmax(axis=1)
    preds['fusion_stack_fbcsp'][te] = StackFusion().fit(ps_tr_base, pm_fbcsp_tr, y_tr).predict_proba(ps_te_base, pm_fbcsp_te).argmax(axis=1)
    preds['fusion_fixed_fbcsp'][te] = (0.5 * ps_te_base + 0.5 * pm_fbcsp_te).argmax(axis=1)

    # Track 2: Subject-Specific Prior Best
    final_prior, p_ev_prior, _, _, _ = _apply_safe_evidence(
        ps_tr_base, ps_te_base, ps_tr, ps_te, pm_prior_tr, pm_prior_te,
        y_tr, train_blocks, cand_scores[prior_m_name], min_gain, min_mi_acc, margin_te,
    )
    preds['fusion_evidence_prior_best'][te] = final_prior
    preds['fusion_stack_prior_best'][te] = StackFusion().fit(ps_tr_base, pm_prior_tr, y_tr).predict_proba(ps_te_base, pm_prior_te).argmax(axis=1)
    preds['fusion_fixed_prior_best'][te] = (0.5 * ps_te_base + 0.5 * pm_prior_te).argmax(axis=1)

    # Track 3: Auto-Adaptive with Safe Floor
    if best_m_name == 'fbcsp':
        final_auto, p_ev_auto = final_fbcsp.copy(), p_ev_fbcsp.copy()
        p_ev_auto_raw = p_ev_fbcsp_raw.copy()
    else:
        oof_gain_auto, coef_auto = true_oof_fusion_gain(ps_tr_base, pm_auto_tr, y_tr, train_blocks, ps_tr)
        p_ev_auto_raw = predict_evidence(ps_te_base, pm_auto_te, coef_auto)
        if oof_gain_auto >= oof_gain_fixed and oof_gain_auto >= min_gain and cand_scores[best_m_name] >= min_mi_acc:
            p_ev_auto = p_ev_auto_raw.copy()
            final_auto = p_ev_auto.argmax(axis=1)
            mask = margin_te >= 0.6
            final_auto[mask] = ps_te_base.argmax(axis=1)[mask]
            p_ev_auto[mask] = ps_te_base[mask]
        else:
            final_auto, p_ev_auto = final_fbcsp.copy(), p_ev_fbcsp.copy()
    preds['fusion_evidence_auto_adaptive'][te] = final_auto
    if extended_meta:
        preds['fusion_evidence_auto_raw'][te] = p_ev_auto_raw.argmax(axis=1)
    preds['fusion_stack_auto_adaptive'][te] = StackFusion().fit(ps_tr_base, pm_auto_tr, y_tr).predict_proba(ps_te_base, pm_auto_te).argmax(axis=1)
    preds['fusion_fixed_auto_adaptive'][te] = (0.5 * ps_te_base + 0.5 * pm_auto_te).argmax(axis=1)

    coefficients = {name: fit_evidence(ps_tr_base, p, y_tr) for name, p in cand_oof.items()}
    methods = {'ss_fbcca': ps_te}
    methods.update({f'mi_{name}': p for name, p in cand_test.items()})
    methods.update({f'evidence_{name}': predict_evidence(ps_te_base, cand_test[name], coef) for name, coef in coefficients.items()})
    methods['fusion_evidence_auto_adaptive'] = p_ev_auto
    methods['fusion_evidence_prior_best'] = p_ev_prior
    methods['fusion_evidence_fbcsp'] = p_ev_fbcsp
    if extended_meta:
        methods['fusion_evidence_auto_raw'] = p_ev_auto_raw
        methods['fusion_evidence_fbcsp_raw'] = p_ev_fbcsp_raw
    diagnostic.fold(context, cand_scores, cand_temps, fold_data['ts_fb'], best_m_name, coefficients)
    diagnostic.predictions(context, trial_ids_te, y_te, ps_te, methods)


def _accumulate_fold_step(seg_folds_prefix, y_tr):
    """Aggregate calibrated segment fold predictions across segments 1..k."""
    if len(seg_folds_prefix) == 1:
        return seg_folds_prefix[0]

    ps_tr, ps_te, ts_acc = accumulate_log_probs(
        [f['ps_tr'] for f in seg_folds_prefix], [f['ps_te'] for f in seg_folds_prefix], y_tr)
    ps_cca_tr, ps_cca_te, _ = accumulate_log_probs(
        [f['ps_cca_tr'] for f in seg_folds_prefix], [f['ps_cca_te'] for f in seg_folds_prefix], y_tr)
    ps_tr_base, ps_te_base, _ = accumulate_log_probs(
        [f['ps_tr_base'] for f in seg_folds_prefix], [f['ps_te_base'] for f in seg_folds_prefix], y_tr)

    cand_oof, cand_test, cand_scores, cand_temps = {}, {}, {}, {}
    for m_key in MI_CANDIDATE_FACTORIES:
        pm_tr, pm_te, tm_m = accumulate_log_probs(
            [f['candidate_oof'][m_key] for f in seg_folds_prefix],
            [f['candidate_test'][m_key] for f in seg_folds_prefix],
            y_tr,
        )
        cand_oof[m_key] = pm_tr
        cand_test[m_key] = pm_te
        cand_temps[m_key] = tm_m
        cand_scores[m_key] = float(np.mean(pm_tr.argmax(axis=1) == y_tr))

    return {
        'ts_fb': ts_acc,
        'ps_tr': ps_tr, 'ps_te': ps_te,
        'ps_cca_tr': ps_cca_tr, 'ps_cca_te': ps_cca_te,
        'ps_tr_base': ps_tr_base, 'ps_te_base': ps_te_base,
        'candidate_oof': cand_oof,
        'candidate_test': cand_test,
        'candidate_scores': cand_scores,
        'candidate_temperatures': cand_temps,
    }


def _append_metric_rows(all_records, preds, y, sub, meta, extended_meta, ch_cfg):
    for m_key, p_arr in preds.items():
        row = {}
        if extended_meta:
            row.update({
                'window_mode': meta['window_mode'],
                'scheme': meta['scheme'],
                'mi_pool': ch_cfg['mi_pool'],
                'ch_select': ch_cfg['ch_select'],
            })
        row.update({
            'window_start_s': meta['window_start_s'],
            'duration_s': meta['duration_s'],
        })
        if extended_meta:
            row.update({
                'decision_time_s': meta['decision_time_s'],
                'ssvep_window': meta['ssvep_window'],
                'mi_window': meta['mi_window'],
                'segment_index': meta['segment_index'],
            })
        row.update({'subject': sub, 'method': m_key, 'accuracy': float(np.mean(p_arr == y))})
        all_records.append(row)


def _make_diag_context(sub, meta, test_b, extended_meta, ch_cfg=None, mi_channels_label=None):
    if mi_channels_label is None:
        mi_channels_label = ch_cfg['mi_pool'] if (extended_meta and ch_cfg is not None) else '18ch_central_raw'
    ctx = dict(subject=sub, window_start_s=meta['window_start_s'], duration_s=meta['duration_s'],
               mi_channels=mi_channels_label, test_block=int(test_b))
    if extended_meta:
        ctx.update(window_mode=meta['window_mode'], decision_time_s=meta['decision_time_s'],
                   ssvep_window=meta['ssvep_window'], mi_window=meta['mi_window'],
                   segment_index=meta['segment_index'])
        if ch_cfg is not None:
            ctx.update(mi_pool=ch_cfg['mi_pool'], ch_select=ch_cfg['ch_select'])
    return ctx


def _precompute_subject_fold_channels(window_configs, loaded_data, subjects, ch_cfg, diagnostic, extended_meta):
    """Select personalized MI channel subsets once per outer fold strictly on training blocks."""
    mi_names = ch_cfg['mi_channels']
    n_pool = len(mi_names)
    if ch_cfg['ch_select'] == 'none':
        return {(sub, int(test_b)): list(range(n_pool))
                for sub in subjects for test_b in np.unique(loaded_data[sub]['blocks'])}

    first_sp = window_configs[0]['specs'][0]
    ss_start = first_sp['ssvep_window'][0]
    mi_start = first_sp['mi_window'][0]
    max_dur = max(sp['cum_duration_s'] for wc in window_configs for sp in wc['specs'])
    ref_dur = min(2.0, max_dur, 4.0 - max(ss_start, mi_start))
    ref_ss_win = (ss_start, round(ss_start + ref_dur, 6))
    ref_mi_win = (mi_start, round(mi_start + ref_dur, 6))
    ref_meta = {
        'window_mode': 'subject_fold_ref',
        'window_start_s': mi_start,
        'duration_s': ref_dur,
        'decision_time_s': round(max(ref_ss_win[1], ref_mi_win[1]), 6),
        'ssvep_window': format_window_span(ref_ss_win),
        'mi_window': format_window_span(ref_mi_win),
        'segment_index': '',
    }

    print(f"Pre-selecting subject_fold channels ({ch_cfg['ch_select']}, pool={ch_cfg['mi_pool']}, "
          f"ref MI={ref_meta['mi_window']}, SSVEP={ref_meta['ssvep_window']})...")
    selected_map = {}
    for sub in subjects:
        s_data = loaded_data[sub]
        y, blocks, ons = s_data['y'], s_data['blocks'], s_data['ons']
        covs_ref, raw_ss_fb, raw_ss_cca = extract_dual_features(
            s_data['epochs'], ons, s_data['ch_labels'], OCC2, mi_names,
            ref_ss_win, ref_mi_win, ALL_MI_BANDS, FBCCA_BANDS, FB_WEIGHTS,
            NOMINAL_FREQUENCIES, FS, HARMONICS,
        )
        for test_b in np.unique(blocks):
            te = np.where(blocks == test_b)[0]
            tr = np.where(blocks != test_b)[0]
            ss_part = _calibrate_ssvep_part(raw_ss_fb, raw_ss_cca, y[tr], tr, te, ss_start, ref_dur)
            sel = select_channels(
                covs_ref[tr], y[tr], blocks[tr], mi_names, ps_tr=ss_part['ps_tr_base'],
                method=ch_cfg['ch_select'], k_min=ch_cfg['ch_k_min'], k_max=ch_cfg['ch_k_max'],
                objective=ch_cfg['ch_objective'], seed=ch_cfg['ch_seed'],
            )
            selected_map[(sub, int(test_b))] = sel['indices']
            ctx = _make_diag_context(sub, ref_meta, test_b, extended_meta, ch_cfg)
            diagnostic.channel_selection(
                ctx, sel['names'], n_pool, sel['best_score'], sel['full_score'],
                sel['n_evals'], ch_cfg['ch_select'], ch_cfg['ch_objective'], 'subject_fold',
            )
    return selected_map


def _resolve_fold_subset(covs, ss_part, y, blocks, tr, test_b, sub, meta,
                         ch_cfg, subject_fold_channels, diagnostic, extended_meta):
    mi_names = ch_cfg['mi_channels']
    if ch_cfg['ch_select'] == 'none':
        return list(range(len(mi_names)))
    if ch_cfg['ch_scope'] == 'subject_fold':
        return subject_fold_channels[(sub, int(test_b))]

    sel = select_channels(
        covs[tr], y[tr], blocks[tr], mi_names, ps_tr=ss_part['ps_tr_base'],
        method=ch_cfg['ch_select'], k_min=ch_cfg['ch_k_min'], k_max=ch_cfg['ch_k_max'],
        objective=ch_cfg['ch_objective'], seed=ch_cfg['ch_seed'],
    )
    ctx = _make_diag_context(sub, meta, test_b, extended_meta, ch_cfg)
    diagnostic.channel_selection(
        ctx, sel['names'], len(mi_names), sel['best_score'], sel['full_score'],
        sel['n_evals'], ch_cfg['ch_select'], ch_cfg['ch_objective'], 'window_fold',
    )
    return sel['indices']


def _evaluate_standard_spec(spec, loaded_data, subjects, subject_prior_best,
                            diagnostic, all_records, extended_meta,
                            ch_cfg, subject_fold_channels):
    t_dur = time.time()
    ss_win = spec['ssvep_window']
    mi_win = spec['mi_window']
    ss_dur = ss_win[1] - ss_win[0]
    mi_names = ch_cfg['mi_channels']
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
    print(f"  Duration: {spec['duration_s']:.1f}s (SSVEP {meta['ssvep_window']}, MI {meta['mi_window']}) ...", end='', flush=True)

    for sub in subjects:
        s_data = loaded_data[sub]
        y, blocks, ons = s_data['y'], s_data['blocks'], s_data['ons']
        covs, raw_ss_fbcca, raw_ss_cca = extract_dual_features(
            s_data['epochs'], ons, s_data['ch_labels'], OCC2, mi_names,
            ss_win, mi_win, ALL_MI_BANDS, FBCCA_BANDS, FB_WEIGHTS,
            NOMINAL_FREQUENCIES, FS, HARMONICS,
        )
        preds = _init_pred_buffers(len(y), extended_meta=extended_meta)
        prior_m_name = subject_prior_best.get(sub, 'fbcsp')

        for test_b in np.unique(blocks):
            te = np.where(blocks == test_b)[0]
            tr = np.where(blocks != test_b)[0]
            ss_part = _calibrate_ssvep_part(raw_ss_fbcca, raw_ss_cca, y[tr], tr, te, ss_win[0], ss_dur)
            sub_idx = _resolve_fold_subset(
                covs, ss_part, y, blocks, tr, test_b, sub, meta,
                ch_cfg, subject_fold_channels, diagnostic, extended_meta,
            )
            covs_fold = covs[:, :, sub_idx, :][:, :, :, sub_idx]
            fold_data = _calibrate_fold(
                raw_ss_fbcca, raw_ss_cca, covs_fold, y, blocks, tr, te,
                ss_win[0], ss_dur, ss_part=ss_part,
            )
            ctx = _make_diag_context(sub, meta, test_b, extended_meta, ch_cfg)
            _run_fold_fusion(
                fold_data, y[tr], y[te], blocks[tr], s_data['trial_ids'][te],
                prior_m_name, spec['duration_s'], len(sub_idx), ctx, diagnostic,
                preds, te, extended_meta=extended_meta,
            )

        _append_metric_rows(all_records, preds, y, sub, meta, extended_meta, ch_cfg)
    print(f" done ({time.time() - t_dur:.1f}s)")


def _evaluate_segmented_group(w_conf, loaded_data, subjects, subject_prior_best,
                              diagnostic, all_records, extended_meta,
                              ch_cfg, subject_fold_channels):
    specs = w_conf['specs']
    n_segs = len(specs)
    step_s = specs[0]['duration_s']
    ss_lat = specs[0]['ssvep_latency_s']
    mi_lat = specs[0]['mi_latency_s']
    mi_names = ch_cfg['mi_channels']

    for sub in subjects:
        t_sub = time.time()
        print(f"  Subject {sub}: {n_segs} segments (step={step_s:.2f}s) ...", end='', flush=True)
        s_data = loaded_data[sub]
        y, blocks, ons = s_data['y'], s_data['blocks'], s_data['ons']
        prior_m_name = subject_prior_best.get(sub, 'fbcsp')

        seg_features = [
            extract_dual_features(
                s_data['epochs'], ons, s_data['ch_labels'], OCC2, mi_names,
                sp['ssvep_window'], sp['mi_window'], ALL_MI_BANDS, FBCCA_BANDS, FB_WEIGHTS,
                NOMINAL_FREQUENCIES, FS, HARMONICS, segment_index=sp['segment_index'],
            )
            for sp in specs
        ]

        step_preds = [_init_pred_buffers(len(y), extended_meta=extended_meta) for _ in range(n_segs)]
        accum_preds = [_init_pred_buffers(len(y), extended_meta=extended_meta) for _ in range(n_segs)]
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

        for test_b in np.unique(blocks):
            te = np.where(blocks == test_b)[0]
            tr = np.where(blocks != test_b)[0]
            fold_segs = []
            for k, sp in enumerate(specs):
                covs_k, raw_fb_k, raw_cca_k = seg_features[k]
                ss_part = _calibrate_ssvep_part(
                    raw_fb_k, raw_cca_k, y[tr], tr, te, sp['ssvep_window'][0], sp['duration_s'],
                )
                sub_idx = _resolve_fold_subset(
                    covs_k, ss_part, y, blocks, tr, test_b, sub, step_metas[k],
                    ch_cfg, subject_fold_channels, diagnostic, extended_meta,
                )
                covs_fold = covs_k[:, :, sub_idx, :][:, :, :, sub_idx]
                f_k = _calibrate_fold(
                    raw_fb_k, raw_cca_k, covs_fold, y, blocks, tr, te,
                    sp['ssvep_window'][0], sp['duration_s'], ss_part=ss_part,
                )
                fold_segs.append(f_k)

                step_ctx = _make_diag_context(sub, step_metas[k], test_b, extended_meta, ch_cfg)
                _run_fold_fusion(
                    f_k, y[tr], y[te], blocks[tr], s_data['trial_ids'][te],
                    prior_m_name, sp['duration_s'], len(sub_idx), step_ctx, diagnostic,
                    step_preds[k], te, extended_meta=extended_meta,
                )

                acc_fold = _accumulate_fold_step(fold_segs, y[tr])
                acc_ctx = _make_diag_context(sub, accum_metas[k], test_b, extended_meta, ch_cfg)
                _run_fold_fusion(
                    acc_fold, y[tr], y[te], blocks[tr], s_data['trial_ids'][te],
                    prior_m_name, sp['cum_duration_s'], len(sub_idx), acc_ctx, diagnostic,
                    accum_preds[k], te, extended_meta=extended_meta,
                )

        for k in range(n_segs):
            _append_metric_rows(all_records, step_preds[k], y, sub, step_metas[k], extended_meta, ch_cfg)
            _append_metric_rows(all_records, accum_preds[k], y, sub, accum_metas[k], extended_meta, ch_cfg)
        print(f" done ({time.time() - t_sub:.1f}s)")


def main():
    args = arguments('adaptive')
    cache_dir, out_dir = args.cache_dir, args.out_dir
    subjects = args.subjects
    subject_prior_best = {
        'FHY': 'fbpower_lda',
        'YDL': 'fbcsp',
        'YZL': 'fbpower_lda',
    }

    mi_pool = getattr(args, 'mi_pool', 'central18')
    ch_cfg = {
        'mi_pool': mi_pool,
        'mi_channels': MI_POOLS[mi_pool],
        'ch_select': getattr(args, 'ch_select', 'none'),
        'ch_k_min': getattr(args, 'ch_k_min', 8),
        'ch_k_max': getattr(args, 'ch_k_max', 12),
        'ch_objective': getattr(args, 'ch_objective', 'fusion_logloss'),
        'ch_scope': getattr(args, 'ch_scope', 'subject_fold'),
        'ch_seed': getattr(args, 'ch_seed', 42),
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
            'ch_labels': list(data['ch_labels']),
        }

    window_configs = windows(args)
    extended_meta = (
        any(wc['mode'] != 'sync' for wc in window_configs)
        or ch_cfg['mi_pool'] != 'central18'
        or ch_cfg['ch_select'] != 'none'
    )

    all_records = []
    diagnostic = Diagnostics(out_dir)
    print(f"Starting Subject-Adaptive MI-SSVEP Fusion (2 Leads: O1, O2 | MI pool={ch_cfg['mi_pool']} | ch_select={ch_cfg['ch_select']})...")
    t0_all = time.time()

    subject_fold_channels = {}
    if ch_cfg['ch_scope'] == 'subject_fold' or ch_cfg['ch_select'] == 'none':
        subject_fold_channels = _precompute_subject_fold_channels(
            window_configs, loaded_data, subjects, ch_cfg, diagnostic, extended_meta,
        )

    for w_conf in window_configs:
        print(f"\n==========================================")
        print(f"WINDOW MODE: {w_conf['mode']} ({w_conf['name']}) - 2 LEADS")
        print(f"==========================================")
        if w_conf['mode'] == 'equal_segmented':
            _evaluate_segmented_group(
                w_conf, loaded_data, subjects, subject_prior_best,
                diagnostic, all_records, extended_meta, ch_cfg, subject_fold_channels,
            )
        else:
            for spec in w_conf['specs']:
                _evaluate_standard_spec(
                    spec, loaded_data, subjects, subject_prior_best,
                    diagnostic, all_records, extended_meta, ch_cfg, subject_fold_channels,
                )

    diagnostic.save()
    df_out = pd.DataFrame(all_records)
    df_out.to_csv(out_dir / 'adaptive_fusion_metrics_2leads.csv', index=False, encoding='utf-8-sig')
    print(f"\nAll evaluations complete in {time.time() - t0_all:.1f}s!")
    print(f"Saved metrics to: {out_dir / 'adaptive_fusion_metrics_2leads.csv'}")


if __name__ == '__main__':
    main()
