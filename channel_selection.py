"""Personalized MI channel selection (Stability-RFE, GA, PSO) for hybrid MI-SSVEP.

Strictly operates inside outer training folds (using inner-LOBO cross-validation)
to avoid outer-test leakage, and evaluates subsets under either fusion Log-Loss
complementarity ('fusion_logloss') or standalone MI Log-Loss ('mi_logloss').
"""
import numpy as np
from scipy import linalg

from windowing import align_proba_4class, apply_cov_car, logp, softmax

CENTRAL_18 = [
    'C3', 'C4', 'FC3', 'FC4', 'CP3', 'CP4', 'C1', 'C2', 'C5', 'C6',
    'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6',
]
CENTRAL_21 = [
    'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
]

MI_POOLS = {
    'central18': CENTRAL_18,
    'central21': CENTRAL_21,
}

VALID_CH_SELECTORS = ('none', 'rfe', 'ga', 'pso')
VALID_CH_OBJECTIVES = ('fusion_logloss', 'mi_logloss')
VALID_CH_SCOPES = ('subject_fold', 'window_fold')


def _shrink(c, amount=0.1):
    c = np.asarray(c, dtype=float)
    n = c.shape[-1]
    eye_scaled = np.trace(c, axis1=-2, axis2=-1)[..., None, None] * np.eye(n) / max(n, 1)
    return (1.0 - amount) * c + amount * eye_scaled


def _normtrace(c):
    tr = np.trace(c, axis1=-2, axis2=-1)[..., None, None]
    return c / np.maximum(tr, 1e-12)


def _cross_entropy(probs, y):
    p_true = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.mean(np.log(p_true)))


def _iter_inner_splits(blocks):
    unique_b = np.unique(blocks)
    for in_b in unique_b:
        val_mask = (blocks == in_b)
        yield np.where(~val_mask)[0], np.where(val_mask)[0]


def _fit_predict_scaled_lda(x_tr, y_tr, x_val, shrinkage=0.5, n_classes=4):
    mu_s = x_tr.mean(axis=0)
    sd_s = x_tr.std(axis=0)
    sd_s = np.where(sd_s == 0.0, 1.0, sd_s)
    xn_tr = (x_tr - mu_s) / sd_s
    xn_val = (x_val - mu_s) / sd_s

    d = xn_tr.shape[1]
    present = [c for c in range(n_classes) if np.any(y_tr == c)]
    full_means = np.zeros((n_classes, d), dtype=float)
    means = np.zeros((len(present), d), dtype=float)
    counts = np.zeros(len(present), dtype=float)
    for idx_c, cls in enumerate(present):
        m = xn_tr[y_tr == cls].mean(axis=0)
        full_means[cls] = m
        means[idx_c] = m
        counts[idx_c] = float(np.sum(y_tr == cls))

    priors = np.maximum(counts / max(len(y_tr), 1), 1e-12)
    centered = xn_tr - full_means[y_tr]
    cov_w = (centered.T @ centered) / max(len(xn_tr), 1)
    cov_shrunk = (
        (1.0 - shrinkage) * cov_w
        + shrinkage * (np.trace(cov_w) / max(d, 1)) * np.eye(d)
        + 1e-12 * np.eye(d)
    )
    coef = np.linalg.solve(cov_shrunk, means.T).T
    intercept = -0.5 * np.sum(means * coef, axis=1) + np.log(priors)
    proba = softmax(xn_val @ coef.T + intercept)
    return align_proba_4class(proba, present, n_classes=n_classes)


def _anova_f_scores(x, y, n_classes=4):
    d = x.shape[1]
    present = [c for c in range(n_classes) if np.any(y == c)]
    if len(present) < 2:
        return np.zeros(d, dtype=float)
    full_means = np.zeros((n_classes, d), dtype=float)
    means = np.zeros((len(present), d), dtype=float)
    counts = np.zeros(len(present), dtype=float)
    for idx_c, cls in enumerate(present):
        m = x[y == cls].mean(axis=0)
        full_means[cls] = m
        means[idx_c] = m
        counts[idx_c] = float(np.sum(y == cls))
    grand = x.mean(axis=0)
    ss_between = np.sum(counts[:, None] * (means - grand) ** 2, axis=0) / max(len(present) - 1, 1)
    ss_within = np.sum((x - full_means[y]) ** 2, axis=0) / max(len(x) - len(present), 1)
    f_vals = np.where(ss_between <= 1e-15, 0.0, ss_between / np.maximum(ss_within, 1e-12))
    return np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_temperature_fast(z, y, bounds=(0.25, 8.0)):
    n = len(y)
    ix = np.arange(n)
    z_y = z[ix, y]
    z_sq = z * z
    b_lo, b_hi = 1.0 / bounds[1], 1.0 / bounds[0]
    beta = 1.0
    for _ in range(7):
        p = softmax(beta * z)
        mean_z = np.sum(p * z, axis=1)
        grad = float(np.mean(mean_z - z_y))
        hess = float(np.mean(np.sum(p * z_sq, axis=1) - mean_z * mean_z))
        beta_next = float(np.clip(beta - grad / max(hess, 1e-8), b_lo, b_hi))
        if abs(beta_next - beta) < 1e-5:
            beta = beta_next
            break
        beta = beta_next
    return 1.0 / beta


def _fit_oof_temperature_fast(z, y, splits, bounds=(0.25, 8.0)):
    if len(splits) < 5:
        tm = _fit_temperature_fast(z, y, bounds=bounds)
        return softmax(z / tm)
    out = np.zeros_like(z)
    for in_tr, in_val in splits:
        if len(in_tr) < 4 or len(np.unique(y[in_tr])) < 2:
            tm = _fit_temperature_fast(z, y, bounds=bounds)
        else:
            tm = _fit_temperature_fast(z[in_tr], y[in_tr], bounds=bounds)
        out[in_val] = softmax(z[in_val] / tm)
    return out


def _fast_fbpower_oof(diag_log_sub, y_tr, train_blocks, splits=None):
    splits = splits if splits is not None else list(_iter_inner_splits(train_blocks))
    x = diag_log_sub.reshape(len(y_tr), -1)
    oof = np.full((len(y_tr), 4), 0.25, dtype=float)
    for in_tr, in_val in splits:
        oof[in_val] = _fit_predict_scaled_lda(x[in_tr], y_tr[in_tr], x[in_val])
    return _fit_oof_temperature_fast(logp(oof), y_tr, splits, bounds=(0.25, 8.0))


def _fast_fbcsp_oof(covs_sub, y_tr, train_blocks, k_feat=8, reg=0.1, splits=None):
    covs_norm = _normtrace(covs_sub)
    n_bands_sub = covs_norm.shape[1] - 1
    n_ch = covs_norm.shape[-1]
    eye_ch = np.eye(n_ch)
    eye_reg = 1e-6 * eye_ch
    n_classes = 4

    unique_b = np.unique(train_blocks)
    n_blk = len(unique_b)
    blk_sums = np.zeros((n_blk, n_bands_sub, n_classes, n_ch, n_ch), dtype=float)
    blk_cnts = np.zeros((n_blk, n_classes), dtype=int)
    for b_pos, in_b in enumerate(unique_b):
        bm = train_blocks == in_b
        for cls in range(n_classes):
            cm = bm & (y_tr == cls)
            cnt = int(np.sum(cm))
            blk_cnts[b_pos, cls] = cnt
            if cnt > 0:
                blk_sums[b_pos, :, cls] = covs_norm[cm, 1:].sum(axis=0)

    tot_sums = blk_sums.sum(axis=0)
    tot_cnts = blk_cnts.sum(axis=0)
    oof = np.full((len(y_tr), n_classes), 0.25, dtype=float)

    for b_pos, in_b in enumerate(unique_b):
        val_mask = (train_blocks == in_b)
        in_tr = np.where(~val_mask)[0]
        in_val = np.where(val_mask)[0]

        tr_sums = tot_sums - blk_sums[b_pos]
        tr_cnts = tot_cnts - blk_cnts[b_pos]
        all_sum = tr_sums.sum(axis=1)
        all_cnt = int(tr_cnts.sum())

        band_feats = []
        for b_idx in range(n_bands_sub):
            band_filters = []
            for cls in range(n_classes):
                c_pos = int(tr_cnts[cls])
                c_neg = all_cnt - c_pos
                if c_pos == 0 or c_neg == 0:
                    band_filters.extend([eye_ch[0], eye_ch[-1]])
                    continue
                mean_pos = tr_sums[b_idx, cls] / c_pos
                mean_neg = (all_sum[b_idx] - tr_sums[b_idx, cls]) / c_neg
                a = (1.0 - reg) * mean_pos + (reg / n_ch) * np.trace(mean_pos) * eye_ch + eye_reg
                b = (1.0 - reg) * mean_neg + (reg / n_ch) * np.trace(mean_neg) * eye_ch + eye_reg
                _, w = linalg.eigh(a, a + b, check_finite=False)
                band_filters.extend([w[:, 0], w[:, -1]])
            w_t = np.asarray(band_filters).T
            z_band = covs_norm[:, b_idx + 1]
            band_feats.append(np.log(np.maximum(np.sum((z_band @ w_t) * w_t, axis=1), 1e-12)))

        x_all = np.concatenate(band_feats, axis=1)
        x_tr = x_all[in_tr]
        f_vals = _anova_f_scores(x_tr, y_tr[in_tr], n_classes=n_classes)
        top_idx = np.argsort(-f_vals, kind='mergesort')[:min(k_feat, x_tr.shape[1])]
        oof[in_val] = _fit_predict_scaled_lda(
            x_tr[:, top_idx], y_tr[in_tr], x_all[in_val][:, top_idx], n_classes=n_classes,
        )

    splits_inner = splits if splits is not None else list(_iter_inner_splits(train_blocks))
    return _fit_oof_temperature_fast(logp(oof), y_tr, splits_inner, bounds=(0.25, 8.0))


def _fit_anchored_b_newton(ls_in, lm_in, y_in, reg=0.05, b_max=3.0):
    n = len(y_in)
    ix = np.arange(n)
    lm_y = lm_in[ix, y_in]
    lm_sq = lm_in * lm_in

    def _loss(b_val):
        p = softmax(ls_in + b_val * lm_in)
        return -float(np.mean(np.log(np.clip(p[ix, y_in], 1e-12, 1.0)))) + reg * b_val * b_val

    loss0 = _loss(0.0)
    b = 0.0
    for _ in range(6):
        p = softmax(ls_in + b * lm_in)
        mean_lm = np.sum(p * lm_in, axis=1)
        grad = float(np.mean(mean_lm - lm_y)) + 2.0 * reg * b
        if b == 0.0 and grad >= 0.0:
            return 0.0
        hess = float(np.mean(np.sum(p * lm_sq, axis=1) - mean_lm * mean_lm)) + 2.0 * reg
        b_next = float(np.clip(b - grad / max(hess, 1e-8), 0.0, b_max))
        if abs(b_next - b) < 1e-6:
            b = b_next
            break
        b = b_next
    return b if _loss(b) < loss0 - 1e-10 else 0.0


def _anchored_fusion_oof_logloss(ps_tr, pm_tr, y_tr, train_blocks, reg=0.05, splits=None):
    ls = logp(ps_tr)
    lm = logp(pm_tr)
    p_oof = np.zeros_like(ps_tr)
    for in_tr, in_val in (splits if splits is not None else _iter_inner_splits(train_blocks)):
        b_opt = _fit_anchored_b_newton(ls[in_tr], lm[in_tr], y_tr[in_tr], reg=reg)
        p_oof[in_val] = softmax(ls[in_val] + b_opt * lm[in_val])
    return _cross_entropy(p_oof, y_tr)


def _eval_subset_objective(idx, covs_tr, y_tr, train_blocks, ps_tr,
                           objective, k_min, n_pool, card_penalty=0.002, splits=None,
                           mi_reref='none'):
    idx_arr = np.asarray(idx, dtype=int)
    covs_sub = covs_tr[:, :, idx_arr, :][:, :, :, idx_arr]
    if mi_reref == 'subset_car':
        covs_sub = apply_cov_car(covs_sub)

    covs_sub_norm = _normtrace(covs_sub[:, 1:])
    diag_sub = np.log(np.maximum(np.diagonal(covs_sub_norm, axis1=-2, axis2=-1), 1e-12))

    pm_power = _fast_fbpower_oof(diag_sub, y_tr, train_blocks, splits=splits)
    nll_power = _cross_entropy(pm_power, y_tr)

    pm_csp = _fast_fbcsp_oof(covs_sub, y_tr, train_blocks, splits=splits)
    nll_csp = _cross_entropy(pm_csp, y_tr)

    if nll_csp <= nll_power:
        pm_best, mi_nll = pm_csp, nll_csp
    else:
        pm_best, mi_nll = pm_power, nll_power

    if objective == 'fusion_logloss' and ps_tr is not None:
        fusion_nll = _anchored_fusion_oof_logloss(ps_tr, pm_best, y_tr, train_blocks, splits=splits)
        base_loss = 0.8 * fusion_nll + 0.2 * mi_nll
    else:
        base_loss = mi_nll

    penalty = card_penalty * (len(idx_arr) - k_min) / max(n_pool, 1)
    return float(base_loss + penalty)


def _channel_importance_scores(covs_sub, y_sub, reg=0.1, mi_reref='none'):
    """Combine multi-band CSP forward spatial pattern energy and trace-normalized log-power F-scores."""
    if mi_reref == 'subset_car':
        covs_sub = apply_cov_car(covs_sub)
    n_ch = covs_sub.shape[-1]
    csp_energy = np.zeros(n_ch, dtype=float)
    f_energy = np.zeros(n_ch, dtype=float)
    covs_norm = _normtrace(covs_sub)
    eye_ch = np.eye(n_ch)

    for band in range(1, covs_norm.shape[1]):
        z = covs_norm[:, band]
        sigma_all = _shrink(z.mean(0), reg) + 1e-6 * eye_ch

        for label in range(4):
            m_pos = y_sub == label
            m_neg = ~m_pos
            if not np.any(m_pos) or not np.any(m_neg):
                continue
            a = _shrink(z[m_pos].mean(0), reg) + 1e-6 * eye_ch
            b = _shrink(z[m_neg].mean(0), reg) + 1e-6 * eye_ch
            eigvals, w = linalg.eigh(a, a + b)
            for col in (0, -1):
                wk = w[:, col]
                denom = float(wk @ sigma_all @ wk) + 1e-8
                pattern = (sigma_all @ wk) / denom
                p_norm = np.linalg.norm(pattern)
                if p_norm > 1e-12:
                    pattern = pattern / p_norm
                weight = abs(float(eigvals[col]) - 0.5)
                csp_energy += weight * (pattern ** 2)

        diag_log = np.log(np.maximum(np.diagonal(covs_sub[:, band], axis1=-2, axis2=-1), 1e-12))
        f_energy += _anova_f_scores(diag_log, y_sub)

    if csp_energy.max() > 0:
        csp_energy = csp_energy / csp_energy.max()
    if f_energy.max() > 0:
        f_energy = f_energy / f_energy.max()
    return 0.5 * csp_energy + 0.5 * f_energy


def _select_stability_rfe(eval_fn, covs_tr, y_tr, train_blocks, k_min, k_max, mi_reref='none'):
    n_pool = covs_tr.shape[-1]
    fold_ranks = np.zeros(n_pool, dtype=float)
    splits = list(_iter_inner_splits(train_blocks))

    for in_tr, _ in splits:
        active = list(range(n_pool))
        rank_cursor = n_pool
        ranks = np.ones(n_pool, dtype=float)

        while len(active) > 2:
            covs_act = covs_tr[np.ix_(in_tr, range(covs_tr.shape[1]), active, active)]
            scores = _channel_importance_scores(covs_act, y_tr[in_tr], mi_reref=mi_reref)
            worst_pos = int(np.argmin(scores))
            elim_ch = active.pop(worst_pos)
            ranks[elim_ch] = float(rank_cursor)
            rank_cursor -= 1

        for rem_ch in active:
            ranks[rem_ch] = 1.5
        fold_ranks += ranks

    mean_ranks = fold_ranks / max(len(splits), 1)
    ranked_indices = [int(i) for i in np.argsort(mean_ranks, kind='mergesort')]

    best_idx, best_score = None, float('inf')
    for k in range(k_min, k_max + 1):
        cand = sorted(ranked_indices[:k])
        score = eval_fn(cand)
        if score < best_score - 1e-12:
            best_score = score
            best_idx = cand
    return best_idx, best_score


def _repair_mask(mask, k_min, k_max, rng):
    out = np.asarray(mask, dtype=bool).copy()
    ones = np.where(out)[0]
    if len(ones) < k_min:
        zeros = np.where(~out)[0]
        add = rng.choice(zeros, size=k_min - len(ones), replace=False)
        out[add] = True
    elif len(ones) > k_max:
        drop = rng.choice(ones, size=len(ones) - k_max, replace=False)
        out[drop] = False
    return out


def _select_ga(eval_fn, n_pool, k_min, k_max, seed=42, pop_size=24, n_gen=20, patience=6):
    rng = np.random.default_rng(seed)

    def _random_ind():
        k = int(rng.integers(k_min, k_max + 1))
        chosen = rng.choice(n_pool, size=k, replace=False)
        mask = np.zeros(n_pool, dtype=bool)
        mask[chosen] = True
        return mask

    pop = [_random_ind() for _ in range(pop_size)]
    best_idx, best_score = None, float('inf')
    stagnant = 0

    for _ in range(n_gen):
        scores = np.array([eval_fn(sorted(np.where(ind)[0].tolist())) for ind in pop])
        order = np.argsort(scores, kind='mergesort')
        gen_best_score = float(scores[order[0]])
        gen_best_idx = sorted(np.where(pop[order[0]])[0].tolist())

        if gen_best_score < best_score - 1e-4:
            best_score = gen_best_score
            best_idx = gen_best_idx
            stagnant = 0
        else:
            if gen_best_score < best_score:
                best_score = gen_best_score
                best_idx = gen_best_idx
            stagnant += 1

        if stagnant >= patience:
            break

        next_pop = [pop[order[0]].copy(), pop[order[1]].copy()] if pop_size >= 2 else [pop[order[0]].copy()]
        while len(next_pop) < pop_size:
            t1 = rng.choice(pop_size, size=min(2, pop_size), replace=False)
            t2 = rng.choice(pop_size, size=min(2, pop_size), replace=False)
            p1 = pop[t1[int(np.argmin(scores[t1]))]]
            p2 = pop[t2[int(np.argmin(scores[t2]))]]

            if rng.random() < 0.8:
                cx = rng.random(n_pool) < 0.5
                c1 = np.where(cx, p1, p2)
                c2 = np.where(cx, p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()

            for child in (c1, c2):
                if rng.random() < 0.50:
                    ones = np.where(child)[0]
                    zeros = np.where(~child)[0]
                    if len(ones) > 0 and len(zeros) > 0:
                        child[rng.choice(ones)] = False
                        child[rng.choice(zeros)] = True
                if rng.random() < 0.25:
                    flip_i = int(rng.integers(0, n_pool))
                    child[flip_i] = ~child[flip_i]
                next_pop.append(_repair_mask(child, k_min, k_max, rng))
        pop = next_pop[:pop_size]

    return best_idx, best_score


def _decode_pso_particle(pos_vec, u_k, k_min, k_max):
    k = int(round(float(np.clip(u_k, k_min, k_max))))
    return sorted(int(i) for i in np.argsort(-pos_vec, kind='mergesort')[:k])


def _select_pso(eval_fn, n_pool, k_min, k_max, seed=42, n_particles=24, n_iter=20, patience=6):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(n_particles, n_pool))
    vel = rng.uniform(-0.5, 0.5, size=(n_particles, n_pool))
    u = rng.uniform(k_min, k_max, size=n_particles)
    vu = rng.uniform(-0.5, 0.5, size=n_particles)

    pbest_pos = pos.copy()
    pbest_u = u.copy()
    pbest_scores = np.full(n_particles, float('inf'))

    gbest_pos = pos[0].copy()
    gbest_u = float(u[0])
    gbest_idx = _decode_pso_particle(pos[0], u[0], k_min, k_max)
    gbest_score = float('inf')
    stagnant = 0

    for it in range(n_iter):
        improved_signif = False
        for i in range(n_particles):
            cand = _decode_pso_particle(pos[i], u[i], k_min, k_max)
            score = eval_fn(cand)
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_pos[i] = pos[i].copy()
                pbest_u[i] = u[i]
            if score < gbest_score:
                if score < gbest_score - 1e-4:
                    improved_signif = True
                gbest_score = score
                gbest_pos = pos[i].copy()
                gbest_u = float(u[i])
                gbest_idx = cand

        stagnant = 0 if improved_signif else stagnant + 1
        if stagnant >= patience:
            break

        w = 0.8 - 0.4 * (it / max(n_iter - 1, 1))
        r1 = rng.random(size=(n_particles, n_pool))
        r2 = rng.random(size=(n_particles, n_pool))
        vel = np.clip(w * vel + 1.5 * r1 * (pbest_pos - pos) + 1.5 * r2 * (gbest_pos - pos), -2.0, 2.0)
        pos = np.clip(pos + vel, -4.0, 4.0)

        ru1 = rng.random(size=n_particles)
        ru2 = rng.random(size=n_particles)
        vu = np.clip(w * vu + 1.5 * ru1 * (pbest_u - u) + 1.5 * ru2 * (gbest_u - u), -1.5, 1.5)
        u = np.clip(u + vu, float(k_min), float(k_max))

    return gbest_idx, gbest_score


def select_channels(covs_tr, y_tr, train_blocks, channel_names, ps_tr=None,
                    method='rfe', k_min=8, k_max=12, objective='fusion_logloss',
                    seed=42, ga_pop=24, ga_gen=20, pso_particles=24, pso_iter=20,
                    mi_reref='none'):
    """Select personalized MI channel subset [k_min, k_max] inside one training fold."""
    n_pool = len(channel_names)
    if covs_tr.shape[-1] != n_pool:
        raise ValueError(f'covs_tr last dim ({covs_tr.shape[-1]}) != len(channel_names) ({n_pool})')
    if method == 'none':
        all_idx = list(range(n_pool))
        return {
            'indices': all_idx,
            'names': list(channel_names),
            'k': n_pool,
            'best_score': 0.0,
            'full_score': 0.0,
            'n_evals': 0,
            'method': 'none',
            'objective': objective,
        }
    if not (2 <= k_min <= k_max <= n_pool):
        raise ValueError(f'Invalid cardinality bounds [{k_min}, {k_max}] for pool size {n_pool}')

    splits = list(_iter_inner_splits(train_blocks))
    eval_cache = {}

    def _cached_eval(cand):
        key = tuple(sorted(int(i) for i in cand))
        if key not in eval_cache:
            eval_cache[key] = _eval_subset_objective(
                key, covs_tr, y_tr, train_blocks, ps_tr,
                objective=objective, k_min=k_min, n_pool=n_pool, splits=splits,
                mi_reref=mi_reref,
            )
        return eval_cache[key]

    full_idx = list(range(n_pool))
    full_score = _eval_subset_objective(
        full_idx, covs_tr, y_tr, train_blocks, ps_tr,
        objective=objective, k_min=k_min, n_pool=n_pool, card_penalty=0.0, splits=splits,
        mi_reref=mi_reref,
    )

    if method == 'rfe':
        best_idx, best_score = _select_stability_rfe(
            _cached_eval, covs_tr, y_tr, train_blocks, k_min, k_max, mi_reref=mi_reref,
        )
    elif method == 'ga':
        best_idx, best_score = _select_ga(
            _cached_eval, n_pool, k_min, k_max, seed=seed, pop_size=ga_pop, n_gen=ga_gen,
        )
    elif method == 'pso':
        best_idx, best_score = _select_pso(
            _cached_eval, n_pool, k_min, k_max, seed=seed, n_particles=pso_particles, n_iter=pso_iter,
        )
    else:
        raise ValueError(f'Unknown channel selection method: {method}')

    return {
        'indices': best_idx,
        'names': [channel_names[i] for i in best_idx],
        'k': len(best_idx),
        'best_score': float(best_score),
        'full_score': float(full_score),
        'n_evals': len(eval_cache),
        'method': method,
        'objective': objective,
    }


def channel_jaccard_stability(fold_channel_lists):
    """Compute mean pairwise Jaccard overlap and per-channel selection counts."""
    sets = [set(chs) for chs in fold_channel_lists]
    freq = {}
    for s in sets:
        for ch in s:
            freq[ch] = freq.get(ch, 0) + 1
    if len(sets) < 2:
        return 1.0, freq
    jaccards = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = len(sets[i] | sets[j])
            jaccards.append(len(sets[i] & sets[j]) / union if union else 1.0)
    return float(np.mean(jaccards)), freq
