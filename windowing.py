"""Window scheduling, dual-modality slicing, and cross-segment evidence accumulation.

Supports:
- sync: synchronized SSVEP/MI windows [start_s, start_s + dur] (legacy default)
- decoupled: physiological latency decoupling with unified decision deadline T
- equal_cumulative: equal effective duration L starting after each modality's latency
- equal_segmented: fixed-step consecutive non-overlapping segments + dynamic accumulation
"""
import numpy as np
from scipy import linalg, optimize, signal, special

VALID_WINDOW_MODES = ('sync', 'decoupled', 'equal_cumulative', 'equal_segmented', 'all_equal')
VALID_FILTER_MODES = ('epoch_zerophase', 'causal', 'window_zerophase')
VALID_MI_REREF_MODES = ('none', 'subset_car')
DEFAULT_FS = 250


def softmax(z):
    return special.softmax(z, axis=1)


def logp(p):
    return np.log(np.clip(p, 1e-8, 1.0))


def covariance(x):
    x = np.asarray(x, dtype=float)
    x = x - x.mean(0)
    return x.T @ x / max(len(x) - 1, 1)


def align_proba_4class(proba, classes, n_classes=4):
    """Map classifier probability columns onto canonical classes 0..n_classes-1."""
    proba = np.asarray(proba, dtype=float)
    classes = np.asarray(classes, dtype=int)
    if len(classes) == n_classes and np.array_equal(classes, np.arange(n_classes)):
        return proba
    out = np.full((len(proba), n_classes), 1e-6, dtype=float)
    for col, cls in enumerate(classes):
        if 0 <= int(cls) < n_classes:
            out[:, int(cls)] = proba[:, col]
    out = np.maximum(out, 1e-12)
    out /= out.sum(axis=1, keepdims=True)
    return out


def apply_subset_car(x):
    """Apply Common Average Reference strictly within the provided channel subset."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        return x
    return x - x.mean(axis=1, keepdims=True)


def apply_cov_car(covs):
    """Apply subset Common Average Reference directly on covariance matrices (H C H)."""
    covs = np.asarray(covs, dtype=float)
    n_ch = covs.shape[-1]
    if n_ch < 2:
        return covs.copy()
    h = np.eye(n_ch) - (1.0 / n_ch)
    return h @ covs @ h


def _ssvep_projection_basis(n_samp, freqs, fs=DEFAULT_FS, max_harmonics=5, max_freq=40.0):
    t = np.arange(n_samp) / fs
    nyq_limit = min(float(max_freq), fs / 2.0 - 1e-6)
    ff = sorted({
        round(float(f) * h, 8)
        for f in freqs
        for h in range(1, int(max_harmonics) + 1)
        if 0.0 < float(f) * h <= nyq_limit
    })
    if not ff:
        return np.empty((n_samp, 0), dtype=float)
    ref = np.column_stack([fun(2 * np.pi * f * t) for f in ff for fun in (np.sin, np.cos)])
    ref = ref - ref.mean(axis=0, keepdims=True)
    u, s, _ = linalg.svd(ref, full_matrices=False)
    return u[:, s > max(float(s.max(initial=0)) * 1e-8, 1e-12)]


def project_out_ssvep(x, freqs, fs=DEFAULT_FS, max_harmonics=5, max_freq=40.0):
    """Orthogonally project out phase-locked SSVEP harmonic components from a slice."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 3 or x.shape[1] < 1:
        return x
    xc = x - x.mean(axis=0, keepdims=True)
    u = _ssvep_projection_basis(len(x), freqs, fs=fs, max_harmonics=max_harmonics, max_freq=max_freq)
    if u.shape[1] == 0:
        return xc
    return xc - u @ (u.T @ xc)


def validate_subject_cache(data, required_channels=None, subject=None):
    """Validate preprocessed cache keys, trial alignment, finite values, labels, and channels."""
    tag = f'subject {subject}' if subject else 'cache'
    required_keys = ('ds_epochs', 'y', 'blocks', 'trial_ids', 'ons', 'ch_labels')
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise ValueError(f'{tag}: missing required cache keys {missing_keys}')

    epochs = data['ds_epochs']
    y_raw = np.asarray(data['y'])
    blocks = np.asarray(data['blocks'])
    trial_ids = data['trial_ids']
    ons_raw = np.asarray(data['ons'])
    ch_labels = [str(c) for c in data['ch_labels']]

    n_trials = len(y_raw)
    if n_trials == 0:
        raise ValueError(f'{tag}: cache contains zero trials')
    if any(len(arr) != n_trials for arr in (epochs, blocks, trial_ids, ons_raw)):
        raise ValueError(
            f'{tag}: misaligned cache lengths ds_epochs={len(epochs)}, y={n_trials}, '
            f'blocks={len(blocks)}, trial_ids={len(trial_ids)}, ons={len(ons_raw)}'
        )

    try:
        ons = np.asarray(ons_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{tag}: ons must be finite numeric sample indices') from exc
    if not np.all(np.isfinite(ons)):
        raise ValueError(f'{tag}: ons contains NaN or Inf')

    try:
        y_num = np.asarray(y_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{tag}: y labels must be integers in {{0, 1, 2, 3}}') from exc
    if not np.all(np.isfinite(y_num)):
        raise ValueError(f'{tag}: y contains NaN or Inf')
    y = y_num.astype(int)
    if not np.array_equal(y_num, y) or not set(np.unique(y).tolist()).issubset({0, 1, 2, 3}):
        raise ValueError(f'{tag}: y labels must be integers in {{0, 1, 2, 3}}, got {np.unique(y_raw).tolist()}')

    n_ch = len(ch_labels)
    if n_ch == 0:
        raise ValueError(f'{tag}: ch_labels is empty')
    if required_channels is not None:
        missing_ch = [c for c in required_channels if c not in ch_labels]
        if missing_ch:
            raise ValueError(f'{tag}: missing required channels {missing_ch}')

    for idx in range(n_trials):
        ep = np.asarray(epochs[idx], dtype=float)
        if ep.ndim != 2 or ep.shape[1] != n_ch:
            raise ValueError(f'{tag}: trial {idx} expected shape (n_samples, {n_ch}), got {ep.shape}')
        if not np.isfinite(ep).all():
            raise ValueError(f'{tag}: trial {idx} contains NaN or Inf')

    return {
        'epochs': epochs,
        'y': y,
        'blocks': blocks,
        'trial_ids': trial_ids,
        'ons': ons,
        'ch_labels': ch_labels,
    }


def fit_temperature(z, y, bounds=(0.005, 8.0)):
    def fun(logt):
        probs = softmax(z / np.exp(logt))
        return -np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)))

    opt = optimize.minimize_scalar(fun, bounds=np.log(bounds), method='bounded')
    return float(np.exp(opt.x)) if opt.success else 1.0


def fit_oof_temperature(z_tr, y_tr, train_blocks, bounds=(0.005, 8.0)):
    """Calibrate training probabilities with inner-OOF temperatures and return full temperature for test."""
    z_tr = np.asarray(z_tr, dtype=float)
    y_tr = np.asarray(y_tr, dtype=int)
    t_full = fit_temperature(z_tr, y_tr, bounds=bounds)
    unique_blocks = np.unique(train_blocks)
    if len(unique_blocks) < 2:
        return softmax(z_tr / t_full), t_full

    p_oof = np.zeros_like(z_tr)
    for in_b in unique_blocks:
        in_val = np.where(train_blocks == in_b)[0]
        in_tr = np.where(train_blocks != in_b)[0]
        if len(in_tr) < 4 or len(np.unique(y_tr[in_tr])) < 2:
            t_in = t_full
        else:
            t_in = fit_temperature(z_tr[in_tr], y_tr[in_tr], bounds=bounds)
        p_oof[in_val] = softmax(z_tr[in_val] / t_in)
    return p_oof, t_full


def reference(n, fs, freq, harmonics=4):
    t = np.arange(n) / fs
    hs = [h for h in range(1, int(harmonics) + 1) if freq * h < fs / 2]
    return np.column_stack([fun(2 * np.pi * freq * h * t) for h in hs for fun in (np.sin, np.cos)])


def cca_scores(x, freqs, fs=250, harmonics=4):
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


def expand_window_modes(modes):
    expanded = []
    for mode in modes:
        items = ('equal_cumulative', 'equal_segmented') if mode == 'all_equal' else (mode,)
        for item in items:
            if item not in expanded:
                expanded.append(item)
    return expanded


def _start_tag(start_s):
    tenths = start_s * 10
    if abs(tenths - round(tenths)) < 1e-6:
        return f'start{int(round(tenths)):02d}'
    return f'start{int(round(start_s * 100)):03d}'


def resolve_window_configs(starts, durations, window_modes=('sync',),
                           ssvep_latency=0.14, mi_latency=0.50,
                           segment_step=0.5, epoch_limit=4.0, min_win_s=0.12):
    """Build uniform window configuration groups for one or more window modes."""
    modes = expand_window_modes(window_modes)
    unique_starts = list(dict.fromkeys(starts))
    default_durs = [i / 2 for i in range(1, 9)]
    explicit_durs = durations is not None
    base_durs = list(dict.fromkeys(durations if explicit_durs else default_durs))
    max_lat = max(ssvep_latency, mi_latency)

    configs = []
    for mode in modes:
        if mode == 'sync':
            for start_s in unique_starts:
                durs = list(dict.fromkeys(d for d in base_durs if start_s + d <= epoch_limit + 1e-9))
                if not durs:
                    continue
                name = _start_tag(start_s)
                specs = [
                    {
                        'mode': 'sync',
                        'scheme': name,
                        'window_start_s': start_s,
                        'duration_s': d,
                        'cum_duration_s': d,
                        'decision_time_s': round(start_s + d, 6),
                        'ssvep_window': (start_s, round(start_s + d, 6)),
                        'mi_window': (start_s, round(start_s + d, 6)),
                        'segment_index': None,
                        'segment_step_s': None,
                    }
                    for d in durs
                ]
                configs.append(dict(name=name, mode='sync', start_s=start_s, durations=durs, specs=specs))

        elif mode == 'decoupled':
            for start_s in unique_starts:
                specs = []
                durs = []
                tag = f'decoupled_{_start_tag(start_s)}'
                for d in base_durs:
                    deadline = round(start_s + d, 6)
                    if deadline > epoch_limit + 1e-9:
                        continue
                    if (deadline - ssvep_latency) < min_win_s or (deadline - mi_latency) < min_win_s:
                        continue
                    durs.append(d)
                    specs.append({
                        'mode': 'decoupled',
                        'scheme': tag,
                        'window_start_s': start_s,
                        'duration_s': d,
                        'cum_duration_s': d,
                        'decision_time_s': deadline,
                        'ssvep_window': (ssvep_latency, deadline),
                        'mi_window': (mi_latency, deadline),
                        'segment_index': None,
                        'segment_step_s': None,
                    })
                if specs:
                    configs.append(dict(
                        name=tag, mode='decoupled', start_s=start_s, durations=durs, specs=specs,
                    ))

        elif mode == 'equal_cumulative':
            durs = list(dict.fromkeys(
                d for d in base_durs if max_lat + d <= epoch_limit + 1e-9 and d >= min_win_s
            ))
            specs = [
                {
                    'mode': 'equal_cumulative',
                    'scheme': 'equal_cumulative',
                    'window_start_s': mi_latency,
                    'duration_s': d,
                    'cum_duration_s': d,
                    'decision_time_s': round(max_lat + d, 6),
                    'ssvep_window': (ssvep_latency, round(ssvep_latency + d, 6)),
                    'mi_window': (mi_latency, round(mi_latency + d, 6)),
                    'segment_index': None,
                    'segment_step_s': None,
                }
                for d in durs
            ]
            if specs:
                configs.append(dict(
                    name='equal_cumulative', mode='equal_cumulative',
                    start_s=mi_latency, durations=durs, specs=specs,
                ))

        elif mode == 'equal_segmented':
            if segment_step < min_win_s or max_lat + segment_step > epoch_limit + 1e-9:
                raise ValueError(f'segment_step={segment_step} does not fit [{max_lat}, {epoch_limit}]')
            max_avail = epoch_limit - max_lat
            if explicit_durs:
                max_avail = min(max_avail, max(max(base_durs), segment_step))
            k_max = max(1, int(np.floor((max_avail + 1e-9) / segment_step)))
            specs = []
            durs = []
            for k in range(1, k_max + 1):
                cum_d = round(k * segment_step, 6)
                ss_a = round(ssvep_latency + (k - 1) * segment_step, 6)
                ss_b = round(ssvep_latency + k * segment_step, 6)
                mi_a = round(mi_latency + (k - 1) * segment_step, 6)
                mi_b = round(mi_latency + k * segment_step, 6)
                durs.append(cum_d)
                specs.append({
                    'mode': 'equal_segmented',
                    'scheme': 'equal_segmented',
                    'window_start_s': mi_a,
                    'duration_s': segment_step,
                    'cum_duration_s': cum_d,
                    'decision_time_s': round(max_lat + cum_d, 6),
                    'ssvep_window': (ss_a, ss_b),
                    'mi_window': (mi_a, mi_b),
                    'ssvep_latency_s': ssvep_latency,
                    'mi_latency_s': mi_latency,
                    'segment_index': k,
                    'segment_step_s': segment_step,
                })
            configs.append(dict(
                name='equal_segmented', mode='equal_segmented',
                start_s=mi_latency, step_s=segment_step, durations=durs, specs=specs,
            ))

    if not configs:
        raise ValueError(f'No valid windows fit epoch_limit={epoch_limit}s for modes={modes}')
    return configs


def sample_bounds(on, start_s, dur_s, fs=DEFAULT_FS, segment_index=None):
    """Compute exact integer sample indices guaranteeing constant length across segments."""
    n_samp = int(round(dur_s * fs))
    if segment_index is not None:
        base_start_s = start_s - (segment_index - 1) * dur_s
        anchor = int(round(on + base_start_s * fs))
        w_start = anchor + (segment_index - 1) * n_samp
        return w_start, w_start + n_samp
    w_start = int(round(on + start_s * fs))
    return w_start, w_start + n_samp


def format_window_span(win):
    return f'[{win[0]:.2f}, {win[1]:.2f}]'


def extract_dual_features(epochs, ons, ch_labels, occ_channels, mi_channels,
                          ssvep_window, mi_window, all_mi_bands, fbcca_bands,
                          fb_weights, nominal_freqs, fs=DEFAULT_FS, harmonics=4,
                          segment_index=None, extract_cca=True,
                          filter_mode='epoch_zerophase', mi_reref='none',
                          mi_decouple_ssvep=False):
    """Extract MI band covariances and SSVEP CCA/FBCCA scores for independent SSVEP/MI windows."""
    if filter_mode not in VALID_FILTER_MODES:
        raise ValueError(f'Unknown filter_mode={filter_mode!r}; expected one of {VALID_FILTER_MODES}')
    if mi_reref not in VALID_MI_REREF_MODES:
        raise ValueError(f'Unknown mi_reref={mi_reref!r}; expected one of {VALID_MI_REREF_MODES}')

    ch_labels = [str(c) for c in ch_labels]
    missing = [c for c in list(mi_channels) + list(occ_channels) if c not in ch_labels]
    if missing:
        raise ValueError(f'Missing required channels in ch_labels: {missing}')

    n_trials = len(epochs)
    if len(ons) != n_trials:
        raise ValueError(f'Mismatched epochs ({n_trials}) and ons ({len(ons)}) lengths')

    ci = [ch_labels.index(c) for c in mi_channels]
    oi = [ch_labels.index(c) for c in occ_channels]

    ss_start_s, ss_end_s = ssvep_window
    mi_start_s, mi_end_s = mi_window
    ss_dur_s = ss_end_s - ss_start_s
    mi_dur_s = mi_end_s - mi_start_s
    ss_n_samp = int(round(ss_dur_s * fs))
    mi_n_samp = int(round(mi_dur_s * fs))
    if mi_n_samp < 3 or ss_n_samp < 3:
        raise ValueError(f'Window sample counts must be >= 3 (got mi={mi_n_samp}, ss={ss_n_samp})')

    covs = np.zeros((n_trials, len(all_mi_bands), len(ci), len(ci)))
    raw_ss_fbcca = np.zeros((n_trials, len(nominal_freqs)))
    raw_ss_cca = np.zeros((n_trials, len(nominal_freqs))) if extract_cca else None

    mi_sos = [signal.butter(4, band, fs=fs, btype='bandpass', output='sos') for band in all_mi_bands]
    fb_sos = [signal.butter(4, [lo, hi], fs=fs, btype='bandpass', output='sos') for lo, hi in fbcca_bands]
    occ_broad_band = [6.0, min(45.0, 0.45 * fs)]
    occ_broad_sos = signal.butter(4, occ_broad_band, fs=fs, btype='bandpass', output='sos') if extract_cca else None
    min_padlen = max(3 * (2 * sos.shape[0] + 1) for sos in (*mi_sos, *fb_sos))
    u_proj = _ssvep_projection_basis(mi_n_samp, nominal_freqs, fs=fs) if mi_decouple_ssvep else None

    for i in range(n_trials):
        ds = np.asarray(epochs[i], dtype=float)
        if ds.ndim != 2 or ds.shape[1] != len(ch_labels):
            raise ValueError(f'Trial {i} shape {ds.shape} does not match {len(ch_labels)} channels')
        if not np.isfinite(ds).all():
            raise ValueError(f'Trial {i} contains NaN or Inf')

        on = float(ons[i])
        mi_a, mi_b = sample_bounds(on, mi_start_s, mi_dur_s, fs=fs, segment_index=segment_index)
        ss_a, ss_b = sample_bounds(on, ss_start_s, ss_dur_s, fs=fs, segment_index=segment_index)
        if not (0 <= mi_a < mi_b <= len(ds)) or (mi_b - mi_a) != mi_n_samp:
            raise ValueError(
                f'MI window [{mi_a}, {mi_b}) (expected {mi_n_samp} samples) out of bounds for trial {i} len={len(ds)}'
            )
        if not (0 <= ss_a < ss_b <= len(ds)) or (ss_b - ss_a) != ss_n_samp:
            raise ValueError(
                f'SSVEP window [{ss_a}, {ss_b}) (expected {ss_n_samp} samples) out of bounds for trial {i} len={len(ds)}'
            )

        if filter_mode == 'epoch_zerophase':
            if len(ds) <= min_padlen:
                raise ValueError(f'Trial {i} length {len(ds)} <= sosfiltfilt padlen {min_padlen}')
            x_mi_src = ds[:, ci]
            if mi_reref == 'subset_car':
                x_mi_src = apply_subset_car(x_mi_src)
            for b_idx, sos in enumerate(mi_sos):
                seg_mi = signal.sosfiltfilt(sos, x_mi_src, axis=0)[mi_a:mi_b]
                if u_proj is not None and u_proj.shape[1] > 0:
                    seg_c = seg_mi - seg_mi.mean(axis=0, keepdims=True)
                    seg_mi = seg_c - u_proj @ (u_proj.T @ seg_c)
                covs[i, b_idx] = covariance(seg_mi)

            x_occ_src = ds[:, oi]
            if extract_cca:
                x_occ_broad = signal.sosfiltfilt(occ_broad_sos, x_occ_src, axis=0)[ss_a:ss_b]
                raw_ss_cca[i] = cca_scores(x_occ_broad, nominal_freqs, fs, harmonics)
            fb_score = np.zeros(len(nominal_freqs))
            for b_idx, sos in enumerate(fb_sos):
                x_sub = signal.sosfiltfilt(sos, x_occ_src, axis=0)[ss_a:ss_b]
                fb_score += fb_weights[b_idx] * (cca_scores(x_sub, nominal_freqs, fs, harmonics) ** 2)
            raw_ss_fbcca[i] = fb_score

        elif filter_mode == 'causal':
            x_mi_avail = ds[:mi_b, ci]
            if mi_reref == 'subset_car':
                x_mi_avail = apply_subset_car(x_mi_avail)
            x_mi_avail = x_mi_avail - x_mi_avail[0:1]
            for b_idx, sos in enumerate(mi_sos):
                seg_mi = signal.sosfilt(sos, x_mi_avail, axis=0)[mi_a:mi_b]
                if u_proj is not None and u_proj.shape[1] > 0:
                    seg_c = seg_mi - seg_mi.mean(axis=0, keepdims=True)
                    seg_mi = seg_c - u_proj @ (u_proj.T @ seg_c)
                covs[i, b_idx] = covariance(seg_mi)

            x_occ_avail = ds[:ss_b, oi]
            x_occ_avail = x_occ_avail - x_occ_avail[0:1]
            if extract_cca:
                x_occ_broad = signal.sosfilt(occ_broad_sos, x_occ_avail, axis=0)[ss_a:ss_b]
                raw_ss_cca[i] = cca_scores(x_occ_broad, nominal_freqs, fs, harmonics)
            fb_score = np.zeros(len(nominal_freqs))
            for b_idx, sos in enumerate(fb_sos):
                x_sub = signal.sosfilt(sos, x_occ_avail, axis=0)[ss_a:ss_b]
                fb_score += fb_weights[b_idx] * (cca_scores(x_sub, nominal_freqs, fs, harmonics) ** 2)
            raw_ss_fbcca[i] = fb_score

        else:  # window_zerophase
            if mi_n_samp <= min_padlen or ss_n_samp <= min_padlen:
                raise ValueError(
                    f'Window length (mi={mi_n_samp}, ss={ss_n_samp}) <= sosfiltfilt padlen {min_padlen}'
                )
            x_central = ds[mi_a:mi_b, ci]
            if mi_reref == 'subset_car':
                x_central = apply_subset_car(x_central)
            for b_idx, sos in enumerate(mi_sos):
                seg_mi = signal.sosfiltfilt(sos, x_central, axis=0)
                if u_proj is not None and u_proj.shape[1] > 0:
                    seg_c = seg_mi - seg_mi.mean(axis=0, keepdims=True)
                    seg_mi = seg_c - u_proj @ (u_proj.T @ seg_c)
                covs[i, b_idx] = covariance(seg_mi)

            x_occ = ds[ss_a:ss_b, oi]
            if extract_cca:
                x_occ_broad = signal.sosfiltfilt(occ_broad_sos, x_occ, axis=0)
                raw_ss_cca[i] = cca_scores(x_occ_broad, nominal_freqs, fs, harmonics)
            fb_score = np.zeros(len(nominal_freqs))
            for b_idx, sos in enumerate(fb_sos):
                x_sub = signal.sosfiltfilt(sos, x_occ, axis=0)
                fb_score += fb_weights[b_idx] * (cca_scores(x_sub, nominal_freqs, fs, harmonics) ** 2)
            raw_ss_fbcca[i] = fb_score

    return covs, raw_ss_fbcca, raw_ss_cca


def accumulate_log_probs(tr_probs_list, te_probs_list, y_tr, bounds=(0.2, 8.0), train_blocks=None):
    """Accumulate Bayesian log-posteriors over 1..k segments with fold-only calibration.

    At k == 1, returns the exact single-segment probabilities untouched so
    accumulated step 1 strictly matches single-segment and 0.5s cumulative baselines.
    """
    if len(tr_probs_list) == 1:
        return tr_probs_list[0], te_probs_list[0], 1.0
    sum_tr = np.sum([logp(p) for p in tr_probs_list], axis=0)
    sum_te = np.sum([logp(p) for p in te_probs_list], axis=0)
    if train_blocks is not None:
        acc_tr, temp = fit_oof_temperature(sum_tr, y_tr, train_blocks, bounds=bounds)
        return acc_tr, softmax(sum_te / temp), temp
    temp = fit_temperature(sum_tr, y_tr, bounds=bounds)
    return softmax(sum_tr / temp), softmax(sum_te / temp), temp
