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
DEFAULT_FS = 250


def softmax(z):
    return special.softmax(z, axis=1)


def logp(p):
    return np.log(np.clip(p, 1e-8, 1.0))


def covariance(x):
    x = x - x.mean(0)
    return x.T @ x / max(len(x) - 1, 1)


def fit_temperature(z, y, bounds=(0.005, 8.0)):
    def fun(logt):
        probs = softmax(z / np.exp(logt))
        return -np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)))

    opt = optimize.minimize_scalar(fun, bounds=np.log(bounds), method='bounded')
    return float(np.exp(opt.x)) if opt.success else 1.0


def reference(n, fs, freq, harmonics=3):
    t = np.arange(n) / fs
    hs = [h for h in range(1, int(harmonics) + 1) if freq * h < fs / 2]
    return np.column_stack([fun(2 * np.pi * freq * h * t) for h in hs for fun in (np.sin, np.cos)])


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
                          fb_weights, nominal_freqs, fs=DEFAULT_FS, harmonics=3,
                          segment_index=None, extract_cca=True):
    """Extract MI band covariances and SSVEP CCA/FBCCA scores for independent SSVEP/MI windows."""
    n_trials = len(epochs)
    ci = [ch_labels.index(c) for c in mi_channels]
    oi = [ch_labels.index(c) for c in occ_channels]

    ss_start_s, ss_end_s = ssvep_window
    mi_start_s, mi_end_s = mi_window
    ss_dur_s = ss_end_s - ss_start_s
    mi_dur_s = mi_end_s - mi_start_s

    covs = np.zeros((n_trials, len(all_mi_bands), len(ci), len(ci)))
    raw_ss_fbcca = np.zeros((n_trials, len(nominal_freqs)))
    raw_ss_cca = np.zeros((n_trials, len(nominal_freqs))) if extract_cca else None

    mi_sos = [signal.butter(4, band, fs=fs, btype='bandpass', output='sos') for band in all_mi_bands]
    fb_sos = [signal.butter(4, [lo, hi], fs=fs, btype='bandpass', output='sos') for lo, hi in fbcca_bands]

    for i in range(n_trials):
        ds = epochs[i]
        on = ons[i]

        mi_a, mi_b = sample_bounds(on, mi_start_s, mi_dur_s, fs=fs, segment_index=segment_index)
        x_central = ds[mi_a:mi_b, ci]
        for b_idx, sos in enumerate(mi_sos):
            xf = signal.sosfiltfilt(sos, x_central, axis=0)
            covs[i, b_idx] = covariance(xf)

        ss_a, ss_b = sample_bounds(on, ss_start_s, ss_dur_s, fs=fs, segment_index=segment_index)
        x_occ = ds[ss_a:ss_b, oi]
        if extract_cca:
            raw_ss_cca[i] = cca_scores(x_occ, nominal_freqs, fs, harmonics)

        fb_score = np.zeros(len(nominal_freqs))
        for b_idx, sos in enumerate(fb_sos):
            x_sub = signal.sosfiltfilt(sos, x_occ, axis=0)
            fb_score += fb_weights[b_idx] * (cca_scores(x_sub, nominal_freqs, fs, harmonics) ** 2)
        raw_ss_fbcca[i] = fb_score

    return covs, raw_ss_fbcca, raw_ss_cca


def accumulate_log_probs(tr_probs_list, te_probs_list, y_tr, bounds=(0.2, 8.0)):
    """Accumulate Bayesian log-posteriors over 1..k segments with fold-only calibration.

    At k == 1, returns the exact single-segment probabilities untouched so
    accumulated step 1 strictly matches single-segment and 0.5s cumulative baselines.
    """
    if len(tr_probs_list) == 1:
        return tr_probs_list[0], te_probs_list[0], 1.0
    sum_tr = np.sum([logp(p) for p in tr_probs_list], axis=0)
    sum_te = np.sum([logp(p) for p in te_probs_list], axis=0)
    temp = fit_temperature(sum_tr, y_tr, bounds=bounds)
    return softmax(sum_tr / temp), softmax(sum_te / temp), temp
