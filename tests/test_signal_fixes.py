"""Regression tests for EEG signal processing, normalization, invariance, and leakage fixes."""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import signal

import run_adaptive_fusion as adaptive
import run_comparison as comparison
import run_reduced_mi_leads as reduced
from windowing import (
    apply_cov_car,
    apply_subset_car,
    covariance,
    extract_dual_features,
    project_out_ssvep,
    validate_subject_cache,
)


def _rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


class SignalProcessingFixTests(unittest.TestCase):
    def test_validate_subject_cache_rejects_invalid_inputs(self):
        labels = np.array(['C3', 'C4', 'O1', 'O2'])
        base = {
            'ds_epochs': np.zeros((4, 300, 4), dtype=float),
            'y': np.array([0, 1, 2, 3]),
            'blocks': np.array([0, 0, 1, 1]),
            'trial_ids': np.array([1, 2, 3, 4]),
            'ons': np.zeros(4, dtype=float),
            'ch_labels': labels,
        }
        parsed = validate_subject_cache(base, required_channels=['C3', 'O1'], subject='S01')
        self.assertEqual(parsed['ch_labels'], ['C3', 'C4', 'O1', 'O2'])

        bad_label = dict(base, y=np.array([0, 1, 2, 4]))
        with self.assertRaises(ValueError):
            validate_subject_cache(bad_label, subject='S01')

        bad_nan = dict(base)
        ep_nan = base['ds_epochs'].copy()
        ep_nan[1, 10, 0] = np.nan
        bad_nan['ds_epochs'] = ep_nan
        with self.assertRaises(ValueError):
            validate_subject_cache(bad_nan, subject='S01')

        with self.assertRaises(ValueError):
            validate_subject_cache(base, required_channels=['Cz'], subject='S01')

    def test_extract_dual_features_rejects_out_of_bounds_windows(self):
        rng = np.random.default_rng(7)
        labels = ['C3', 'C4', 'O1', 'O2']
        epochs = rng.normal(size=(4, 250, 4))
        mi_bands = [(8, 30), (8, 12)]
        fb_bands = [(6, 45), (9, 45)]
        fb_weights = np.array([1.25, 0.67])
        freqs = [12.0, 10.0, 60.0 / 7.0, 7.5]

        with self.assertRaises(ValueError):
            extract_dual_features(
                epochs, np.zeros(4), labels, ['O1', 'O2'], ['C3', 'C4'],
                (0.5, 1.5), (0.5, 1.5), mi_bands, fb_bands, fb_weights, freqs,
            )

        with self.assertRaises(ValueError):
            extract_dual_features(
                epochs, np.full(4, -50.0), labels, ['O1', 'O2'], ['C3', 'C4'],
                (0.0, 0.5), (0.0, 0.5), mi_bands, fb_bands, fb_weights, freqs,
            )

    def test_causal_mode_has_zero_future_leakage_and_epoch_mode_handles_short_slices(self):
        rng = np.random.default_rng(11)
        labels = ['C3', 'C4', 'O1', 'O2']
        epochs = rng.normal(size=(2, 600, 4))
        ons = np.full(2, 100.0)
        mi_bands = [(8, 30), (8, 12)]
        fb_bands = [(6, 45), (9, 45)]
        fb_weights = np.array([1.25, 0.67])
        freqs = [12.0, 10.0, 60.0 / 7.0, 7.5]

        cov_c1, fb_c1, cca_c1 = extract_dual_features(
            epochs, ons, labels, ['O1', 'O2'], ['C3', 'C4'],
            (0.14, 0.64), (0.50, 1.00), mi_bands, fb_bands, fb_weights, freqs,
            filter_mode='causal',
        )
        epochs_mut = epochs.copy()
        epochs_mut[:, 350:, :] += 500.0  # strictly after mi_b=350 and ss_b=260
        cov_c2, fb_c2, cca_c2 = extract_dual_features(
            epochs_mut, ons, labels, ['O1', 'O2'], ['C3', 'C4'],
            (0.14, 0.64), (0.50, 1.00), mi_bands, fb_bands, fb_weights, freqs,
            filter_mode='causal',
        )
        np.testing.assert_allclose(cov_c1, cov_c2, atol=1e-12)
        np.testing.assert_allclose(fb_c1, fb_c2, atol=1e-12)
        np.testing.assert_allclose(cca_c1, cca_c2, atol=1e-12)

        # epoch_zerophase cleanly extracts a 0.10s (25 samples < sosfiltfilt padlen=27) window
        cov_short, fb_short, cca_short = extract_dual_features(
            epochs, ons, labels, ['O1', 'O2'], ['C3', 'C4'],
            (0.14, 0.24), (0.50, 0.60), mi_bands, fb_bands, fb_weights, freqs,
            filter_mode='epoch_zerophase',
        )
        self.assertEqual(cov_short.shape, (2, 2, 2, 2))
        self.assertTrue(np.isfinite(fb_short).all() and np.isfinite(cca_short).all())

    def test_broadband_cca_rejects_slow_drift_and_decouple_car_identities(self):
        fs = 250
        t = np.arange(800) / fs
        freqs = [12.0, 10.0, 60.0 / 7.0, 7.5]
        labels = ['C3', 'C4', 'FC3', 'O1', 'O2']
        sig = np.zeros((1, 800, 5), dtype=float)
        # Strong <2 Hz drift on occipital channels + pure 10 Hz SSVEP
        drift = 200.0 * np.sin(2 * np.pi * 1.2 * t) + 150.0 * t
        sig[0, :, 3] = drift + 2.0 * np.sin(2 * np.pi * 10.0 * t)
        sig[0, :, 4] = drift + 2.0 * np.cos(2 * np.pi * 10.0 * t)
        # Central channels have 10 Hz + 20 Hz SSVEP leakage + 18 Hz motor oscillation
        sig[0, :, 0] = 5.0 * np.sin(2 * np.pi * 10.0 * t) + 3.0 * np.sin(2 * np.pi * 20.0 * t) + np.sin(2 * np.pi * 18.0 * t)
        sig[0, :, 1] = 2.0 * np.sin(2 * np.pi * 10.0 * t) + np.cos(2 * np.pi * 18.0 * t)
        sig[0, :, 2] = np.sin(2 * np.pi * 18.0 * t + 0.5)

        _, _, raw_cca = extract_dual_features(
            sig, np.array([100.0]), labels, ['O1', 'O2'], ['C3', 'C4', 'FC3'],
            (0.0, 1.0), (0.0, 1.0), [(8, 30), (8, 12)], [(6, 45)], np.array([1.25]),
            freqs, fs=fs, harmonics=4,
        )
        self.assertEqual(int(np.argmax(raw_cca[0])), 1)  # 10 Hz class

        # Orthogonal SSVEP projection removes 10/20 Hz power while leaving 18 Hz intact
        pure_ssvep = sig[0, 100:350, :1] - np.sin(2 * np.pi * 18.0 * t[100:350])[:, None]
        projected = project_out_ssvep(pure_ssvep, freqs, fs=fs)
        self.assertLess(np.var(projected), 1e-10)

        # Subset CAR identity: signal-domain CAR == covariance-domain H C H
        cov_raw, _, _ = extract_dual_features(
            sig, np.array([100.0]), labels, ['O1', 'O2'], ['C3', 'C4', 'FC3'],
            (0.0, 1.0), (0.0, 1.0), [(8, 30), (8, 12)], [(6, 45)], np.array([1.25]),
            freqs, fs=fs, mi_reref='none',
        )
        cov_car_sig, _, _ = extract_dual_features(
            sig, np.array([100.0]), labels, ['O1', 'O2'], ['C3', 'C4', 'FC3'],
            (0.0, 1.0), (0.0, 1.0), [(8, 30), (8, 12)], [(6, 45)], np.array([1.25]),
            freqs, fs=fs, mi_reref='subset_car',
        )
        np.testing.assert_allclose(apply_cov_car(cov_raw), cov_car_sig, atol=1e-10)

    def test_fbpower_and_tangent_scale_invariance_and_rcsp_4ch(self):
        rng = np.random.default_rng(31)
        n_tr, n_te, n_bands, n_ch = 32, 16, 7, 4
        y_tr = np.tile(np.arange(4), n_tr // 4)

        def _rand_spd(n):
            a = rng.normal(size=(n, n_bands, n_ch, n_ch))
            return np.einsum('nbik,nbjk->nbij', a, a) + 0.1 * np.eye(n_ch)

        cov_tr = _rand_spd(n_tr)
        cov_te = _rand_spd(n_te)
        cov_te_scaled = cov_te * 25.0  # Simulate 5x amplitude / impedance drift

        for cls in (adaptive.FBPowerLDAModel, adaptive.TangentLogRegModel,
                    reduced.FBPowerLDAModel, reduced.TangentLogRegModel):
            m = cls().fit(cov_tr, y_tr)
            p1 = m.predict_proba(cov_te)
            p2 = m.predict_proba(cov_te_scaled)
            np.testing.assert_allclose(p1, p2, atol=1e-6)

        rcsp = adaptive.RCSPModel().fit(cov_tr, y_tr)
        self.assertEqual(rcsp.w.shape[0], 8)  # 4 classes * 2 extreme filters on 4ch

        # Missing-class safety on FBCSPLDAModel
        y_partial = np.array([0, 0, 1, 1, 2, 2] * 2)
        fbcsp = comparison.FBCSPLDAModel(k=8).fit(cov_tr[:12], y_partial)
        p_miss = fbcsp.predict_proba(cov_te)
        self.assertEqual(p_miss.shape, (n_te, 4))
        np.testing.assert_allclose(p_miss.sum(axis=1), 1.0, atol=1e-6)

    def test_subject_fold_channel_selection_invariant_to_cli_durations_and_start_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(77)
            labels = adaptive.CENTRAL + ['FCz', 'Cz', 'CPz', 'O1', 'Oz', 'O2']
            n_trials = 24
            np.savez(
                root / 'YDL_preprocessed.npz',
                ds_epochs=rng.normal(size=(n_trials, 1100, len(labels))),
                y=np.tile(np.arange(4), n_trials // 4),
                blocks=np.repeat(np.arange(3), n_trials // 3),
                ons=np.zeros(n_trials, dtype=float),
                trial_ids=np.arange(1, n_trials + 1),
                ch_labels=np.array(labels),
            )

            out_a = root / 'run_a'
            out_b = root / 'run_b'
            subprocess.run([
                sys.executable, 'run_adaptive_fusion.py',
                '--cache-dir', str(root), '--out-dir', str(out_a),
                '--subjects', 'YDL', '--starts', '0.5', '0.0', '--durations', '0.5',
                '--mi-pool', 'central21', '--ch-select', 'rfe', '--ch-scope', 'subject_fold',
            ], check=True, capture_output=True, text=True)
            subprocess.run([
                sys.executable, 'run_adaptive_fusion.py',
                '--cache-dir', str(root), '--out-dir', str(out_b),
                '--subjects', 'YDL', '--starts', '0.0', '0.5', '--durations', '0.5', '1.5',
                '--mi-pool', 'central21', '--ch-select', 'rfe', '--ch-scope', 'subject_fold',
            ], check=True, capture_output=True, text=True)

            diag_a = _rows(out_a / 'channel_selection_diagnostics.csv')
            diag_b = _rows(out_b / 'channel_selection_diagnostics.csv')
            self.assertEqual(
                [(r['test_block'], r['selected_channels']) for r in diag_a],
                [(r['test_block'], r['selected_channels']) for r in diag_b],
            )
            acc_a = {
                (r['window_start_s'], r['method']): r['accuracy']
                for r in _rows(out_a / 'adaptive_fusion_metrics_2leads.csv')
                if float(r['duration_s']) == 0.5
            }
            acc_b = {
                (r['window_start_s'], r['method']): r['accuracy']
                for r in _rows(out_b / 'adaptive_fusion_metrics_2leads.csv')
                if float(r['duration_s']) == 0.5
            }
            self.assertEqual(acc_a, acc_b)

    def test_comparison_feature_fusion_uses_48dim_oof_slots(self):
        rng = np.random.default_rng(91)
        n_trials, n_bands, n_ch = 24, 7, 18
        a = rng.normal(size=(n_trials, n_bands, n_ch, n_ch))
        covs = np.einsum('nbik,nbjk->nbij', a, a) + 0.1 * np.eye(n_ch)
        raw_cca = rng.uniform(size=(n_trials, 4))
        raw_fbcca = rng.uniform(size=(n_trials, 4))
        y = np.tile(np.arange(4), n_trials // 4)
        blocks = np.repeat(np.arange(3), n_trials // 3)
        tr = np.where(blocks != 0)[0]
        te = np.where(blocks == 0)[0]

        fd = comparison._calibrate_comparison_fold(covs, raw_cca, raw_fbcca, y, blocks, tr, te)
        self.assertEqual(fd['mi_feats_tr'].shape, (len(tr), 48))
        self.assertEqual(fd['mi_feats_te'].shape, (len(te), 48))
        ff = comparison.FeatureFusionModel(k=8, C=0.1).fit(fd['mi_feats_tr'], fd['ss_raw_tr'], y[tr])
        p = ff.predict_proba(fd['mi_feats_te'], fd['ss_raw_te'])
        self.assertEqual(p.shape, (len(te), 4))
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
