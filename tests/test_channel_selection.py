"""Unit and synthetic end-to-end tests for CENTRAL_21 channel selection (RFE, GA, PSO)."""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from channel_selection import (
    CENTRAL_18,
    CENTRAL_21,
    channel_jaccard_stability,
    select_channels,
)
from runtime import arguments


def _rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def _make_synthetic_21ch_cache(root, subject='YDL', n_trials=24, length=600):
    rng = np.random.default_rng(77)
    labels = CENTRAL_21 + ['O1', 'Oz', 'O2']
    y = np.tile(np.arange(4), n_trials // 4)
    blocks = np.repeat(np.arange(3), n_trials // 3)
    cache = Path(root) / f'{subject}_preprocessed.npz'
    np.savez(
        cache,
        ds_epochs=rng.normal(size=(n_trials, length, len(labels))),
        y=y,
        blocks=blocks,
        ons=np.zeros(n_trials, dtype=float),
        trial_ids=np.array([f'rec:{i}' for i in range(1, n_trials + 1)]),
        ch_labels=np.array(labels),
    )
    return cache


class ChannelSelectionTests(unittest.TestCase):
    def test_central21_structure_and_zero_occipital_overlap(self):
        occ_9 = {'Pz', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2'}
        self.assertEqual(len(CENTRAL_21), 21)
        self.assertEqual(len(set(CENTRAL_21)), 21)
        self.assertTrue(set(CENTRAL_21).isdisjoint(occ_9))
        self.assertTrue({'FCz', 'Cz', 'CPz'}.issubset(CENTRAL_21))
        self.assertTrue(set(CENTRAL_18).issubset(CENTRAL_21))

    def test_rfe_ga_pso_recover_planted_signal_channels(self):
        rng = np.random.default_rng(123)
        n_trials, n_bands, n_ch = 72, 7, 21
        y = np.tile(np.arange(4), n_trials // 4)
        blocks = np.repeat(np.arange(3), n_trials // 3)
        planted = [1, 3, 5, 8, 10, 12, 15, 17]

        # Build non-saturated, band-specific class covariance matrices strictly on 8 planted channels
        covs = np.zeros((n_trials, n_bands, n_ch, n_ch), dtype=float)
        class_pairs = [(1, 3), (5, 8), (10, 12), (15, 17)]
        for i in range(n_trials):
            cls = int(y[i])
            a, b_idx = class_pairs[cls]
            shared_noise = rng.normal(0.0, 0.45, size=n_ch)
            for b in range(n_bands):
                diag = np.exp(shared_noise + rng.normal(0.0, 0.35, size=n_ch))
                if b == (cls + 1):
                    diag[a] *= 1.8
                    diag[b_idx] *= 1.8
                mat = np.diag(diag)
                if b == (cls + 1):
                    cov_val = 0.60 * np.sqrt(diag[a] * diag[b_idx])
                    mat[a, b_idx] = cov_val
                    mat[b_idx, a] = cov_val
                covs[i, b] = mat

        # Weak uninformative SSVEP prior so fusion_logloss relies cleanly on MI gains
        ps_tr = np.full((n_trials, 4), 0.25)

        rfe1 = select_channels(
            covs, y, blocks, CENTRAL_21, ps_tr=ps_tr,
            method='rfe', k_min=8, k_max=12, objective='fusion_logloss', seed=42,
        )
        rfe2 = select_channels(
            covs, y, blocks, CENTRAL_21, ps_tr=ps_tr,
            method='rfe', k_min=8, k_max=12, objective='fusion_logloss', seed=42,
        )
        self.assertEqual(rfe1['indices'], rfe2['indices'])

        for method, res, min_recall in [('rfe', rfe1, 6)] + [
            (m, select_channels(
                covs, y, blocks, CENTRAL_21, ps_tr=ps_tr,
                method=m, k_min=8, k_max=12, objective='mi_logloss', seed=42,
                ga_pop=24, ga_gen=18, pso_particles=24, pso_iter=18,
            ), 5)
            for m in ('ga', 'pso')
        ]:
            idx = res['indices']
            self.assertGreaterEqual(len(idx), 8, method)
            self.assertLessEqual(len(idx), 12, method)
            self.assertEqual(idx, sorted(set(idx)), method)
            self.assertLess(res['best_score'], res['full_score'], f'{method} failed to improve over full 21')
            recalled = len(set(idx) & set(planted))
            self.assertGreaterEqual(recalled, min_recall, f'{method} recalled {recalled}/8 from {idx}')

        jacc, freq = channel_jaccard_stability([['C3', 'Cz'], ['C3', 'CPz']])
        self.assertAlmostEqual(jacc, 1.0 / 3.0)
        self.assertEqual(freq['C3'], 2)

        none_res = select_channels(covs, y, blocks, CENTRAL_21, method='none')
        self.assertEqual(none_res['indices'], list(range(21)))

    def test_runtime_validates_ch_k_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / 'YDL_preprocessed.npz').touch()
            argv = [
                'run', '--cache-dir', tmp, '--out-dir', str(cache / 'out'),
                '--subjects', 'YDL', '--mi-pool', 'central18',
                '--ch-select', 'rfe', '--ch-k-min', '12', '--ch-k-max', '20',
            ]
            with patch('sys.argv', argv), self.assertRaises(SystemExit):
                arguments('adaptive')

    def test_end_to_end_adaptive_fusion_central21_rfe_ga_pso(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_synthetic_21ch_cache(root, subject='YDL', n_trials=24, length=600)

            for selector in ('rfe', 'ga', 'pso'):
                out_dir = root / f'out_{selector}'
                subprocess.run([
                    sys.executable, 'run_adaptive_fusion.py',
                    '--cache-dir', str(root),
                    '--out-dir', str(out_dir),
                    '--subjects', 'YDL',
                    '--starts', '0.5',
                    '--durations', '0.5',
                    '--mi-pool', 'central21',
                    '--ch-select', selector,
                    '--ch-k-min', '8',
                    '--ch-k-max', '12',
                    '--ch-objective', 'fusion_logloss',
                    '--ch-scope', 'subject_fold',
                ], check=True, capture_output=True, text=True)

                ch_diag = _rows(out_dir / 'channel_selection_diagnostics.csv')
                self.assertEqual(len(ch_diag), 3)
                for row in ch_diag:
                    self.assertEqual(int(row['n_pool']), 21)
                    self.assertGreaterEqual(int(row['n_selected']), 8)
                    self.assertLessEqual(int(row['n_selected']), 12)
                    self.assertEqual(row['ch_method'], selector)

                summaries = _rows(out_dir / 'adaptive_fusion_metrics_2leads.csv')
                trials = _rows(out_dir / 'trial_predictions.csv')
                for method in (
                    'ss_fbcca',
                    'fusion_evidence_fbcsp_raw',
                    'fusion_evidence_auto_raw',
                    'fusion_evidence_auto_adaptive',
                ):
                    expected = float(next(r['accuracy'] for r in summaries if r['method'] == method))
                    sel = [t for t in trials if t['method'] == method]
                    self.assertEqual(len(sel), 24)
                    observed = sum(t['correct'] == 'True' for t in sel) / 24
                    self.assertEqual(observed, expected)


if __name__ == '__main__':
    unittest.main()
