"""Unit and end-to-end tests for equal-duration cumulative and segmented latency-shifted windowing."""
import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from runtime import arguments
import run_reduced_mi_leads as reduced
from windowing import resolve_window_configs, sample_bounds


def _rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def _make_synthetic_cache(root, subject='YDL', n_trials=24, length=1100, onset=0.0):
    rng = np.random.default_rng(19)
    labels = reduced.ALL_MI_CHANNELS + ['O1', 'Oz', 'O2']
    y = np.tile(np.arange(4), n_trials // 4)
    blocks = np.repeat(np.arange(3), n_trials // 3)
    cache = Path(root) / f'{subject}_preprocessed.npz'
    np.savez(
        cache,
        ds_epochs=rng.normal(size=(n_trials, length, len(labels))),
        y=y,
        blocks=blocks,
        ons=np.full(n_trials, onset, dtype=float),
        trial_ids=np.arange(101, 101 + n_trials),
        ch_labels=np.array(labels),
    )
    return cache


class WindowingUnitTests(unittest.TestCase):
    def test_resolve_equal_cumulative_and_segmented(self):
        configs = resolve_window_configs(
            starts=[0.5, 0.0], durations=None, window_modes=('all_equal',),
        )
        self.assertEqual([c['mode'] for c in configs], ['equal_cumulative', 'equal_segmented'])

        cum = configs[0]
        self.assertEqual(cum['durations'], [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        for sp in cum['specs']:
            self.assertAlmostEqual(sp['ssvep_window'][0], 0.14)
            self.assertAlmostEqual(sp['mi_window'][0], 0.50)
            self.assertAlmostEqual(sp['ssvep_window'][1] - sp['ssvep_window'][0], sp['duration_s'])
            self.assertAlmostEqual(sp['mi_window'][1] - sp['mi_window'][0], sp['duration_s'])

        seg = configs[1]
        self.assertEqual(len(seg['specs']), 7)
        self.assertEqual(seg['specs'][0]['ssvep_window'], (0.14, 0.64))
        self.assertEqual(seg['specs'][0]['mi_window'], (0.50, 1.00))
        self.assertEqual(seg['specs'][1]['ssvep_window'], (0.64, 1.14))
        self.assertEqual(seg['specs'][1]['mi_window'], (1.00, 1.50))
        self.assertEqual(seg['specs'][-1]['ssvep_window'], (3.14, 3.64))
        self.assertEqual(seg['specs'][-1]['mi_window'], (3.50, 4.00))

    def test_resolve_decoupled_filters_invalid_deadlines(self):
        configs = resolve_window_configs(
            starts=[0.0, 0.14], durations=[0.5, 1.0], window_modes=('decoupled',),
        )
        self.assertEqual(configs[0]['durations'], [1.0])  # start=0.0, dur=0.5 gives T=0.50 (0s MI)
        self.assertEqual(configs[1]['durations'], [0.5, 1.0])

    def test_sample_bounds_contiguous_under_half_sample_onset(self):
        seg = resolve_window_configs([0.5], None, ('equal_segmented',))[0]
        ss_bounds = [
            sample_bounds(1245.5, sp['ssvep_window'][0], sp['duration_s'], fs=250, segment_index=sp['segment_index'])
            for sp in seg['specs']
        ]
        mi_bounds = [
            sample_bounds(1245.5, sp['mi_window'][0], sp['duration_s'], fs=250, segment_index=sp['segment_index'])
            for sp in seg['specs']
        ]
        for a, b in ss_bounds + mi_bounds:
            self.assertEqual(b - a, 125)
        for idx in range(len(ss_bounds) - 1):
            self.assertEqual(ss_bounds[idx][1], ss_bounds[idx + 1][0])
            self.assertEqual(mi_bounds[idx][1], mi_bounds[idx + 1][0])

    def test_runtime_accepts_new_window_modes_and_custom_subject(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / 'S01_preprocessed.npz').touch()
            argv = [
                'run', '--cache-dir', tmp, '--out-dir', str(cache / 'out'),
                '--subjects', 'S01', '--window-mode', 'all_equal', '--durations', '0.5', '1.0',
            ]
            with patch('sys.argv', argv):
                args = arguments('adaptive')
            self.assertEqual(args.subjects, ['S01'])
            self.assertEqual([w['mode'] for w in args.resolved_windows], ['equal_cumulative', 'equal_segmented'])

    def test_end_to_end_equal_schemes_and_step1_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_synthetic_cache(root, subject='YDL', n_trials=24, length=1100, onset=0.0)

            out_adapt = root / 'adapt_eq'
            subprocess.run([
                sys.executable, 'run_adaptive_fusion.py',
                '--cache-dir', str(root), '--out-dir', str(out_adapt),
                '--subjects', 'YDL', '--window-mode', 'all_equal', '--durations', '0.5', '1.0',
            ], check=True, capture_output=True, text=True)

            summaries = _rows(out_adapt / 'adaptive_fusion_metrics_2leads.csv')
            trials = _rows(out_adapt / 'trial_predictions.csv')

            # Expect cumulative (2 durs) + segmented_step (2 segs) + segmented_accum (2 segs) = 6 conditions * 15 methods
            modes_seen = {r['window_mode'] for r in summaries}
            self.assertEqual(modes_seen, {'equal_cumulative', 'equal_segmented_step', 'equal_segmented_accum'})

            # Step-1 invariant: at 0.5s, equal_cumulative == equal_segmented_step == equal_segmented_accum
            for method in ('ss_fbcca', 'fusion_evidence_fbcsp', 'fusion_evidence_auto_adaptive'):
                by_mode = {
                    m: float(next(
                        r['accuracy'] for r in summaries
                        if r['window_mode'] == m and float(r['duration_s']) == 0.5 and r['method'] == method
                    ))
                    for m in modes_seen
                }
                self.assertEqual(by_mode['equal_cumulative'], by_mode['equal_segmented_step'])
                self.assertEqual(by_mode['equal_segmented_step'], by_mode['equal_segmented_accum'])

                # Verify trial_predictions matches summary accuracy for every condition row
                for row in (r for r in summaries if r['method'] == method):
                    expected = float(row['accuracy'])
                    sel = [
                        t for t in trials
                        if t['window_mode'] == row['window_mode']
                        and t['segment_index'] == row['segment_index']
                        and t['duration_s'] == row['duration_s']
                        and t['method'] == method
                    ]
                    self.assertEqual(len(sel), 24)
                    observed = sum(t['correct'] == 'True' for t in sel) / 24
                    self.assertEqual(observed, expected)

            # Reduced leads end-to-end with equal schemes
            out_red = root / 'reduced_eq'
            subprocess.run([
                sys.executable, 'run_reduced_mi_leads.py',
                '--cache-dir', str(root), '--out-dir', str(out_red),
                '--subjects', 'YDL', '--window-mode', 'all_equal', '--durations', '0.5', '1.0',
                '--n-jobs', '1',
            ], check=True, capture_output=True, text=True)
            red_rows = _rows(out_red / 'reduced_mi_leads_metrics.csv')
            self.assertEqual(
                {r['window_mode'] for r in red_rows},
                {'equal_cumulative', 'equal_segmented_step', 'equal_segmented_accum'},
            )
            # Step-1 invariant for reduced leads
            for ch in ('2ch_C3_CP3', '18ch_central_raw'):
                step1 = [
                    float(next(
                        r['acc_auto_evidence_fusion'] for r in red_rows
                        if r['window_mode'] == m and float(r['duration_s']) == 0.5 and r['mi_channels'] == ch
                    ))
                    for m in ('equal_cumulative', 'equal_segmented_step', 'equal_segmented_accum')
                ]
                self.assertEqual(step1[0], step1[1])
                self.assertEqual(step1[1], step1[2])

            # Comparison script end-to-end with equal_segmented
            out_comp = root / 'comp_eq'
            subprocess.run([
                sys.executable, 'run_comparison.py',
                '--cache-dir', str(root), '--out-dir', str(out_comp),
                '--subjects', 'YDL', '--window-mode', 'equal_segmented', '--durations', '0.5', '1.0',
            ], check=True, capture_output=True, text=True)
            comp_rows = _rows(out_comp / 'all_metrics.csv')
            self.assertEqual(
                {r['window_mode'] for r in comp_rows},
                {'equal_segmented_step', 'equal_segmented_accum'},
            )
            self.assertTrue((out_comp / 'mean_summary_2leads_equal_segmented_accum.csv').is_file())

    def test_decoupled_analysis_script_equal_schemes(self):
        script_path = Path(r'E:\Program\My_AntiG\Project\MI_SSVEP_analysis\20261115_decoupled_alignment\run_decoupled_analysis.py')
        if not script_path.is_file():
            self.skipTest('Analysis script not present at expected workspace path')

        spec = importlib.util.spec_from_file_location('run_decoupled_analysis', script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(29)
            labels = mod.OCC_9 + mod.CENTRAL_18
            # deduplicate while preserving order (Pz etc. are disjoint from CENTRAL_18)
            labels = list(dict.fromkeys(labels))
            n_trials = 24
            y = np.tile(np.arange(4), n_trials // 4)
            blocks = np.repeat(np.arange(3), n_trials // 3)
            np.savez(
                root / 'S01_preprocessed.npz',
                ds_epochs=rng.normal(size=(n_trials, 1100, len(labels))),
                y=y,
                blocks=blocks,
                ons=np.zeros(n_trials, dtype=float),
                trial_ids=np.arange(1, n_trials + 1),
                ch_labels=np.array(labels),
            )

            cum_rows = mod.evaluate_window_condition(
                'S01', 'Equal_Cumulative', 'equal_cumulative', 0.5, 1.0, (0.14, 0.64), (0.50, 1.00),
                cache_dir=root,
            )
            seg_rows = mod.evaluate_segmented_condition(
                'S01', segment_step=0.5, ss_lat=0.14, mi_lat=0.50, max_cum_dur=1.0, cache_dir=root,
            )
            self.assertEqual(len(cum_rows), 3)
            self.assertEqual(len(seg_rows), 3 * 4)  # 3 tiers * 2 segments * 2 modes (step, accum)

            for tier in ('Tier1_Full_Leads', 'Tier2_O1O2_Full_MI', 'Tier3_O1O2_C3CP3'):
                r_cum = next(r for r in cum_rows if r['tier'] == tier)
                r_step = next(r for r in seg_rows if r['tier'] == tier and r['mode'] == 'Equal_Segmented_Step' and '[seg 1]' in r['mi_window'])
                r_acc = next(r for r in seg_rows if r['tier'] == tier and r['mode'] == 'Equal_Segmented_Accum' and '[seg 1..1]' in r['mi_window'])
                for col in ('FBCCA', 'FBCSP_Evidence', 'Auto_Evidence', 'Auto_Evidence_Gate'):
                    self.assertEqual(r_cum[col], r_step[col])
                    self.assertEqual(r_step[col], r_acc[col])


if __name__ == '__main__':
    unittest.main()
