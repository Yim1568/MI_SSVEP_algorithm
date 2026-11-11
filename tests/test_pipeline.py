"""Synthetic end-to-end regression; no participant data required."""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import run_reduced_mi_leads as reduced


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


class PipelineTests(unittest.TestCase):
    def test_diagnostics_preserve_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(12)
            labels = reduced.ALL_MI_CHANNELS + ['O1','Oz','O2']
            y = np.tile(np.arange(4), 6)
            blocks = np.repeat(np.arange(3), 8)
            cache = root/'YDL_preprocessed.npz'
            np.savez(cache, ds_epochs=rng.normal(size=(24,1100,len(labels))),
                     y=y, blocks=blocks, ons=np.zeros(24,dtype=int),
                     trial_ids=np.arange(101,125), ch_labels=np.array(labels))
            without = reduced.process_subject_window('YDL',0,.5,cache)
            with_diagnostics = reduced.process_subject_window('YDL',0,.5,cache,root/'reduced')
            self.assertEqual(without, with_diagnostics)
            self.assertEqual(len(rows(root/'reduced/fold_diagnostics.csv')),8*3*4)
            for row in rows(root/'reduced/block_metrics.csv'):
                self.assertEqual(int(row['net_gain_count']),int(row['method_correct'])-int(row['fbcca_correct']))
            out = root/'adaptive'
            subprocess.run([sys.executable,'run_adaptive_fusion.py','--cache-dir',str(root),
                            '--out-dir',str(out),'--subjects','YDL','--starts','0',
                            '--durations','.5'],check=True, capture_output=True, text=True)
            self.assertEqual(len(rows(out/'fold_diagnostics.csv')),3*4)
            trials = rows(out/'trial_predictions.csv')
            summaries = rows(out/'adaptive_fusion_metrics_2leads.csv')
            for method in ['fusion_evidence_auto_adaptive','fusion_evidence_prior_best','fusion_evidence_fbcsp']:
                selected = [r for r in trials if r['method']==method]
                self.assertEqual(len(selected),24)
                observed = sum(r['correct']=='True' for r in selected)/24
                expected = float(next(r['accuracy'] for r in summaries if r['method']==method))
                self.assertEqual(observed, expected)


if __name__ == '__main__':
    unittest.main()
