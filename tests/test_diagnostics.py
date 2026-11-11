import tempfile
import unittest
import csv
from pathlib import Path
from diagnostics import Diagnostics


class DiagnosticTests(unittest.TestCase):
    def test_ties_and_correction_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Diagnostics(tmp)
            context = dict(subject='synthetic', window_start_s=0.5, duration_s=1, test_block=2)
            scores = {'power': .5, 'csp': .5, 'rcsp': .4, 'tangent': .3}
            # baseline correct at 0,2; fusion correct at 1,2: one rescue and one harm.
            ps = [[.8,.1,.05,.05],[.8,.1,.05,.05],[.1,.8,.05,.05]]
            pf = [[.1,.8,.05,.05],[.1,.8,.05,.05],[.1,.8,.05,.05]]
            d.fold(context, scores, {k: 1 for k in scores}, .1, 'power',
                   {k:[1,.2,0,0,0,0] for k in scores})
            d.predictions(context, [11,12,13], [0,1,1], ps, {'fusion':pf})
            d.save()
            rows = list(csv.DictReader((Path(tmp)/'fold_diagnostics.csv').read_text(encoding='utf-8-sig').splitlines()))
            self.assertEqual(len(rows), 4)
            self.assertEqual(float(rows[0]['top_gap']), 0)
            self.assertEqual(rows[0]['tied_best'], 'power|csp')
            blocks = list(csv.DictReader((Path(tmp)/'block_metrics.csv').read_text(encoding='utf-8-sig').splitlines()))
            self.assertEqual(int(blocks[0]['corrections']), 1)
            self.assertEqual(int(blocks[0]['damages']), 1)
            self.assertEqual(int(blocks[0]['net_gain_count']), 0)
            trials = list(csv.DictReader((Path(tmp)/'trial_predictions.csv').read_text(encoding='utf-8-sig').splitlines()))
            self.assertEqual([int(r['trial_id']) for r in trials], [11,12,13])

    def test_reject_misaligned_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Diagnostics(tmp)
            with self.assertRaises(ValueError):
                d.predictions({}, [1], [0], [[1,0,0,0]], {'bad': []})


if __name__ == '__main__':
    unittest.main()
