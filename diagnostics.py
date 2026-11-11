"""Read-only diagnostics: never feeds outer-test outcomes back into selection."""
import csv
import json
import platform
import hashlib
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path


def write_manifest(out, args):
    packages = {}
    for name in ['numpy', 'scipy', 'pandas', 'scikit-learn', 'pyriemann', 'joblib']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    (Path(out) / 'run_manifest.json').write_text(json.dumps({
        'arguments': vars(args), 'python': platform.python_version(),
        'packages': packages, 'schema_version': 1,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in Path(__file__).parent.glob('*.py')},
        'selection': 'inner-block-OOF MI accuracy; stable insertion-order ties',
    }, indent=2, default=str), encoding='utf-8')


class Diagnostics:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.folds, self.trials, self.blocks = [], [], []

    def fold(self, context, scores, temperatures, ss_temperature, selected, coefficients):
        ranked = sorted(scores, key=scores.get, reverse=True)
        best, second = ranked[:2]
        ties = '|'.join(k for k in scores if scores[k] == scores[best])
        for rank, name in enumerate(ranked, 1):
            coef = coefficients[name]
            self.folds.append(dict(context, candidate=name, inner_mi_accuracy=float(scores[name]),
                rank=rank, selected=name == selected, selected_model=selected,
                best_model=best, runner_up_model=second, best_score=float(scores[best]),
                runner_up_score=float(scores[second]), top_gap=float(scores[best]-scores[second]),
                tied_best=ties, mi_temperature=float(temperatures[name]),
                ss_temperature=float(ss_temperature), evidence_a=float(coef[0]),
                evidence_b=float(coef[1]), **{f'bias_{c}':float(coef[c+2]) for c in range(4)}))

    def predictions(self, context, trial_ids, targets, baseline, methods):
        n = len(targets)
        if not n or len(trial_ids) != n or len(baseline) != n:
            raise ValueError('Trial IDs, targets and baseline must align and be nonempty')
        if any(len(p) != n for p in methods.values()):
            raise ValueError('Prediction rows do not align with test trials')
        baseline_labels = [max(range(4), key=lambda c: row[c]) for row in baseline]
        for method, probabilities in methods.items():
            corrections = damages = ss_count = correct_count = 0
            for i, row in enumerate(probabilities):
                if len(row) != 4:
                    raise ValueError('Expected four class probabilities')
                pred = max(range(4), key=lambda c: row[c])
                target = int(targets[i])
                ss_ok, ok = baseline_labels[i] == target, pred == target
                rescue, harm = (not ss_ok and ok), (ss_ok and not ok)
                corrections += rescue
                damages += harm
                ss_count += ss_ok
                correct_count += ok
                self.trials.append(dict(context, method=method, trial_id=int(trial_ids[i]),
                    target=target, prediction=pred, fbcca_prediction=baseline_labels[i],
                    correct=ok, correction=rescue, damage=harm,
                    **{f'p{c}':float(row[c]) for c in range(4)},
                    **{f'fbcca_p{c}':float(baseline[i][c]) for c in range(4)}))
            self.blocks.append(dict(context, method=method, n_trials=n,
                fbcca_correct=ss_count, method_correct=correct_count,
                fbcca_accuracy=ss_count/n, accuracy=correct_count/n,
                corrections=corrections, damages=damages,
                net_gain_count=corrections-damages, gain_accuracy=(corrections-damages)/n))

    def save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        for filename, rows in [('fold_diagnostics.csv', self.folds),
                               ('trial_predictions.csv', self.trials),
                               ('block_metrics.csv', self.blocks)]:
            if rows:
                with (self.directory / filename).open('w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
