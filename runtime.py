"""Portable command-line configuration; each run gets a fresh output directory."""
import argparse
from datetime import datetime
from pathlib import Path
from diagnostics import write_manifest


def arguments(kind):
    parser = argparse.ArgumentParser(description=f'MI/SSVEP {kind} evaluation')
    parser.add_argument('--cache-dir', type=Path, required=True,
                        help='Directory containing SUBJECT_preprocessed.npz files')
    parser.add_argument('--out-dir', type=Path, default=None,
                        help='New output directory (must not already exist)')
    parser.add_argument('--subjects', nargs='+', choices=['YDL','YZL','FHY'], default=['YDL','YZL','FHY'])
    parser.add_argument('--starts', nargs='+', type=float, choices=[0.0,0.5], default=[0.5,0.0])
    parser.add_argument('--durations', nargs='+', type=float, default=None)
    if kind == 'reduced':
        parser.add_argument('--n-jobs', type=int, default=6)
    args = parser.parse_args()
    args.subjects = list(dict.fromkeys(args.subjects))
    if args.durations and any(d <= 0 or d > 4 for d in args.durations):
        parser.error('durations must be in (0, 4] seconds')
    for sub in args.subjects:
        if not (args.cache_dir / f'{sub}_preprocessed.npz').is_file():
            parser.error(f'Missing cache for {sub}')
    args.out_dir = args.out_dir or Path('outputs') / f'{kind}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    args.out_dir.mkdir(parents=True, exist_ok=False)
    args.resolved_windows = windows(args)
    write_manifest(args.out_dir, args)
    return args


def windows(args):
    configs = []
    for start in dict.fromkeys(args.starts):
        durations = args.durations if args.durations is not None else [i/2 for i in range(1,9)]
        durations = list(dict.fromkeys(d for d in durations if start+d <= 4))
        if not durations:
            raise ValueError(f'No durations fit the 4-second epoch at start={start}')
        configs.append(dict(name=f'start{int(start*10):02d}', start_s=start, durations=durations))
    return configs
