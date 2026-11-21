"""Portable command-line configuration; each run gets a fresh output directory."""
import argparse
from datetime import datetime
from pathlib import Path
from diagnostics import write_manifest
from windowing import VALID_WINDOW_MODES, resolve_window_configs
from channel_selection import (
    MI_POOLS,
    VALID_CH_OBJECTIVES,
    VALID_CH_SCOPES,
    VALID_CH_SELECTORS,
)


def _valid_start(val):
    s = float(val)
    if not (0.0 <= s < 4.0):
        raise argparse.ArgumentTypeError('starts must be in [0, 4) seconds')
    return s


def _valid_latency(val):
    lat = float(val)
    if not (0.0 <= lat < 4.0):
        raise argparse.ArgumentTypeError('latency must be in [0, 4) seconds')
    return lat


def arguments(kind):
    parser = argparse.ArgumentParser(description=f'MI/SSVEP {kind} evaluation')
    parser.add_argument('--cache-dir', type=Path, required=True,
                        help='Directory containing SUBJECT_preprocessed.npz files')
    parser.add_argument('--out-dir', type=Path, default=None,
                        help='New output directory (must not already exist)')
    parser.add_argument('--subjects', nargs='+', default=['YDL', 'YZL', 'FHY'])
    parser.add_argument('--starts', nargs='+', type=_valid_start, default=[0.5, 0.0])
    parser.add_argument('--durations', nargs='+', type=float, default=None)
    parser.add_argument('--window-mode', nargs='+', choices=VALID_WINDOW_MODES, default=['sync'],
                        help='Window alignment scheme: sync, decoupled, equal_cumulative, equal_segmented, all_equal')
    parser.add_argument('--ssvep-latency', type=_valid_latency, default=0.14,
                        help='SSVEP physiological latency onset in seconds (default: 0.14)')
    parser.add_argument('--mi-latency', type=_valid_latency, default=0.50,
                        help='MI physiological latency onset in seconds (default: 0.50)')
    parser.add_argument('--segment-step', type=float, default=0.50,
                        help='Segment step in seconds for equal_segmented mode (default: 0.50)')
    parser.add_argument('--mi-pool', choices=tuple(MI_POOLS), default='central18',
                        help='MI electrode pool: central18 (legacy default) or central21 (3x7 sensorimotor)')
    parser.add_argument('--ch-select', choices=VALID_CH_SELECTORS, default='none',
                        help='Personalized MI channel selection method: none, rfe, ga, pso')
    parser.add_argument('--ch-k-min', type=int, default=8,
                        help='Minimum selected MI channels when --ch-select != none (default: 8)')
    parser.add_argument('--ch-k-max', type=int, default=12,
                        help='Maximum selected MI channels when --ch-select != none (default: 12)')
    parser.add_argument('--ch-objective', choices=VALID_CH_OBJECTIVES, default='fusion_logloss',
                        help='Channel selection optimization objective: fusion_logloss or mi_logloss')
    parser.add_argument('--ch-scope', choices=VALID_CH_SCOPES, default='subject_fold',
                        help='Channel selection scope: subject_fold (once per fold) or window_fold')
    parser.add_argument('--ch-seed', type=int, default=42,
                        help='Random seed for stochastic channel selectors GA/PSO (default: 42)')
    if kind == 'reduced':
        parser.add_argument('--n-jobs', type=int, default=6)
    args = parser.parse_args()
    args.subjects = list(dict.fromkeys(args.subjects))
    if args.durations and any(d <= 0 or d > 4 for d in args.durations):
        parser.error('durations must be in (0, 4] seconds')
    if args.segment_step <= 0 or args.segment_step > 4:
        parser.error('segment-step must be in (0, 4] seconds')
    n_pool = len(MI_POOLS[args.mi_pool])
    if not (2 <= args.ch_k_min <= args.ch_k_max <= n_pool):
        parser.error(f'ch-k-min/max must satisfy 2 <= k_min <= k_max <= {n_pool} for mi-pool={args.mi_pool}')
    for sub in args.subjects:
        if not (args.cache_dir / f'{sub}_preprocessed.npz').is_file():
            parser.error(f'Missing cache for {sub}')
    try:
        args.resolved_windows = windows(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.out_dir = args.out_dir or Path('outputs') / f'{kind}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_manifest(args.out_dir, args)
    return args


def windows(args):
    return resolve_window_configs(
        starts=args.starts,
        durations=args.durations,
        window_modes=getattr(args, 'window_mode', ['sync']),
        ssvep_latency=getattr(args, 'ssvep_latency', 0.14),
        mi_latency=getattr(args, 'mi_latency', 0.50),
        segment_step=getattr(args, 'segment_step', 0.50),
    )
