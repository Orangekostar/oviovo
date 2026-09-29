#!/usr/bin/env python3
"""Run the bounded, frozen-evidence paired study without neural inference."""

import argparse
import fcntl
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['bind', 'recover', 'cal', 'freeze', 'replica', 'diagnose', 'report', 'publish', 'all'], default='all')
    parser.add_argument('--wave1-binding', default='/mnt/shared/ww/ovimap-a7-evidence-upgrade-wave1/attempt_001/binding.json')
    parser.add_argument('--output-root', default='/mnt/shared/ww/ovimap-paired-evidence-v1/attempt_001')
    parser.add_argument('--spec', default=str(ROOT / 'docs/paper/static_ovmap/paired_evidence_v1/PROTOCOL_SPEC.json'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--path-map', help='Explicit old-root/new-root JSON; creates only absent root aliases, never overwrites')
    args = parser.parse_args()
    if args.threads <= 0 or args.workers <= 0:
        parser.error('threads and workers must be positive')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.path_map:
        from src.static_ovmap.paired_evidence_study.path_map import apply_path_map
        print({'path_map': apply_path_map(args.path_map)}, flush=True)
    from src.static_ovmap.paired_evidence_study.binding import bind
    from src.static_ovmap.paired_evidence_study.workflow import run
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.coordinator.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = bind(args.spec, args.wave1_binding, root)
        run(binding, args.phase, args.threads, args.workers)
    print({'phase': args.phase, 'status': 'FINISHED'}, flush=True)


if __name__ == '__main__':
    main()
