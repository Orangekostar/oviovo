#!/usr/bin/env python3
"""Run an exact-source native mapper with a post-completion query export hook."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.query_history import capture_history, instrument_mapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapper', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, native_args = parser.parse_known_args()
    if native_args[:1] != ['--']:
        parser.error('native mapper arguments must follow --')
    source = args.mapper.read_bytes()
    if hashlib.sha256(source).hexdigest() != args.expected_sha256:
        raise ValueError('native mapper source changed after command construction')
    if args.output.exists():
        raise FileExistsError(args.output)
    captured = []

    def capture(scope):
        if not (scope['select_by_vis'] or scope['select_combine']) or scope['select_by_viewcov']:
            raise ValueError('capture contract requires native top-area retention')
        start = time.perf_counter()
        receipt = capture_history(scope['inst_dict'], scope['inst_sem_dict'], args.output,
                                  max_top_vis=scope['max_top_vis'])
        receipt.update({'scene': scope['args'].scene_num,
                        'native_mapper_sha256': args.expected_sha256,
                        'native_cache_sha256': sha256_file(scope['inst_sem_f']),
                        'full_cache_sha256': sha256_file(args.output/'full_query_cache.pkl'),
                        'wrapper_sha256': sha256_file(__file__),
                        'capture_module_sha256': sha256_file(Path(__file__).resolve().parents[2]/'src/static_ovmap/query_history.py'),
                        'capture_seconds': time.perf_counter()-start,
                        'instrumentation': 'append one callback at normal end of main; original file unchanged'})
        (args.output/'history_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        captured.append(receipt)

    code = instrument_mapper(source.decode(), str(args.mapper))
    sys.argv = [str(args.mapper), *native_args[1:]]
    sys.path.insert(0, str(args.mapper.parent))
    exec(code, {'__name__': '__main__', '__file__': str(args.mapper), '_static_capture': capture})
    if len(captured) != 1:
        raise RuntimeError('native mapper did not produce exactly one complete history')


if __name__ == '__main__':
    main()
