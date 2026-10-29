"""CPU-only recovery of actual absent-GT auxiliary recall failures in a live stream."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

from .common import (canonical_digest, read, source_identity, verified, write)


def recover(binding):
    from .score_compat import install
    install()
    from .standard_score import score_scene
    root = Path(binding['output_root'])
    protocol = verified(root / 'protocol_lock.json')
    outcomes = []
    for benchmark, scope in protocol['scopes'].items():
        for scene in scope['scenes']:
            scored = root / 'benchmark/evaluation' / benchmark / scene / 'receipt.json'
            frozen = root / 'benchmark/predictions' / benchmark / scene / 'receipt.json'
            gt = root / 'benchmark/GT' / benchmark / scene / 'receipt.json'
            failure = root / 'failures' / benchmark / scene / 'predict/traceback.txt'
            if scored.is_file() or not all(p.is_file() for p in (frozen, gt, failure)):
                continue
            traceback = failure.read_text()
            if 'UnboundLocalError' not in traceback or 'ar_current' not in traceback:
                continue
            identity = canonical_digest(dict(predictions=verified(frozen)['identity'],
                GT=verified(gt)['identity'], repair=source_identity(('score_compat',))))
            receipt = root / 'review/score_recovery' / benchmark / (scene + '.json')
            prior = verified(receipt) if receipt.is_file() else None
            if prior and prior['input_identity'] != identity:
                raise ValueError('Preserve the previous scoring recovery before changing its identity')
            attempts = (prior or {}).get('attempts', [])
            if len(attempts) >= 2:
                continue
            started = time.monotonic()
            try:
                result = score_scene(binding, benchmark, scene,
                    formal_connection=benchmark == 'scannet200_val' and scene == scope['scenes'][0])
                attempt = dict(status='DONE', number=len(attempts) + 1, score=result['identity'])
            except Exception as exc:
                attempt = dict(status='FAILED_RUNTIME', number=len(attempts) + 1,
                               error_type=type(exc).__name__, error=str(exc))
            attempt['elapsed_seconds'] = time.monotonic() - started
            outcome = write(receipt, dict(status=attempt['status'], input_identity=identity,
                benchmark=benchmark, scene=scene, attempts=attempts + [attempt],
                original_failure_preserved=str(failure), prediction_freeze=verified(frozen)['identity'],
                compatibility_proof='review/G3_evaluator_auxiliary_compat.json',
                numerical_producer_files_changed=False, new_GPU_calls=0, new_training_updates=0))
            outcomes.append(outcome)
            print('SCORE_RECOVERY', benchmark, scene, attempt['status'], flush=True)
    return outcomes


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args(argv)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    root = Path(args.output_root).resolve()
    folder = root / 'bootstrap/score_recovery'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'monitor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('SCORE_RECOVERY_ALREADY_MONITORED', flush=True)
            return 0
        write(folder / 'process.json', dict(status='WATCHING' if args.watch else 'ONE_PASS',
            pid=os.getpid(), CPU_workers=1, BLAS_threads=1, GPU_workers=0,
            producer=source_identity(('score_recovery', 'score_compat'))))
        while True:
            recover(verified(root / 'binding.json'))
            if not args.watch:
                break
            from .postprocess import alive
            marker = read(root / 'bootstrap/streaming_standard_process.json')
            if not alive(marker):
                break
            time.sleep(20)
        write(folder / 'exit.json', dict(status='DONE', actual_exit_code=0,
            only_recovery_monitor_complete=True, scientific_completion_not_claimed=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
