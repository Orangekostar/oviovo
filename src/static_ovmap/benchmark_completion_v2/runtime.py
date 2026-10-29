"""Real child exit receipts; single inherited GPU lock and finite job identities."""
import contextlib
import fcntl
import os
from pathlib import Path
import subprocess
import time

from .common import verified, write, canonical_digest, ConsumedInputs, StageFailure


@contextlib.contextmanager
def gpu_lock(binding):
    path = binding['asset_config'].get('gpu_lock') or f"/mnt/shared/ww/ovimap-module-validation-v1/.visual-gpu-{binding['gpu']}.lock"
    with Path(path).open('a') as stream:
        print('WAIT_GPU_LOCK', path, flush=True)
        fcntl.flock(stream, fcntl.LOCK_EX)
        print('GPU_LOCK_ACQUIRED', flush=True)
        yield


def execute(binding, job_name, argv, *, cwd, env=None, inputs=None, outputs=(), gpu=False, child_gpu_lock=False):
    root = Path(binding['output_root'])
    folder = root / 'jobs' / job_name
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / 'receipt.json'
    argv = list(map(str, argv))
    selected_env = {key: value for key, value in (env or {}).items() if key.startswith('OVIMAP_')
                    or key in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'OMP_NUM_THREADS',
                               'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
    identity = canonical_digest(dict(argv=argv, cwd=str(cwd), env=selected_env, inputs=inputs or [], outputs=list(map(str, outputs))))
    prior = verified(receipt) if receipt.exists() else None
    index = ConsumedInputs(root)
    if prior:
        if prior['input_identity'] != identity:
            raise ValueError('Job changed; preserve previous receipt before affected repair: ' + job_name)
        if prior['status'] == 'DONE':
            for row in prior['outputs']:
                index.identity(row['path'], row)
            return prior
        if len(prior['attempts']) >= 2:
            raise StageFailure('Finite same-identity retry budget exhausted: ' + job_name,
                               status=prior['status'], missing=[str(receipt)])
    process_path = folder / 'process.json'
    if process_path.is_file():
        active = verified(process_path)
        proc = Path('/proc') / str(active['pid']) / 'cmdline'
        if active['input_identity'] == identity and proc.is_file():
            actual = proc.read_bytes().split(b'\0')
            if [item.decode() for item in actual if item] == active['argv']:
                raise StageFailure('The identical scene job is already running', status='BLOCKED_RESOURCE', missing=[str(process_path)])
    if gpu:
        check_gpu_budget(binding)
    attempts = (prior or {}).get('attempts', [])
    attempt = dict(number=len(attempts) + 1, started_unix=time.time())
    context = gpu_lock(binding) if gpu and not child_gpu_lock else contextlib.nullcontext()
    with context:
        start = time.monotonic()
        log = folder / f"attempt_{attempt['number']:02d}.log"
        with log.open('w') as stream:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
            write(folder / 'process.json', dict(status='RUNNING', input_identity=identity, pid=process.pid,
                argv=argv, cwd=str(cwd), started_unix=attempt['started_unix'], gpu_lock_held=gpu))
            returncode = process.wait()
        attempt.update(returncode=returncode, elapsed_seconds=time.monotonic() - start, log=str(log))
    if returncode:
        write(receipt, dict(status='FAILED_RUNTIME', input_identity=identity, attempts=attempts + [attempt],
            argv=argv, cwd=str(cwd), env=selected_env, inputs=inputs or [], outputs=[], actual_child_exit=returncode,
            gpu_worker_launches=int(gpu)))
        raise StageFailure('Real child failed: ' + job_name, status='FAILED_RUNTIME', missing=[str(log)])
    missing = [str(path) for path in outputs if not Path(path).is_file()]
    if missing:
        write(receipt, dict(status='FAILED_RUNTIME', input_identity=identity, attempts=attempts + [attempt],
            argv=argv, cwd=str(cwd), env=selected_env, inputs=inputs or [], outputs=[], actual_child_exit=0,
            gpu_worker_launches=int(gpu),
            error='Child exited zero but mandatory outputs are absent', missing=missing))
        raise StageFailure('Child omitted mandatory outputs: ' + job_name, status='FAILED_RUNTIME', missing=missing)
    records = [index.identity(path) for path in outputs]
    index.save()
    return write(receipt, dict(status='DONE', input_identity=identity, argv=argv, cwd=str(cwd), env=selected_env,
        inputs=inputs or [], outputs=records, attempts=attempts + [attempt], actual_child_exit=0,
        gpu_worker_launches=int(gpu), elapsed_seconds=attempt['elapsed_seconds']))


def gpu_seconds(root):
    root = Path(root)
    total = 0.
    for path in (root / 'jobs').glob('**/receipt.json'):
        row = verified(path)
        if row.get('gpu_worker_launches') or path.parent.name in ('native_query', 'cropformer', 'F_MV1000'):
            total += sum(attempt['elapsed_seconds'] for attempt in row['attempts'])
    # These real early jobs predate the common subprocess launcher. Include their
    # entire allocated worker wall time, conservatively including CPU overhead.
    for name in ('model_lock_exit.json', 'legacy_predict_exit.json'):
        path = root / 'bootstrap' / name
        if path.is_file():
            row = __import__('json').loads(path.read_text())
            total += float(row.get('seconds', row.get('elapsed_seconds', 0.)))
    return total


def check_gpu_budget(binding):
    path = Path(binding['output_root']) / 'runtime_constraints.json'
    constraint = verified(path) if path.exists() else {}
    maximum = constraint.get('max_gpu_hours')
    if maximum is not None and gpu_seconds(binding['output_root']) >= maximum * 3600:
        raise StageFailure('Actual explicit GPU quota reached at a checkpoint boundary',
                           status='BLOCKED_RESOURCE', missing=[str(path)])
