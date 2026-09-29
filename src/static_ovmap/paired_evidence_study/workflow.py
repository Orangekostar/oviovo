"""Scene-isolated CPU work with serial CAL and transfer barriers."""

import concurrent.futures
import multiprocessing
import os
import resource
import time
from pathlib import Path

from src.static_ovmap.composition_study.io import write_once

from .binding import read_json_or_gzip as read
from .evaluation import configurations, evaluate, pool_methods, predict
from .evidence import recover
from .selection import freeze


def _scene_job(binding, scene, configs, mode, threads):
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = str(threads)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=threads):
        start = time.monotonic()
        if mode == 'recover':
            result = recover(binding, scene)
        elif mode == 'predict':
            result = predict(binding, scene, configs)
        else:
            result = evaluate(binding, scene, configs)
        return {k: v for k, v in {**result, 'job_seconds': time.monotonic() - start,
                'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}.items() if k != 'inputs'}


def scenes_parallel(binding, scenes, configs, mode, threads, workers):
    output = Path(binding['output_root']) / 'execution'
    results = []
    # Spawn avoids inheriting evaluator globals, BLAS locks or annotation state.
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(scenes)),
            mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as executor:
        pending = {executor.submit(_scene_job, binding, s, configs, mode, threads): s for s in scenes}
        for future in concurrent.futures.as_completed(pending):
            scene = pending[future]
            try:
                result = future.result()
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                result = {'scene': scene, 'status': 'FAILED', 'error': f'{type(error).__name__}: {error}'}
            results.append(result)
            print({'phase': mode, 'scene': scene, 'status': result.get('status', 'LOCKED'),
                   'seconds': result.get('job_seconds')}, flush=True)
    stamp = time.time_ns()
    write_once(output / f'{mode}_{stamp}.json', {'phase': mode, 'workers': workers, 'threads_per_process': threads,
                                               'results': results, 'physical_model_loads': 0})
    failures = [r for r in results if r.get('status') == 'FAILED']
    if failures:
        raise RuntimeError(f'Independent scene jobs finished; failed leaves: {failures}')
    return results


def run(binding, phase, threads=4, workers=3):
    spec = read(binding['spec'])
    root = Path(binding['output_root'])
    cal, replica = spec['datasets']['calibration'], spec['datasets']['replica']
    if phase in ('recover', 'all'):
        scenes_parallel(binding, cal, [], 'recover', threads, workers)
    if phase in ('cal', 'all') and not (root / 'transfer_lock.json').exists():
        configs = configurations(spec)
        scenes_parallel(binding, cal, [], 'recover', threads, workers)
        scenes_parallel(binding, cal, configs, 'predict', threads, workers)
        scenes_parallel(binding, cal, configs, 'evaluate', threads, workers)
        pool_methods(binding, 'cal', [c['id'] for c in configs])
    if phase in ('freeze', 'all'):
        freeze(binding)
    if phase in ('replica', 'all'):
        lock = read(root / 'transfer_lock.json')
        configs = configurations(spec, lock['choices'])
        if lock['combination']['status'] == 'MEASURED':
            configs.append({k: lock['combination'][k] for k in ('id', 'method', 'parameter')})
        scenes_parallel(binding, replica, [], 'recover', threads, workers)
        scenes_parallel(binding, replica, configs, 'predict', threads, workers)
        scenes_parallel(binding, replica, configs, 'evaluate', threads, workers)
        pool_methods(binding, 'replica', [c['id'] for c in configs])
    if phase in ('diagnose', 'all'):
        from .diagnostics import diagnose
        diagnose(binding)
    if phase in ('report', 'all'):
        from .reporting import report
        report(binding)
    if phase in ('publish', 'all'):
        from .publication import publish
        publish(binding)
