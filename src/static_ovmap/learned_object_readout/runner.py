"""Public fixed-stage controller; scientific training runs in the bound FC env."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

from .binding import bind
from .common import PHASES, REPO, verified, write


def _data_initializer():
    cpus = sorted(os.sched_getaffinity(0))[:4]
    slot = (multiprocessing.current_process()._identity[0]-1) % len(cpus)
    os.sched_setaffinity(0, {cpus[slot]})


def _data_job(args):
    from .data import prepare_family
    return prepare_family(*args)


def generate_data(binding, *, holdout=False):
    from .data import prepare_dataset
    from .sensor import SensorReader
    root = Path(binding['output_root']); split = verified(root/'split_manifest.json'); started = time.perf_counter()
    roles = ('holdout',) if holdout else ('train','dev')
    if not holdout:
        calibration = {r['scene']:SensorReader(r['files']['.sens']).calibration() for r in split['selected']}
        write(root/'data/calibration_inventory.json', dict(status='COMPLETE',scenes=calibration,holdout_annotation_reads=0))
    role_by_scene = {s:role for role in roles for s in split['roles'][role]}
    queue = [(binding,r,role_by_scene[r['scene']]) for r in split['selected'] if r['scene'] in role_by_scene]
    with ProcessPoolExecutor(max_workers=4, initializer=_data_initializer) as pool:
        for result in pool.map(_data_job, queue):
            print('DATA_READY',result['scene'],result['base_objects'],flush=True)
    prepare_dataset(binding, holdout=holdout)
    result = verified(root/('holdout_data_profile.json' if holdout else 'data_profile.json'))
    timing = root/'data'/('holdout_generation_runtime.json' if holdout else 'generation_runtime.json')
    if not timing.exists():
        write(timing,dict(status='COMPLETE',profile=result['identity'],roles=roles,CPU_workers=4,
                          elapsed_seconds=time.perf_counter()-started))
    return result


def fc_worker(binding, action):
    from static_ovmap.backbone_wave1.runtime import execute
    root = Path(binding['output_root'])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=binding['gpu'], PYTHONPATH=str(REPO/'src'),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    argv = [binding['FC_python'], '-u', '-m', 'static_ovmap.learned_object_readout.worker',
            '--output-root', str(root), '--action', action]
    return execute(argv, REPO, root/'logs'/(action+'.log'), env=env)


def perform(binding, phase):
    root = Path(binding['output_root'])
    if phase == 'bind': return binding
    if phase == 'data':
        from .inventory import materialize
        materialize(binding)
        return generate_data(binding)
    if phase in ('features','engineer','train','nominate','repeat'):
        fc_worker(binding, phase)
        targets = dict(features='features/train-dev.json', engineer='engineering/receipt.json',
                       train='training/seed17/receipt.json', nominate='dev_nomination.json', repeat='repeat_nomination.json')
        return verified(root/targets[phase])
    if phase == 'holdout':
        generate_data(binding, holdout=True); fc_worker(binding,'holdout')
        return verified(root/'recognition/holdout.json')
    if phase == 'predict':
        from .prediction import prepare_regression, build_predictions
        from .output_checks import verify_zero_updates
        prepare_regression(binding); verify_zero_updates(binding); fc_worker(binding,'predict')
        return build_predictions(binding)
    if phase == 'evaluate':
        from .evaluation import evaluate
        return evaluate(binding)
    if phase == 'report':
        from .reporting import report
        return report(binding)
    raise ValueError('Unknown fixed phase')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True); parser.add_argument('--parent-root', required=True)
    parser.add_argument('--observation-root', required=True); parser.add_argument('--output-root', required=True)
    parser.add_argument('--storage-root'); parser.add_argument('--data-root', action='append', default=[])
    parser.add_argument('--path-map'); parser.add_argument('--gpu'); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--phase', choices=(*PHASES,'all'), default='all')
    args = parser.parse_args(argv)
    binding = bind(args.spec,args.parent_root,args.observation_root,args.output_root,storage_root=args.storage_root,
                   path_map=args.path_map,gpu=args.gpu,data_roots=args.data_root)
    root = Path(binding['output_root']); phases = PHASES if args.phase == 'all' else (args.phase,)
    started = time.perf_counter()
    actual_argv = sys.argv if argv is None else [parser.prog,*argv]
    for phase in phases:
        begin = time.perf_counter(); print('PHASE_START',phase,flush=True)
        try:
            result = perform(binding,phase)
            write(root/'phases'/(phase+'.json'),dict(status='COMPLETE', phase=phase, source_binding=binding['identity'],
                  result_identity=result['identity'], invocation_elapsed_seconds=time.perf_counter()-begin,
                  resume=args.resume, argv=actual_argv))
        except BaseException as exc:
            log = root/'logs'/(phase+'.log')
            detail = repr(exc)+(log.read_text(errors='replace')[-12000:] if log.exists() else '')
            status = next((s for s in ('BLOCKED_TRAINING_DATA','BLOCKED_CALIBRATION','BLOCKED_RUNTIME') if s in detail),None)
            if status is None:
                status = 'BLOCKED_TRAINING_DATA' if phase=='data' else 'BLOCKED_RUNTIME' if phase in ('features','engineer','train','repeat') else 'PARTIAL_RESULTS'
            write(root/'execution_failure.json',dict(status=status, phase=phase, error=repr(exc),
                  failed_leaf_log=str(log) if log.exists() else None,required_metrics=None,
                  missing_metric_reason='Mandatory phase did not complete; no baseline substituted for missing predictions',
                  exit_code=1, argv=actual_argv, elapsed_seconds=time.perf_counter()-started, deployment='N0_UNCHANGED'))
            raise
    status = 'SCIENCE_COMPLETE' if args.phase == 'all' else 'PHASE_COMPLETE'
    if args.phase == 'all':
        result = verified(root/'result_store.json')
        if result['status'] != 'SCIENCE_COMPLETE' or result['scene_method_coverage'] != 286 or result['full_cohort_pool_coverage'] != 22:
            raise RuntimeError('PARTIAL_RESULTS: full learned study coverage not verified')
    receipt = write(root/'execution_last.json',dict(status=status, exit_code=0, argv=actual_argv, phase=args.phase,
                    result_identity=result['identity'], elapsed_seconds=time.perf_counter()-started))
    print('TERMINAL',status,receipt['identity'],flush=True)
    return receipt
