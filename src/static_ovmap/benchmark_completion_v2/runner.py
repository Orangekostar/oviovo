"""Independent frozen phase DAG; scientific imports occur only after terminal gate."""
import argparse
from pathlib import Path
import sys
import time

from .common import (REPO, read, verified, write, StageFailure, canonical_digest, digest_file)
from .lifecycle import terminal_resume

PHASES = ('bind', 'protocol', 'cache_post', 'method_lock', 'legacy_predict', 'legacy_score',
          'benchmark_inputs', 'baselines', 'benchmark_predict', 'benchmark_score',
          'statistics', 'tables', 'review', 'publish')


def parser():
    p = argparse.ArgumentParser(description='Frozen OVMap benchmark completion v2')
    p.add_argument('--semantic-root')
    p.add_argument('--structure-root')
    p.add_argument('--output-root', required=True)
    p.add_argument('--path-map')
    p.add_argument('--asset-config')
    p.add_argument('--spec', default=str(REPO / 'configs/static_ovmap/benchmark_completion_v2.json'))
    p.add_argument('--phase', choices=('all', *PHASES), default='all')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--gpu')
    p.add_argument('--max-gpu-hours', type=float)
    p.add_argument('--refresh-assets', action='store_true')
    return p


def _model_lock(binding):
    root = Path(binding['output_root'])
    if (root / 'model_lock.json').is_file():
        lock = verified(root / 'model_lock.json')
        from .common import source_identity
        if lock['producer'] != source_identity(('registry', 'region_inputs')):
            raise ValueError('Actual frozen model producer changed')
        # Actual state and parity already verified by the immutable receipt;
        # actual downstream head loading checks the recorded transport again.
        return lock
    import os
    from .runtime import gpu_lock
    from .registry import lock, configure_torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(binding['gpu'])
    configure_torch()
    with gpu_lock(binding):
        result, _ = lock(binding)
    return result


def _scene_matrix(binding, *, inputs_only=False, score=False):
    root = Path(binding['output_root'])
    protocol = verified(root / 'protocol_lock.json')
    completed, blocked = [], []
    for benchmark, scope in protocol['scopes'].items():
        if scope['status'] != 'DONE':
            blocked.append(dict(benchmark=benchmark, status=scope['status'], missing=scope.get('missing', [])))
            continue
        for i, scene in enumerate(scope['scenes']):
            try:
                if benchmark == 'replica48' and not score:
                    from .replica_io import prepare_geometry
                    prepare_geometry(binding, scene)
                if not score:
                    from .capacity import check_scene
                    check_scene(binding,benchmark,scene)
                if score:
                    prediction = root / 'benchmark/predictions' / benchmark / scene / 'receipt.json'
                    try:
                        prediction.lstat()
                    except FileNotFoundError:
                        raise StageFailure('Scene prediction must be frozen before scoring', missing=[dict(
                            path=str(prediction), command='--phase benchmark_predict --resume')]) from None
                    from .semantic_refresh import prepare_standard_score
                    prepare_standard_score(binding, benchmark, scene)
                    if benchmark == 'replica48':
                        from .replica_io import prepare_ground_truth
                        prepare_ground_truth(binding, scene)
                    from .standard_score import score_scene
                    result = score_scene(binding, benchmark, scene, formal_connection=i == 0 and benchmark == 'scannet200_val')
                elif inputs_only:
                    from .native_inputs import build
                    result = build(binding, benchmark, scene)
                else:
                    from .benchmark import predict_scene
                    from .semantic_refresh import resume_standard_scene
                    result = resume_standard_scene(binding, benchmark, scene)
                    if result is None:
                        result = predict_scene(binding, benchmark, scene)
                    from .semantic_refresh import prepare_standard_score
                    prepare_standard_score(binding, benchmark, scene)
                    # Freeze every available variant first, then score the same
                    # scene before streaming the next one. An interrupted full
                    # run thus retains scored formal scenes, not just inputs.
                    if benchmark == 'replica48':
                        from .replica_io import prepare_ground_truth
                        prepare_ground_truth(binding, scene)
                    from .standard_score import score_scene
                    score_scene(binding, benchmark, scene,
                                formal_connection=i == 0 and benchmark == 'scannet200_val')
                if not inputs_only and benchmark=='replica48' and i==0:
                    from .replica_validation import verify_scene
                    verify_scene(binding,scene)
                if not inputs_only:
                    from .capacity import observe_scene
                    observe_scene(binding,benchmark,scene)
                row = dict(benchmark=benchmark, scene=scene, identity=result['identity'], status=result['status'])
                if result['status'] == 'DONE':
                    completed.append(row)
                else:
                    blocked.append(dict(row, missing=result.get('missing_variants', []),
                                        reason=result.get('missing_reason')))
            except StageFailure as exc:
                blocked.append(dict(benchmark=benchmark, scene=scene, status=exc.status, reason=str(exc), missing=exc.missing))
            except Exception as exc:
                import traceback
                failure = root / 'failures' / benchmark / scene / ('score' if score else 'inputs' if inputs_only else 'predict')
                failure.mkdir(parents=True, exist_ok=True)
                (failure / 'traceback.txt').write_text(traceback.format_exc())
                blocked.append(dict(benchmark=benchmark, scene=scene, status='FAILED_RUNTIME',
                                    reason=type(exc).__name__ + ': ' + str(exc), traceback=str(failure / 'traceback.txt')))
            write(root / 'progress' / ('benchmark_score.json' if score else 'benchmark_inputs.json' if inputs_only else 'benchmark_predict.json'),
                dict(status='RUNNING', completed=completed, blocked=blocked,
                    expected_scenes=sum(len(s.get('scenes', [])) for s in protocol['scopes'].values())))
    return dict(status='DONE' if not blocked else 'FAILED_RUNTIME' if any(r['status'] == 'FAILED_RUNTIME' for r in blocked)
                else 'BLOCKED_RESOURCE' if any(r['status'] == 'BLOCKED_RESOURCE' for r in blocked) else 'BLOCKED_DEPENDENCY',
                completed=completed, blocked=blocked)


def dispatch(phase, binding, args):
    root = Path(binding['output_root'])
    if phase in ('protocol', 'legacy_predict', 'legacy_score', 'benchmark_inputs',
                 'benchmark_predict', 'benchmark_score') and (root / 'protocol_lock.json').is_file():
        from .semantic_refresh import check_alias
        check_alias(binding, verified(root / 'protocol_lock.json'))
    if phase in ('benchmark_predict', 'benchmark_score', 'baselines', 'statistics', 'tables'):
        from .score_compat import install
        install()
    if phase == 'bind':
        return binding
    if phase == 'protocol':
        if (root / 'protocol_lock.json').is_file():
            lock = verified(root / 'protocol_lock.json')
            from .common import source_identity, ConsumedInputs
            if lock['producer'] != source_identity(('protocol',)):
                raise ValueError('Protocol producer changed; an explicit affected repair is required')
            index = ConsumedInputs(root)
            for descriptor in lock['official_sources'].values():
                index.identity(descriptor['path'], descriptor)
            return lock
        from .protocol import build
        return build(binding)
    if phase == 'cache_post':
        from .calibration import run
        return run(binding)
    if phase == 'method_lock':
        return _model_lock(binding)
    if phase == 'legacy_predict':
        from .semantic_refresh import prepare_legacy_prediction
        prepare_legacy_prediction(binding)
        if (root / 'legacy/prediction_lock.json').is_file():
            return verified(root / 'legacy/prediction_lock.json')
        from .legacy import predict
        from .registry import configure_torch, load_models
        from .runtime import gpu_lock, check_gpu_budget
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(binding['gpu'])
        configure_torch()
        check_gpu_budget(binding)
        with gpu_lock(binding):
            models, _ = load_models(binding)
            protocol = verified(root / 'protocol_lock.json')
            return predict(binding, models, replica_base_mask=protocol['legacy']['replica51']['base_mask'])
    if phase == 'legacy_score':
        from .semantic_refresh import prepare_legacy_score
        prepare_legacy_score(binding)
        from .scoring import legacy_score
        return legacy_score(binding)
    if phase == 'benchmark_inputs':
        # Establish each first formal input connection before the full streaming
        # prediction matrix. Subsequent scenes build the same bundle on demand;
        # never precompute312 maps before testing the first real source pipeline.
        from .native_inputs import build
        protocol = verified(root / 'protocol_lock.json')
        rows, blocked = {}, []
        for benchmark, scope in protocol['scopes'].items():
            try:
                rows[benchmark] = build(binding, benchmark, scope['scenes'][0])
            except StageFailure as exc:
                blocked.append(dict(benchmark=benchmark, reason=str(exc), status=exc.status, missing=exc.missing))
        return dict(status='BLOCKED_DEPENDENCY' if blocked else 'DONE', first_formal_bundles=rows, blocked=blocked,
            coverage='FIRST_FORMAL_CONNECTION;REMAINING_BUNDLES_BUILT_IN_STREAMING_PREDICTION_PHASE')
    if phase in ('benchmark_predict', 'benchmark_score'):
        return _scene_matrix(binding, score=phase == 'benchmark_score')
    if phase == 'baselines':
        from .baselines import run
        return run(binding)
    if phase == 'statistics':
        from .statistics import paired_bootstrap
        rows, missing = {}, []
        for benchmark in ('scannet200_val', 'replica48'):
            try:
                rows[benchmark] = paired_bootstrap(binding, benchmark)
            except StageFailure as exc:
                missing.append(dict(benchmark=benchmark, status=exc.status, reason=str(exc), missing=exc.missing))
        return dict(status='BLOCKED_DEPENDENCY' if missing else 'DONE', benchmarks=rows, missing=missing)
    if phase == 'tables':
        from .tables import build
        return build(binding)
    if phase == 'review':
        from .review import run
        return run(binding)
    if phase == 'publish':
        from .publication import publish
        return publish(binding, args)
    raise ValueError('Unknown phase')


def main(argv=None):
    args = parser().parse_args(argv)
    root = Path(args.output_root).resolve()
    if (root / 'asset_refresh_pending.json').is_file() and not args.refresh_assets:
        raise ValueError('An interrupted asset refresh requires --refresh-assets with the same asset-config')
    prior_paths = read(root / 'task_paths.json') if (root / 'task_paths.json').is_file() else {}
    if args.resume and not args.asset_config:
        args.asset_config = prior_paths.get('asset_config')
    if args.resume and args.max_gpu_hours is None and (root / 'runtime_constraints.json').is_file():
        args.max_gpu_hours = verified(root / 'runtime_constraints.json').get('max_gpu_hours')
    if args.max_gpu_hours is not None and args.max_gpu_hours <= 0:
        raise ValueError('Actual explicit GPU quota must be positive')
    early = terminal_resume(args)
    if early is not None:
        return early
    import os
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    from .binding import bind
    if args.resume and (root / 'binding.json').is_file() and not args.refresh_assets:
        binding = verified(root / 'binding.json')
        if args.asset_config and canonical_digest(read(args.asset_config)) != binding['asset_identity']:
            raise ValueError('New asset configuration requires --refresh-assets; completed CPU calibration remains reusable')
        if args.path_map and read(args.path_map) != binding['path_map']:
            raise ValueError('Changed path mapping requires --refresh-assets')
        for argument, key in ((args.semantic_root, 'semantic_root'), (args.structure_root, 'structure_root')):
            if argument and str(Path(argument).resolve()) != str(Path(binding[key]).resolve()):
                raise ValueError('Explicit resume root differs from the locked scientific input')
        if args.gpu is not None and str(args.gpu) != str(binding['gpu']):
            raise ValueError('Resume cannot move the inherited GPU')
    else:
        if args.refresh_assets and (root / 'binding.json').is_file():
            from .refresh import bind_assets
            binding = bind_assets(args)
        else:
            binding = bind(args)
    write(root / 'task_paths.json', dict(semantic_root=binding['semantic_root'], structure_root=binding['structure_root'],
        output_root=str(root), asset_config=str(Path(args.asset_config).resolve()) if args.asset_config else None,
        path_map=binding['path_map'], controller_python=sys.executable))
    write(root / 'runtime_constraints.json', dict(status='DONE', max_gpu_hours=args.max_gpu_hours,
        source='EXPLICIT_USER_CLI' if args.max_gpu_hours is not None else 'NO_USER_GPU_HOUR_QUOTA',
        GPU_workers=1, CPU_scorers=3, BLAS_threads=1, finite_scope=True))
    phases = PHASES if args.phase == 'all' else (args.phase,)
    if args.refresh_assets and args.phase == 'all':
        refresh = verified(root / 'asset_refresh.json')
        phases = tuple(p for p in phases if p in refresh['affected_phases'])
    failed = False
    for phase in phases:
        start = time.monotonic()
        print('PHASE', phase, flush=True)
        try:
            result = dispatch(phase, binding, args)
            status = result['status']
            row = dict(status=status, result_identity=result.get('identity'), result=result)
        except StageFailure as exc:
            status = exc.status
            row = dict(status=status, reason=str(exc), missing=exc.missing)
        except Exception as exc:
            import traceback
            status = 'FAILED_RUNTIME'
            folder = root / 'failures' / phase
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'traceback.txt').write_text(traceback.format_exc())
            row = dict(status=status, reason=type(exc).__name__ + ': ' + str(exc), traceback=str(folder / 'traceback.txt'))
        write(root / 'phases' / (phase + '.json'), dict(phase=phase, elapsed_seconds=time.monotonic() - start, **row))
        failed |= status not in ('DONE', 'NOT_APPLICABLE')
        print('PHASE_EXIT', phase, status, flush=True)
    if 'publish' in phases and (root / 'publication/final.json').is_file():
        from .publication import finish
        finish(binding, args)
    return 20 if failed else 0
