"""Actionable remaining work from actual quotas, file availability and retries."""
from pathlib import Path
import shutil

from .common import verified


def resource_block(binding):
    from .runtime import gpu_seconds
    root=Path(binding['output_root'])
    path=root/'runtime_constraints.json'
    maximum=verified(path).get('max_gpu_hours') if path.is_file() else None
    if maximum is not None and gpu_seconds(root)>=maximum*3600:
        return dict(status='BLOCKED_RESOURCE',runnable=False,reason='ACTUAL_GPU_QUOTA_EXHAUSTED',missing=[str(path)])
    reserve=int(binding['specification']['runtime']['minimal_disk_reserve_GiB']*2**30)
    estimates=[verified(p)['estimate']['estimated_bytes'] for p in (root/'resources/capacity').glob('*/*.json')]
    required=reserve+2*max(estimates,default=0)
    if shutil.disk_usage(root).free<required:
        return dict(status='BLOCKED_RESOURCE',runnable=False,reason='ACTUAL_SCENE_DISK_RESERVE_EXHAUSTED',
                    required_free_bytes=required)
    return None


def scene_state(binding, benchmark, scene, blocked_resource=None):
    from .capacity import visual_inputs_complete
    root=Path(binding['output_root'])
    if blocked_resource and not visual_inputs_complete(root,benchmark,scene):
        return dict(blocked_resource)
    if benchmark=='scannet200_val':
        inventory={}
        for path in (root/'protocol/scannet_asset_inventory.json',root/'assets/refreshed_scannet_inventory.json'):
            if path.is_file():
                inventory.update(verified(path)['scenes'].get(scene,{}).get('files',{}))
        suffixes=('.sens','_vh_clean_2.ply','.txt','.aggregation.json',
                  '_vh_clean_2.0.010000.segs.json','_vh_clean_2.labels.ply')
        for suffix in suffixes:
            supplied=inventory.get(suffix)
            if supplied and Path(supplied['path']).is_file():
                continue
            path=root/'assets/datasets/scannet/scans'/scene/(scene+suffix+'.download.json')
            if path.is_file():
                attempt=verified(path)
                if attempt['status']!='DONE' and len(attempt['attempts'])>=2:
                    return dict(status=attempt['status'],runnable=False,reason='OFFICIAL_ASSET_ACCESS_RETRIES_EXHAUSTED',
                                missing=[str(path)])
    for name in ('cropformer','native_map','F_MV1000','native_query'):
        path=root/'jobs/benchmark'/benchmark/scene/name/'receipt.json'
        if path.is_file():
            job=verified(path)
            if job['status']=='FAILED_RUNTIME' and len(job['attempts'])>=2:
                return dict(status='FAILED_RUNTIME',runnable=False,reason='SAME_IDENTITY_RUNTIME_RETRIES_EXHAUSTED',
                            missing=[str(path)])
    freeze=root/'benchmark/predictions'/benchmark/scene/'receipt.json'
    return dict(status='BLOCKED_RESOURCE',runnable=True,
        reason='QUEUED_CPU_SCORING_OF_FROZEN_PREDICTIONS' if freeze.is_file() else 'QUEUED_FINITE_SINGLE_WORKER_PIPELINE',
        command='--phase benchmark_score --resume' if freeze.is_file() else '--phase benchmark_predict --resume')


def _replica_method_state(binding, scene, method, dependencies, stages, blocked_resource):
    from .common import StageFailure
    root = Path(binding['output_root'])
    try:
        report = dependencies(binding, scene)
    except StageFailure as exc:
        return dict(status=exc.status, runnable=False, reason=str(exc), missing=exc.missing)
    except ValueError as exc:
        return dict(status='FAILED_RUNTIME', runnable=False, reason=str(exc), missing=[])
    if report['missing']:
        return dict(status='BLOCKED_DEPENDENCY', runnable=False,
                    reason='FULL_' + method.upper() + '_ASSETS_MISSING', missing=report['missing'])
    for stage in (*stages, 'score'):
        name = 'shared' if stage == 'text' else scene
        path = root / 'jobs/external' / method / 'replica48' / name / stage / 'receipt.json'
        if path.is_file():
            job = verified(path)
            if job['status'] != 'DONE' and len(job['attempts']) >= 2:
                return dict(status=job['status'], runnable=False,
                    reason='SAME_IDENTITY_FULL_METHOD_RETRIES_EXHAUSTED', missing=[str(path)])
    pipeline = root / 'external' / method / 'pipeline/replica48' / scene / 'receipt.json'
    if blocked_resource and not pipeline.is_file():
        return dict(blocked_resource)
    return dict(status='BLOCKED_RESOURCE', runnable=True,
        reason='QUEUED_FULL_' + method.upper() + ('_CPU_SCORE' if pipeline.is_file() else '_PIPELINE'),
        command='--phase baselines --resume')


def maskclustering_state(binding, scene, blocked_resource=None):
    from .mask_clustering_full import dependencies, STAGES
    return _replica_method_state(binding, scene, 'MaskClustering', dependencies, STAGES, blocked_resource)


def openmask_state(binding, scene, blocked_resource=None):
    from .openmask_full import dependencies, STAGES
    return _replica_method_state(binding, scene, 'OpenMask3D', dependencies, STAGES, blocked_resource)
