"""Pinned full OpenMask3D arbitrary-scene pipeline with Replica-only I/O."""
import argparse
import contextlib
import copy
import fcntl
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

from .common import (REPO, ConsumedInputs, StageFailure, source_identity,
                     verified, write)
from .openmask_io import checkout, geometry, ply_options, source

STAGES = ('masks', 'features', 'text', 'query')
CONFIG = 'openmask3d/configs/openmask3d_inference.yaml'


def paths(binding, scene):
    work = Path(binding['output_root']) / 'external/OpenMask3D/workspace'
    return dict(masks=work / scene / 'masks' / (scene + '_masks.pt'),
                features=work / scene / 'features' / (scene + '_openmask3d_features.npy'),
                text=work / 'shared/replica48_text.npy', query=work / scene / 'prediction.npz')


def _source_records(binding):
    repo, _ = checkout(binding)
    from .external import PINS
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only',
        PINS['OpenMask3D'][1], '--', 'openmask3d'], cwd=repo, text=True).splitlines()
    names = [name for name in names if Path(name).suffix in ('.py', '.yaml', '.yml')]
    names += ['run_openmask3d_single_scene.sh']
    from .common import digest_file
    return [dict(path=str(source(binding, name)), sha256=digest_file(repo / name)) for name in names]


def locked_configuration(binding):
    import yaml
    _, assets = checkout(binding)
    original = yaml.safe_load(source(binding, CONFIG).read_text())
    if assets.get('original_config'):
        supplied = Path(assets['original_config'])
        if not supplied.is_file():
            raise StageFailure('Configured OpenMask3D original config is missing', missing=[str(supplied)])
        value = yaml.safe_load(supplied.read_text())
        def without_paths(config):
            config = copy.deepcopy(config)
            config.pop('data', None)  # Camera and dataset paths are the recorded Replica port.
            config['external'].pop('sam_checkpoint', None)
            for name in ('experiment_name', 'output_directory'):
                config['output'].pop(name, None)
            return config
        if without_paths(value) != without_paths(original):
            raise ValueError('Replica port cannot change original OpenMask3D numeric hyperparameters')
    source(binding, 'run_openmask3d_single_scene.sh')
    return dict(proposal=dict(num_queries=120, use_dbscan=True, dbscan_eps=.95,
                             train_mode=False, test_mode='test'),
                features=original['openmask3d'], sam_model_type=original['external']['sam_model_type'],
                clip_model=original['external']['clip_model'], save_crops=original['output']['save_crops'],
                optimize_gpu_usage=original['gpu']['optimize_gpu_usage'],
                prompt='a {} in a scene', confidence='UNCHANGED_AUTHOR_CONSTANT1')


def dependencies(binding, scene):
    repo, assets = checkout(binding)
    locked_configuration(binding)
    root = Path(binding['output_root'])
    weights = root / 'assets/weights/OpenMask3D'
    resolved = dict(python=assets.get('python') or root / 'assets/envs/OpenMask3D/bin/python',
        mask_ckpt_replica=assets.get('mask_ckpt_replica') or weights / 'arbitrary_scene_mask3d.ckpt',
        sam_checkpoint=assets.get('sam_checkpoint') or weights / 'sam_vit_h_4b8939.pth',
        clip_assets=assets.get('clip_assets') or weights / 'ViT-L-14-336px.pt',
        original_config=assets.get('original_config') or repo / CONFIG)
    resolved = {key: Path(value) for key, value in resolved.items()}
    missing = [dict(asset=key, expected=str(path)) for key, path in resolved.items() if not path.is_file()]
    options = ply_options(binding, scene)
    if not any(path.is_file() for path in options):
        missing.append(dict(asset='author_colored_evaluation_PLY', expected=list(map(str, options))))
    from .replica_io import _tuple_path
    try:
        _tuple_path(binding, scene)  # Existence only; GT labels are not opened here.
    except StageFailure as exc:
        missing.append(dict(asset='author_instance_GT', expected=exc.missing))
    from .external import PINS
    return dict(status='BLOCKED_DEPENDENCY' if missing else 'DONE', missing=missing,
        assets={key: str(path) for key, path in resolved.items()}, checkout=str(repo),
        method='OpenMask3D (Replica dataset port)', frozen_commit=PINS['OpenMask3D'][1],
        full_method_inference_completed=False)


@contextlib.contextmanager
def _local_clip(path):
    import clip
    original = clip.load
    def load(name, *args, **kwargs):
        if name != 'ViT-L/14@336px':
            raise ValueError('Original OpenMask3D CLIP architecture changed')
        return original(str(path), *args, **kwargs)
    clip.load = load
    try:
        yield
    finally:
        clip.load = original


@contextlib.contextmanager
def _author_context(repo, argv):
    import collections
    import collections.abc
    import importlib.machinery
    import importlib.util
    old_cwd, old_path, old_argv = Path.cwd(), list(sys.path), list(sys.argv)
    sequence_missing = not hasattr(collections, 'Sequence')
    previous_datasets = {name: module for name, module in sys.modules.items()
                         if name == 'datasets' or name.startswith('datasets.')}
    try:
        # Unchanged MinkowskiEngine0.5.4 imports the pre-Python3.10 alias.
        if sequence_missing:
            collections.Sequence = collections.abc.Sequence
        # The author's namespace must precede the host's regular datasets package.
        for name in previous_datasets:
            sys.modules.pop(name)
        spec = importlib.machinery.ModuleSpec('datasets', loader=None, is_package=True)
        spec.submodule_search_locations = [str(repo / 'openmask3d/class_agnostic_mask_computation/datasets')]
        sys.modules['datasets'] = importlib.util.module_from_spec(spec)
        os.chdir(repo / 'openmask3d')
        sys.path[:0] = [str(repo / 'openmask3d/class_agnostic_mask_computation'), str(repo)]
        sys.argv = argv
        yield
    finally:
        for name in list(sys.modules):
            if name == 'datasets' or name.startswith('datasets.'):
                sys.modules.pop(name)
        sys.modules.update(previous_datasets)
        if sequence_missing:
            collections.__dict__.pop('Sequence', None)
        os.chdir(old_cwd)
        sys.path[:] = old_path
        sys.argv[:] = old_argv


def run_worker(binding, scene, stage):
    if stage not in STAGES:
        raise ValueError('Unknown full OpenMask3D stage')
    repo, _ = checkout(binding)
    _source_records(binding)
    locked_configuration(binding)
    scope = verified(Path(binding['output_root']) / 'protocol_lock.json')['scopes']['replica48']
    if stage != 'text' and scene not in scope['scenes']:
        raise ValueError('OpenMask3D scene is outside the frozen Replica scope')
    files = paths(binding, scene or scope['scenes'][0])
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    if stage == 'query':
        import numpy as np
        from .openmask_io import query
        prediction = query(binding, masks_path=files['masks'], features_path=files['features'],
                           text_path=files['text'])
        np.savez_compressed(files['query'], **prediction)
        return
    assets = dependencies(binding, scene or scope['scenes'][0])['assets']
    if stage == 'text':
        import clip
        import numpy as np
        from .openmask_io import text_embeddings
        with _local_clip(assets['clip_assets']):
            model, _ = clip.load('ViT-L/14@336px', device='cuda:0')
            np.save(files['text'], text_embeddings(binding, model, 'cuda:0'))
        return
    from .replica_layout import prepare
    layout = prepare(binding, scene)
    ply = geometry(binding, scene)['geometry']['path']
    if stage == 'masks':
        entry = source(binding, 'openmask3d/class_agnostic_mask_computation/get_masks_single_scene.py')
        arguments = ['general.experiment_name=replica48', 'general.checkpoint=' + assets['mask_ckpt_replica'],
            'general.train_mode=false', 'data.test_mode=test', 'model.num_queries=120',
            'general.use_dbscan=true', 'general.dbscan_eps=0.95', 'general.save_visualizations=false',
            'general.scene_path=' + ply, 'general.mask_save_dir=' + str(files['masks'].parent),
            'hydra.run.dir=' + str(files['masks'].parent / 'hydra_outputs')]
        with _author_context(repo, [str(entry), *arguments]):
            runpy.run_path(str(entry), run_name='__main__')
        return
    folder = Path(layout['layout_root'])
    entry = source(binding, 'openmask3d/compute_features_single_scene.py')
    arguments = ['data.masks.masks_path=' + str(files['masks']),
        'data.camera.poses_path=' + str(folder / 'pose'),
        'data.camera.intrinsic_path=' + str(folder / 'intrinsic/intrinsic_color.txt'),
        f'data.camera.intrinsic_resolution=[{layout["camera"]["h"]},{layout["camera"]["w"]}]',
        'data.depths.depths_path=' + str(folder / 'depth'),
        'data.depths.depth_scale=' + str(layout['camera']['scale']), 'data.depths.depths_ext=.png',
        'data.images.images_path=' + str(folder / 'color'), 'data.images.images_ext=.jpg',
        'data.point_cloud_path=' + ply, 'output.output_directory=' + str(files['features'].parent),
        'output.save_crops=false', 'external.sam_checkpoint=' + assets['sam_checkpoint'],
        'gpu.optimize_gpu_usage=false', 'hydra.run.dir=' + str(files['features'].parent / 'hydra_outputs')]
    with _author_context(repo, [str(entry), *arguments]), _local_clip(assets['clip_assets']):
        import torch
        import openmask3d.utils as utils
        original = utils.get_free_gpu
        # The original selector inspects all physical GPUs. The inherited lock
        # and CUDA_VISIBLE_DEVICES select the only device this worker may use.
        utils.get_free_gpu = lambda min_mem=7000: torch.device('cuda:0')
        try:
            runpy.run_path(str(entry), run_name='__main__')
        finally:
            utils.get_free_gpu = original


def check_capacity(binding, scene):
    from .capacity import estimate_scene
    from .replica_layout import _raw_root
    from .runtime import check_gpu_budget
    check_gpu_budget(binding)
    inherited = dict(binding.get('inherited') or {}, data_roots=[str(_raw_root(binding, scene))])
    estimate = estimate_scene(dict(binding, inherited=inherited), 'replica48', scene)
    root = Path(binding['output_root'])
    reserve = int(binding['specification']['runtime']['minimal_disk_reserve_GiB'] * 2**30)
    required = reserve + 2 * estimate['estimated_bytes']
    free = shutil.disk_usage(root).free
    path = root / 'resources/capacity/OpenMask3D_replica48' / (scene + '.json')
    record = write(path, dict(status='DONE' if free >= required else 'BLOCKED_RESOURCE',
        method='OpenMask3D', benchmark='replica48', scene=scene, free_bytes=free,
        required_free_bytes=required, minimal_reserve_bytes=reserve, next_scene_buffer_multiplier=2,
        estimate=estimate, producer=source_identity(('openmask_full', 'capacity'))))
    if free < required:
        raise StageFailure('Full OpenMask3D needs20GiB reserve plus twice the next-scene estimate',
                           status='BLOCKED_RESOURCE', missing=[str(path)])
    return record


def _run_scene(binding, scene):
    from .runtime import execute
    report = dependencies(binding, scene)
    if report['missing']:
        raise StageFailure('Full OpenMask3D Replica dependencies are missing', missing=report['missing'])
    root = Path(binding['output_root'])
    previous = root / 'external/OpenMask3D/pipeline/replica48' / scene / 'receipt.json'
    if not previous.is_file() or verified(previous)['status'] != 'DONE':
        check_capacity(binding, scene)
    from .replica_layout import prepare
    layout = prepare(binding, scene)
    original_ply = geometry(binding, scene)
    assets, files = report['assets'], paths(binding, scene)
    index = ConsumedInputs(root)
    sources = _source_records(binding)
    shared = [dict(sources=sources, protocol=verified(root / 'protocol_lock.json')['identity'],
                   producer=source_identity(('openmask_full', 'openmask_io'))),
              index.identity(assets['python']), index.identity(assets['clip_assets'])]
    local = [dict(layout=layout['identity']), original_ply['geometry']]
    jobs = {}
    for stage in STAGES:
        command = [assets['python'], '-m', 'static_ovmap.benchmark_completion_v2.openmask_full',
                   '--output-root', str(root), '--worker-stage', stage]
        if stage != 'text':
            command += ['--scene', scene]
        env = dict(os.environ, PYTHONPATH=str(REPO / 'src') + ':' + str(REPO),
                   CUDA_VISIBLE_DEVICES=str(binding['gpu']) if stage != 'query' else '',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        env['PATH'] = str(Path(assets['python']).parent) + ':' + env.get('PATH', '')
        env['LD_LIBRARY_PATH'] = str(Path(assets['python']).parents[1] / 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
        inputs = list(shared)
        if stage != 'text':
            inputs += local + [dict(previous_stages={key: value['identity'] for key, value in jobs.items()})]
        if stage == 'masks':
            inputs += [index.identity(assets['mask_ckpt_replica'])]
        elif stage == 'features':
            inputs += [index.identity(assets['sam_checkpoint']), index.identity(files['masks'])]
        elif stage == 'query':
            inputs += [index.identity(files[name]) for name in ('masks', 'features', 'text')]
        job = f'external/OpenMask3D/replica48/{"shared" if stage == "text" else scene}/{stage}'
        jobs[stage] = execute(binding, job, command, cwd=REPO, env=env, inputs=inputs,
                              outputs=[files[stage]], gpu=stage != 'query')
    index.save()
    return write(previous, dict(status='DONE', method='OpenMask3D (Replica dataset port)', scene=scene,
        full_original_inference_stages=True, actual_jobs={key: job['identity'] for key, job in jobs.items()},
        prediction=index.identity(files['query']), layout=layout['identity'], geometry=original_ply['geometry'],
        source=sources, assets=assets, original_numeric_configuration=locked_configuration(binding),
        GT_labels_used_for_prediction=False, formal_G3_verified=False, new_training_updates=0,
        common_official_scoring_pending=True))


def run_scene(binding, scene):
    folder = Path(binding['output_root']) / 'external/OpenMask3D'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'driver.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise StageFailure('A full OpenMask3D driver is already running',
                               status='BLOCKED_RESOURCE', missing=[str(folder / 'driver.lock')])
        return _run_scene(binding, scene)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--scene')
    parser.add_argument('--worker-stage', choices=STAGES)
    args = parser.parse_args(argv)
    binding = verified(Path(args.output_root) / 'binding.json')
    if args.worker_stage:
        run_worker(binding, args.scene, args.worker_stage)
        return 0
    if not args.scene:
        parser.error('--scene is required for a full scene pipeline')
    try:
        run_scene(binding, args.scene)
    except StageFailure as exc:
        write(Path(args.output_root) / 'external/OpenMask3D/pipeline/replica48' / args.scene / 'blocked.json',
              dict(status=exc.status, reason=str(exc), missing=exc.missing))
        print(exc.status, str(exc), exc.missing, flush=True)
        return 20
    return 0


if __name__ == '__main__':
    sys.exit(main())
