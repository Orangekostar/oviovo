"""Original full MaskClustering stages with a task-owned Replica I/O boundary."""
import argparse
import ast
import contextlib
import fcntl
import hashlib
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import types

from .common import (REPO, ConsumedInputs, Resolver, StageFailure, digest_file,
                     read, source_identity, verified, write)
from .external import PINS

STAGES = ('frontend', 'cluster', 'features', 'text', 'query')
FILES = ('configs/scannet.json', 'dataset/scannet.py', 'main.py', 'mask_predict.py',
         'utils/config.py', 'utils/geometry.py', 'utils/mask_backprojection.py',
         'utils/post_process.py', 'graph/construction.py', 'graph/iterative_clustering.py',
         'graph/node.py', 'semantics/get_open-voc_features.py',
         'semantics/extract_label_featrues.py', 'semantics/open-voc_query.py')


def workspace(binding):
    return Path(binding['output_root']) / 'external/MaskClustering/workspace'


def _checkout(binding):
    assets = binding['asset_config'].get('external', {}).get('MaskClustering', {})
    repo = Path(assets.get('checkout') or Path(binding['output_root']) / 'assets/external/MaskClustering')
    if not (repo / '.git').exists():
        raise StageFailure('Pinned full MaskClustering source is missing', missing=[str(repo)])
    pin = PINS['MaskClustering'][1]
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if actual != pin:
        raise ValueError('Full MaskClustering checkout differs from its frozen commit')
    return repo, assets


def _source_records(repo):
    records = []
    pin = PINS['MaskClustering'][1]
    for name in FILES:
        path = repo / name
        expected = subprocess.check_output(['git', 'show', pin + ':' + name], cwd=repo)
        sha = digest_file(path)
        if sha != hashlib.sha256(expected).hexdigest():
            raise ValueError('Full MaskClustering author source changed: ' + name)
        records.append(dict(path=str(path), sha256=sha))
    return records


def locked_args(binding, scene):
    repo, assets = _checkout(binding)
    original = read(repo / 'configs/scannet.json')
    pinned = subprocess.check_output(['git', 'show', PINS['MaskClustering'][1] + ':configs/scannet.json'], cwd=repo)
    if digest_file(repo / 'configs/scannet.json') != hashlib.sha256(pinned).hexdigest():
        raise ValueError('Original MaskClustering numeric configuration changed')
    if assets.get('original_config') and not Path(assets['original_config']).is_file():
        raise StageFailure('Configured original MaskClustering config is missing',
                           missing=[assets['original_config']])
    supplied = read(assets['original_config']) if assets.get('original_config') else original
    path_fields = {'cropformer_path', 'dataset'}
    if ({k: v for k, v in supplied.items() if k not in path_fields}
            != {k: v for k, v in original.items() if k not in path_fields}):
        raise ValueError('Replica port cannot change original MaskClustering numeric hyperparameters')
    values = dict(original, dataset='replica48', config='replica48', seq_name=scene,
                  seq_name_list=scene, debug=False)
    return argparse.Namespace(**values)


def _frontend(binding):
    inherited = binding.get('inherited') or {}
    scenes = inherited.get('cohorts', {}).get('replica8', [])
    if not scenes:
        return {}
    resolver = Resolver(binding.get('path_map'))
    record = inherited.get('scenes', {}).get(scenes[0], {}).get('context')
    if not record:
        return {}
    return resolver.rewrite(verified(resolver.resolve(record['path'])))['runtime']


def dependencies(binding, scene):
    repo, assets = _checkout(binding)
    locked_args(binding, scene)
    root = Path(binding['output_root'])
    inherited = _frontend(binding)
    weights = root / 'assets/weights/MaskClustering'
    paths = dict(python=assets.get('python') or root / 'assets/envs/MaskClustering/bin/python',
        cropformer_python=assets.get('cropformer_python') or inherited.get('frontend_python')
            or root / 'assets/envs/MaskClustering-frontend/bin/python',
        cropformer_root=assets.get('cropformer_root') or inherited.get('cropformer_root')
            or repo / 'third_party/detectron2/projects/CropFormer',
        cropformer_checkpoint=assets.get('cropformer_checkpoint') or weights / 'Mask2Former_hornet_3x_576d0b.pth',
        clip_assets=assets.get('clip_assets') or weights / 'open_clip_pytorch_model.bin',
        original_config=assets.get('original_config') or repo / 'configs/scannet.json')
    paths = {k: Path(v) for k, v in paths.items()}
    if paths['clip_assets'].is_dir():
        paths['clip_assets'] /= 'open_clip_pytorch_model.bin'
    paths['frontend_config'] = paths['cropformer_root'] / 'configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml'
    paths['frontend_predictor'] = paths['cropformer_root'] / 'demo_cropformer/predictor.py'
    missing = [dict(asset=k, expected=str(p)) for k, p in paths.items()
               if not (p.is_dir() if k == 'cropformer_root' else p.is_file())]
    dataset = binding['asset_config'].get('datasets', {}).get('replica48', {})
    geometry = [root / 'assets/datasets/replica48/geometry' / (scene + '.ply'),
                root / 'assets/datasets/replica48/replica_3d' / (scene + '.pth')]
    if dataset.get('evaluation_mesh_root'):
        geometry.insert(0, Path(dataset['evaluation_mesh_root']) / (scene + '.ply'))
    provided = dataset.get('gt_root') or dataset.get('preprocessed_root')
    if provided:
        geometry += [Path(provided) / (scene + '.pth'), Path(provided) / 'replica_3d' / (scene + '.pth')]
    if not any(p.is_file() for p in geometry):
        missing.append(dict(asset='author_evaluation_point_domain', expected=list(map(str, geometry))))
    tuples = [root / 'assets/datasets/replica48/replica_3d' / (scene + '.pth')]
    if provided:
        tuples[:0] = [Path(provided) / (scene + '.pth'), Path(provided) / 'replica_3d' / (scene + '.pth')]
    if not any(p.is_file() for p in tuples):
        missing.append(dict(asset='author_instance_GT', expected=list(map(str, tuples))))
    return dict(status='BLOCKED_DEPENDENCY' if missing else 'DONE', missing=missing,
        assets={k: str(v) for k, v in paths.items()}, checkout=str(repo),
        method='MaskClustering (Replica dataset port)', frozen_commit=PINS['MaskClustering'][1],
        full_method_inference_completed=False)


@contextlib.contextmanager
def _author_context(repo, work, args, factory):
    previous_cwd, previous_path, previous_argv = Path.cwd(), list(sys.path), list(sys.argv)
    previous = {name: value for name, value in sys.modules.items()
                if name == 'utils' or name.startswith('utils.')}
    package = types.ModuleType('utils')
    package.__path__ = [str(repo / 'utils')]
    package.__package__ = 'utils'
    module = types.ModuleType('utils.config')
    module.get_args = lambda: args
    module.get_dataset = factory
    work.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(work)
        sys.path.insert(0, str(repo))
        # The author's namespace directory otherwise loses to src/utils/__init__.py.
        for name in previous:
            sys.modules.pop(name)
        sys.modules['utils'] = package
        sys.modules['utils.config'] = module
        yield
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
        sys.argv[:] = previous_argv
        for name in list(sys.modules):
            if name == 'utils' or name.startswith('utils.'):
                sys.modules.pop(name)
        sys.modules.update(previous)


@contextlib.contextmanager
def _local_clip(path):
    import open_clip
    original = open_clip.create_model_and_transforms
    def create(name, *args, **kwargs):
        if name != 'ViT-H-14' or kwargs.get('pretrained') != 'laion2b_s32b_b79k':
            raise ValueError('Full MaskClustering CLIP architecture/tag changed')
        kwargs['pretrained'] = str(path)
        return original(name, *args, **kwargs)
    open_clip.create_model_and_transforms = create
    try:
        yield
    finally:
        open_clip.create_model_and_transforms = original


def run_worker(binding, scene, stage):
    if stage not in STAGES:
        raise ValueError('Unknown full MaskClustering stage')
    repo, _ = _checkout(binding)
    _source_records(repo)
    args = locked_args(binding, scene)
    root = Path(binding['output_root'])
    scope = verified(root / 'protocol_lock.json')['scopes']['replica48']
    if stage != 'text' and scene not in scope['scenes']:
        raise ValueError('MaskClustering scene is outside the official frozen scope')
    work = workspace(binding)
    text = work / 'data/text_features/replica48.npy'
    def factory(request):
        if request.seq_name != scene:
            raise ValueError('Original stage requested an unfrozen scene')
        from .mask_clustering_io import make_dataset
        return make_dataset(binding, scene, text_features_path=text)
    with _author_context(repo, work, args, factory):
        if stage in ('cluster', 'query'):
            source = repo / ('main.py' if stage == 'cluster' else 'semantics/open-voc_query.py')
            runpy.run_path(str(source), run_name='__main__')
            return
        resolved = dependencies(binding, scene or scope['scenes'][0])['assets']
        if stage == 'frontend':
            from .replica_layout import prepare
            layout = prepare(binding, scene)
            project = Path(resolved['cropformer_root'])
            sys.path[:0] = [str(project), str(project / 'demo_cropformer')]
            sys.argv = [str(repo / 'mask_predict.py'), '--config-file', resolved['frontend_config'],
                '--root', str(Path(layout['layout_root']).parent), '--seq_name_list', scene,
                '--image_path_pattern', 'color/*0.jpg', '--dataset', 'replica48',
                '--opts', 'MODEL.WEIGHTS', resolved['cropformer_checkpoint']]
            runpy.run_path(str(repo / 'mask_predict.py'), run_name='__main__')
        else:
            with _local_clip(resolved['clip_assets']):
                if stage == 'features':
                    runpy.run_path(str(repo / 'semantics/get_open-voc_features.py'), run_name='__main__')
                else:
                    import numpy as np
                    import torch
                    import open_clip
                    from open_clip import tokenizer
                    source = repo / 'semantics/extract_label_featrues.py'
                    nodes = [n for n in ast.parse(source.read_text()).body
                             if isinstance(n, ast.FunctionDef) and n.name in ('load_clip', 'extract_text_feature')]
                    namespace = dict(np=np, torch=torch, open_clip=open_clip, tokenizer=tokenizer)
                    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
                    namespace['model'] = namespace['load_clip']()
                    text.parent.mkdir(parents=True, exist_ok=True)
                    namespace['extract_text_feature'](str(text), scope['text_names'])


def check_capacity(binding, scene):
    """The external method owns a separate cache and must reserve its own work."""
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
    path = root / 'resources/capacity/MaskClustering_replica48' / (scene + '.json')
    record = write(path, dict(status='DONE' if free >= required else 'BLOCKED_RESOURCE',
        method='MaskClustering', benchmark='replica48', scene=scene, free_bytes=free,
        required_free_bytes=required, minimal_reserve_bytes=reserve, next_scene_buffer_multiplier=2,
        estimate=estimate, producer=source_identity(('mask_clustering_full', 'capacity'))))
    if free < required:
        raise StageFailure('Full MaskClustering needs20GiB reserve plus twice the next-scene estimate',
                           status='BLOCKED_RESOURCE', missing=[str(path)])
    return record


def _run_scene(binding, scene):
    from .runtime import execute
    report = dependencies(binding, scene)
    if report['missing']:
        raise StageFailure('Full MaskClustering Replica dependencies are missing',
                           missing=report['missing'])
    from .replica_layout import prepare
    from .replica_io import prepare_geometry
    root = Path(binding['output_root'])
    previous = root / 'external/MaskClustering/pipeline/replica48' / scene / 'receipt.json'
    if not previous.is_file() or verified(previous)['status'] != 'DONE':
        check_capacity(binding, scene)
    layout = prepare(binding, scene)
    geometry = prepare_geometry(binding, scene)
    repo, _ = _checkout(binding)
    assets = report['assets']
    index = ConsumedInputs(root)
    sources = _source_records(repo)
    shared = [dict(sources=sources, protocol=verified(root / 'protocol_lock.json')['identity'],
                   producer=source_identity(('mask_clustering_full', 'mask_clustering_io'))),
              index.identity(assets['original_config']), index.identity(assets['clip_assets'])]
    local = [dict(layout=layout['identity']), index.identity(geometry)]
    work = workspace(binding)
    work.mkdir(parents=True, exist_ok=True)
    objects = Path(layout['layout_root']) / 'output/object/replica48'
    text = work / 'data/text_features/replica48.npy'
    outputs = dict(frontend=[Path(layout['layout_root']) / 'output/mask' / (str(i) + '.png')
                            for i in layout['selected_frame_ids']],
        cluster=[objects / 'object_dict.npy', work / 'data/prediction/replica48_class_agnostic' / (scene + '.npz')],
        features=[objects / 'open-vocabulary_features.npy'], text=[text],
        query=[work / 'data/prediction/replica48' / (scene + '.npz')])
    jobs = {}
    for stage in STAGES:
        python = assets['cropformer_python'] if stage == 'frontend' else assets['python']
        command = [python, '-m', 'static_ovmap.benchmark_completion_v2.mask_clustering_full',
                   '--output-root', str(root), '--worker-stage', stage]
        if stage != 'text':
            command += ['--scene', scene]
        env = dict(os.environ, PYTHONPATH=str(REPO / 'src') + ':' + str(REPO),
                   CUDA_VISIBLE_DEVICES=str(binding['gpu']) if stage != 'query' else '',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        env['PATH'] = str(Path(python).parent) + ':' + env.get('PATH', '')
        env['LD_LIBRARY_PATH'] = str(Path(python).parents[1] / 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
        inputs = list(shared)
        if stage != 'text':
            inputs += local + [dict(previous_stages={k: v['identity'] for k, v in jobs.items()})]
        if stage == 'frontend':
            inputs += [index.identity(assets[k]) for k in
                       ('cropformer_checkpoint', 'frontend_config', 'frontend_predictor')]
        if stage == 'query':
            inputs += [index.identity(text)]
        job = f'external/MaskClustering/replica48/{"shared" if stage == "text" else scene}/{stage}'
        jobs[stage] = execute(binding, job, command, cwd=work, env=env, inputs=inputs,
                              outputs=outputs[stage], gpu=stage != 'query')
    index.save()
    return write(root / 'external/MaskClustering/pipeline/replica48' / scene / 'receipt.json',
        dict(status='DONE', method='MaskClustering (Replica dataset port)', scene=scene,
             full_original_inference_stages=True, actual_jobs={k: v['identity'] for k, v in jobs.items()},
             prediction=index.identity(outputs['query'][0]), layout=layout['identity'],
             geometry=index.identity(geometry), source=sources, assets=assets,
             original_numeric_configuration=vars(locked_args(binding, scene)),
             GT_labels_used_for_prediction=False, formal_G3_verified=False,
             new_training_updates=0, common_official_scoring_pending=True))


def run_scene(binding, scene):
    folder = Path(binding['output_root']) / 'external/MaskClustering'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'driver.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise StageFailure('A full MaskClustering driver is already running',
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
        write(Path(args.output_root) / 'external/MaskClustering/pipeline/replica48' / args.scene / 'blocked.json',
              dict(status=exc.status, reason=str(exc), missing=exc.missing))
        print(exc.status, str(exc), exc.missing, flush=True)
        return 20
    return 0


if __name__ == '__main__':
    sys.exit(main())
