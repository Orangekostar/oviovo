"""New finite-pose input exports and true OVI native map capture; no G1 recovery."""
import os
from pathlib import Path
import sys
import time

from .common import (REPO, verified, write, read, ConsumedInputs, canonical_digest,
                     source_identity, StageFailure, Resolver)


def uniform_frames(finite_frame_ids, maximum=200):
    ids = list(map(int, finite_frame_ids))
    if maximum != 200 or ids != sorted(set(ids)):
        raise ValueError('Frozen rule requires sorted unique source IDs and maximum200')
    n, k = len(ids), min(200, len(ids))
    return [] if not k else [ids[0]] if k == 1 else [ids[j * (n - 1) // (k - 1)] for j in range(k)]


def _scannet_frames(binding, scene):
    import cv2
    import numpy as np
    import zlib
    from .acquisition import scannet_scene
    from static_ovmap.module_validation.scannet_frames import sensor_header, _matrix, _unpack, _read
    root = Path(binding['output_root'])
    acquisition = scannet_scene(binding, scene)
    sensor = Path(acquisition['files']['.sens']['path'])
    index = ConsumedInputs(root)
    descriptor = index.identity(sensor)
    folder = root / 'benchmark/frames/scannet200_val' / scene
    path = folder / 'frame_plan.json'
    identity = canonical_digest(dict(sensor=descriptor['sha256'], producer=source_identity(('scene_bundle',)), rule=binding['specification']['standard']['frame_rule']))
    if path.exists():
        prior = verified(path)
        if prior['input_identity'] != identity:
            raise ValueError('Completed frame export changed; preserve before affected repair')
        for row in prior['outputs']:
            index.identity(row['path'], row)
        return prior
    poses, offsets = {}, {}
    with sensor.open('rb') as stream:
        header = sensor_header(stream)
        for i in range(header['source_frame_count']):
            pose = _matrix(stream)
            _, _, color_size, depth_size = _unpack(stream, 'QQQQ')
            if stream.tell() + color_size + depth_size > sensor.stat().st_size:
                raise ValueError('Truncated official sensor frame')
            if np.isfinite(pose).all():
                poses[i] = pose.copy()
                offsets[i] = (stream.tell(), color_size, depth_size)
            stream.seek(color_size + depth_size, 1)
        frames = uniform_frames(sorted(poses))
        if not frames:
            raise StageFailure('Official scene contains no finite-pose source frame', missing=[str(sensor)])
        for name in ('color', 'depth', 'pose', 'intrinsic'):
            (folder / name).mkdir(parents=True, exist_ok=True)
        outputs = []
        for name, matrix in header['matrices'].items():
            target = folder / 'intrinsic' / (name + '.txt')
            np.savetxt(target, matrix, fmt='%.9g')
            outputs.append(index.identity(target))
        for i in frames:
            offset, color_size, depth_size = offsets[i]
            stream.seek(offset)
            color = _read(stream, color_size)
            rgb = cv2.imdecode(np.frombuffer(color, np.uint8), cv2.IMREAD_COLOR)
            if rgb is None or rgb.shape[:2] != (header['color_height'], header['color_width']):
                raise ValueError('Official original JPEG dimensions differ from sensor header')
            depth = np.frombuffer(zlib.decompress(_read(stream, depth_size)), dtype='<u2')
            depth = depth.reshape(header['depth_height'], header['depth_width'])
            color_path, depth_path, pose_path = folder / 'color' / f'{i}.jpg', folder / 'depth' / f'{i}.png', folder / 'pose' / f'{i}.txt'
            color_path.write_bytes(color)
            if not cv2.imwrite(str(depth_path), depth):
                raise OSError('Depth PNG export failed')
            np.savetxt(pose_path, poses[i], fmt='%.9g')
            outputs.extend(index.identity(p) for p in (color_path, depth_path, pose_path))
    result = write(path, dict(status='DONE', input_identity=identity, scene=scene, benchmark='scannet200_val',
        source=descriptor, frame_ids=frames, finite_pose_frames=len(poses), source_frame_count=header['source_frame_count'],
        source_frame_rule=binding['specification']['standard']['frame_rule'], outputs=outputs,
        native_data_folder=str(folder.parent), native_dataset='scannet_nyu',
        depth_scale=header['depth_shift'], color='ORIGINAL_JPEG_BYTES_UNCHANGED',
        pose_intrinsic='ORIGINAL_FP32_ROUNDTRIP', GT_labels_read=False))
    index.save()
    return result


def prepare_frames(binding, benchmark, scene):
    if benchmark == 'scannet200_val':
        return _scannet_frames(binding, scene)
    if benchmark != 'replica48':
        raise ValueError('Unknown standard benchmark')
    import numpy as np
    root = Path(binding['output_root'])
    roots = (binding['inherited'] or {}).get('data_roots', [])
    source = next((Path(p) / scene for p in roots if (Path(p) / scene / 'traj.txt').is_file()
                   and (Path(p) / 'cam_params.json').is_file()), None)
    if source is None:
        raise StageFailure('Official Replica RGB-D/pose source is absent', missing=[scene + '/traj.txt'])
    index = ConsumedInputs(root)
    inputs = [index.identity(source / 'traj.txt'), index.identity(source.parent / 'cam_params.json')]
    poses = np.loadtxt(source / 'traj.txt').reshape(-1, 4, 4)
    finite = [i for i, pose in enumerate(poses) if np.isfinite(pose).all()
              and (source / 'results' / f'frame{i:06d}.jpg').is_file()
              and (source / 'results' / f'depth{i:06d}.png').is_file()]
    frames = uniform_frames(finite)
    if not frames:
        raise StageFailure('No finite-pose Replica source RGB-D frames', missing=[str(source)])
    outputs = [index.identity(source / 'results' / f'{kind}{i:06d}.{suffix}')
               for i in frames for kind, suffix in (('frame', 'jpg'), ('depth', 'png'))]
    result = write(root / 'benchmark/frames/replica48' / scene / 'frame_plan.json', dict(status='DONE',
        input_identity=canonical_digest(dict(inputs=inputs, frame_ids=frames, producer=source_identity(('scene_bundle',)))),
        scene=scene, benchmark=benchmark, frame_ids=frames, finite_pose_frames=len(finite),
        source_frame_count=len(poses), source_frame_rule=binding['specification']['standard']['frame_rule'],
        inputs=inputs, outputs=outputs, native_data_folder=str(source.parent), native_dataset='replica',
        GT_labels_read=False))
    index.save()
    return result


def prepare_native_scene(binding, benchmark, scene):
    """Run original CropFormer, depth fusion and native association, exactly once."""
    from .runtime import execute
    from static_ovmap.module_validation.boundary_jobs import verify_capture
    root = Path(binding['output_root'])
    protocol = verified(root / 'protocol_lock.json')
    if scene not in protocol['scopes'][benchmark]['scenes']:
        raise ValueError('Native scene is outside locked official scope')
    frame_plan = prepare_frames(binding, benchmark, scene)
    index = ConsumedInputs(root)
    resolver = Resolver(binding['path_map'])
    cohort = 'replica8' if benchmark == 'replica48' else 'scannet_cf18'
    first = binding['inherited']['cohorts'][cohort][0]
    context = resolver.rewrite(verified(resolver.resolve(binding['inherited']['scenes'][first]['context']['path'])))
    original = context['actual_capture_command']
    native_entry = Path(original['command'][1])
    original_env = original['environment']
    runtime = context['runtime']
    for field in ('cropformer_config', 'cropformer_weights'):
        if not Path(runtime[field]).is_file():
            raise StageFailure('Actual original CropFormer asset is absent', missing=[runtime[field]])
    frame_ids = frame_plan['frame_ids']
    base = root / 'benchmark/native' / benchmark / scene
    frontend = base / 'frontend'
    frontend.mkdir(parents=True, exist_ok=True)
    data = Path(frame_plan['native_data_folder']) / scene
    images = [data / ('results' if benchmark == 'replica48' else 'color') /
              (f'frame{i:06d}.jpg' if benchmark == 'replica48' else f'{i}.jpg') for i in frame_ids]
    front_py, front_root = Path(runtime['frontend_python']), Path(runtime['cropformer_root'])
    common_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(binding['gpu']),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    front_env = dict(common_env, PATH=str(front_py.parent) + ':' + os.environ.get('PATH', ''),
        LD_LIBRARY_PATH=str(front_py.parents[1] / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''),
        PYTHONPATH=str(front_root))
    argv = [front_py, front_root / 'demo_cropformer/demo_from_dirs.py', '--config-file', runtime['cropformer_config'],
        '--input', *images, '--output', frontend, '--confidence-threshold', '0.5', '--out-type', '0',
        '--opts', 'MODEL.WEIGHTS', runtime['cropformer_weights']]
    frontend_outputs = [frontend / (f'frame{i:06d}.png' if benchmark == 'replica48' else f'{i}.png') for i in frame_ids]
    front = execute(binding, f'benchmark/{benchmark}/{scene}/cropformer', argv, cwd=front_root,
        env=front_env, inputs=[index.identity(runtime[k]) for k in ('cropformer_config', 'cropformer_weights')]
        + [dict(frame_plan=frame_plan['identity'])], outputs=frontend_outputs, gpu=True)
    # Native Replica loader looks up frameN.png? Preserve the actual author
    # naming produced from JPEG basenames; the generator's path policy is bound.
    command = list(original['command'])
    def option(name, value):
        if name in command:
            command[command.index(name) + 1] = str(value)
        else:
            command.extend([name, str(value)])
    command[:2] = [str(runtime['mapping_python']), '-m']
    command.insert(2, 'static_ovmap.benchmark_completion_v2.native_mapper')
    for name, value in {'--dataset':frame_plan['native_dataset'], '--scene_num':scene,
            '--data_folder':frame_plan['native_data_folder'], '--result_folder':base / 'mapper',
            '--start':frame_ids[0], '--end':frame_ids[-1] + 1, '--step':1, '--num_threads':3,
            '--temp_panoptics_folder':frontend, '--temp_geometrics_folder':base / 'geometrics',
            '--intermediate_seg_folder':base / 'segments'}.items():
        option(name, value)
    if '--use_temp_geometrics' in command:
        command.remove('--use_temp_geometrics')
    for flag in ('--save_temp_geometrics', '--skip_feature_extraction'):
        if flag not in command:
            command.append(flag)
    recipe = dict(id='NATIVE_BENCHMARK', family='control', depth_fusion='native', association='native',
        frontend='cropformer', diagnostic_root=str(base / 'diagnostics'), order_diagnostics=False,
        paired_frontend_root=str(base / 'unused_paired_frontend'))
    write(base / 'recipe.json', recipe)
    map_env = dict(common_env, **original_env)
    map_env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', OVMAP_BENCHMARK_FRAME_PLAN=str(Path(frame_plan['outputs'][0]['path']).parent.parent / 'frame_plan.json')
        if benchmark == 'scannet200_val' else str(root / 'benchmark/frames/replica48' / scene / 'frame_plan.json'),
        OVMAP_BENCHMARK_NATIVE_ENTRY=str(native_entry), OVMAP_BENCHMARK_NATIVE_ENTRY_SHA256=index.identity(native_entry)['sha256'],
        OVIMAP_MODULE_VALIDATION_CAPTURE_ROOT=str(base / 'capture'), OVIMAP_MODULE_VALIDATION_HELPER_ROOT=str(REPO),
        OVIMAP_BACKBONE_RECIPE=str(base / 'recipe.json'), OVIMAP_NATIVE_UPSTREAM=str(native_entry.parent.parent))
    map_env['PYTHONPATH'] = ':'.join([str(REPO / 'src'), str(REPO), original_env['PYTHONPATH']])
    capture_path = base / 'capture' / scene / 'manifest.json'
    deferred = base / 'diagnostics/native_deferred_metadata.json'
    mapped = execute(binding, f'benchmark/{benchmark}/{scene}/native_map', command, cwd=native_entry.parent.parent,
        env=map_env, inputs=[index.identity(native_entry), dict(frontend=front['identity'], frame_plan=frame_plan['identity']),
            index.identity(base / 'recipe.json'), index.identity(Path(__file__).with_name('native_mapper.py'))],
        outputs=[capture_path, deferred])
    capture = verify_capture(capture_path, allow_skipped=True)
    if capture['scheduled_frame_ids'] != frame_ids:
        raise ValueError('Native map changed the frozen uniform finite-pose frame plan')
    result = write(base / 'receipt.json', dict(status='DONE', benchmark=benchmark, scene=scene,
        frame_plan=frame_plan['identity'], frontend=front['identity'], native_map=mapped['identity'],
        capture_manifest=index.identity(capture_path), deferred_metadata=index.identity(deferred),
        all_original_native_owners=True, G1_recovery=False, GT_labels_read=False,
        new_CropFormer_frames=len(frame_ids), boundary_adaptation='only explicit frame iterator and count; same native numerical operators'))
    index.save()
    return result
