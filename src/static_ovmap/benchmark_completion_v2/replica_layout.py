"""GT-free Replica RGBD layout for unmodified author step10 readers."""
from pathlib import Path

from .common import (ConsumedInputs, Resolver, StageFailure, canonical_digest,
                     read, source_identity, verified, write)


def _raw_root(binding, scene):
    config = binding['asset_config'].get('datasets', {}).get('replica48', {})
    declared = [config['raw_root']] if config.get('raw_root') else []
    declared += (binding.get('inherited') or {}).get('data_roots', [])
    resolver = Resolver(binding.get('path_map'))
    choices = [Path(resolver.resolve(value)) for value in declared]
    found = next((p for p in choices if (p / 'cam_params.json').is_file()
                  and (p / scene / 'traj.txt').is_file()), None)
    if found is None:
        raise StageFailure('Replica original camera/trajectory root is missing', missing=list(map(str, choices)))
    return found


def _link(path, source):
    if path.is_symlink():
        if path.resolve() != source.resolve():
            raise ValueError('Replica staged image target changed: ' + str(path))
    elif path.exists():
        raise ValueError('An unstaged file occupies the Replica image path: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(source.resolve())


def prepare(binding, scene):
    import numpy as np
    root = Path(binding['output_root'])
    protocol = verified(root / 'protocol_lock.json')
    if scene not in protocol['scopes']['replica48']['scenes']:
        raise ValueError('Replica scene is outside the locked official scope')
    raw = _raw_root(binding, scene)
    camera_path, trajectory = raw / 'cam_params.json', raw / scene / 'traj.txt'
    camera = read(camera_path)['camera']
    values = np.asarray([camera[k] for k in ('w', 'h', 'fx', 'fy', 'cx', 'cy', 'scale')], np.float64)
    if (not np.isfinite(values).all() or min(values[[0, 1, 2, 3, 6]]) <= 0
            or values[0] != int(values[0]) or values[1] != int(values[1])):
        raise ValueError('Original Replica camera metadata is invalid')
    poses = np.loadtxt(trajectory, dtype=np.float64, ndmin=2)
    if poses.shape[1] != 16 or not len(poses) or not np.isfinite(poses).all():
        raise ValueError('Original Replica c2w trajectory is invalid')
    poses = poses.reshape(-1, 4, 4)
    if not np.allclose(poses[:, 3, :], [0, 0, 0, 1], rtol=0, atol=1e-12):
        raise ValueError('Original Replica c2w homogeneous coordinates are invalid')
    frames = []
    missing = []
    for frame in range(len(poses)):
        color = raw / scene / 'results' / f'frame{frame:06d}.jpg'
        depth = raw / scene / 'results' / f'depth{frame:06d}.png'
        frames.append((frame, color, depth))
        missing += [str(p) for p in (color, depth) if not p.is_file()]
    if missing:
        raise StageFailure('Original Replica frame files are missing', missing=missing)
    index = ConsumedInputs(root)
    inputs = [index.identity(camera_path), index.identity(trajectory)]
    selected = list(range(0, len(poses), 10))
    consumed = {str(frame): dict(rgb=index.identity(color), depth=index.identity(depth))
                for frame, color, depth in frames if frame % 10 == 0}
    folder = root / 'external/replica_rgbd' / scene
    receipt = folder / 'layout.json'
    identity = canonical_digest(dict(inputs=inputs, consumed=consumed, frame_count=len(poses),
        protocol=protocol['identity'], producer=source_identity(('replica_layout',)),
        rule='ALL_ORIGINAL_FRAMES_AND_C2W;AUTHOR_STEP10;NO_REENCODING_OR_GT'))
    old = verified(receipt) if receipt.is_file() else None
    if old and old['input_identity'] != identity:
        raise ValueError('Frozen Replica RGBD source or producer changed; preserve before affected repair')
    for frame, color, depth in frames:
        _link(folder / 'color' / f'{frame}.jpg', color)
        _link(folder / 'depth' / f'{frame}.png', depth)
    if old:
        for output in old['outputs']:
            index.identity(output['path'], output)
        index.save()
        return old
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[0, 0], intrinsic[1, 1] = camera['fx'], camera['fy']
    intrinsic[0, 2], intrinsic[1, 2] = camera['cx'], camera['cy']
    outputs = []
    for name in ('intrinsic_color.txt', 'intrinsic_depth.txt'):
        path = folder / 'intrinsic' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, intrinsic, fmt='%.17g')
        outputs.append(index.identity(path))
    for frame, pose in enumerate(poses):
        path = folder / 'pose' / f'{frame}.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, pose, fmt='%.17g')
        outputs.append(index.identity(path))
    index.save()
    return write(receipt, dict(status='DONE', scope='RGBD_IO_ONLY', scene=scene,
        input_identity=identity, layout_root=str(folder), original_root=str(raw), inputs=inputs,
        camera=camera, original_frame_count=len(poses), selected_frame_ids=selected,
        consumed_frames=consumed, outputs=outputs, image_encoding_changed=False,
        pose_convention='ORIGINAL_C2W_FLOAT64_17_DIGITS', depth_divisor=camera['scale'],
        GT_labels_read=False, evaluation_point_domain_ready=False,
        full_method_inference_completed=False, new_GPU_calls=0, new_training_updates=0))
