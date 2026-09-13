"""Adapter for trusted local native OVI pickle caches; never guesses history."""
from collections import Counter
import hashlib
from pathlib import Path
import pickle

import numpy as np

from .contracts import Observation


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_native_binding(binding, cache_sha256, config_sha256, text_space, feature_dim):
    if binding.get('native_cache_sha256') != cache_sha256:
        raise ValueError('native binding cache hash mismatch')
    if binding.get('source_config_sha256') != config_sha256:
        raise ValueError('native binding source config hash mismatch')
    if binding.get('feature_space_id') != text_space:
        raise ValueError('native and text feature space mismatch')
    if binding.get('feature_dim') != feature_dim:
        raise ValueError('native feature dimension mismatch')


def load_native_cache(path, *, scene_id, feature_space_id, source_config_hash,
                      history_scope):
    if history_scope not in ('full_query_history', 'retained_native_top10',
                              'retained_native_last8', 'unknown_retention'):
        raise ValueError('explicit supported history_scope is required')
    path = Path(path)
    with path.open('rb') as handle:
        native = pickle.load(handle)
    bank, colors = {}, {}
    for instance_id, record in native.items():
        features = np.asarray(record['feat'])
        frames = record['frame_id']
        area = record['vis_area']
        boxes = record['box_2d']
        if features.ndim != 2 or not (len(features) == len(frames) == len(area) == len(boxes)):
            raise ValueError('native record arrays do not align')
        if not isinstance(instance_id, (int, np.integer)) or instance_id <= 0:
            raise ValueError('native instance IDs must be positive integers')
        color = np.asarray(record['color'])
        if color.shape != (3,) or not np.issubdtype(color.dtype, np.integer) or np.any((color < 0) | (color > 255)):
            raise ValueError('native color must be an integer RGB triplet')
        colors[str(instance_id)] = color.tolist()
        bank[int(instance_id)] = [Observation(
            scene_id=scene_id, instance_id=int(instance_id), frame_id=int(frames[i]),
            source_query_id=f'{int(instance_id)}:{i}', feature_space_id=feature_space_id,
            feature=features[i], visible_area_px=float(area[i]),
            crop_bbox_xyxy=tuple(int(v) for v in boxes[i]),
            source_config_hash=source_config_hash) for i in range(len(frames))]
        # A camera pose alone is not an object-relative viewing direction.
        # Independent geometry support cannot be inferred from semantic queries.
    counts = Counter(len(obs) for obs in bank.values())
    return bank, {'source_path': str(path.resolve()), 'source_sha256': sha256_file(path),
                  'history_scope': history_scope, 'instance_colors': colors,
                  'query_count_histogram': dict(sorted(counts.items())),
                  'instance_count': len(bank), 'stored_query_count': sum(map(len, bank.values())),
                  'instances_above_k8': sum(len(obs) > 8 for obs in bank.values()),
                  'mode': 'STATIC_OFFLINE_READOUT'}
