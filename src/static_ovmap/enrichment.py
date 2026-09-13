"""Measured context-crop proxies and final-map directions for offline selection."""
from dataclasses import replace

import numpy as np


def crop_quality(rgb, depth, bbox):
    """Use the encoder's historical exclusive slicing, without a bbox fix.

    Both quantities describe the context bbox, not an unavailable source mask.
    The fixed sharpness scale is 100 intensity-units squared (8-bit RGB).
    """
    rgb, depth = np.asarray(rgb), np.asarray(depth)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or depth.shape != rgb.shape[:2]:
        raise ValueError('RGB uint8 and aligned depth are required')
    x1, y1, x2, y2 = map(int, bbox)
    if not (0 <= x1 <= x2 < rgb.shape[1] and 0 <= y1 <= y2 < rgb.shape[0]):
        raise ValueError('bbox outside image')
    d = depth[y1:y2, x1:x2]
    result = {'depth_valid_ratio': None, 'sharpness_proxy': None,
              'mask_quality_proxy': None}
    if d.size:
        result['depth_valid_ratio'] = float((np.isfinite(d) & (d > 0)).mean())
    crop = rgb[y1:y2, x1:x2].astype(np.float64)
    if crop.shape[0] >= 3 and crop.shape[1] >= 3:
        gray = crop @ np.array([.299, .587, .114])
        lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2]
               + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
        variance = float(lap.var())
        result['sharpness_proxy'] = variance / (variance + 100.)
    return result


def view_direction(camera_center, object_center):
    direction = np.asarray(camera_center, dtype=float) - np.asarray(object_center, dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError('centers must be finite XYZ')
    length = float(np.linalg.norm(direction))
    return None if length < 1e-8 else tuple(direction / length)


def apply_enrichment(bank, data, cache_sha256):
    if data.get('native_cache_sha256') != cache_sha256:
        raise ValueError('enrichment cache identity mismatch')
    records = data['observations']
    observed_ids = {obs.source_query_id for observations in bank.values() for obs in observations}
    if set(records) - observed_ids:
        raise ValueError('enrichment contains unknown query identity')
    result = {}
    fields = ('depth_valid_ratio', 'sharpness_proxy', 'mask_quality_proxy',
              'camera_direction', 'view_bin')
    for instance, observations in bank.items():
        result[instance] = []
        for obs in observations:
            record = records.get(obs.source_query_id)
            if record is None:
                result[instance].append(obs)
                continue
            if record.get('frame_id') != obs.frame_id or record.get('instance_id') != instance:
                raise ValueError('enrichment query/frame/instance identity mismatch')
            values = {name: record[name] for name in fields if name in record}
            result[instance].append(replace(obs, **values))
    return result
