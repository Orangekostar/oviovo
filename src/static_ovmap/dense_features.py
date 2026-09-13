"""Frame-window layout and query-independent native owner feature pools."""
import numpy as np


def window_boxes(height, width, crop=224, stride=112):
    if min(height, width) < crop or min(crop, stride) <= 0 or stride > crop:
        raise ValueError('positive overlapping crop layout within the image required')
    h_count = (height-crop+stride-1)//stride+1
    w_count = (width-crop+stride-1)//stride+1
    result = []
    for y in range(h_count):
        for x in range(w_count):
            y1, x1 = min(y*stride, height-crop), min(x*stride, width-crop)
            result.append((x1, y1, x1+crop, y1+crop))
    return result


def pool_owner_features(features, owners, requested):
    features, owners = np.asarray(features), np.asarray(owners)
    if features.ndim != 3 or features.shape[:2] != owners.shape or not np.isfinite(features).all():
        raise ValueError('finite HWC features must align with native owner image')
    result = {}
    for owner in requested:
        if owner <= 0:
            raise ValueError('unknown owner cannot supply a semantic observation')
        mask = owners == owner
        if not mask.any():
            result[int(owner)] = None
            continue
        y, x = np.nonzero(mask)
        values = features[mask].astype(np.float32)
        quadrants = (y >= (y.min()+y.max()+1)/2).astype(int)*2 + (x >= (x.min()+x.max()+1)/2)
        prototypes = []
        for quadrant in range(4):
            selected = quadrants == quadrant
            if selected.any():
                prototypes.append({'feature': values[selected].mean(axis=0), 'pixels': int(selected.sum()),
                                   'image_quadrant': quadrant})
        result[int(owner)] = {'feature': values.mean(axis=0), 'pixels': len(values), 'prototypes': prototypes}
    return result
