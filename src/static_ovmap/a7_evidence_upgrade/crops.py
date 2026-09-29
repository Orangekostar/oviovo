"""Recognition-only mask overrides; never weaken original target validation."""

import numpy as np


def recognition_crops(rgb, original_bbox, semantic_mask):
    rgb, mask = np.asarray(rgb), np.asarray(semantic_mask)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("recognition RGB requires HWC uint8")
    if mask.dtype != np.bool_ or mask.shape != rgb.shape[:2]:
        raise ValueError("semantic mask must be full-resolution boolean")
    if len(original_bbox) != 4 or any(int(v) != v for v in original_bbox):
        raise ValueError("recognition crop requires original integer bbox")
    x1, y1, x2, y2 = map(int, original_bbox)
    height, width = mask.shape
    if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height):
        raise ValueError("original bbox is not a nondegenerate in-image min/max box")
    foreground = rgb.copy()
    foreground[~mask] = 0
    six, geometries, support = [], [], []
    for layer in range(3):
        xpad, ypad = int(.1 * layer * (x2 - x1)), int(.1 * layer * (y2 - y1))
        left, top = max(0, x1 - xpad), max(0, y1 - ypad)
        right, bottom = min(width - 1, x2 + xpad), min(height - 1, y2 + ypad)
        pixels = int(mask[top:bottom, left:right].sum())
        if right <= left or bottom <= top or pixels == 0:
            raise ValueError("empty raw or semantic foreground crop")
        geometries.append((left, top, right, bottom))
        support.append(pixels)
        for image in (rgb, foreground):
            crop = image[top:bottom, left:right].copy()
            crop.setflags(write=False)
            six.append(crop)
    return {"six": tuple(six), "geometries": tuple(geometries), "foreground_pixels": support}
