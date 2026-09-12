"""Current-visit semantic evidence without historical geometry retirement."""

import numpy as np

from src.oviv2.surface_multiview_semantics import aggregate_views, diverse_views


def unique_region_pixels(region_pixels, image_size):
    """Map each pixel to its sole region slot, or -1 for absent/ambiguous."""
    result = np.full(image_size, -1, np.int32)
    counts = np.zeros(image_size, np.int32)
    for slot, pixels in enumerate(region_pixels):
        pixels = np.unique(pixels)
        if pixels.dtype.kind not in "iu" or np.any(
            (pixels < 0) | (pixels >= image_size)
        ):
            raise ValueError("integer pixels inside the image required")
        counts[pixels] += 1
        result[pixels] = slot
    result[counts != 1] = -1
    return result


def current_view_embedding(features, quality, frame_ids, directions, *, k=4):
    """At least two real current frames; at most k diverse quality-weighted views."""
    features, quality = np.asarray(features), np.asarray(quality)
    frames, directions = np.asarray(frame_ids), np.asarray(directions)
    if (
        features.ndim != 2
        or quality.shape != (len(features),)
        or frames.shape != quality.shape
        or directions.shape != (len(features), 3)
    ):
        raise ValueError("aligned current observations required")
    if (
        not np.isfinite(features).all()
        or not np.isfinite(quality).all()
        or np.any(quality < 0)
    ):
        raise ValueError("finite features and nonnegative quality required")
    _, inverse, counts = np.unique(frames, return_inverse=True, return_counts=True)
    valid = (quality > 0) & (counts[inverse] == 1)
    if valid.sum() < 2:
        return None
    selected = diverse_views(directions[valid], quality[valid], k=k)
    return aggregate_views(features[valid][selected], quality[valid][selected])
