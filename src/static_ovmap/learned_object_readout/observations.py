"""Class-independent fixed views/sites and exact feature-grid sampling."""
import numpy as np


def normalized_grid(uv, original_hw, resized_hw, padded_hw):
    uv = np.asarray(uv, np.float64)
    h, w = original_hw; hr, wr = resized_hw; hp, wp = padded_hw
    if uv.ndim != 2 or uv.shape[1] != 2 or min(h, w, hr, wr, hp, wp) <= 0:
        raise ValueError('Explicit image transform and pixel coordinates required')
    res = (uv + .5) * np.array([wr/w, hr/h]) - .5
    return 2 * (res + .5) / np.array([wp, hp]) - 1


def sample_raw(raw, uv, original_hw, resized_hw, padded_hw):
    import torch
    from torch.nn import functional as F
    uv = np.asarray(uv, np.float64)
    h, w = original_hw
    valid = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    safe = np.where(valid[:, None], uv, 0)
    grid = torch.as_tensor(normalized_grid(safe, original_hw, resized_hw, padded_hw),
                           dtype=raw.dtype, device=raw.device).reshape(1, -1, 1, 2)
    if raw.ndim != 4 or raw.shape[0] != 1:
        raise ValueError('One shared raw image grid required')
    sampled = F.grid_sample(raw, grid, mode='bilinear', padding_mode='zeros', align_corners=False)[0, :, :, 0].T
    mask = torch.as_tensor(valid, dtype=torch.bool, device=raw.device)
    return torch.where(mask[:, None], sampled, 0), mask


def site_indices(xyz, area, maximum):
    """Area quantiles without duplicating any physical site."""
    xyz, area = np.asarray(xyz), np.asarray(area, np.float64)
    if not len(area):
        return np.empty(0, np.int64)
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    if len(order) <= maximum:
        return order
    mass = np.cumsum(area[order]); total = mass[-1]
    if not np.isfinite(total) or total <= 0:
        raise ValueError('Positive physical site area required')
    picks = np.searchsorted(mass, (np.arange(maximum) + .5) * total / maximum)
    return order[np.unique(np.minimum(picks, len(order)-1))]


def choose_views(frame_ids, masks, visibility, area, maximum=8):
    frame_ids = list(map(int, frame_ids))
    area = np.asarray(area, np.float64)
    qualified = [i for i, mask in enumerate(masks) if full_qualified(mask)]
    if not qualified:
        return []
    anchor = min(qualified, key=lambda i: (-int(masks[i].sum()), frame_ids[i]))
    chosen = [anchor]
    covered = np.asarray(visibility[anchor], bool).copy()
    while len(chosen) < maximum and len(chosen) < len(qualified):
        remaining = [i for i in qualified if i not in chosen]
        best = min(remaining, key=lambda i: (-float(area[np.asarray(visibility[i], bool) & ~covered].sum()),
                                            -int(masks[i].sum()), frame_ids[i]))
        chosen.append(best); covered |= visibility[best]
    return chosen


def full_qualified(mask):
    y, x = np.nonzero(mask)
    return bool(len(x) >= 100 and x.max()-x.min()+1 >= 2 and y.max()-y.min()+1 >= 2)
