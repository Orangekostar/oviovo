"""Synthetic reference rules only; not the production SAM-V/mesh integration."""
from __future__ import annotations
from typing import Mapping, Sequence
import numpy as np


def interior_points(mask: np.ndarray, maximum: int = 3) -> list[tuple[int, int]]:
    from scipy.ndimage import distance_transform_edt, label
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or maximum < 1:
        raise ValueError('2D mask and positive maximum required')
    components, count = label(mask, np.ones((3, 3), dtype=int))
    if not count:
        return []
    choices = []
    for component in range(1, count + 1):
        rows = np.flatnonzero(components == component)
        choices.append((-len(rows), int(rows[0]), component))
    chosen = min(choices)[2]
    distance = distance_transform_edt(np.pad(components == chosen, 1))[1:-1, 1:-1]
    yx = np.argwhere(distance >= 2.0)
    if not len(yx):
        return []
    values = distance[yx[:, 0], yx[:, 1]]
    first = int(np.flatnonzero(values == values.max())[0])
    selected = [first]
    while len(selected) < min(maximum, len(yx)):
        d2 = ((yx[:, None, :] - yx[np.asarray(selected)][None, :, :]) ** 2).sum(-1).min(1)
        d2[selected] = -1
        selected.append(int(np.flatnonzero(d2 == d2.max())[0]))
    return [(int(yx[j, 1]), int(yx[j, 0])) for j in selected]


def order_window(frame_ids: Sequence[int], anchor: int, second: int) -> tuple[list[int], int, int]:
    ids = list(map(int, frame_ids))
    if len(set(ids)) != len(ids) or anchor not in ids or second not in ids or anchor == second:
        raise ValueError('distinct frames and both mandatory frames required')
    ids.sort()
    return ids, ids.index(anchor), ids.index(second)


def transform_points(points: np.ndarray, original_hw: tuple[int, int]) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    h, w = original_hw
    if p.ndim != 2 or p.shape[1] != 2 or h <= 0 or w <= 0:
        raise ValueError('invalid points/image')
    if np.any(p < 0) or np.any(p[:, 0] >= w) or np.any(p[:, 1] >= h):
        raise ValueError('original coordinates out of bounds')
    return p * np.array([1024.0 / w, 1024.0 / h])


def split_panorama(logits: np.ndarray, frames: int) -> np.ndarray:
    x = np.asarray(logits)
    if x.ndim != 2 or frames < 1 or x.shape[1] % frames:
        raise ValueError('panorama must split into exact view tiles')
    h, width = x.shape
    return x.reshape(h, frames, width // frames).transpose(1, 0, 2).copy()


def count_view_votes(source_maps: Sequence[np.ndarray], valid_maps: Sequence[np.ndarray],
                     masks: Sequence[np.ndarray], row_count: int) -> tuple[np.ndarray, np.ndarray]:
    if not (len(source_maps) == len(valid_maps) == len(masks)):
        raise ValueError('aligned view lists required')
    n = np.zeros(row_count, dtype=np.int64)
    k = np.zeros(row_count, dtype=np.int64)
    for source, valid, mask in zip(source_maps, valid_maps, masks, strict=True):
        source, valid, mask = np.asarray(source), np.asarray(valid, bool), np.asarray(mask, bool)
        if source.shape != valid.shape or mask.shape != source.shape:
            raise ValueError('aligned view rasters required')
        ids = source[valid]
        if np.any(ids < 0) or np.any(ids >= row_count):
            raise ValueError('invalid visible source index')
        counts = np.bincount(ids, minlength=row_count)
        hits = np.bincount(ids[mask[valid]], minlength=row_count)
        visible = counts > 0
        n += visible
        k += visible & (2 * hits >= counts)
    return n, k


def arbitrate(base: np.ndarray, target_ids: Sequence[int], counts: Mapping[int, tuple[np.ndarray, np.ndarray]],
              domains: Mapping[int, np.ndarray], editable: np.ndarray,
              anchors: Mapping[int, int] | None = None) -> np.ndarray:
    """Integer vote comparisons implement thresholds 2/3 and 1/3 exactly."""
    base = np.asarray(base, np.int64)
    editable = np.asarray(editable, bool)
    if base.ndim != 1 or editable.shape != base.shape:
        raise ValueError('aligned base/edit domain required')
    ids = sorted(set(map(int, target_ids)))
    if len(ids) != len(target_ids) or any(i <= 0 for i in ids):
        raise ValueError('unique positive target owners required')
    out = base.copy()
    for r in np.flatnonzero(editable):
        contenders = []
        for owner in ids:
            n, k = counts[owner]
            if domains[owner][r] and n[r] >= 2 and 3 * k[r] >= 2 * n[r]:
                contenders.append(owner)
        if contenders:
            best = [contenders[0]]
            for owner in contenders[1:]:
                bn, bk = counts[best[0]]
                n, k = counts[owner]
                lhs, rhs = int(k[r]) * int(bn[r]), int(bk[r]) * int(n[r])
                if lhs > rhs:
                    best = [owner]
                elif lhs == rhs:
                    best.append(owner)
            if len(best) == 1:
                out[r] = best[0]
            # Any exact tie retains base, including a base owner not among contenders.
        elif int(base[r]) in ids:
            owner = int(base[r]); n, k = counts[owner]
            if domains[owner][r] and n[r] >= 2 and 3 * k[r] <= n[r]:
                out[r] = 0
    for row, owner in (anchors or {}).items():
        if not (0 <= row < len(base)) or int(base[row]) != int(owner) or owner not in ids:
            raise ValueError('prompt source must belong to its original selected owner')
        out[row] = owner
    return out


def labels_for_partition(owners: np.ndarray, old_labels: Mapping[int, int],
                         updates: Mapping[int, int] | None = None) -> np.ndarray:
    owners = np.asarray(owners, np.int64)
    classes = {**old_labels, **(updates or {})}
    out = np.zeros(owners.shape, np.int64)
    for owner in np.unique(owners[owners > 0]):
        if int(owner) not in classes:
            raise ValueError('unknown owner without an inherited class')
        out[owners == owner] = classes[int(owner)]
    return out


def paired_labels(old_class: int, class_ids: Sequence[int], old_scores: Sequence[np.ndarray | None],
                  new_scores: Sequence[np.ndarray | None], margin: float = .01) -> tuple[int, int, bool]:
    ids = list(class_ids)
    if old_class not in ids or len(old_scores) != 2 or len(new_scores) != 2:
        raise ValueError('two fixed views and valid old class required')
    if any(x is None for x in [*old_scores, *new_scores]):
        return old_class, old_class, False
    results = []
    for source in (old_scores, new_scores):
        array = np.asarray(source, np.float64)
        if array.shape != (2, len(ids)) or not np.isfinite(array).all():
            raise ValueError('invalid full vocabulary scores')
        avg = array.mean(0); c = int(avg.argmax())
        results.append(ids[c] if ids[c] != old_class and avg[c] - avg[ids.index(old_class)] >= margin else old_class)
    return results[0], results[1], True


def pilot_target(candidate: Mapping[str, Mapping[str, float]], baseline: Mapping[str, Mapping[str, float]],
                 cf_d2: Mapping[str, float], eps: float = 1e-10) -> bool:
    metrics = ('apall', 'ap50', 'ap25', 'miou', 'macc')
    for cohort in ('replica_probe2', 'cf_probe2'):
        for metric in metrics:
            value, reference = candidate[cohort][metric], baseline[cohort][metric]
            if not np.isfinite(value) or value < reference - eps:
                return False
    cf = candidate['cf_probe2']
    return cf['apall'] > cf_d2['apall'] + eps and cf['ap50'] >= cf_d2['ap50'] - eps
