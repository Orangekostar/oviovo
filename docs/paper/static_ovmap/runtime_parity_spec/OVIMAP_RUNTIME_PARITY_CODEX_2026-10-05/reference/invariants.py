"""Tiny package-level examples; NOT replacements for the real projector/worker.

The bbox routine demonstrates the ideal-pinhole conservative rectangle only.
It does not certify parent FP32 ray arithmetic, triangle tie behavior, or v2 masks.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from typing import Iterable
import numpy as np

@dataclass(frozen=True)
class View:
    pixels: int
    frame: int
    digest: str

    @property
    def rank(self):
        return (-self.pixels, self.frame, self.digest)


def retain_top3(stream: Iterable[View]) -> list[View]:
    kept: list[View] = []
    for view in stream:
        if view.pixels < 0 or view.frame < 0:
            raise ValueError('invalid view')
        if len(kept) == 3 and view.rank[:2] > kept[-1].rank[:2]:
            continue
        kept.append(view)
        kept.sort(key=lambda v: v.rank)
        del kept[3:]
    return kept


def ideal_candidate_roi(camera_corners: np.ndarray, K: np.ndarray,
                        shape: tuple[int, int], guard: int = 2):
    """Return half-open (x0,y0,x1,y1), None, or FULL for uncertain geometry.

    Caller supplies already camera-space corners under exact pinhole assumptions.
    Full production implementation must address reference-origin/rotation rounding.
    """
    points = np.asarray(camera_corners, np.float64)
    K = np.asarray(K, np.float64)
    h, w = shape
    if points.shape != (8, 3) or K.shape != (3, 3) or min(h, w) <= 0 or guard < 0:
        raise ValueError('bad shape')
    if not np.isfinite(points).all() or not np.isfinite(K).all():
        return 'FULL'
    if not np.array_equal(K[2], [0, 0, 1]):
        return 'FULL'
    if np.max(points[:, 2]) <= 1e-6:
        return None
    if np.min(points[:, 2]) <= 1e-6:
        return 'FULL'
    uvw = points @ K.T
    uv = uvw[:, :2] / uvw[:, 2:]
    lo = np.floor(np.nextafter(uv.min(0), -np.inf)).astype(np.int64) - guard
    hi = np.ceil(np.nextafter(uv.max(0), np.inf)).astype(np.int64) + guard + 1
    x0, y0 = max(0, int(lo[0])), max(0, int(lo[1]))
    x1, y1 = min(w, int(hi[0])), min(h, int(hi[1]))
    return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1, y1)


SCIENCE_FIELDS = ('scene', 'owner', 'frame', 'mask_digest', 'geometry_digest',
                  'camera_digest', 'rgb_digest', 'depth_digest', 'bbox', 'pixels',
                  'selection_rank', 'operator_semantics', 'vocabulary_digest')


def science_key(record: dict) -> str:
    """A schematic key; actual production fields are in the task contract."""
    missing = set(SCIENCE_FIELDS) - set(record)
    if missing:
        raise ValueError('missing scientific fields: '+','.join(sorted(missing)))
    payload = {key: record[key] for key in SCIENCE_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def summarize_times(rows: list[dict], scenes: list[str], repeats: int = 2) -> dict:
    """All cells required; equal-scene means, never minima."""
    keys = [(r['scene'], r['repeat']) for r in rows]
    expected = {(s, r) for s in scenes for r in range(1, repeats+1)}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('incomplete/duplicated timing grid')
    values = {(r['scene'], r['repeat']): float(r['seconds']) for r in rows}
    if not all(np.isfinite(v) and v > 0 for v in values.values()):
        raise ValueError('invalid measured time')
    means = {s: sum(values[s, r] for r in range(1, repeats+1))/repeats for s in scenes}
    return {'scene_means': means, 'cohort_mean': sum(means.values())/len(means)}


def balanced_schedule(scenes: list[str], arms: list[str]):
    return [(s, a, r) for r in (1, 2)
            for s in (scenes if r == 1 else list(reversed(scenes)))
            for a in (arms if r == 1 else list(reversed(arms)))]


def candidate_choice(means: dict[str, float], order: list[str], band: float = .03):
    if not means or not all(np.isfinite(t) and t > 0 for t in means.values()):
        raise ValueError('invalid candidate time')
    if not set(means) <= set(order):
        raise ValueError('unknown variant')
    threshold = min(means.values()) * (1+band)
    return next(name for name in order if name in means and means[name] <= threshold)
