"""Small specification fixtures; NOT the production repair/evaluation implementation.

They intentionally use no server data, model weights, Open3D or benchmark GT.
Codex must test the real production call sites separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import json
import math
import numpy as np

SPEC = json.loads(Path(__file__).with_name('PROTOCOL_SPEC.json').read_text())


def pose_banks(frames: Sequence[tuple[int, np.ndarray]]) -> tuple[list[int], list[int]]:
    """Greedy first representative bins, then fixed even/odd bank assignment."""
    reps: list[tuple[int, np.ndarray]] = []
    seen: set[int] = set()
    for fid, value in sorted(frames, key=lambda row: row[0]):
        p = np.asarray(value, dtype=np.float64)
        if fid in seen or p.shape != (4, 4) or not np.isfinite(p).all():
            raise ValueError('invalid unique camera input')
        seen.add(fid)
        if not np.allclose(p[3], [0, 0, 0, 1], atol=1e-12):
            raise ValueError('nonhomogeneous camera')
        if not np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-5):
            raise ValueError('nonrigid camera')
        matched = False
        for _, q in reps:
            angle = math.degrees(math.acos(float(np.clip((np.trace(q[:3, :3].T @ p[:3, :3])-1)/2, -1, 1))))
            if (np.linalg.norm(p[:3, 3]-q[:3, 3]) <= SPEC['observer']['pose_translation_m']
                    and angle <= SPEC['observer']['pose_rotation_degrees']):
                matched = True
                break
        if not matched:
            reps.append((fid, p.copy()))
    limit = SPEC['observer']['max_frames']
    if len(reps) > limit:
        ids = [k * (len(reps)-1) // (limit-1) for k in range(limit)]
        reps = [reps[i] for i in ids]
    selected = [fid for fid, _ in reps]
    return selected[::2], selected[1::2]


def representative_source_row(faces: np.ndarray, primitive: np.ndarray, uv: np.ndarray) -> np.ndarray:
    faces, primitive, uv = np.asarray(faces), np.asarray(primitive), np.asarray(uv)
    if uv.shape != (len(primitive), 2) or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('unaligned triangle hits')
    if not np.issubdtype(primitive.dtype, np.integer) or np.any(primitive < 0) or np.any(primitive >= len(faces)):
        raise ValueError('invalid hit primitive')
    weights = np.column_stack((1-uv.sum(1), uv))
    tol = SPEC['observer']['barycentric_validity_tolerance']
    if not np.isfinite(weights).all() or (weights < -tol).any() or (weights > 1+tol).any():
        raise ValueError('invalid barycentric weights')
    rows = faces[primitive]
    tied = weights == weights.max(axis=1, keepdims=True)
    return np.where(tied, rows, np.iinfo(np.int64).max).min(1)


def unit_summary(labels: Iterable[int]) -> tuple[int, float] | None:
    a = np.asarray(list(labels), dtype=np.int64)
    cfg = SPEC['observer']
    positive = a[a > 0]
    if (len(a) < cfg['min_visible_pixels'] or len(positive) < cfg['min_labeled_pixels']
            or len(positive) / max(1, len(a)) < cfg['min_labeled_fraction']):
        return None
    ids, counts = np.unique(positive, return_counts=True)
    best = int(np.argmax(counts))  # unique IDs are sorted, ties use smallest ID
    fraction = float(counts[best] / len(positive))
    return (int(ids[best]), fraction) if fraction >= cfg['dominance'] else None


def pair_vote(a: tuple[int, float] | None, b: tuple[int, float] | None) -> tuple[int, int] | None:
    if a is None or b is None:
        return None
    if a[0] == b[0]:
        return 1, 0
    if min(a[1], b[1]) >= SPEC['observer']['separation_dominance']:
        return 0, 1
    return 0, 0


def rates(votes: Sequence[tuple[int, int] | None]) -> tuple[int, float | None, float | None]:
    known = [x for x in votes if x is not None]
    if not known:
        return 0, None, None
    return len(known), float(np.mean([x[0] for x in known])), float(np.mean([x[1] for x in known]))


def verified_pairs(pair_rates: Sequence[tuple[int, float | None, float | None]], edit_cost: float) -> bool:
    cfg = SPEC['repair']
    if not 0 <= edit_cost <= 1:
        raise ValueError('invalid edit cost')
    if not pair_rates or any(n < cfg['minimum_verification_views_per_pair'] or s is None or d is None for n, s, d in pair_rates):
        return False
    same = float(np.mean([x[1] for x in pair_rates]))
    separate = float(np.mean([x[2] for x in pair_rates]))
    score = same - cfg['separation_penalty'] * separate - cfg['edit_penalty'] * edit_cost
    return same >= cfg['minimum_same_rate'] and separate <= cfg['maximum_separate_rate'] and score >= cfg['minimum_repair_score']


def clique_prefixes(seed: str, ordered_neighbors: Sequence[str], positive_edges: set[frozenset[str]]) -> list[tuple[str, ...]]:
    group = [seed]
    result = []
    for x in ordered_neighbors:
        if x in group:
            continue
        trial = group + [x]
        if (sum(u.startswith('I:') for u in trial) > SPEC['support']['max_incumbents_per_hypothesis']
                or not all(frozenset((a, b)) in positive_edges for a, b in combinations(trial, 2))):
            continue
        group.append(x)
        result.append(tuple(group))
        if len(group) == SPEC['support']['max_units_per_hypothesis']:
            break
    return result


@dataclass(frozen=True)
class Operation:
    units: tuple[str, ...]
    retained_owner: int
    semantic_label: int


def apply_operations(g: np.ndarray, semantic: np.ndarray, painted: np.ndarray, raw: np.ndarray,
                     units: Mapping[str, np.ndarray], operations: Sequence[Operation]) -> tuple[np.ndarray, np.ndarray]:
    g, semantic, painted, raw = map(np.asarray, (g, semantic, painted, raw))
    if not (g.shape == semantic.shape == painted.shape == raw.shape) or g.ndim != 1:
        raise ValueError('unaligned source rows')
    if len(operations) > SPEC['support']['max_applied_operations']:
        raise ValueError('edit budget exceeded')
    owners, classes = g.copy(), semantic.copy()
    consumed: set[str] = set()
    used_rows: set[int] = set()
    for operation in operations:
        if (len(set(operation.units)) != len(operation.units) or len(operation.units) < 2
                or len(operation.units) > SPEC['support']['max_units_per_hypothesis']
                or consumed.intersection(operation.units)):
            raise ValueError('overlapping or invalid operation')
        hosts = [x for x in operation.units if x.startswith('I:')]
        if len(hosts) > 1:
            raise ValueError('cannot merge incumbent cores')
        if operation.retained_owner <= 0 or operation.semantic_label <= 0:
            raise ValueError('positive owner/class required')
        if hosts:
            host = int(hosts[0].split(':')[1])
            if operation.retained_owner != host:
                raise ValueError('host ID must remain')
            if not np.all(semantic[painted == host] == operation.semantic_label):
                raise ValueError('structural host class must remain')
        elif operation.retained_owner <= max(np.max(g, initial=0), np.max(raw, initial=0), np.max(painted, initial=0)):
            raise ValueError('union needs a fresh ID')
        for name in operation.units:
            if name.startswith('I:'):
                continue
            rr = np.asarray(units[name], dtype=np.int64)
            if (rr.ndim != 1 or len(np.unique(rr)) != len(rr) or np.any(rr < 0) or np.any(rr >= len(g))
                    or np.any(painted[rr] != 0) or np.any(raw[rr] == 0) or used_rows.intersection(map(int, rr))):
                raise ValueError('residual support conflicts with immutable rows')
            owners[rr] = operation.retained_owner
            classes[rr] = operation.semantic_label
            used_rows.update(map(int, rr))
        consumed.update(operation.units)
    if not np.array_equal(owners[painted > 0], g[painted > 0]):
        raise AssertionError('incumbent core moved')
    return owners, classes


def reread_decision(old_class_index: int, full: np.ndarray, regions: np.ndarray,
                    *, distinct_nonfull: int) -> tuple[int, bool, bool]:
    """Rows already content-deduplicated; class indices follow bound vocabulary order."""
    full, regions = np.asarray(full, dtype=float), np.asarray(regions, dtype=float)
    if full.ndim != 2 or regions.ndim != 2 or full.shape[1] != regions.shape[1]:
        raise ValueError('aligned full score vectors required')
    if not np.isfinite(full).all() or not np.isfinite(regions).all():
        raise ValueError('finite scores required')
    if not len(full):
        return old_class_index, False, False
    cfg = SPEC['reread']
    q = full.mean(0)
    proposed = int(q.argmax())
    simple = proposed != old_class_index and q[proposed]-q[old_class_index] >= cfg['aggregate_margin_over_incumbent']
    stable = (simple and len(full) == 2 and len(regions) >= cfg['stable_min_distinct_regions']
              and distinct_nonfull >= cfg['stable_min_extra_regions']
              and bool(np.all(full.argmax(1) == proposed))
              and bool(np.all(full[:, proposed]-full[:, old_class_index] >= cfg['stable_full_view_margin_min'])))
    if stable:
        stable = (np.mean(regions.argmax(1) == proposed) >= cfg['stable_vote_fraction']
                  and np.median(regions[:, proposed]-regions[:, old_class_index]) >= cfg['stable_median_margin_min'])
    return proposed, bool(simple), bool(stable)


def target_met(candidate: Mapping, reference: Mapping, d2: Mapping) -> bool:
    cfg = SPEC['selection']
    eps = cfg['tolerance_fraction']
    metrics = SPEC['evaluation']['metrics']
    for record in (candidate, reference, d2):
        for cohort in SPEC['cohorts']:
            for m in metrics:
                value = record[cohort][m]
                if value is None or not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('full finite fraction metrics required')
    return (all(candidate[c][m] >= reference[c][m]-eps for c in SPEC['cohorts'] for m in metrics)
            and candidate['scannet_cf18']['apall'] > d2['scannet_cf18']['apall']+eps
            and candidate['scannet_cf18']['ap50'] >= d2['scannet_cf18']['ap50']-eps)
