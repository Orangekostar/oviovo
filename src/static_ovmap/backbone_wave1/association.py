"""Current-frame object matching against the read-only native prior probe."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .fusion import spatial_key


@dataclass(frozen=True)
class Assignments:
    forward: set
    backward: set
    mutual: set
    forward_coverages: np.ndarray
    reverse_coverages: np.ndarray


def _positive_assignment(benefit, eligible):
    n, m = benefit.shape
    if not n or not m:
        return set()
    scores = np.full((n, m + n), -1e6, np.float64)
    scores[:, :m] = np.where(eligible, benefit, -1e6)
    scores[np.arange(n), m + np.arange(n)] = 0.
    rows, cols = linear_sum_assignment(-scores)
    return {(int(i), int(j)) for i, j in zip(rows, cols)
            if j < m and eligible[i, j] and benefit[i, j] > 0}


def directional_assignments(intersections, local_areas, global_areas):
    inter = np.asarray(intersections, np.float64)
    local, glob = np.asarray(local_areas, np.float64), np.asarray(global_areas, np.float64)
    if inter.ndim != 2 or inter.shape != (len(local), len(glob)):
        raise ValueError("intersection shape mismatch")
    if any(not np.isfinite(x).all() or np.any(x < 0) for x in (inter, local, glob)):
        raise ValueError("finite nonnegative counts required")
    if np.any(inter.sum(1) > local + 1e-9) or np.any(inter.sum(0) > glob + 1e-9):
        raise ValueError("intersections exceed disjoint support")
    forward = np.divide(inter, local[:, None], out=np.zeros_like(inter), where=local[:, None] > 0)
    reverse = np.divide(inter, glob[None, :], out=np.zeros_like(inter), where=glob[None, :] > 0)
    f_pairs = _positive_assignment(forward - .2, (inter >= 100) & (forward > .2))
    b_pairs = {(j, i) for i, j in _positive_assignment(
        reverse.T - .2, (inter.T >= 100) & (reverse.T > .2))}
    return Assignments(f_pairs, b_pairs, f_pairs & b_pairs, forward, reverse)


@dataclass(frozen=True)
class ObjectPlan:
    local_groups: list
    global_owners: list
    local_areas: list
    global_areas: list
    intersections: list
    forward_coverages: list
    reverse_coverages: list
    forward_pairs: list
    backward_pairs: list
    accepted_pairs: list
    planned_owners: dict


def plan_objects(fragments, depth_m, prior_owner, *, mode):
    if mode not in ("object_forward", "object_bidirectional"):
        raise ValueError("unsupported directional association mode")
    depth, prior = np.asarray(depth_m), np.asarray(prior_owner)
    if depth.ndim != 2 or depth.shape != prior.shape:
        raise ValueError("aligned depth and probe required")
    valid = np.isfinite(depth) & (depth > 0) & (depth < 50)
    supports = {}
    for fragment in fragments:
        if fragment.is_thing and fragment.input_group > 0:
            if fragment.mask.shape != depth.shape:
                raise ValueError("fragment support shape mismatch")
            supports.setdefault(fragment.input_group, np.zeros_like(valid))[:] |= fragment.mask
    groups = sorted(supports, key=lambda k: spatial_key(supports[k]))
    owners = sorted(int(k) for k in np.unique(prior) if k > 0)
    local = [int((supports[k] & valid).sum()) for k in groups]
    global_areas = [int((prior == k).sum()) for k in owners]
    intersections = np.zeros((len(groups), len(owners)), np.int64)
    for i, group in enumerate(groups):
        values, counts = np.unique(prior[supports[group] & valid], return_counts=True)
        observed = dict(zip(values.tolist(), counts.tolist()))
        intersections[i] = [observed.get(k, 0) for k in owners]
    result = directional_assignments(intersections, local, global_areas)
    accepted = result.forward if mode == "object_forward" else result.mutual
    pairs = lambda values: [[groups[i], owners[j]] for i, j in sorted(values)]
    planned = {group: 0 for group in groups}
    planned.update({groups[i]: owners[j] for i, j in accepted})
    return ObjectPlan(groups, owners, local, global_areas, intersections.tolist(),
                      result.forward_coverages.tolist(), result.reverse_coverages.tolist(),
                      pairs(result.forward), pairs(result.backward), pairs(accepted), planned)
