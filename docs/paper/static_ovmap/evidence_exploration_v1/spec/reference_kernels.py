"""Small NumPy specifications for the proposed study, NOT production model code.

These kernels do not load FC/AnyUp, raycast, inspect private data, or score a map.
Production adapters must still be verified in the actual repository environment.
"""
from __future__ import annotations
from typing import Mapping, Sequence
import numpy as np

METRICS = ('apall', 'ap50', 'ap25', 'miou', 'macc')
NUMERIC_TOL = 1e-10

def occupancy_trigger(occupancy: np.ndarray, hard_support: int) -> tuple[bool, float]:
    w = np.asarray(occupancy, dtype=np.float64)
    if w.size == 0 or not np.isfinite(w).all() or np.any((w < 0) | (w > 1)):
        raise ValueError('occupancy must be finite probabilities')
    if isinstance(hard_support, bool) or int(hard_support) != hard_support or hard_support < 0:
        raise ValueError('hard support must be a nonnegative integer')
    mass = float(w.sum())
    if mass <= 0:
        raise ValueError('nonempty bound region cannot have zero area mass')
    purity = float((w * w).sum() / (mass + 1e-8))
    return (hard_support < 4 or purity < 0.5), purity


def contrast_decision(
    target: Sequence[float], controls: np.ndarray | None, purity: float,
    *, triggered: bool, strength: float = 0.25, threshold: float = 0.01,
) -> dict:
    """Nontriggered identity; missing controls give the margin-only control.

    ``accepted`` is independent of ``feature_available``. Scores are NOT probabilities.
    """
    s = np.asarray(target, dtype=np.float64)
    if s.ndim != 1 or len(s) < 2 or not np.isfinite(s).all():
        raise ValueError('a finite full-class score vector is required')
    if not 0 <= purity <= 1 or not 0 <= strength <= 1 or threshold < 0:
        raise ValueError('invalid contrast constants')
    r = s.copy()
    if triggered and controls is not None:
        c = np.asarray(controls, dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != len(s) or not np.isfinite(c).all():
            raise ValueError('controls must have the same class order and width')
        if len(c):
            b = c.max(axis=0)
            r -= strength * (1 - purity) * (b - b.mean())
    top = int(np.argmax(r))
    ordered = np.sort(r)
    margin = float(ordered[-1] - ordered[-2])
    return {'feature_available': True, 'proposed_class_index': top,
            'accepted': bool(not triggered or margin >= threshold),
            'margin': margin, 'scores': r}


def geometry_factor(
    owner_key: Sequence[float], unknown_key: Sequence[float],
    query_depth: Sequence[float], key_owner_depth: Sequence[float], *, use_depth: bool = True,
) -> np.ndarray:
    """Unknown depth is neutral; other-owner evidence is soft, never -infinity."""
    own, unknown = np.asarray(owner_key, float), np.asarray(unknown_key, float)
    dq, dk = np.asarray(query_depth, float), np.asarray(key_owner_depth, float)
    if own.ndim != 1 or unknown.shape != own.shape or dk.shape != own.shape or dq.ndim != 1:
        raise ValueError('owner/key and query shapes differ')
    if (not np.isfinite(own).all() or not np.isfinite(unknown).all()
            or np.any(own < 0) or np.any(unknown < 0) or np.any(own + unknown > 1 + 1e-6)):
        raise ValueError('inconsistent owner occupancies')
    out = np.broadcast_to(0.05 + 0.95 * np.clip(own + unknown, 0, 1),
                          (len(dq), len(own))).copy()
    if use_depth:
        valid = np.isfinite(dq[:, None]) & (dq[:, None] > 0) & np.isfinite(dk[None, :]) & (dk[None, :] > 0)
        safeq = np.where(np.isfinite(dq) & (dq > 0), dq, 1.)
        safek = np.where(np.isfinite(dk) & (dk > 0), dk, 1.)
        z = np.minimum(np.abs(safeq[:, None] - safek[None, :]) / (0.02 + 0.02 * safeq[:, None]), 3)
        out *= np.where(valid, np.exp(-z), 1.)
    return out


def reweight_attention(attention: np.ndarray, factor: np.ndarray) -> np.ndarray:
    a, g = np.asarray(attention, float), np.asarray(factor, float)
    if a.ndim != 2 or g.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(g).all():
        raise ValueError('finite matching query-key arrays required')
    if np.any(a < 0) or np.any(g <= 0) or np.any(a.sum(axis=1) <= 0):
        raise ValueError('invalid attention or all-masked query')
    weighted = a * g
    return weighted / weighted.sum(axis=1, keepdims=True)


def target_flags(
    candidate: Mapping[str, Mapping[str, float]],
    b1: Mapping[str, Mapping[str, float]], b0: Mapping[str, Mapping[str, float]],
) -> dict:
    for record in (candidate, b1, b0):
        for cohort in ('replica8', 'scannet_cf18'):
            if set(METRICS) - set(record[cohort]):
                raise ValueError('complete five-metric record required')
            if any(not np.isfinite(record[cohort][m]) or not 0 <= record[cohort][m] <= 1 for m in METRICS):
                raise ValueError('metrics must be finite fractions')
    rep_ok = all(candidate['replica8'][m] >= b1['replica8'][m] - NUMERIC_TOL for m in METRICS)
    cf_ok = all(candidate['scannet_cf18'][m] >= b1['scannet_cf18'][m] - NUMERIC_TOL for m in METRICS)
    cf = candidate['scannet_cf18']; d2 = b0['scannet_cf18']
    above_d2 = cf['apall'] > d2['apall'] + NUMERIC_TOL and cf['ap50'] >= d2['ap50'] - NUMERIC_TOL
    passed = bool(rep_ok and cf_ok and above_d2)
    return {'replica_nondecrease_all5': bool(rep_ok), 'cf_nondecrease_all5': bool(cf_ok),
            'cf_AP_conditions': bool(above_d2), 'target_met': passed,
            'material_target_met': bool(passed and cf['apall'] - d2['apall'] >= .001 - NUMERIC_TOL)}


def choose_method(records: Mapping[str, Mapping], simplicity_order: Sequence[str]) -> str:
    b1, b0 = records['EV01_G1_V2'], records['EV00_D2']
    possible = [k for k in simplicity_order if k in records and k not in ('EV00_D2', 'EV01_G1_V2')
                and target_flags(records[k], b1, b0)['target_met']]
    if not possible:
        return 'EV01_G1_V2'
    best = possible[0]
    def values(k):
        rec = records[k]
        return (rec['scannet_cf18']['apall'], rec['scannet_cf18']['ap50'],
                rec['replica8']['apall'], rec['scannet_cf18']['miou'])
    for key in possible[1:]:
        for x, y in zip(values(key), values(best)):
            if x > y + NUMERIC_TOL:
                best = key; break
            if y > x + NUMERIC_TOL:
                break
    return best
