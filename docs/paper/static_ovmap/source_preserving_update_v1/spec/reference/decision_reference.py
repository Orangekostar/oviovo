"""Small mathematical specification, NOT a production pipeline or measured method.

No repository, filesystem, model, GT or evaluator access is performed here.
Actual parent reconstruction and same-view acquisition require the Codex adapter.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
from scipy.special import softmax, log_softmax

NEW_METHODS = (
    'SU04_F_REPLACE', 'SU05_F_BLEND', 'SU06_F_COARSE',
    'SU07_GLOBAL_BLEND', 'SU08_PAIRED_DELTA',
)
METRICS = ('apall', 'ap50', 'ap25', 'miou', 'macc')


def _vector(x: Sequence[float], n: int | None = None) -> np.ndarray:
    a = np.asarray(x, np.float64)
    if a.ndim != 1 or a.size < 2 or (n is not None and len(a) != n) or not np.isfinite(a).all():
        raise ValueError('Expected a finite full-vocabulary score vector')
    return a


def source_components(scores: Mapping[str, Sequence[float] | None],
                      temperatures: Mapping[str, float]):
    """Return (p0, q, pF, wF) for the reviewed D2 grouping."""
    available = [n for n in ('N', 'Q', 'F') if scores.get(n) is not None]
    if not available:
        return None, None, None, 0.0
    size = len(_vector(scores[available[0]]))
    p = {}
    for n in available:
        t = float(temperatures[n])
        if not np.isfinite(t) or t <= 0:
            raise ValueError('Temperature must be finite and positive')
        p[n] = softmax(_vector(scores[n], size) / t)
    nq = [p[n] for n in ('N', 'Q') if n in p]
    q = np.mean(nq, axis=0) if nq else None
    if q is not None:
        q = q / q.sum()
    f = p.get('F')
    groups = [x for x in (q, f) if x is not None]
    p0 = np.mean(groups, axis=0)
    p0 = p0 / p0.sum()
    wf = 0.0 if f is None else 1.0 / len(groups)
    return p0, q, f, wf


def mean_view_scores(scores: Sequence[Sequence[float]]) -> np.ndarray:
    a = np.asarray(scores, np.float64)
    if a.ndim != 2 or len(a) not in (1, 2) or a.shape[1] < 2 or not np.isfinite(a).all():
        raise ValueError('Exactly one or two successful paired FULL views required')
    return a.mean(axis=0)


def update_probability(method: str, scores, temperatures, a, c) -> np.ndarray:
    if method not in NEW_METHODS:
        raise ValueError('Unknown fixed update rule')
    p0, q, f, wf = source_components(scores, temperatures)
    if p0 is None or f is None:
        raise ValueError('Common-domain updates require historical F')
    a, c = _vector(a, len(p0)), _vector(c, len(p0))
    tf = float(temperatures['F'])
    pa, pc = softmax(a / tf), softmax(c / tf)
    if method == 'SU07_GLOBAL_BLEND':
        beta = .5 * wf
        out = (1-beta)*p0 + beta*pa
    else:
        if method == 'SU04_F_REPLACE':
            dense = pa
        elif method == 'SU05_F_BLEND':
            dense = .5*f + .5*pa
        elif method == 'SU06_F_COARSE':
            dense = .5*f + .5*pc
        else:
            residual = (a-c)/tf
            if np.ptp(residual) == 0:
                return p0.copy()
            dense = softmax(log_softmax(_vector(scores['F'], len(p0))/tf) + .5*residual)
        out = dense if q is None else (1-wf)*q + wf*dense
    return out / out.sum()


@dataclass(frozen=True)
class Decision:
    class_id: int
    probability: np.ndarray | None
    changed: bool
    reason: str


def decide(method: str, ids, old_class: int, scores, temperatures, *,
           common_domain: bool, a=None, c=None, protected=False,
           parent_hard_class=None, parent_stable_class=None) -> Decision:
    ids = tuple(int(x) for x in ids)
    if len(ids) < 2 or len(set(ids)) != len(ids) or any(x <= 0 for x in ids) or old_class not in ids:
        raise ValueError('Original positive ordered vocabulary required')
    p0, _, _, _ = source_components(scores, temperatures)
    if p0 is not None and len(p0) != len(ids):
        raise ValueError('Vocabulary length mismatch')
    if not common_domain or protected:
        return Decision(old_class, p0, False, 'KEEP_OUTSIDE_COMMON_DOMAIN')
    if method in ('SU02_HARD_MATCHED', 'SU03_STABLE_MATCHED'):
        label = parent_hard_class if method == 'SU02_HARD_MATCHED' else parent_stable_class
        if label not in ids:
            raise ValueError('A validated historical decision is required')
        return Decision(label, None, label != old_class, 'MATCHED_PARENT_REPLAY')
    p = update_probability(method, scores, temperatures, a, c)
    label = ids[int(np.argmax(p))]
    return Decision(label, p, label != old_class, 'FULL_DISTRIBUTION_ARGMAX')


def target_flags(candidate, g1, d2, *, epsilon=1e-10):
    for r in (candidate, g1, d2):
        for cohort in ('replica8', 'scannet_cf18'):
            for metric in METRICS:
                v = r[cohort][metric]
                if v is None or not np.isfinite(v) or not 0 <= v <= 1:
                    raise ValueError('Finite complete unrounded fractions required')
    rep = all(candidate['replica8'][m] >= g1['replica8'][m]-epsilon for m in METRICS)
    cf = all(candidate['scannet_cf18'][m] >= g1['scannet_cf18'][m]-epsilon for m in METRICS)
    crossing = (candidate['scannet_cf18']['apall'] > d2['scannet_cf18']['apall']+epsilon
                and candidate['scannet_cf18']['ap50'] >= d2['scannet_cf18']['ap50']-epsilon)
    passed = rep and cf and crossing
    return {'replica_all5':rep, 'cf_all5':cf, 'CF_D2_crossing':crossing, 'target_met':passed,
            'material_target_met':passed and candidate['scannet_cf18']['apall']-d2['scannet_cf18']['apall'] >= .001-epsilon}
