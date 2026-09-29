"""Float64 probability operators with exact identity boundaries."""

import numpy as np
from scipy.special import log_softmax, softmax


def probability(value):
    p = np.asarray(value, np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("positive finite probability vector required")
    if not np.isclose(p.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("probability mass differs from one")
    return p


def pool(scores, temperatures, names):
    available = [name for name in names if scores.get(name) is not None]
    if not available:
        return None
    values = [softmax(np.asarray(scores[n], np.float64) / temperatures[n]) for n in available]
    p = np.mean(values, axis=0)
    return p / p.sum()


def candidate_set(base, paired_ovr):
    p = probability(base)
    indices = np.argsort(-p, kind="stable")[:5].tolist()
    if paired_ovr is not None:
        indices.append(int(np.argmax(paired_ovr)))
    return np.array(sorted(set(indices)), np.int64)


def paired_residual(fc, ovr, tf, to=None):
    if fc is None or ovr is None:
        return None
    f, o = np.asarray(fc, np.float64), np.asarray(ovr, np.float64)
    if f.shape != o.shape or not np.isfinite([tf, tf if to is None else to]).all() or tf <= 0:
        raise ValueError("invalid paired evidence")
    if to is None:
        return (o - f) / tf
    if to <= 0:
        raise ValueError("invalid O temperature")
    return log_softmax(o / to) - log_softmax(f / tf)


def ratio_update(base, residual, eta, indices=None):
    if base is None:
        return None
    p = probability(base)
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("invalid eta")
    if residual is None or eta == 0:
        return p.copy()
    r = np.asarray(residual, np.float64)
    ix = np.arange(len(p)) if indices is None else np.asarray(indices, np.int64)
    if r.shape != p.shape or not np.isfinite(r).all() or not len(ix):
        raise ValueError("invalid residual/candidate shape")
    if len(np.unique(ix)) != len(ix) or ix.min() < 0 or ix.max() >= len(p):
        raise ValueError("invalid candidate indices")
    if np.ptp(r[ix]) == 0:
        return p.copy()
    out = p.copy()
    mass = p[ix].sum()
    out[ix] = mass * softmax(np.log(p[ix]) + eta * r[ix])
    # Correct only inside C; never renormalize unchanged outside probabilities.
    out[ix[np.argmax(out[ix])]] += mass - out[ix].sum()
    return out


def blend(base, other, value):
    if not 0 <= value <= 1:
        raise ValueError("invalid interpolation parameter")
    if base is None or other is None or value == 0:
        return None if base is None else np.asarray(base, np.float64).copy()
    return (1 - value) * np.asarray(base) + value * np.asarray(other)


def simple_dependence(scores, temperatures, base, method):
    if base is None:
        return None
    names = [n for n in ('N', 'Q', 'F') if scores.get(n) is not None]
    z = np.mean([np.asarray(scores[n]) / temperatures[n] for n in names], axis=0)
    if method == 'PE_D1_LOGPOOL':
        return softmax(z)
    if method == 'PE_D1_ANCHORED':
        return softmax(.5 * (z + np.log(base)))
    if method == 'PE_D2_GROUPED':
        groups = [pool(scores, temperatures, group) for group in (('N', 'Q'), ('F',))]
        p = np.mean([g for g in groups if g is not None], axis=0)
        return p / p.sum()
    raise ValueError("unknown simple dependence operator")
