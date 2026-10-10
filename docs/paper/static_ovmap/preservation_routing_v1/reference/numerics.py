"""Small reference arithmetic, NOT the production FC/MA/readout implementation."""
from __future__ import annotations
import math
import numpy as np


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norm <= 1e-12):
        raise ValueError('finite nonzero vectors required')
    return x / norm


def residual(teacher: np.ndarray, current: np.ndarray, initial: np.ndarray) -> np.ndarray:
    return unit(np.asarray(teacher) + (np.asarray(current) - np.asarray(initial)))


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    shifted = x - x.max(axis=-1, keepdims=True)
    value = np.exp(shifted)
    return value / value.sum(axis=-1, keepdims=True)


def absolute_route(energy: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count-normalized null route; J counts valid groups, NOT nonzero rho."""
    energy = np.asarray(energy, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    if energy.ndim != 2 or rho.shape != (energy.shape[1],):
        raise ValueError('query-by-group energies and one rho per group required')
    if not np.isfinite(energy).all() or not np.isfinite(rho).all() or np.any((rho < 0) | (rho > 1)):
        raise ValueError('finite energy and rho in [0,1] required')
    if not energy.shape[1]:
        return np.ones(energy.shape[0]), np.zeros_like(energy)
    log_rho = np.full_like(rho, -np.inf)
    np.log(rho, out=log_rho, where=rho > 0)
    logits = energy + log_rho - math.log(energy.shape[1])
    joined = np.concatenate((np.zeros((len(energy), 1)), logits), axis=1)
    values = softmax(joined)
    return values[:, 0], values[:, 1:]


def delta_output(base: np.ndarray, group_values: np.ndarray, alpha: np.ndarray,
                 linear: np.ndarray) -> np.ndarray:
    """No intermediate normalization or bias: null mass is not cancelled."""
    base = np.asarray(base, np.float64)
    pooled = (np.asarray(alpha) @ np.asarray(group_values)).mean(0)
    return unit(base + .5 * (np.asarray(linear) @ pooled))


def r_gate(student: dict, teacher: dict, net_corrections: int, tolerance: float = 1e-10) -> bool:
    return (student['A'] >= teacher['A'] + .005 - tolerance
            and student['M'] >= teacher['M'] - tolerance
            and student['C'] >= teacher['C'] - tolerance
            and net_corrections > 0)


def choose_checkpoint(rows: list[dict], tolerance: float = 1e-10) -> dict | None:
    """Sort by predeclared criteria, discarding untrained step zero."""
    best = None
    directions = [('A', 1), ('M', 1), ('C', 1), ('CE', -1)]
    for row in sorted((r for r in rows if r['step'] >= 250), key=lambda r: r['step']):
        if best is None:
            best = row
            continue
        for key, direction in directions:
            change = direction * (row[key] - best[key])
            if abs(change) > tolerance:
                if change > 0:
                    best = row
                break
    return best
