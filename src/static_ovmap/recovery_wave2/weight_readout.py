"""Exact endpoint interpolation, including inherited missing-source behavior."""

import numpy as np


def interpolate_probabilities(equal, grouped, gamma):
    if gamma not in (1 / 3, .4, .45, .5):
        raise ValueError("gamma is outside the fixed recovery protocol")
    if (equal is None) != (grouped is None):
        raise ValueError("endpoint source availability differs")
    if equal is None:
        return None
    equal, grouped = np.asarray(equal, np.float64), np.asarray(grouped, np.float64)
    if equal.ndim != 1 or not len(equal) or equal.shape != grouped.shape:
        raise ValueError("endpoint probability shapes differ")
    for value in (equal, grouped):
        if (not np.isfinite(value).all() or np.any(value < 0)
                or not np.isclose(value.sum(), 1., rtol=0, atol=1e-12)):
            raise ValueError("endpoint probabilities must be finite normalized vectors")
    if gamma == 1 / 3 or np.array_equal(equal, grouped):
        return equal.copy()
    if gamma == .5:
        return grouped.copy()
    t = 6 * gamma - 2
    return (1 - t) * equal + t * grouped


def weight_decisions(equal, grouped, gamma, valid_ids, incumbent):
    owners = {str(owner) for owner in incumbent}
    if set(equal) != owners or set(grouped) != owners:
        raise ValueError("endpoint decisions do not cover the original owner registry")
    ids = np.asarray(valid_ids, np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or np.any(ids <= 0):
        raise ValueError("invalid frozen category order")
    labels, decisions = {}, {}
    for owner, native_label in incumbent.items():
        e, d = equal[str(owner)], grouped[str(owner)]
        if e["available_sources"] != d["available_sources"]:
            raise ValueError("endpoint source availability differs")
        probabilities = interpolate_probabilities(e["probabilities"], d["probabilities"], gamma)
        if probabilities is not None and len(probabilities) != len(ids):
            raise ValueError("endpoint probabilities leave the frozen vocabulary")
        label = int(native_label if probabilities is None else ids[int(probabilities.argmax())])
        labels[int(owner)] = label
        decisions[str(owner)] = {
            "gamma": gamma, "available_sources": list(e["available_sources"]),
            "probabilities": None if probabilities is None else probabilities.tolist(),
            "label": label, "all_unavailable_fallback": probabilities is None,
            "changed_label_vs_FC_EQ": label != e["label"],
            "changed_label_vs_D2": label != d["label"],
        }
    return labels, decisions
