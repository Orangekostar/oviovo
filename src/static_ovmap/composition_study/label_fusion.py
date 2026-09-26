"""GT-free all-owner agreement and equal probability mixtures."""

import numpy as np
from scipy.special import softmax

SOURCES = ("N0", "Q_GAIN", "S_SIGLIP2_AREA")
METHODS = ("CP_M1_AGREE_KEEP", "CP_M2_EQUAL_RAW", "CP_M2_EQUAL_CAL")


def fuse_labels(native_labels, sources, valid_ids, method, temperatures=None):
    ids = tuple(map(int, valid_ids))
    if method not in METHODS or set(sources) != set(SOURCES):
        raise ValueError("unlisted fusion method or source set")
    if not ids or len(set(ids)) != len(ids) or min(ids) <= 0:
        raise ValueError("invalid class vocabulary")
    temperatures = (
        {name: 0.07 for name in SOURCES} if method != METHODS[2] else temperatures
    )
    if (
        temperatures is None
        or set(temperatures) != set(SOURCES)
        or any(not np.isfinite(value) or value <= 0 for value in temperatures.values())
    ):
        raise ValueError("source temperatures must be positive finite scalars")
    columns = {}
    for name in SOURCES:
        source_ids = tuple(map(int, sources[name]["valid_ids"]))
        if len(source_ids) != len(ids) or set(source_ids) != set(ids):
            raise ValueError("source vocabulary differs")
        columns[name] = [source_ids.index(value) for value in ids]
        if set(map(int, sources[name]["objects"])) != set(native_labels):
            raise ValueError("source owner registry differs")
    labels, decisions = {}, {}
    for owner, incumbent in sorted(native_labels.items()):
        available, scores, source_labels = [], {}, {}
        for name in SOURCES:
            objects = sources[name]["objects"]
            row = objects.get(owner, objects.get(str(owner)))
            source_labels[name] = int(row["label"])
            if not row["available"]:
                continue
            values = np.asarray(row["scores"], dtype=np.float64)
            if values.shape != (len(ids),) or not np.isfinite(values).all():
                raise ValueError(
                    "available source requires a complete finite score vector"
                )
            values = values[columns[name]]
            if row["label"] != ids[int(np.argmax(values))] or row["label"] <= 0:
                raise ValueError("source scores do not reproduce its positive label")
            available.append(name)
            scores[name] = values
        label, probabilities, weights = int(incumbent), None, {}
        if method == METHODS[0]:
            if not {"Q_GAIN", "S_SIGLIP2_AREA"} <= set(available):
                reason = "SOURCE_UNAVAILABLE"
            elif source_labels["Q_GAIN"] != source_labels["S_SIGLIP2_AREA"]:
                reason = "DISAGREEMENT_KEEP"
            else:
                label = source_labels["Q_GAIN"]
                reason = (
                    "AGREEMENT_UNCHANGED" if label == incumbent else "AGREEMENT_REPLACE"
                )
        elif not available:
            reason = "NO_AVAILABLE_SOURCE"
        else:
            probabilities = np.mean(
                [softmax(scores[name] / temperatures[name]) for name in available],
                axis=0,
            )
            probabilities /= probabilities.sum()
            label = ids[int(np.argmax(probabilities))]
            probabilities = probabilities.tolist()
            weights = {name: 1.0 / len(available) for name in available}
            reason = "EQUAL_AVAILABLE_PROBABILITY_MIXTURE"
        labels[owner] = label
        decisions[owner] = {
            "label": label,
            "native_label": int(incumbent),
            "changed": label != incumbent,
            "available_sources": available,
            "source_labels": source_labels,
            "source_weights": weights,
            "reason": reason,
            "probabilities": probabilities,
        }
    return labels, decisions
