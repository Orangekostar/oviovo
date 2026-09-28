"""Explicit source-subset ablations, independent of historical method guards."""

import numpy as np
from scipy.special import softmax


def fuse(native_labels, sources, valid_ids, selected, temperatures, *, hard=False, empty_class0=False):
    ids, selected = tuple(map(int, valid_ids)), tuple(selected)
    if not ids or len(set(ids)) != len(ids) or min(ids) <= 0:
        raise ValueError("invalid frozen vocabulary")
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(sources):
        raise ValueError("invalid selected source set")
    columns = {}
    for name in selected:
        source = sources[name]
        source_ids = tuple(map(int, source["valid_ids"]))
        if len(source_ids) != len(ids) or set(source_ids) != set(ids):
            raise ValueError("source vocabulary mismatch")
        if set(map(int, source["objects"])) != set(native_labels):
            raise ValueError("source owner registry mismatch")
        columns[name] = [source_ids.index(value) for value in ids]
        if not hard and (not np.isfinite(temperatures[name]) or temperatures[name] <= 0):
            raise ValueError("temperature must be finite and positive")
    labels, details = {}, {}
    for owner, incumbent in sorted(native_labels.items()):
        available, values, votes = [], [], []
        for name in selected:
            objects = sources[name]["objects"]
            row = objects.get(owner, objects.get(str(owner)))
            if not row["available"]:
                if row["scores"] is not None:
                    raise ValueError("unavailable source has fabricated scores")
                continue
            scores = np.asarray(row["scores"], np.float64)
            if scores.shape != (len(ids),) or not np.isfinite(scores).all():
                raise ValueError("available source requires complete finite scores")
            scores = scores[columns[name]]
            if ids[int(scores.argmax())] != row["label"]:
                raise ValueError("source scores do not reproduce label")
            available.append(name)
            votes.append(int(row["label"]))
            if not hard:
                values.append(softmax(scores / temperatures[name]))
        probabilities = None
        if not available:
            label = 0 if empty_class0 else int(incumbent)
        elif hard:
            counts = np.asarray([votes.count(label) for label in ids])
            tied = [label for label, count in zip(ids, counts, strict=True) if count == counts.max()]
            native_vote = votes[available.index("N0")] if "N0" in available else None
            label = native_vote if native_vote in tied else tied[0]
            probabilities = counts / counts.sum()
        else:
            probabilities = np.mean(values, axis=0)
            probabilities /= probabilities.sum()
            label = ids[int(probabilities.argmax())]
        labels[owner] = label
        details[owner] = {"label": label, "native_label": int(incumbent), "available_sources": available,
                          "probabilities": None if probabilities is None else probabilities.tolist(),
                          "source_labels": dict(zip(available, votes, strict=True)), "changed": label != incumbent}
    return labels, details
