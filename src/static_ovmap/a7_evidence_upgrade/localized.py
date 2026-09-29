"""Fixed E04 candidate and common-view decision semantics; no learned gates."""

import hashlib

import numpy as np


def freeze_candidates(scene, valid_ids, sources, labels, details, views):
    order = {c: i for i, c in enumerate(valid_ids)}
    eligible = []
    for owner, winner in labels.items():
        selected_views = views.get(f"owner:{owner}", [])
        if not selected_views:
            continue
        if len(selected_views) > 3:
            raise ValueError("original static view cap exceeded")
        genuine = []
        for name in ("N0", "Q_GAIN", "S_SIGLIP2_AREA"):
            row = sources[name]["objects"][str(owner)]
            if row["available"] and row["label"] in order:
                genuine.append(row["label"])
        if len(genuine) < 2 or len(set(genuine)) < 2:
            continue
        if winner not in order:
            raise ValueError("eligible A7 owner has no positive winner")
        classes = sorted(set(genuine + [winner]), key=order.__getitem__)
        full = np.asarray(details[owner]["probabilities"], np.float64)
        if full.shape != (len(valid_ids),) or not np.isfinite(full).all() or np.any(full < 0):
            raise ValueError("actual A7 distribution missing")
        p = full[[order[c] for c in classes]]
        if p.sum() <= 0:
            raise ValueError("A7 shortlist has zero mass")
        p /= p.sum()
        if classes[int(p.argmax())] != winner:
            raise ValueError("shortlist does not preserve A7 winner")
        key = hashlib.sha256(f"{scene}|AW_E04|{owner}".encode()).hexdigest()
        eligible.append({"owner_id": owner, "hash": key, "candidates": classes,
                         "request_ids": list(selected_views), "a7_winner": winner,
                         "a7_restricted_probabilities": p.tolist(), "original_source_labels": genuine})
    eligible.sort(key=lambda r: (r["hash"], r["owner_id"]))
    return {"eligible_count": len(eligible), "selected": eligible[:64], "cap": 64,
            "selection_basis": "original_sources_and_static_views_only"}


def decide(candidates, winner, probabilities, supports, areas):
    """Each support row contains all class scores, or None for failed calls."""
    p, weights = np.asarray(probabilities, np.float64), np.asarray(areas, np.float64)
    if len(candidates) != len(set(candidates)) or winner not in candidates:
        raise ValueError("invalid frozen shortlist")
    if p.shape != (len(candidates),) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("invalid restricted A7 probabilities")
    if weights.shape != (len(supports),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("invalid original visibility weights")

    def choose(scores):
        tied = [c for c, score in zip(candidates, scores, strict=True) if score == max(scores)]
        return winner if winner in tied else tied[0]

    shortlist = choose(p)
    if shortlist != winner:
        raise ValueError("shortlist changed base decision")
    common, rows = [], []
    for i, view in enumerate(supports):
        if len(view) != len(candidates):
            raise ValueError("candidate support row is incomplete")
        if any(v is None for v in view):
            continue
        values = np.asarray(view, np.float64)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("invalid score times IoU support")
        common.append(i)
        rows.append(values)
    result = {"SHORTLIST": winner, "SPATIAL": winner, "MIX50": winner,
              "common_view_indices": common, "q": None, "status": "NO_LOCALIZED_EVIDENCE"}
    if not rows:
        return result
    evidence = np.average(np.stack(rows), axis=0, weights=weights[common])
    result["evidence"] = evidence.tolist()
    if evidence.sum() <= 1e-12:
        return result
    q = evidence / evidence.sum()
    result.update(SPATIAL=choose(q), MIX50=choose((p + q) / 2), q=q.tolist(), status="LOCALIZED_EVIDENCE")
    return result
