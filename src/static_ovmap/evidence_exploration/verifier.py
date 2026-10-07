"""Prespecified decisions and unrounded retrospective target gate."""

import numpy as np


METRICS = ("apall", "ap50", "ap25", "miou", "macc")
TOL = 1e-10


def decide(baseline, valid_ids, *, triggered, purity, target_scores=None,
           controls=(), decision="none", representation_fallback=None):
    ids = np.asarray(valid_ids, np.int64)
    if decision not in {"none", "margin", "contrast_margin"} or not 0 <= purity <= 1:
        raise ValueError("decision must follow the fixed experimental rule")
    out = {"feature_available": bool(baseline["available"]), "proposed_class": baseline["label"],
           "accepted": bool(baseline["available"]), "defer_reason": None,
           "raw_cosines": baseline["scores"], "decision_scores": baseline["scores"],
           "margin": None, "triggered": bool(triggered), "purity": float(purity),
           "baseline_vector_sha256": baseline.get("aggregate_vector_sha256"),
           "representation_fallback": representation_fallback}
    if not triggered or not baseline["available"] or representation_fallback is not None:
        if representation_fallback is not None and representation_fallback != "REPRESENTATION_UNAVAILABLE_KEEP_B1":
            raise ValueError("undeclared representation fallback")
        return out
    s = np.asarray(baseline["scores"] if target_scores is None else target_scores, np.float64)
    if s.shape != ids.shape or len(ids) < 2 or not np.isfinite(s).all():
        raise ValueError("target requires finite complete original-class-order cosines")
    r = s.copy()
    c = np.asarray(controls, np.float64)
    if c.size:
        if c.ndim != 2 or c.shape[1] != len(ids) or not np.isfinite(c).all():
            raise ValueError("controls require the same representation and complete vocabulary")
        if decision == "contrast_margin":
            b = c.max(axis=0)
            r -= .25 * (1 - purity) * (b - b.mean())
    ordered = np.sort(r)
    margin = float(ordered[-1] - ordered[-2])
    accepted = decision == "none" or margin >= .01
    out.update(raw_cosines=s.tolist(), decision_scores=r.tolist(), proposed_class=int(ids[int(r.argmax())]),
               feature_available=True, margin=margin, accepted=bool(accepted),
               defer_reason=None if accepted else "LOW_MARGIN")
    return out


def target_flags(candidate, b1, b0):
    for record in (candidate, b1, b0):
        for cohort in ("replica8", "scannet_cf18"):
            if set(METRICS) - set(record[cohort]):
                raise ValueError("target requires all five metrics on both complete cohorts")
            if any(record[cohort][m] is None or not np.isfinite(record[cohort][m])
                   or not 0 <= record[cohort][m] <= 1 for m in METRICS):
                raise ValueError("selection metrics require finite unrounded fractions")
    rep = all(candidate["replica8"][m] >= b1["replica8"][m] - TOL for m in METRICS)
    cf = all(candidate["scannet_cf18"][m] >= b1["scannet_cf18"][m] - TOL for m in METRICS)
    ap = (candidate["scannet_cf18"]["apall"] > b0["scannet_cf18"]["apall"] + TOL
          and candidate["scannet_cf18"]["ap50"] >= b0["scannet_cf18"]["ap50"] - TOL)
    passed = rep and cf and ap
    return {"replica_nondecrease_all5": rep, "cf_nondecrease_all5": cf, "cf_AP_conditions": ap,
            "target_met": passed, "material_target_met": passed and
            candidate["scannet_cf18"]["apall"] - b0["scannet_cf18"]["apall"] >= .001 - TOL}


def choose_method(records, simplicity_order):
    b1, b0 = records["EV01_G1_V2"], records["EV00_D2"]
    possible = [m for m in simplicity_order if m in records and m not in ("EV00_D2", "EV01_G1_V2")
                and target_flags(records[m], b1, b0)["target_met"]]
    if not possible:
        return "EV01_G1_V2"
    def values(method):
        r = records[method]
        return (r["scannet_cf18"]["apall"], r["scannet_cf18"]["ap50"],
                r["replica8"]["apall"], r["scannet_cf18"]["miou"])
    best = possible[0]
    for method in possible[1:]:
        for x, y in zip(values(method), values(best), strict=True):
            if x > y + TOL:
                best = method
                break
            if y > x + TOL:
                break
    return best
