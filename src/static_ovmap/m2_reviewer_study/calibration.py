"""CAL-only shared source-NLL and single-source cosine calibration."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from src.static_ovmap.composition_study.temperature import CAL_SCENES


def fit_shared(examples_by_source, training_scenes, valid_ids):
    scenes, ids = tuple(training_scenes), tuple(valid_ids)
    if not scenes or not set(scenes) <= set(CAL_SCENES):
        raise ValueError("only declared CAL scenes may fit temperatures")
    if not examples_by_source:
        raise ValueError("shared fit requires participating sources")
    arrays, info = {}, {}
    for name, examples in examples_by_source.items():
        rows = [r for r in examples if r["scene_id"] in scenes and r.get("available", True) and r["gt_label"] in ids]
        keys = [(r["scene_id"], r["owner_id"]) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate calibration object")
        scores = np.asarray([r["scores"] for r in rows], np.float64).reshape(-1, len(ids))
        targets = np.asarray([ids.index(r["gt_label"]) for r in rows], np.int64)
        groups = [np.asarray([i for i, r in enumerate(rows) if r["scene_id"] == scene], np.int64) for scene in scenes]
        arrays[name] = (scores, targets, [g for g in groups if len(g)])
        info[name] = {"objects": len(rows), "classes": len({r["gt_label"] for r in rows}), "example_ids": [list(k) for k in keys],
                      "objects_per_scene": {s: sum(r["scene_id"] == s for r in rows) for s in scenes}}

    def losses(log_t):
        values = {}
        for name, (scores, targets, groups) in arrays.items():
            logits = scores / np.exp(log_t)
            nll = logsumexp(logits, axis=1) - logits[np.arange(len(targets)), targets]
            values[name] = float(np.mean([nll[g].mean() for g in groups])) if groups else None
        return values

    result = {"status": "UNCALIBRATED_DEFAULT_T", "temperature": .07, "training_scenes": list(scenes), "sources": info,
              "objective": "equal_mean_of_each_source_scene_balanced_NLL", "nll_before": None, "nll_after": None,
              "bound_hit": None, "reason": "INSUFFICIENT_OBJECT_OR_CLASS_SUPPORT"}
    if any(not np.isfinite(a[0]).all() for a in arrays.values()):
        return {**result, "reason": "NONFINITE_INPUT"}
    before = losses(np.log(.07))
    result["source_nll_before"] = before
    result["nll_before"] = result["nll_after"] = float(np.mean(list(before.values()))) if all(v is not None for v in before.values()) else None
    if any(v["objects"] < 5 or v["classes"] < 2 for v in info.values()):
        return result
    bounds = np.log([.01, 2.])
    try:
        fitted = minimize_scalar(lambda t: np.mean(list(losses(t).values())), method="bounded", bounds=bounds, options={"maxiter": 64, "xatol": 1e-4})
    except (RuntimeError, ValueError, FloatingPointError) as error:
        return {**result, "reason": "OPTIMIZER_FAILURE", "error": str(error)}
    if not fitted.success or not np.isfinite(fitted.fun) or not bounds[0] <= fitted.x <= bounds[1]:
        return {**result, "reason": "OPTIMIZER_FAILURE", "optimizer_message": str(fitted.message)}
    return {**result, "status": "FITTED", "temperature": float(np.exp(fitted.x)), "nll_after": float(fitted.fun),
            "source_nll_after": losses(fitted.x), "reason": None, "evaluations": int(fitted.nfev),
            "optimizer_message": str(fitted.message), "bound_hit": "LOWER" if abs(fitted.x - bounds[0]) <= 1e-4 else "UPPER" if abs(fitted.x - bounds[1]) <= 1e-4 else None}
