"""Evaluation-side, scene-balanced scalar fitting on locked CAL evidence only."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

CAL_SCENES = ("scene0056_00", "scene0534_00")


def fit_temperature(examples, training_scenes, valid_ids):
    training_scenes = tuple(training_scenes)
    if not training_scenes or not set(training_scenes) <= set(CAL_SCENES):
        raise ValueError("temperature fitting accepts declared CAL scenes only")
    ids = tuple(map(int, valid_ids))
    rows = [
        row
        for row in examples
        if row["scene_id"] in training_scenes
        and row.get("available", True)
        and row.get("gt_label") in ids
    ]
    keys = [(row["scene_id"], row["owner_id"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("temperature examples must be unique object correspondences")
    scores = np.asarray([row["scores"] for row in rows], np.float64).reshape(
        -1, len(ids)
    )
    if not np.isfinite(scores).all():
        raise ValueError("temperature scores must be finite")
    targets = np.asarray([ids.index(row["gt_label"]) for row in rows], np.int64)
    groups = [
        np.asarray(
            [i for i, row in enumerate(rows) if row["scene_id"] == scene], np.int64
        )
        for scene in training_scenes
    ]
    groups = [group for group in groups if len(group)]

    def nll(log_temperature):
        logits = scores / np.exp(log_temperature)
        losses = logsumexp(logits, axis=1) - logits[np.arange(len(rows)), targets]
        return float(np.mean([losses[group].mean() for group in groups]))

    initial = nll(np.log(0.07)) if rows else None
    result = {
        "status": "UNCALIBRATED_DEFAULT_T",
        "temperature": 0.07,
        "training_scenes": list(training_scenes),
        "objects": len(rows),
        "classes": len({row["gt_label"] for row in rows}),
        "objects_per_scene": {
            scene: sum(row["scene_id"] == scene for row in rows)
            for scene in training_scenes
        },
        "example_ids": [[scene, owner] for scene, owner in keys],
        "nll_before": initial,
        "nll_after": initial,
        "bound_hit": None,
        "reason": "INSUFFICIENT_OBJECT_OR_CLASS_SUPPORT",
    }
    if len(rows) < 5 or result["classes"] < 2:
        return result
    bounds = tuple(np.log([0.01, 2.0]))
    try:
        fitted = minimize_scalar(
            nll, method="bounded", bounds=bounds, options={"xatol": 1e-4, "maxiter": 64}
        )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        return {**result, "reason": "OPTIMIZER_FAILURE", "error": str(error)}
    if (
        not fitted.success
        or not np.isfinite(fitted.fun)
        or not bounds[0] <= fitted.x <= bounds[1]
    ):
        return {
            **result,
            "reason": "OPTIMIZER_FAILURE",
            "optimizer_message": str(fitted.message),
        }
    return {
        **result,
        "status": "FITTED",
        "temperature": float(np.exp(fitted.x)),
        "nll_after": float(fitted.fun),
        "reason": None,
        "evaluations": int(fitted.nfev),
        "bound_hit": "LOWER"
        if abs(fitted.x - bounds[0]) <= 1e-4
        else "UPPER"
        if abs(fitted.x - bounds[1]) <= 1e-4
        else None,
    }
