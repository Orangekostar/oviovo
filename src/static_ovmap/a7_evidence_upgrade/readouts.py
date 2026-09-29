"""Protocol-fixed final Q readouts; no controller, selection or image inference."""

import numpy as np


def query_readout(features, areas, mode):
    features = np.asarray(features, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)
    if features.ndim != 2 or not len(features) or not features.shape[1] or areas.shape != (len(features),):
        raise ValueError("readout requires nonempty aligned retained features and areas")
    if not np.isfinite(features).all() or not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError("retained inputs must be finite with positive original areas")
    norms = np.linalg.norm(features, axis=1)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("retained input has invalid zero or nonfinite direction")
    original = np.average(features, axis=0, weights=areas)
    original_norm = np.linalg.norm(original)
    if not np.isfinite(original_norm) or original_norm <= 0:
        raise ValueError("original available area aggregate has no valid direction")
    original /= original_norm
    vectors = features / norms[:, None]
    n = len(features)
    weights = np.full(n, 1 / n, dtype=np.float64)
    iterations, converged, small_sample = 0, True, None
    if mode == "RAW_EQ":
        result = features.mean(axis=0)
    elif mode == "UNIT_EQ":
        result = vectors.mean(axis=0)
    elif mode == "GMED":
        result = vectors.mean(axis=0)
        if n <= 2:
            small_sample = "ONE_VIEW_DIRECTION" if n == 1 else "TWO_VIEW_MIDPOINT"
        else:
            converged = False
            for iterations in range(1, 65):
                distances = np.sqrt(np.square(vectors - result).sum(axis=1) + 1e-8)
                weights = 1 / distances
                weights /= weights.sum()
                updated = np.sum(vectors * weights[:, None], axis=0)
                movement = float(np.linalg.norm(updated - result))
                result = updated  # Accept first, then test; never project inside the loop.
                if movement <= 1e-8:
                    converged = True
                    break
    else:
        raise ValueError(f"unknown protocol readout: {mode}")
    norm = float(np.linalg.norm(result))
    fallback = norm <= 1e-12
    direction = original.copy() if fallback else result / norm
    return direction, {"mode": mode, "retained_views": n, "iterations": iterations,
        "converged": converged, "small_sample": small_sample,
        "epsilon": 1e-4 if mode == "GMED" else None,
        "pre_normalization_norm": norm,
        "numerical_fallback": "ORIGINAL_AREA_DIRECTION" if fallback else None,
        "original_view_norms": norms.tolist(), "view_norm_min": float(norms.min()),
        "view_norm_max": float(norms.max()), "view_norm_std_population": float(norms.std()),
        "original_area_weights": (areas / areas.sum()).tolist(),
        "influence_weights": weights.tolist(), "effective_sample_size": float(1 / np.square(weights).sum()),
        "influence_domain": "raw_features" if mode == "RAW_EQ" else "unit_features",
        "influence_is_correctness_probability": False, "image_forwards": 0}
