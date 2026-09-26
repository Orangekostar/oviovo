"""New CAL-only composition gate and nomination; no old retention override."""

import math

from .temperature import CAL_SCENES

CONTROLS = ("N0", "Q_COMBINE", "Q_GAIN", "S_SIGLIP2_AREA")
COMPOSITIONS = (
    "CP_M1_AGREE_KEEP",
    "CP_M2_EQUAL_RAW",
    "CP_M2_EQUAL_CAL",
    "CP_M3_COMBINE_S2",
    "CP_M4_GAIN_S2",
    "CP_M5_MIX50_NATIVE",
    "CP_M6_MIX50_S2",
)
TOLERANCE = 1e-10


def cal_summaries(rows):
    seen = set()
    for row in rows:
        if row["role"] != "compose_cal" or row["scene_id"] not in CAL_SCENES:
            raise ValueError("nomination and gate consume CAL rows only")
        key = row["scene_id"], row["method_id"]
        if key in seen:
            raise ValueError("duplicate CAL method/scene record")
        seen.add(key)
    summaries = {}
    for method in (*CONTROLS, *COMPOSITIONS):
        selected = [row for row in rows if row["method_id"] == method]
        complete = len(selected) == 2 and all(
            row.get("status", "COMPLETE") == "COMPLETE"
            and all(
                row.get("metrics", {}).get(key) is not None
                and math.isfinite(row["metrics"][key])
                for key in ("uap", "miou")
            )
            for row in selected
        )
        summaries[method] = {
            "complete": complete,
            "scene_count": len(selected),
            "metrics": {
                key: sum(row["metrics"][key] for row in selected) / 2
                if complete
                else None
                for key in ("uap", "miou")
            },
            "changed_owners": sum(row.get("changed_owners", 0) for row in selected),
            "mean_logical_requests": sum(
                row.get("logical_requests", 0) for row in selected
            )
            / 2
            if complete
            else None,
        }
    return summaries


def weak_dom(left, right):
    if not left["complete"] or not right["complete"]:
        return False
    deltas = [left["metrics"][key] - right["metrics"][key] for key in ("uap", "miou")]
    return all(value >= -TOLERANCE for value in deltas) and any(
        value > TOLERANCE for value in deltas
    )


def m6_gate(rows):
    means = cal_summaries(rows)
    checks = {
        "M3_vs_COMBINE": weak_dom(means[COMPOSITIONS[3]], means["Q_COMBINE"]),
        "M4_vs_GAIN": weak_dom(means[COMPOSITIONS[4]], means["Q_GAIN"]),
        "M5_vs_COMBINE": weak_dom(means[COMPOSITIONS[5]], means["Q_COMBINE"]),
    }
    enabled = (checks["M3_vs_COMBINE"] or checks["M4_vs_GAIN"]) and checks[
        "M5_vs_COMBINE"
    ]
    return {
        "enabled": enabled,
        "status": "REQUIRED" if enabled else "NOT_REQUIRED_BY_COMPOSITION_CAL_GATE",
        "checks": checks,
        "inputs": means,
        "metric_tolerance": TOLERANCE,
        "role": "compose_cal",
    }


def _best(methods, means, *, require_intervention=False):
    winner = None
    for method in methods:
        row = means[method]
        if not row["complete"] or (require_intervention and row["changed_owners"] == 0):
            continue
        if winner is None:
            winner = method
            continue
        previous = means[winner]
        for value, old in (
            (row["metrics"]["uap"], previous["metrics"]["uap"]),
            (row["metrics"]["miou"], previous["metrics"]["miou"]),
            (-row["mean_logical_requests"], -previous["mean_logical_requests"]),
        ):
            if value > old + TOLERANCE:
                winner = method
                break
            if value < old - TOLERANCE:
                break
    return winner


def _recommendation(candidate, reference):
    if candidate is None or reference is None or not reference["complete"]:
        return "INCONCLUSIVE"
    if weak_dom(candidate, reference):
        return "MEAN_GAIN"
    deltas = [
        candidate["metrics"][key] - reference["metrics"][key] for key in ("uap", "miou")
    ]
    return "TRADEOFF" if any(value > TOLERANCE for value in deltas) else "NO_MEAN_GAIN"


def nominate(rows, *, m6_enabled):
    methods = COMPOSITIONS if m6_enabled else COMPOSITIONS[:-1]
    expected = {
        (scene, method) for scene in CAL_SCENES for method in (*CONTROLS, *methods)
    }
    if {(row["scene_id"], row["method_id"]) for row in rows} != expected:
        raise ValueError("nomination requires complete configured CAL record coverage")
    means = cal_summaries(rows)
    nominee = _best(methods, means, require_intervention=True)
    single = _best(CONTROLS, means)
    comparison = None if nominee is None else means[nominee]
    return {
        "nominee": nominee,
        "best_single_CAL": single,
        "cal_means": means,
        "status": "NOMINATED" if nominee else "NO_EFFECTIVE_INTERVENTION",
        "confirmation_required": nominee is not None,
        "confirmation_methods": [
            *CONTROLS,
            nominee,
            *([COMPOSITIONS[5]] if nominee == COMPOSITIONS[6] else []),
        ]
        if nominee
        else [],
        "calibration_recommendation": {
            "versus_N0": _recommendation(comparison, means["N0"]),
            "versus_best_single": _recommendation(
                comparison, None if single is None else means[single]
            ),
        },
        "metric_tolerance": TOLERANCE,
        "fixed_order": list(methods),
    }
