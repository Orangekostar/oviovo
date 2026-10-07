"""One retrospective uniform ID and prespecified bounded paired timing slate."""

from pathlib import Path

from static_ovmap.cvpr_compact.area_fallback_experiment import seal
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.recovery_wave2.binding import read

from .verifier import target_flags, choose_method


def select(binding, root):
    root = Path(root)
    store = read(root/"result_store.json")
    _verified_identity(store)
    if store["scene_method_coverage"] != 234 or store["full_cohort_pool_coverage"] != 18:
        raise ValueError("selection requires all nine complete outputs on both full cohorts")
    spec = binding["specification"]
    records = {m["id"]: {c: next(x["metrics"] for x in store["pooled_metrics"] if x["method"] == m["id"] and x["cohort"] == c)
        for c in binding["cohorts"]} for m in spec["methods"]}
    flags = {name: target_flags(values, records["EV01_G1_V2"], records["EV00_D2"]) for name, values in records.items()}
    chosen = choose_method(records, spec["selection"]["simplicity_order"])
    passing = [name for name, flag in flags.items() if flag["target_met"]]
    timing_candidate = chosen if passing else spec["timing_no_pass_priority"][0]
    timing_methods = list(dict.fromkeys(["EV01_G1_V2", timing_candidate, *spec["timing_comparators"][timing_candidate]]))[:4]
    result = seal({"status": "RESEARCH_SELECTION_FROZEN", "result_store_identity": store["identity"],
        "recommended_method": chosen, "passing_methods": passing, "target_flags": flags,
        "metrics": records, "timing_candidate": timing_candidate, "timing_methods": timing_methods,
        "timing_scene_order_round1": binding["cohorts"]["replica8"],
        "timing_scene_order_round2": list(reversed(binding["cohorts"]["replica8"])),
        "timing_method_order_round2": list(reversed(timing_methods)), "exposed_cohorts": True,
        "independent_confirmation": False, "deployment": "N0_UNCHANGED"})
    path = root/"selection.json"
    if path.exists():
        previous = read(path)
        _verified_identity(previous)
        if previous != result:
            raise ValueError("frozen selection changed; preserve original selection and disclose a patch")
    else:
        atomic_write_json(path, result)
    print("RESEARCH_SELECTION", chosen, "target passing", passing, "timing", timing_methods, flush=True)
    return result
