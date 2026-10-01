"""Fixed common-availability exports isolate recovery coverage from class decisions."""

from pathlib import Path

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest

from .binding import read
from .evaluation import pool_cohort
from .light import build_scene


CONDITIONS = {"RW_DIAG_U2_COMMON": {"gamma": .5, "recovery": "U2"},
              "RW_DIAG_U3_COMMON": {"gamma": .5, "recovery": "U3"}}


def common_coverage(binding, cohort, *, workers=3):
    from .workflow import _light_eval_job, parallel_jobs

    root, scenes = Path(binding["output_root"]), binding["datasets"][cohort]
    subdir, scopes = "extra_conditions/DIAGNOSTIC_COMMON_COVERAGE", {}
    for scene in scenes:
        source_root = root / "recovery" / scene / "BB00_NATIVE"
        fc = read(source_root / "fc_recovery_receipt.json")
        two, three = (read(fc["sources"][arm]["path"])["objects"] for arm in ("U2", "U3"))
        scopes[scene] = sorted(int(owner) for owner in two if two[owner]["available"] and three[owner]["available"])
        destination = root / "light" / scene / "BB00_NATIVE" / subdir / "receipt.json"
        if destination.is_file():
            old = read(destination)
            if old["diagnostic_covered_owners"] != scopes[scene]:
                raise ValueError("locked same-coverage diagnostic changed its evidence-only scope")
        else:
            build_scene(binding, scene, conditions_override=CONDITIONS, output_subdir=subdir,
                        recovery_owner_filter=scopes[scene])
    parallel_jobs(_light_eval_job, [(binding, scene, "BB00_NATIVE", None, subdir) for scene in scenes], workers)
    pooled = pool_cohort(binding, cohort, scenes, list(CONDITIONS))
    result = {"status": "COMPLETE", "cohort": cohort, "scene_order": scenes, "covered_owners": scopes,
        "pools": pooled["methods"], "covered_owners_selected_without_GT": True,
        "new_image_encodings": 0, "new_region_poolings": 0, "selection_input": False,
        "purpose": "FIXED_U2_U3_COMMON_SOURCE_SUCCESS_TO_EXPOSE_CAP_OR_FAILURE_COVERAGE_DIFFERENCES"}
    result["identity"] = canonical_digest(result)
    atomic_write_json(root / "mechanisms" / (cohort + "_common_coverage.json"), result)
    return result
