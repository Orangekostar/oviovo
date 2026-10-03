"""Commit a complete, tested experiment before main semantic execution."""

from pathlib import Path
import subprocess
import time

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read

from .projected_views import _verified_identity
from .protocol import PACKAGE, experiment_matrix, load_spec
from .runtime import FREEZE_FILE, require_frozen_execution


OWNED_PATHS = (
    "src/static_ovmap/cvpr_compact", "scripts/evaluation/run_ovimap_cvpr_compact.py",
    "tests/evaluation/test_cvpr_compact_contracts.py", "configs/static_ovmap/cvpr_compact_tables_v1.json",
    "docs/paper/static_ovmap/cvpr_compact_tables_v1",
    "docs/superpowers/plans/2026-10-03-cvpr-compact-tables.md",
    "docs/superpowers/specs/2026-10-03-cvpr-compact-tables-design.md",
)


def implementation_inventory(binding, *, index=None):
    from .workflow import _require_complete_implementation

    repo = Path(binding["repository_root"])
    _require_complete_implementation(repo)
    index = index or ConsumptionIndex()
    paths = sorted((repo / "src/static_ovmap/cvpr_compact").glob("*.py"))
    paths += [repo / OWNED_PATHS[i] for i in (1, 2, 3)]
    paths += sorted(path for path in PACKAGE.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    return [{**index.identity(path), "path": path.relative_to(repo).as_posix()} for path in paths]


def verified_validation(binding, name, inventory, index):
    root = Path(binding["output_root"])
    path = root / "validation" / (name + ".json")
    index.identity(path)
    proof = read(path)
    _verified_identity(proof)
    if proof["implementation_sources"] != inventory:
        raise ValueError("validation source inventory is stale: " + name)
    if name == "final_contract_tests":
        if (proof["status"] != "PASS" or proof["stage"] != "FINAL_PRODUCTION_VALIDATION"
                or proof["exit_code"] != 0 or not proof["tests_passed"]
                or proof["argv"][-3:] != ["pytest", "-q", "tests/evaluation/test_cvpr_compact_contracts.py"]):
            raise ValueError("final focused production validation is missing or failed")
    elif (proof["status"] != "COMPLETE" or proof["scene"] != load_spec(binding["spec"])["smoke_scene"]
            or proof["logical_condition_count"] != 8 or proof["new_development_map_count"] != 0
            or proof["main_performance_outcomes_seen"]):
        raise ValueError("the final smoke must reuse only the existing eight-condition development map")
    for item in proof.get("inputs", []) + proof.get("outputs", []):
        index.identity(item["path"], item)
    return index.identity(path)


def main_outcomes_exist(binding):
    root = Path(binding["output_root"])
    spec = load_spec(binding["spec"])
    scenes = [row["scene"] for row in experiment_matrix(spec)["anchors"]]
    return any((root / directory / scene / "receipt.json").exists()
               for scene in scenes for directory in ("predictions", "evaluation", "recovery", "native_recovery")) or any(
                   (root / "readouts" / scene / "BB00_NATIVE/receipt.json").exists() for scene in scenes) or any(
                   (root / "timing").glob("*/CT_*/receipt.json"))


def freeze_record(binding, *, amendment=None):
    from .native_region_worker import NATIVE_VL_SHA256

    _verified_identity(binding)
    spec = load_spec(binding["spec"])
    repo, root = Path(binding["repository_root"]), Path(binding["output_root"])
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip() != spec["branch"]:
        raise ValueError("implementation freeze must use the specified isolated branch")
    subprocess.run(["git", "merge-base", "--is-ancestor", spec["base_commit"], "HEAD"], cwd=repo, check=True)
    seen = main_outcomes_exist(binding)
    if seen and amendment is None:
        raise RuntimeError("main semantic work exists before an initial freeze; cannot certify preregistration")
    if amendment is not None:
        if not amendment.get("reason") or not amendment.get("affected_descendants") or amendment["main_outcomes_seen"] != seen:
            raise ValueError("technical amendment must state its reason, actual outcome exposure and affected descendants")
        for relative in amendment["affected_descendants"]:
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or path.parts[0] not in (
                    "readouts", "contexts", "projected_views", "recovery", "native_recovery", "predictions", "evaluation", "pools", "diagnostics", "tables", "reports"):
                raise ValueError("amendment may only name affected semantic descendants; reserved timings cannot be replayed")
    index = ConsumptionIndex(root / "validation/input_verifications.json")
    inventory = implementation_inventory(binding, index=index)
    tests = verified_validation(binding, "final_contract_tests", inventory, index)
    smoke = verified_validation(binding, "final_smoke", inventory, index)
    anchors, sequences = [], []
    for row in experiment_matrix(spec)["anchors"]:
        scene = row["scene"]
        context_path = root / "anchor_contexts" / (scene + ".json")
        context = read(context_path)
        _verified_identity(context)
        capture = read(context["capture_manifest"])
        _verified_identity(capture)
        mapping = read(context["parent_map_receipt"])
        if (mapping["status"] != "COMPLETE" or mapping["map_id"] != "BB00_NATIVE"
                or len(capture["scheduled_frame_ids"]) != 200
                or capture["scheduled_frame_ids"] != context["schedule"]
                or capture["scene_id"] != scene or not capture["completed_frame_ids"]):
            raise ValueError("freeze needs its complete, exact same-scene BB00 anchor: " + scene)
        anchors.append({"scene": scene, "context": index.identity(context_path),
                        "capture": index.identity(context["capture_manifest"]), "map": index.identity(context["parent_map_receipt"])})
        sequences.append({**row, "scheduled_frame_ids": capture["scheduled_frame_ids"],
                          "completed_frame_ids": capture["completed_frame_ids"]})
    reference = read(root / "external/reference.json")
    _verified_identity(reference)
    if not reference["all_30_pdf_values_agree"] or reference["provided_file_overwritten"]:
        raise ValueError("freeze requires the explicit verified external PDF choice and preserved original transcription")
    models = {scene: {"native": row["models"]["native"], "checkpoint": row["checkpoint"],
                     "FC_physical_model_identity": row["FC_physical_model_identity"], "FC_text": row["FC_text"],
                     "native_extension": row["native_extension"]}
              for scene, row in binding["scenes"].items()}
    result = {"status": "IMPLEMENTATION_FROZEN", "task_id": spec["task_id"], "branch": spec["branch"],
        "binding_identity": binding["identity"], "spec_sha256": binding["spec_identity"]["sha256"],
        "matrix_identity": experiment_matrix(spec)["identity"], "primary_method": "CT_A3_ER",
        "sequences": sequences, "methods": spec["methods"], "models": models,
        "FC": binding["fc"], "FC_asset_receipt": index.identity(binding["fc"]["receipt"]),
        "bound_input_manifest": index.identity(root / "resolved_inputs.json"), "native_vl_sha256": NATIVE_VL_SHA256,
        "numerical_contracts": {key: spec[key] for key in ("frame_sampling", "geometry", "models", "recovery", "projection", "evaluation", "timing")},
        "final_temperatures": binding["final_temperatures"], "data_exposure": binding["exposure_ledger"],
        "effective_producer_sources": binding["effective_producer_sources"], "anchor_inputs": anchors,
        "validation": {"focused_tests": tests, "existing_map_smoke": smoke},
        "external_reference": index.identity(root / "external/reference.json"),
        "implementation_sources": inventory, "deployment": "N0_UNCHANGED",
        "main_outcomes_seen": seen, "untouched_confirmation_claim": False,
        "amendment": amendment, "created_at_unix": time.time()}
    result["identity"] = canonical_digest(result)
    index.write_memo(root / "validation/input_verifications.json")
    return result


def ensure_freeze(binding, *, amendment=None):
    spec = load_spec(binding["spec"])
    repo = Path(binding["repository_root"])
    path = repo / FREEZE_FILE
    if path.is_file() and amendment is None:
        return {"status": "IMPLEMENTATION_FROZEN", "revision": require_frozen_execution(binding, spec["cohorts"]["replica8"][0])}
    if amendment is not None:
        previous = read(path)
        _verified_identity(previous)
        amendment = {**amendment, "previous_freeze_identity": previous["identity"],
                     "previous_revision": subprocess.check_output(["git", "log", "-1", "--format=%H", "--", FREEZE_FILE], cwd=repo, text=True).strip()}
        atomic_write_json(path.parent / ("previous_" + previous["identity"] + ".json"), previous)
    frozen = freeze_record(binding, amendment=amendment)
    atomic_write_json(path, frozen)
    owned = [*OWNED_PATHS, str(path.parent.relative_to(repo))]
    subprocess.run(["git", "add", "--", *owned], cwd=repo, check=True)
    subprocess.run(["git", "diff", "--cached", "--check", "--", *owned], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--only", "-m", "research: freeze complete compact-table experiment" if amendment is None
                    else "research: freeze documented compact-table technical amendment", "--", *owned], cwd=repo, check=True)
    revision = require_frozen_execution(binding, spec["cohorts"]["replica8"][0])
    return {"status": "IMPLEMENTATION_FROZEN", "revision": revision, "identity": frozen["identity"]}
