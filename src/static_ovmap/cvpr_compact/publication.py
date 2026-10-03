"""Measured compact artifacts, primary evidence audit and normal branch publication."""

import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read

from .freezing import OWNED_PATHS, implementation_inventory, verified_validation
from .partial_execution import PARTIAL_LOCK, PARTIAL_RESULT, _pool_partition, semantic_block, validate_partial_scope
from .projected_views import _verified_identity
from .protocol import PACKAGE, experiment_matrix, load_spec
from .runtime import require_frozen_execution
from .tables import _complete, _evidence, collect_actual_physical_costs, main_tables, render_tables, result_store
from .timing import _validate_unmeasured_block, aggregate_timings, recovery_arm


RELEASE_ROOT = "artifacts/static_ovmap/cvpr_compact_tables_v1"
REPORT_NAMES = tuple("COMPACT_TABLES_" + name + ".md" for name in ("RESULTS", "HANDOFF", "SELECTION", "CLAIMS"))
LIMIT_BYTES = 50 * 1024**2
SECRET_KEYS = {"password", "api_key", "access_token", "refresh_token", "authorization", "credentials", "scannet_password"}


def requirement_catalog():
    items = []
    for filename in ("CODEX_FINAL_EXECUTION_EN.md", "IMPLEMENTATION_CONTRACTS.md", "TABLE_CONTRACTS.md", "SOURCE_EVIDENCE.md"):
        text = (PACKAGE / filename).read_text()
        heading, block, start = "", [], 0
        for number, line in enumerate([*text.splitlines(), ""], 1):
            if line.startswith("#"):
                heading = line
            if line.strip():
                if not block:
                    start = number
                block.append(line)
            elif block:
                quoted = "\n".join(block)
                items.append({"id": filename + ":" + str(start), "source": filename,
                              "line": start, "heading": heading, "requirement": quoted})
                block = []
    def leaves(value, prefix):
        if isinstance(value, dict):
            for key, child in value.items():
                leaves(child, prefix + "." + key)
        else:
            items.append({"id": prefix, "source": prefix.split(".")[0] + ".json", "requirement": value})
    for filename in ("PROTOCOL_SPEC", "TABLE_BINDINGS"):
        leaves(read(PACKAGE / (filename + ".json")), filename)
    result = {"requirements": items, "package": [{"path": path.name, "sha256": canonical_digest(read(path))}
        for path in (PACKAGE / "PROTOCOL_SPEC.json", PACKAGE / "TABLE_BINDINGS.json")]}
    result["identity"] = canonical_digest(result)
    return result


def _document(path, index, status=None):
    index.identity(path)
    row = read(path)
    _verified_identity(row)
    if status is not None and row["status"] != status:
        raise ValueError("incomplete actual publication evidence: " + str(path))
    for item in row.get("inputs", []) + row.get("outputs", []):
        index.identity(item["path"], item)
    return row


def verify_visual_qa(binding, table_receipt, index):
    root = Path(binding["output_root"])
    proof = _document(root / "validation/final_table_visual_qa.json", index, "PASS")
    if (proof["reviewer"] != "PRIMARY_CODEX" or not proof["actual_measured_pdf_inspected"]
            or proof["tables_identity"] != table_receipt["tables_identity"]
            or proof["table_receipt_identity"] != table_receipt["identity"]
            or proof["table_row_counts"] != {"table1": 6, "table2": 6, "table3": 4}
            or proof["minimum_body_font_pt"] < 8.5 or proof["overfull_boxes"] != 0
            or not proof["readable"] or not proof["rows_and_captions_unclipped"]):
        raise ValueError("final actual tables lack primary visual QA under the fixed layout contract")
    pdf = index.identity(root / "tables/table_layout_preview.pdf")
    if proof["pdf"] != pdf or not proof["rendered_pages"]:
        raise ValueError("visual QA PDF or inspected rendered pages differ from the measured tables")
    for item in proof["rendered_pages"]:
        index.identity(item["path"], item)
    return proof


def audit_measurements(binding, *, index=None):
    spec, root = load_spec(binding["spec"]), Path(binding["output_root"])
    revision = require_frozen_execution(binding, spec["cohorts"]["replica8"][0])
    index = index or ConsumptionIndex(root / "validation/input_verifications.json")
    inventory = implementation_inventory(binding, index=index)
    tests = verified_validation(binding, "final_contract_tests", inventory, index)
    smoke = verified_validation(binding, "final_smoke", inventory, index)
    matrix = experiment_matrix(spec)
    scene_receipts, locks, pools, anchor_proofs = {}, {}, {}, []
    for anchor in matrix["anchors"]:
        scene, cohort = anchor["scene"], anchor["cohort"]
        context = _document(root / "anchor_contexts" / (scene + ".json"), index)
        mapping_path = context["parent_map_receipt"]
        mapping = read(mapping_path)
        if mapping["status"] != "COMPLETE" or mapping["map_id"] != "BB00_NATIVE":
            raise ValueError("actual publication anchor is incomplete")
        capture = _document(context["capture_manifest"], index)
        if capture["scheduled_frame_ids"] != context["schedule"] or len(context["schedule"]) != 200 or capture["scene_id"] != scene:
            raise ValueError("actual publication anchor schedule or scene changed")
        anchor_proofs.append({**anchor, "context": index.identity(root / "anchor_contexts" / (scene + ".json")),
                              "map": index.identity(mapping_path), "capture": index.identity(context["capture_manifest"])})
        locks[scene] = _document(root / "predictions" / scene / "receipt.json", index)
        methods = {row["method"] for row in matrix["outputs"] if row["scene"] == scene}
        if locks[scene]["status"] == PARTIAL_LOCK:
            methods = set(validate_partial_scope(spec, locks[scene]))
            failed = _document(root / "recovery" / scene / "receipt.json", index, "FAILED")
            projected = _document(root / "projected_views" / scene / "receipt.json", index, "COMPLETE")
            manifest = _document(projected["manifest"], index)
            for arm, block in locks[scene]["blocked_sources"].items():
                if semantic_block(failed, manifest, recovery_arm(arm)) != block:
                    raise ValueError("publication block differs from the actual unchanged selected-mask failures")
        elif locks[scene]["status"] != "PREDICTIONS_LOCKED":
            raise ValueError("publication cannot infer blocked predictions from incomplete execution")
        if (locks[scene]["scene"] != scene or locks[scene]["cohort"] != cohort
                or set(locks[scene]["predictions"]) != methods or locks[scene]["output_count"] != len(methods)
                or not locks[scene]["baseline_array_and_rank_parity"]):
            raise ValueError("publication requires every independently available fixed prediction and unchanged A0")
        scene_receipts[scene] = _evidence(root / "evaluation" / scene / "receipt.json", index)
        if (scene_receipts[scene]["prediction_lock_identity"] != locks[scene]["identity"]
                or scene_receipts[scene].get("blocked_methods", {}) != locks[scene].get("blocked_methods", {})):
            raise ValueError("publication scoring differs from its exact locked method/blocked scope")
    partitions = {}
    for cohort in spec["cohorts"]:
        selected = {scene: scene_receipts[scene] for scene in spec["cohorts"][cohort]}
        methods, blocked = _pool_partition(spec, cohort, selected)
        manifest = _evidence(root / "pools" / cohort / "receipt.json", index)
        if (manifest["scene_order"] != spec["cohorts"][cohort] or set(manifest["methods"]) != set(methods)
                or manifest.get("blocked_methods", {}) != blocked):
            raise ValueError("publication cohort manifest changed actual full ordered pool availability")
        partitions[cohort] = (set(methods), blocked, manifest)
    for planned in matrix["pools"]:
        key = (planned["cohort"], planned["method"])
        if key[1] in partitions[key[0]][1]:
            if (root / "pools" / key[0] / (key[1] + ".json")).exists():
                raise ValueError("a blocked full pool must not have a substitute scientific receipt")
            continue
        pools[key] = _complete(root / "pools" / key[0] / (key[1] + ".json"), index)
        if pools[key] != partitions[key[0]][2]["methods"][key[1]]:
            raise ValueError("publication pool differs from its full-cohort manifest")
    actual_store = result_store(spec, scene_receipts, pools, root)
    store = _evidence(root / "tables/result_store.json", index)
    if actual_store != store:
        raise ValueError("published store differs from the actual full ordered scene/pool receipts")
    timings = []
    for planned in matrix["timings"]:
        row = _document(root / "timing" / planned["scene"] / planned["arm"] / "receipt.json", index)
        if row["status"] == "BLOCKED_UNMEASURED":
            _validate_unmeasured_block(row)
            block = locks[planned["scene"]].get("blocked_sources", {}).get(planned["arm"])
            if block != row["technical_block"]:
                raise ValueError("publication cold block differs from the actual scientific dependency")
        elif row["status"] != "COMPLETE":
            raise ValueError("publication requires every actual fixed cold observation or proved unmeasured block")
        timings.append(row)
    timing = _evidence(root / "timing/pool.json", index)
    aggregate = aggregate_timings(spec, timings)
    if any(timing[key] != aggregate[key] for key in aggregate if key != "identity"):
        raise ValueError("published timing pool differs from its24 fixed measured/blocked positions")
    diagnosis = _evidence(root / "diagnostics/replica8_recovery.json", index)
    all_diagnosis = _evidence(root / "diagnostics/receipt.json", index)
    if all_diagnosis["scene_order"] != [row["scene"] for row in matrix["anchors"]]:
        raise ValueError("publication diagnosis must include all26 scenes")
    from .diagnostics import aggregate_recovery_diagnostics
    diagnostics = {anchor["scene"]: _evidence(root / "diagnostics" / (anchor["scene"] + ".json"), index)
                   for anchor in matrix["anchors"]}
    if (all_diagnosis["scene_identities"] != {scene: row["identity"] for scene, row in diagnostics.items()}
            or diagnosis != aggregate_recovery_diagnostics(spec, [diagnostics[s] for s in spec["cohorts"]["replica8"]])):
        raise ValueError("published diagnosis differs from its full fixed post-lock scene evidence")
    reference = _document(root / "external/reference.json", index, "AUTHOR_REPORTED_NOT_REPRODUCED")
    tables = _evidence(root / "tables/tables_main.json", index)
    if tables != main_tables(spec, store, reference, diagnosis, timing, root):
        raise ValueError("actual table cells differ from their typed measured and attributed evidence")
    for filename, content in render_tables(tables).items():
        if (root / "tables" / filename).read_text() != content:
            raise ValueError("numeric LaTeX differs from its single result store")
    table_receipt = _evidence(root / "tables/receipt.json", index)
    visual = verify_visual_qa(binding, table_receipt, index)
    costs = _evidence(root / "tables/costs.json", index)
    expected_costs = {(row["scene"], row["method_id"]) for row in store["scene_metrics"] if row["status"] == "COMPLETE"}
    if ({(row["scene"], row["method_id"]) for row in costs["method_dependencies"]} != expected_costs
            or len(costs["method_dependencies"]) != len(expected_costs)):
        raise ValueError("every actual fixed prediction must retain its method dependency costs")
    expected_blocked = [{key: row[key] for key in ("scene", "cohort", "method_id", "status", "unavailable_reason",
        "blocked_condition", "technical_block", "receipt_path", "receipt_identity")}
        for row in store["scene_metrics"] if row["status"] == "BLOCKED_TECHNICAL"]
    if (costs.get("blocked_method_dependencies", []) != expected_blocked
            or costs["physical_payments"] != collect_actual_physical_costs(binding, [r["scene"] for r in matrix["anchors"]], index=index)):
        raise ValueError("published costs changed blocked dependencies or original physical worker payments")
    reports = _evidence(root / "reports/receipt.json", index)
    if reports["primary_method"] != "CT_A3_ER" or reports["deployment"] != "N0_UNCHANGED":
        raise ValueError("publication cannot change the fixed primary method or deployment")
    from .reports import analyze_results, render_reports
    report_revision = reports["freeze_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", report_revision):
        raise ValueError("reports must retain their actual full generation-freeze revision")
    repo = Path(binding["repository_root"])
    subprocess.run(["git", "merge-base", "--is-ancestor", report_revision, "HEAD"], cwd=repo, check=True)
    producer = index.identity(Path(__file__).with_name("reports.py"), reports["producer"])
    committed = subprocess.check_output(["git", "show", report_revision + ":src/static_ovmap/cvpr_compact/reports.py"], cwd=repo)
    if hashlib.sha256(committed).hexdigest() != producer["sha256"]:
        raise ValueError("report generator differs from its actual committed generation revision")
    analysis = _evidence(root / "reports/analysis.json", index)
    if analysis != analyze_results(spec, store, diagnosis, timing):
        raise ValueError("published analysis changed actual paired effects or unavailable comparisons")
    for name, content in render_reports(binding, store, diagnosis, timing, reference, analysis, report_revision,
                                         cost_data=costs, scene_diagnostics=diagnostics).items():
        path = Path(binding["repository_root"]) / "docs/paper/static_ovmap" / name
        index.identity(path)
        if path.read_text() != content:
            raise ValueError("published report differs from its measured typed evidence: " + name)
    blocked_dependencies = ["scene/" + row["scene"] + "/" + row["method_id"] for row in expected_blocked]
    blocked_dependencies += ["pool/" + row["cohort"] + "/" + row["method_id"]
        for row in store["pooled_metrics"] if row["status"] == "BLOCKED_TECHNICAL"]
    blocked_dependencies += ["timing/" + row["scene"] + "/" + row["arm"]
        for row in timings if row["status"] == "BLOCKED_UNMEASURED"]
    coverage = {"implementation_status": "IMPLEMENTATION_FROZEN", "scientific_status": store["status"],
        "anchor_count": len(anchor_proofs), "prediction_count": store["main_scene_outputs"]["complete"], "pool_count": len(pools),
        "main_scene_outputs": store["main_scene_outputs"], "internal_pools": store["internal_pools"],
        "cold_timing_leaves": {"complete": timing["leaf_count"], "blocked": timing.get("blocked_leaf_count", 0), "required": 24},
        "blocked_required_dependencies": blocked_dependencies, "scientific_complete": not blocked_dependencies,
        "objective_completion_status": "PENDING_PRIMARY_REQUIREMENT_AUDIT",
        "timing_status": timing["status"], "timing_count": timing["leaf_count"], "fixed_matrix_identity": matrix["identity"],
        "benchmark_coverage": reports["benchmark_coverage"],
        "external_comparison_provenance_status": reference["status"], "external_protocol_status": reference["external_protocol_status"],
        "external_full_scorer_protocol_verified": reference["full_scorer_protocol_independently_verified"],
        "performance_outcome": reports["performance_outcome"], "publication_status": "PENDING_PUSH_VERIFICATION",
        "primary_method": "CT_A3_ER", "deployment": "N0_UNCHANGED", "freeze_revision": revision,
        "focused_tests": tests, "existing_map_smoke": smoke, "visual_QA_identity": visual["identity"],
        "actual_anchor_proofs": anchor_proofs}
    coverage["identity"] = canonical_digest(coverage)
    atomic_write_json(root / "publication/scientific_coverage.json", coverage)
    index.write_memo(root / "validation/input_verifications.json")
    return coverage


def check_public_json(value):
    if isinstance(value, dict):
        if SECRET_KEYS & {key.lower() for key in value}:
            raise ValueError("compact publication contains an access-secret field")
        for child in value.values():
            check_public_json(child)
    elif isinstance(value, list):
        for child in value:
            check_public_json(child)
    elif isinstance(value, str) and re.search(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|://[^/\s:]+:[^/\s]+@)", value):
        raise ValueError("compact publication contains an apparent credential value")


def build_release(binding):
    root, repo = Path(binding["output_root"]), Path(binding["repository_root"])
    index = ConsumptionIndex(root / "validation/input_verifications.json")
    coverage = audit_measurements(binding, index=index)
    destination = repo / RELEASE_ROOT
    copied, retained = [], {}

    def copy(path, relative=None):
        path = Path(path)
        identity = index.identity(path)
        relative = relative or path.relative_to(root).as_posix()
        target = destination / relative
        if path.suffix == ".json":
            check_public_json(read(path))
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json" and identity["bytes"] > 65536:
            target = target.with_suffix(".json.gz")
            target.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            encoding = "gzip; original JSON bytes preserved exactly"
        else:
            shutil.copyfile(path, target)
            encoding = "IDENTITY"
        copied.append({"source": identity, "release": {**index.identity(target), "path": target.relative_to(repo).as_posix()},
                       "encoding": encoding})

    def keep(item):
        verified = index.identity(item["path"], item)
        retained[verified["path"]] = verified

    for filename in ("matrix.json", "resolved_inputs.json", "exposure_ledger.json"):
        copy(root / filename)
    for item in binding["inputs"]:
        keep(item)
    copy(root / "publication/scientific_coverage.json", "scientific_coverage.json")
    for directory in ("tables", "reports"):
        for path in sorted((root / directory).iterdir()):
            if path.suffix in (".json", ".csv", ".tex", ".pdf"):
                copy(path)
    copy(root / "external/reference.json")
    for filename in ("final_contract_tests.json", "final_smoke.json", "final_table_visual_qa.json"):
        copy(root / "validation" / filename)
    for scene in [row["scene"] for row in experiment_matrix(load_spec(binding["spec"]))["anchors"]]:
        context = read(root / "contexts" / (scene + ".json"))
        copy(root / "contexts" / (scene + ".json"))
        for name, source in context["sources"].items():
            copy(source["path"], "sources/" + scene + "/" + name + ".json")
        query_path = context.get("native_query_receipt") or str(Path(context["native_query_root"]) / "native_query_receipt.json")
        query = read(query_path)
        copy(query_path, "sources/" + scene + "/native_query_receipt.json")
        copy(query["query_decisions"]["path"], "sources/" + scene + "/query_decisions.json")
        for directory in ("projected_views", "recovery", "native_recovery", "predictions", "evaluation"):
            receipt = root / directory / scene / "receipt.json"
            copy(receipt)
            document = read(receipt)
            for item in document.get("inputs", []) + document.get("outputs", []):
                keep(item)
        projected = read(root / "projected_views" / scene / "receipt.json")
        copy(projected["manifest"], "projected_views/" + scene + "/manifest.json")
        regions = root / "recovery" / scene / "regions/receipt.json"
        if regions.is_file():
            copy(regions)
            fc = read(regions)
            for name, source in fc["sources"].items():
                copy(source["path"], "sources/" + scene + "/" + name + ".json")
        else:
            failed = read(root / "recovery" / scene / "receipt.json")
            locked = read(root / "predictions" / scene / "receipt.json")
            if failed.get("status") != "FAILED" or locked.get("status") != PARTIAL_LOCK or not locked.get("blocked_methods"):
                raise ValueError("missing regions receipt requires the actual proved all-failed scientific block")
        independent = root / "recovery" / (scene + "_U2") / "receipt.json"
        if independent.is_file():
            copy(independent)
            fc = read(independent)
            for name, source in fc["sources"].items():
                copy(source["path"], "sources/" + scene + "/INDEPENDENT_" + name + ".json")
            for item in fc.get("inputs", []) + fc.get("outputs", []):
                keep(item)
        native = read(root / "native_recovery" / scene / "receipt.json")
        copy(native["source"]["path"], "sources/" + scene + "/G1_NATIVE.json")
        evaluated = read(root / "evaluation" / scene / "receipt.json")
        copy(root / "predictions" / scene / "existing_D2_probability_audit.json")
        for row in evaluated["rows"]:
            method = row["method"]
            manifest = read(root / "predictions" / scene / method / "manifest.json")
            copy(root / "predictions" / scene / method / "manifest.json")
            scoring = read(row["evaluation_receipt"])
            copy(row["evaluation_receipt"], "evaluation/" + scene + "/" + method + "/scoring.json")
            for filename in ("matches.json.gz", "trace.json.gz"):
                copy(Path(scoring["manifest"]).with_name(filename), "evaluation/" + scene + "/" + method + "/" + filename)
            # Scoring views include actual positive-label decisions; unknown0 is
            # represented explicitly by the locked evaluation registry checks.
            decisions = {"scene": scene, "method": method, "prediction_key": manifest["prediction_key"],
                "record_key": manifest["record_key"], "instance_ranks": manifest["instance_ranks"],
                "all_positive_owner_labels": manifest["metadata"]["owner_semantic_decisions"],
                "official_view": scoring["view"], "registry_checks": evaluated["registry_checks"][method],
                "metadata": manifest["metadata"], "array_identities": {key: manifest[key] for key in ("owner_ids", "semantic_labels")}}
            check_public_json(decisions)
            path = root / "publication/decisions" / scene / (method + ".json")
            atomic_write_json(path, decisions)
            copy(path, "decisions/" + scene + "/" + method + ".json")
        copy(root / "diagnostics" / (scene + ".json"))
        if scene in binding["cohorts"]["scannet_cf18"]:
            for receipt in (root / "prepare" / scene / "receipt.json", root / "frontend" / scene / "cropformer/receipt.json",
                            root / "maps" / scene / "BB00_NATIVE/map_receipt.json", root / "readouts" / scene / "BB00_NATIVE/receipt.json",
                            root / "readouts" / scene / "BB00_NATIVE/fc/receipt.json"):
                copy(receipt)
    for cohort in binding["cohorts"]:
        for path in sorted((root / "pools" / cohort).glob("*.json")):
            if path.name != "input_verifications.json":
                copy(path)
    copy(root / "diagnostics/replica8_recovery.json")
    copy(root / "diagnostics/receipt.json")
    copy(root / "timing/pool.json")
    for planned in experiment_matrix(load_spec(binding["spec"]))["timings"]:
        path = root / "timing" / planned["scene"] / planned["arm"] / "receipt.json"
        copy(path)
        row = read(path)
        if row.get("status") == "BLOCKED_UNMEASURED":
            _validate_unmeasured_block(row)
            if (path.parent / "call/receipt.json").exists():
                raise ValueError("unreserved blocked timing cannot publish a physical call")
        else:
            copy(row["recovery_receipt"])
        for item in row["inputs"] + row["outputs"]:
            keep(item)
    costs = read(root / "tables/costs.json")
    for stage in costs["physical_payments"]["workers"]:
        receipt = stage["receipt"]
        copy(receipt["path"], "cost_receipts/" + receipt["sha256"] + ".json")
    reconstruction = {"status": "EXACT_SHARED_ARTIFACT_IDENTITIES", "artifacts": list(retained.values()),
        "rebuild_argv": ["/home/ww/miniconda3/envs/ovimap-map/bin/python", "scripts/evaluation/run_ovimap_cvpr_compact.py",
                         "--spec", "configs/static_ovmap/cvpr_compact_tables_v1.json", "--phase", "all", "--resume"],
        "cwd": str(repo), "freeze_revision": coverage["freeze_revision"], "path_maps": binding["path_map"],
        "instructions": "Restore the immutable raw/model/parent dependencies with these exact hashes. Use repeatable --path-map OLD=NEW for relocated roots. "
                        "The recorded production command reconstructs missing permitted semantic descendants with unchanged inputs. "
                        "Cold timings are original physical observations; restore their exact receipts and do not replay reserved measurements.",
        "cold_timings_are_observations_not_reconstructible_samples": True}
    check_public_json(reconstruction)
    atomic_write_json(destination / "shared_artifacts.json", reconstruction)
    catalog = requirement_catalog()
    atomic_write_json(destination / "validation/requirement_catalog.json", catalog)
    files = sorted(path for path in destination.rglob("*") if path.is_file() and path.name not in ("bundle.json", "primary_review.json"))
    total = sum(path.stat().st_size for path in files)
    if total >= LIMIT_BYTES:
        raise RuntimeError("actual compact release exceeds50MiB; preserve evidence and compact its representation")
    manifest = {"status": "COMPACT_RELEASE_BUILT", "scientific_coverage_identity": coverage["identity"],
        "files": [{**index.identity(path), "path": path.relative_to(repo).as_posix()} for path in files],
        "original_copies": copied, "bytes_excluding_bundle_and_primary_review": total,
        "limit_bytes": LIMIT_BYTES, "primary_review": "PENDING", "publication": "PENDING_PUSH_VERIFICATION"}
    manifest["identity"] = canonical_digest(manifest)
    atomic_write_json(destination / "bundle.json", manifest)
    index.write_memo(root / "validation/input_verifications.json")
    return manifest


def verify_primary_review(binding, bundle):
    repo, root = Path(binding["repository_root"]), Path(binding["output_root"])
    index = ConsumptionIndex(root / "validation/input_verifications.json")
    review = _document(root / "validation/primary_review.json", index)
    coverage = _document(root / "publication/scientific_coverage.json", index)
    actual_blocks = set(coverage.get("blocked_required_dependencies", []))
    catalog = requirement_catalog()
    expected = {row["id"] for row in catalog["requirements"]}
    if (review["reviewer"] != "PRIMARY_CODEX" or review["requirement_catalog_identity"] != catalog["identity"]
            or review["bundle_identity"] != bundle["identity"]
            or {row["requirement_id"] for row in review["requirements"]} != expected
            or len(review["requirements"]) != len(expected)):
        raise ValueError("primary review must cover every package requirement against the current full release")
    blocked_requirements, covered_blocks = set(), set()
    for row in review["requirements"]:
        if row["status"] not in ("PROVEN_COMPLETE", "BLOCKED_TECHNICAL") or not row["evidence"] or not row["finding"]:
            raise ValueError("uncertain, indirect or missing requirement evidence cannot pass publication")
        dependencies = set(row.get("blocked_dependencies", []))
        if row["status"] == "BLOCKED_TECHNICAL":
            if not dependencies or not dependencies <= actual_blocks:
                raise ValueError("blocked requirement must identify an actual audited fixed scientific dependency")
            blocked_requirements.add(row["requirement_id"])
            covered_blocks.update(dependencies)
        elif dependencies:
            raise ValueError("a proven complete requirement cannot contain unresolved dependencies")
        for item in row["evidence"]:
            index.identity(item["path"], item)
    if (set(review["unresolved_required_items"]) != blocked_requirements or covered_blocks != actual_blocks
            or len(review["unresolved_required_items"]) != len(blocked_requirements)):
        raise ValueError("primary review must retain every actual blocked requirement and scientific dependency")
    if blocked_requirements:
        if review["status"] != "PASS_WITH_TECHNICAL_BLOCKS" or review.get("objective_complete") is not False:
            raise ValueError("conditional publication must not claim overall objective completion")
    elif review["status"] != "PASS":
        raise ValueError("complete publication requires its explicit primary PASS review")
    expected_files = {item["path"] for item in bundle["files"]}
    expected_files.update(row["path"] for row in implementation_inventory(binding, index=index))
    expected_files.update("docs/paper/static_ovmap/" + name for name in REPORT_NAMES)
    reviewed = set()
    for row in review["reviewed_files"]:
        path = Path(row["path"])
        if not path.resolve().is_relative_to(repo.resolve()):
            raise ValueError("reviewed publication file is outside the task repository")
        index.identity(path, row)
        reviewed.add(path.resolve().relative_to(repo.resolve()).as_posix())
    if not expected_files <= reviewed:
        raise ValueError("primary review omitted actual implementation, reports or release files")
    return review


def verify_remote_sha(local, output, branch):
    rows = [line.split() for line in output.splitlines() if line.strip()]
    if (not re.fullmatch(r"[0-9a-f]{40}", local) or len(rows) != 1 or len(rows[0]) != 2
            or rows[0] != [local, "refs/heads/" + branch]):
        raise ValueError("full40 local and named remote branch SHAs do not agree")
    return local


def publish(binding):
    spec = load_spec(binding["spec"])
    repo, root = Path(binding["repository_root"]), Path(binding["output_root"])
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip() != spec["branch"]:
        raise ValueError("normal publication must use the specified research branch")
    bundle = build_release(binding)
    review = verify_primary_review(binding, bundle)
    destination = repo / RELEASE_ROOT
    atomic_write_json(destination / "validation/primary_review.json", review)
    if sum(path.stat().st_size for path in destination.rglob("*") if path.is_file()) >= LIMIT_BYTES:
        raise ValueError("full compact artifact set including audit exceeds50MiB")
    paths = [*OWNED_PATHS, RELEASE_ROOT, *["docs/paper/static_ovmap/" + name for name in REPORT_NAMES]]
    subprocess.run(["git", "add", "--", *paths], cwd=repo, check=True)
    subprocess.run(["git", "diff", "--cached", "--check", "--", *paths], cwd=repo, check=True)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *paths], cwd=repo).returncode
    if changed not in (0, 1):
        raise RuntimeError("cannot inspect the reviewed staged publication")
    if changed:
        subprocess.run(["git", "commit", "--only", "-m", "research: publish measured compact CVPR tables and primary audit", "--", *paths], cwd=repo, check=True)
    local = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    push = ["git", "push", "origin", "HEAD:refs/heads/" + spec["branch"]]
    remote = ["git", "ls-remote", "origin", "refs/heads/" + spec["branch"]]
    result = {"status": "PUSH_PENDING", "branch": spec["branch"], "local_sha": local,
        "remote_sha": None, "binding_identity": binding["identity"], "review_identity": review["identity"],
        "bundle_identity": bundle["identity"], "commands": [push, remote], "normal_push_without_force": True,
        "post_push_receipt_is_external": True, "retry_command": " ".join(push), "started_at_unix": time.time()}
    path = root / "publication/final.json"
    atomic_write_json(path, result)
    try:
        subprocess.run(push, cwd=repo, check=True, capture_output=True, text=True)
        found = subprocess.run(remote, cwd=repo, check=True, capture_output=True, text=True)
        result.update(status="PUSH_VERIFIED", remote_sha=verify_remote_sha(local, found.stdout, spec["branch"]))
    except (subprocess.CalledProcessError, ValueError) as exc:
        result.update(status="PUSH_FAILED", error=str(exc), command_stderr=getattr(exc, "stderr", None))
        raise
    finally:
        result["identity"] = canonical_digest({k: v for k, v in result.items() if k != "identity"})
        atomic_write_json(path, result)
    return result
