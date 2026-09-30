"""Measured compact release, four reports, and external verified-push receipt."""

from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from static_ovmap.m2_reviewer_study.binding import InputIndex
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from .binding import read


def net_gain(metrics, baseline):
    da, dm = 100 * (metrics["apall"] - baseline["apall"]), 100 * (metrics["miou"] - baseline["miou"])
    if da > 0 and dm > 0 and max(da, dm) > .05:
        return "MEASURED_NET_GAIN"
    if da * dm < 0:
        return "TRADEOFF"
    return "NO_MEASURED_NET_GAIN"


def _table(pools):
    lines = ["| Map | Readout | APall | AP50 | AP25 | mIoU | mAcc |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in sorted(pools, key=lambda r: (r["map_id"], r["method"])):
        metrics = row["metrics"]
        lines.append("| " + " | ".join([row["map_id"], row["method"], *[
            f'{100 * metrics[key]:.2f}' if metrics[key] is not None else "undefined"
            for key in ("apall", "ap50", "ap25", "miou", "macc")]]) + " |")
    return "\n".join(lines)


def render(binding, spec):
    root, repo = Path(binding["output_root"]), Path(binding["repository_root"])
    release = repo / spec["publication"]["repo_artifacts"]
    docs = repo / "docs/paper/static_ovmap"
    release.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    rows, pools, diagnostics, interventions, maps, costs = [], [], [], [], [], []
    for path in (root / "readouts").glob("*/*/evaluation_rows.json"):
        rows.extend(read(path)["rows"])
    for path in (root / "pools").glob("*/*/*/OFFICIAL_CURRENT_CLASS.json"):
        value = read(path)
        value.update(cohort=path.relative_to(root / "pools").parts[0], map_id=path.relative_to(root / "pools").parts[1])
        pools.append(value)
    for path in (root / "readouts").glob("*/*/raw_geometry_diagnostics.json"):
        diagnostics.append(read(path))
    for path in (root / "readouts").glob("*/*/intervention.json"):
        interventions.append(read(path))
    for path in (root / "maps").glob("*/*/map_receipt.json"):
        maps.append(read(path))
    for path in (root / "readouts").glob("*/*/receipt.json"):
        value = read(path)
        nq, fc = read(path.parent / "native_query/native_query_receipt.json"), read(path.parent / "fc/receipt.json")
        costs.append({"scene": value["scene"], "map_id": value["map_id"],
            "required_image_encodings": value["required_image_encodings"],
            "attributable_standalone_seconds": value["attributable_map_plus_readout_seconds"],
            "timing_missing_components": value["standalone_timing_missing_components"],
            "physical_native_crop_encodings": nq["physical_image_encodings"],
            "physical_FC_image_encodings": fc["physical_image_encodings"],
            "physical_FC_region_poolings": fc["physical_region_poolings"],
            "native_query_wall_seconds": nq["elapsed_seconds"], "FC_wall_seconds": fc["elapsed_seconds"],
            "Q_logical_attempts": nq["query_logical_ledger"]["attempts"],
            "source_availability": value["source_availability"], "cap_exclusions": value["fc_target_cap_exclusions"]})
    bridge = read(root / "bridge_parity.json") if (root / "bridge_parity.json").is_file() else {"status": "NOT_MEASURED"}
    selection = read(root / "selection.json") if (root / "selection.json").is_file() else None
    expected = len(spec["map_variants"]) * 4
    if selection:
        expected += 8 * len(selection["replica_recipes"]) + (4 if any(r["id"] == "BBX_COMPOSE" for r in selection["replica_recipes"]) else 0)
    success = [m for m in maps if m["status"] == "COMPLETE"]
    primary = [r for r in rows if r["rank_mode"] == "OFFICIAL_CURRENT_CLASS"]
    expected_pairs = {(scene, recipe["id"]) for scene in spec["datasets"]["development"] for recipe in spec["map_variants"]}
    if selection:
        expected_pairs |= {(scene, recipe["id"]) for scene in spec["datasets"]["replica"] for recipe in selection["replica_recipes"]}
        if any(r["id"] == "BBX_COMPOSE" for r in selection["replica_recipes"]):
            expected_pairs |= {(scene, "BBX_COMPOSE") for scene in spec["datasets"]["development"]}
    measured_pairs = {(m["scene"], m["map_id"]) for m in success}
    expected_rows = {(scene, map_id, method) for scene, map_id in expected_pairs for method in spec["semantics"]["readouts"]}
    measured_rows = {(r["scene"], r["map_id"], r["method"]) for r in primary}
    complete = bool(selection) and measured_pairs == expected_pairs and measured_rows == expected_rows and bridge["status"] == "VERIFIED"
    baseline = next((r["metrics"] for r in pools if r["cohort"] == "replica" and r["map_id"] == "BB00_NATIVE" and r["method"] == "D2"), None)
    nominee = next((r["metrics"] for r in pools if selection and r["cohort"] == "replica" and r["map_id"] == selection["nominee"] and r["method"] == "D2"), None)
    conclusion = net_gain(nominee, baseline) if nominee and baseline else "NOT_MEASURED"
    status = ("COMPLETE_MEASURED_NET_GAIN" if conclusion == "MEASURED_NET_GAIN" else "COMPLETE_NO_NET_GAIN") if complete else "INCOMPLETE"
    per_class = []
    for row in rows:
        receipt = read(row["evaluation_receipt"])
        per_class.append({"scene": row["scene"], "map_id": row["map_id"], "method": row["method"],
            "rank_mode": row["rank_mode"], "confusion": receipt["confusion"],
            "context": receipt["context"], "view": receipt["view"]})
    external = {"output_root": str(root), "native_build_root": spec["native_build_root"],
        "sam2_source": binding["sam2"]["code"], "assets_root": binding["assets_root"],
        "map_outputs": [{"scene": m["scene"], "map_id": m["map_id"], "capture": m.get("capture_manifest"),
                         "status": m["status"], "input_identity": m.get("input_identity")} for m in maps],
        "restore": "Restore the external map/capture/source/cache arrays at the recorded absolute paths and verify their receipts; weights/data are not in Git."}
    events = []
    for mapping in success:
        capture_path = Path(mapping["capture_manifest"])
        capture = read(capture_path)
        for frame in capture["frames"]:
            state = read(capture_path.parent / frame["native_state"]["path"])["native_state"]
            association = state.get("association", {})
            events.append({"scene": mapping["scene"], "map_id": mapping["map_id"], "frame_id": frame["frame_id"],
                "namespaces": {"current_2D_groups": sorted(set(row["input_instance_label"] for row in state["segments"])),
                    "superpoint_labels": [row["registered_label"] for row in state["segments"]],
                    "count_object_owners": sorted(set(row["instance_label"] for row in state["label_instances"]))},
                "fragments": len(state["segments"]), "aliases": state["aliases"],
                "association": {k: v for k, v in association.items() if k not in {"prior_owners", "alias_tokens"}},
                "realized_owner_discrepancies": [row for row in state["segments"] if row.get("owner_discrepancy")],
                "native_selected_requests": len(frame["native_selected_request_ids"]),
                "admissible_requests": len(frame["requests"]), "source_map_state_id": frame["map_state_id"]})
    semantic_differences = []
    from static_ovmap.module_validation.scannet_study import load_prediction, project_values
    for path in (root / "readouts").glob("*/*/receipt.json"):
        receipt = read(path)
        anchor = load_prediction(receipt["predictions"]["NATIVE_READOUT"])
        with np.load(path.parent / "projection/projection.npz", allow_pickle=False) as arrays:
            nearest, matched = arrays["nearest"], arrays["matched"]
        baseline_root = root / "readouts" / receipt["scene"] / "BB00_NATIVE"
        for method, manifest in receipt["predictions"].items():
            prediction = load_prediction(manifest)
            changed = prediction.semantic_labels != anchor.semantic_labels
            row = {"scene": receipt["scene"], "map_id": receipt["map_id"], "readout": method,
                "same_map_semantic_changed_source_rows": int(changed.sum()),
                "same_map_geometry_and_owners_exact": prediction.geometry == anchor.geometry and np.array_equal(prediction.owner_ids, anchor.owner_ids)}
            if (baseline_root / "receipt.json").is_file():
                base_receipt = read(baseline_root / "receipt.json")
                base = load_prediction(base_receipt["predictions"][method])
                with np.load(baseline_root / "projection/projection.npz", allow_pickle=False) as arrays:
                    base_labels = project_values(base.semantic_labels, arrays["nearest"], arrays["matched"])
                current_labels = project_values(prediction.semantic_labels, nearest, matched)
                row["cross_map_same_readout_changed_target_labels"] = int(np.count_nonzero(current_labels != base_labels))
            if method == "D2":
                fc_decisions, d2_decisions = read(path.parent / "decisions/FC_EQ.json"), read(path.parent / "decisions/D2.json")
                differences = [np.max(np.abs(np.asarray(fc_decisions[k]["probabilities"]) - np.asarray(value["probabilities"])))
                               for k, value in d2_decisions.items() if value["probabilities"] is not None]
                row["D2_vs_FC_EQ_max_probability_difference"] = float(max(differences, default=0.))
                row["D2_vs_FC_EQ_changed_owner_labels"] = sum(value["label"] != fc_decisions[k]["label"] for k, value in d2_decisions.items())
            semantic_differences.append(row)
    physical_frontends = [read(path) for path in (root / "frontend").glob("*/SAM2_PAIRED/receipt.json")]
    failures = [read(path) for path in (root / "execution").glob("*.json") if read(path).get("failures")]
    provenance = {"base_commit": spec["base_commit"], "upstream_commit": spec["upstream_commit"],
        "implementation_commits": sorted(set(read(path)["implementation_commit"] for path in (root / "execution").glob("*.json")
                                      if "implementation_commit" in read(path))),
        "consumed_code_and_model_identities": binding["inputs"], "map_source_and_output_identities": [
            {"scene": m["scene"], "map_id": m["map_id"], "inputs": m["inputs"], "outputs": m["outputs"]} for m in success]}
    outputs = {"resolved_inputs.json": binding, "experiment_matrix.json": read(root / "matrix.json"),
        "bridge_parity.json": bridge, "scene_rows.json": primary, "secondary_rank_rows.json": [r for r in rows if r not in primary],
        "dataset_pools.json": pools, "intervention_summary.json": interventions,
        "raw_geometry_diagnostics.json": diagnostics, "costs.json": costs,
        "semantic_differences.json": semantic_differences, "code_model_output_provenance.json": provenance,
        "failures.json": failures, "frontend_physical_costs.json": [{"scene": r["scene"], "status": r["status"],
            "counters": r["counters"], "elapsed_seconds": r["elapsed_seconds"],
            "peak_gpu_allocated_bytes": r["peak_gpu_allocated_bytes"], "identity": r["identity"]} for r in physical_frontends],
        "per_class_confusions.json": per_class, "external_artifacts.json": external,
        "completion.json": {"status": status, "implementation": "IMPLEMENTED_SCOPED_RUNTIME_VERIFIED",
            "experimental_coverage": {"successful_map_scenes": len(success), "expected": expected,
                "primary_rows": len(primary), "expected_primary_rows": 3 * expected},
            "scientific": conclusion, "publication": "EXTERNAL_RECEIPT_REQUIRED",
            "all_scenes_exposed": True, "deployment": "N0_UNCHANGED"}}
    for leaf in ("candidate_freeze.json", "selection.json", "bridge_diagnosis.json", "transfer_freeze_commit.json"):
        if (root / leaf).is_file():
            outputs[leaf] = read(root / leaf)
    for name, data in outputs.items():
        atomic_write_json(release / name, data)
    with gzip.open(release / "native_event_ledger.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(events, stream, sort_keys=True, allow_nan=False)
    for leaf in ("native_trace/summary.json", "sam2_preflight/receipt.json"):
        path = root / "prepare" / leaf
        if path.is_file():
            atomic_write_json(release / "validation" / (Path(leaf).parent.name + ".json"), read(path))
    build = Path(spec["native_build_root"]) / "native_build_receipt.json"
    if build.is_file():
        atomic_write_json(release / "validation/native_build_receipt.json", read(build))
    test_path = root / "review/scoped_tests.json"
    if test_path.is_file():
        atomic_write_json(release / "validation/scoped_tests.json", read(test_path))
    reports = {
        "BACKBONE_WAVE1_RESULTS.md": f"# Backbone wave-1 measured results\n\nStatus: `{status}`. Successful map-scene configurations: {len(success)}/{expected}; primary official rows: {len(primary)}/{3 * expected}. All data are exposed. Deployment: N0_UNCHANGED.\n\n## Development Official Pools (%)\n\n" + _table([r for r in pools if r["cohort"] == "development"]) +
            "\n\n## Replica Official Pools (%)\n\n" + _table([r for r in pools if r["cohort"] == "replica"]) +
            f"\n\nNominee measurement label: `{conclusion}`. Bridge: `{bridge['status']}`. APall uses the actual released overlap vector in the evaluation contexts, .50 through .90; AP25 is separate. Pools use the released evaluator over complete ordered cohorts. Semantic metrics sum scene confusion matrices.\n\n" +
            "Raw numeric geometry and official native triangle-first-color paint are separate. Class-agnostic recall uses strict IoU > .25/.5/.75, maximum-cardinality one-to-one matches and every eligible GT in the denominator. Missed GT best-IoU is zero. Surface precision/completeness use every exported vertex and every valid target vertex at strict 5cm; duplicated native vertices are retained consistently. Detailed raw geometry, GT-indexed gains/losses and predicted overlap associations are in the compact release.\n\n" +
            "Costs distinguish actual forwards from content-deduplicated standalone obligations. Common SAM predictions are required by both RAW/GEOM. N/Q/FC costs are required by both fused readouts. Timing is measured stage attribution, not a cold end-to-end rerun; missing timing components are disclosed per job. No significance or independent-generalization claim is made.\n",
        "BACKBONE_WAVE1_HANDOFF.md": "# Backbone wave-1 handoff\n\n" + f"Repository: `{repo}`\nExternal attempt: `{root}`\nIsolated upstream: `{spec['upstream_worktree']}`\nIsolated native build: `{spec['native_build_root']}`\n\n" +
            "Apply the original module_validation_v1 patch, then third_party_patches/ovimap/backbone_wave1_v1/backbone_wave1_v1.patch to the pinned upstream. The old native extension remains intact. SAM uses its pinned source checkout and historical environment; optional connected-components CUDA extension absence uses the official loader fallback and is recorded.\n\n" +
            f"```bash\n{spec['runtime_default']} scripts/evaluation/run_ovimap_backbone_wave1.py --phase all --spec docs/paper/static_ovmap/backbone_wave1_v1/PROTOCOL_SPEC.json --output-root {root} --gpu 2 --mapping-workers 2 --mapping-threads 8 --evaluation-workers 3 --resume\n```\n\n" +
            "Report and publish never launch mapping/models. The coordinator, map outputs and GPU use real locks. Interrupted maps restart from frame zero with previous attempts retained. Restore external arrays/data/weights from external_artifacts.json and validate consumed receipts. Implementation and transfer-freeze commits are in execution/freeze receipts; the final publication SHA is external in publication/final.json.\n",
        "BACKBONE_WAVE1_SELECTION.md": "# Backbone wave-1 selection\n\n" +
            (f"Frozen nominee: `{selection['nominee']}`. Transfer map IDs: " + ", ".join(r["id"] for r in selection["replica_recipes"]) + ".\n\n" if selection else "Selection is incomplete; no Replica nominee is asserted.\n\n") +
            "Selection uses D2 four-scene official development pooling only. APall band .05pp, mIoU band .1pp, AP50 band .1pp, then required standalone image encodings, median attributable time, changed block count and method ID. Strict metric ranking, banded ranking and every tie step are frozen in candidate_freeze.json/selection.json before composition/Replica. Individual positive gain is not required for composition; both families require complete inputs and actual raw partition intervention. Replica results do not refit weights, temperatures or the Q model.\n",
        "BACKBONE_WAVE1_CLAIMS.md": "# Backbone wave-1 claims\n\n" + f"Completion status: `{status}`. Measured nominee label: `{conclusion}`.\n\n" +
            "Supported implementation claims: the new native extension exposes a read-only prior probe and enforces object plans through candidate filtering, mode4 counts and compatible aliases; real short-trace receipts report realized ownership. Simultaneous fusion uses unchanged original depth support. Bounded SAM uses current-only forward outputs, five-frame resets and shared RAW-defined discoveries, with geometry checked against previous RAW support.\n\n" +
            "Scientific mechanism claims require the measured raw geometry ledgers and relevant simple-control comparisons. A class-conditioned rank or semantic coverage change alone does not establish stronger geometry. A stronger-backbone claim is not automatically authorized by AP gain. Mixed APall/mIoU signs are a tradeoff. Negative complete studies remain complete without a deployment change.\n\n" +
            "Untested/out of scope: independent confirmation; B02/B04/B06-B12; full OVRCOAT wrapper; SAM3; new temperature/Q/quality-head fitting; deployment improvement. Four development and eight Replica scenes are historically exposed; extra visual inference is explicitly accounted. Numeric owner IDs are not correspondence across maps.\n"}
    for name, value in reports.items():
        (docs / name).write_text(value, encoding="utf-8")
    index = InputIndex()
    included = [path for path in release.rglob("*") if path.is_file() and path.name != "release_manifest.json"]
    included += [docs / name for name in reports]
    for path in included:
        if path.stat().st_size >= 95 * 1024 ** 2:
            raise ValueError("compact release exceeds the per-file size contract")
        index.identity(path)
    atomic_write_json(release / "release_manifest.json", {"entries": index.entries(),
        "total_bytes": sum(path.stat().st_size for path in included), "raw_arrays_in_Git": False})
    atomic_write_json(root / "report/render_receipt.json", {"status": status, "release": str(release),
        "reports": list(reports), "output_identities": index.entries()})
    return outputs["completion.json"]


def publish(binding, spec):
    repo, root = Path(binding["repository_root"]), Path(binding["output_root"])
    completion = read(repo / spec["publication"]["repo_artifacts"] / "completion.json")
    if not completion["status"].startswith("COMPLETE"):
        raise ValueError("unfinished study cannot be published as completed")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    if branch != spec["publication"]["branch"]:
        raise ValueError("publication branch differs from the fixed study branch")
    scope = [spec["implementation_root"], spec["runner"], spec["repository_spec_root"],
             "third_party_patches/ovimap/backbone_wave1_v1", spec["publication"]["repo_artifacts"],
             *["docs/paper/static_ovmap/" + name for name in spec["publication"]["reports"]],
             "tests/evaluation/test_backbone_wave1_kernels.py", "tests/evaluation/test_backbone_wave1_frontend.py",
             "tests/evaluation/test_backbone_wave1_readouts.py"]
    subprocess.run(["git", "add", "--", *scope], cwd=repo, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo).returncode:
        subprocess.run(["git", "commit", "-m", "research: publish measured backbone-wave1 evidence"], cwd=repo, check=True)
    local = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    receipt = {"branch": branch, "local_sha": local, "remote_sha": None, "status": "PUSH_PENDING",
               "release_path": spec["publication"]["repo_artifacts"]}
    path = root / spec["publication"]["external_receipt"]
    try:
        subprocess.run(["git", "push", "origin", "HEAD:refs/heads/" + branch], cwd=repo, check=True)
        remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/" + branch], cwd=repo, text=True).split()[0]
        if remote != local:
            raise ValueError("full local/remote publication SHA mismatch")
        receipt.update(status="PUSH_VERIFIED", remote_sha=remote)
    except BaseException as exc:
        receipt.update(status="BLOCKED_PUSH", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(path, receipt)
    return receipt
