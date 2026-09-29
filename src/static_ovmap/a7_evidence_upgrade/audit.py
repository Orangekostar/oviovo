"""Final whole-matrix, source/fallback, geometry and rank checks on real files."""

import math
from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.composition_study.object_evidence import owner_labels
from src.static_ovmap.m2_reviewer_study.binding import InputIndex
from src.static_ovmap.m2_reviewer_study.fusion import fuse
from src.static_ovmap.m2_reviewer_study.scores import read_scene
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.scannet_study import load_prediction

from .calibration import base_temperatures, source_path
from .evaluation import RANKS
from .selection import lock_transfer

METRICS = ("apall", "ap50", "ap25", "miou", "macc")


def measured_methods(binding):
    root = Path(binding["output_root"])
    nomination, pair = read_json(root / "nomination.json"), read_json(root / "composition.json")
    blocked = nomination["blocked_methods"]
    if blocked:
        access = read_json(root / "e04/access_probe.json")
        if access["status"] != "BLOCKED_ASSET_ACCESS" or set(blocked) != set(access["affected_methods"]):
            raise ValueError("blocked method registry has no matching real access evidence")
    methods = [m for m in binding["methods"] if m not in blocked]
    if pair["status"] == "PAIR_FROZEN":
        methods.append(binding["optional_method"])
    return methods, blocked


def validate_calibration(binding, index):
    """Verify saved examples and fold membership without fitting again."""
    root = Path(binding["output_root"])
    spec, parent = read_json(binding["spec"]), read_json(binding["reviewer_binding"])
    scenes = spec["datasets"]["calibration"]
    geometry_path = Path(parent["composition_root"]) / "calibration/examples.json"
    index.identity(geometry_path)
    correspondences = read_json(geometry_path)["correspondences"]
    checked = {}
    for definition in spec["source_variants"]:
        variant = definition["id"]
        fit_path = root / "calibration" / (variant + ".json")
        index.identity(fit_path)
        fit = read_json(fit_path)
        expected = []
        for scene in scenes:
            path = source_path(binding, scene, variant)
            index.identity(path)
            source = read_json(path)
            for row in correspondences[scene]:
                obj = source["objects"][str(row["owner_id"])]
                if row["correspondence"] == "unique" and row["geometry_iou"] > .5 and obj["available"]:
                    expected.append({"scene_id": scene, "owner_id": row["owner_id"],
                                     "gt_label": row["gt_label"], "available": True, "scores": obj["scores"]})
        if fit["examples"] != expected or fit["Replica_fitting"] is not False or set(fit["folds"]) != set(scenes):
            raise ValueError("scalar examples/folds differ from original strict CAL geometry matches")
        for name, trained in [(s, [other for other in scenes if other != s]) for s in scenes] + [("final", scenes)]:
            value = fit["final"] if name == "final" else fit["folds"][name]
            rows = [r for r in expected if r["scene_id"] in trained and r["gt_label"] in source["valid_ids"]]
            keys = [[r["scene_id"], r["owner_id"]] for r in rows]
            classes = len({r["gt_label"] for r in rows})
            if value["training_scenes"] != trained or value["example_ids"] != keys or value["objects"] != len(rows) or value["classes"] != classes:
                raise ValueError("scalar fold leaks held-out scene or misreports example support")
            temperature = value["temperature"]
            if not math.isfinite(temperature) or not .01 <= temperature <= 2:
                raise ValueError("scalar outside prescribed bounds")
            if (len(rows) < 5 or classes < 2) and (temperature != .07 or value["status"] != "UNCALIBRATED_DEFAULT_T"):
                raise ValueError("insufficient scalar support did not use the prescribed default")
        checked[variant] = {"examples": len(expected), "opposite_scene_folds_verified": True, "final_CAL_only": True}
    return checked


def validate_matrix(binding):
    root, index = Path(binding["output_root"]), InputIndex()
    path = root / "audit/matrix.json"
    if path.exists():
        previous = read_json(path)
        for item in previous["inputs"]:
            index.identity(item["path"], item)
        return previous
    lock_transfer(binding)
    index.identity(__file__)
    spec = read_json(binding["spec"])
    methods, blocked = measured_methods(binding)
    # Fail before expensive validation when even one expected result is absent.
    expected_paths = [root / "rows" / s / m / (rank + ".json")
                      for s in binding["scenes"] for m in methods for rank in RANKS]
    expected_paths += [root / "pooled" / split / m / (rank + ".json")
                       for split in ("cal", "replica") for m in methods for rank in RANKS]
    missing = [str(p) for p in expected_paths if not p.is_file()]
    if missing:
        raise ValueError(f"final matrix incomplete: {len(missing)} missing rows/pools; first={missing[0]}")
    parent = read_json(binding["reviewer_binding"])
    calibration_checks = validate_calibration(binding, index)
    definitions = {v["id"]: v for v in spec["source_variants"]}
    pair = read_json(root / "composition.json")
    scene_rows, pool_rows, fallbacks, owner_checks = [], [], 0, 0
    rows_by_key = {}
    overlaps = None
    for scene in binding["scenes"]:
        evidence = read_scene(parent, scene, index)
        native, ids = evidence.native, evidence.sources["N0"]["valid_ids"]
        baseline_labels = owner_labels(native)
        sources = {**evidence.sources, "N0": evidence.cosine_native}
        variant_sources = {}
        for variant in definitions:
            source_file = source_path(binding, scene, variant)
            index.identity(source_file)
            source = read_json(source_file)
            if source["valid_ids"] != ids or source["native_record_key"] != native.record_key or set(source["objects"]) != set(map(str, baseline_labels)):
                raise ValueError("source vocabulary/geometry/owner registry differs")
            if canonical_digest({k: v for k, v in source.items() if k != "identity"}) != source["identity"]:
                raise ValueError("source identity changed")
            for obj in source["objects"].values():
                if obj["available"]:
                    scores = np.asarray(obj["scores"], np.float64)
                    if scores.shape != (len(ids),) or not np.isfinite(scores).all() or ids[int(scores.argmax())] != obj["label"]:
                        raise ValueError("genuine source score/label mismatch")
                elif obj["scores"] is not None:
                    raise ValueError("unavailable source supplied fabricated evidence")
            variant_sources[variant] = source
        frozen_ranks = dict(native.instance_ranks)
        for method in methods:
            locked_path = root / "locked" / scene / (method + ".json")
            index.identity(locked_path)
            locked = read_json(locked_path)
            labels = {int(k): v for k, v in locked["labels"].items()}
            if set(labels) != set(baseline_labels):
                raise ValueError("excluded/unsupported owners disappeared from complete map")
            prediction_path = Path(locked["prediction_manifest"])
            index.identity(prediction_path)
            prediction = load_prediction(prediction_path)
            manifest = read_json(prediction_path)
            index.identity(prediction_path.parent / manifest["arrays"]["path"], manifest["arrays"])
            if prediction.prediction_key != locked["prediction_identity"] or prediction.geometry != native.geometry or prediction.instance_ranks != native.instance_ranks:
                raise ValueError("locked prediction geometry/ranks/identity differ")
            if not np.array_equal(prediction.owner_ids, native.owner_ids):
                raise ValueError("actual owner array changed")
            lookup = np.zeros(int(native.owner_ids.max()) + 1, np.int64)
            for owner, label in labels.items():
                if label != 0 and label not in ids:
                    raise ValueError("prediction class outside frozen vocabulary")
                lookup[owner] = label
            expected_semantic = lookup[native.owner_ids]
            background = native.owner_ids == 0
            expected_semantic[background] = native.semantic_labels[background]
            if not np.array_equal(prediction.semantic_labels, expected_semantic):
                raise ValueError("saved complete-map semantics differ from locked labels")
            owner_checks += len(labels)
            selected, temperatures = dict(sources), base_temperatures(binding, scene)
            replacement_variants = []
            if method.endswith("_DIRECT"):
                variant = method.removesuffix("_DIRECT")
                for owner, label in labels.items():
                    obj = variant_sources[variant]["objects"][str(owner)]
                    expected = obj["label"] if obj["available"] else baseline_labels[owner]
                    fallbacks += not obj["available"]
                    if label != expected:
                        raise ValueError("DIRECT fallback is not genuine source top1 or N0")
            elif method.endswith("_A7"):
                replacement_variants = [method.removesuffix("_A7")]
            elif method == "AW_COMBO_QR":
                replacement_variants = [pair["q_variant"], pair["region_variant"]]
            if replacement_variants:
                for variant in replacement_variants:
                    slot = definitions[variant]["slot"]
                    fit = read_json(root / "calibration" / (variant + ".json"))
                    temperatures[slot] = (fit["folds"][scene] if binding["scenes"][scene]["role"] == "CAL" else fit["final"])["temperature"]
                    selected[slot] = variant_sources[variant]
                actual, _ = fuse(baseline_labels, selected, ids, tuple(selected), temperatures)
                if actual != labels:
                    raise ValueError("A7 replacement/frozen scalar reconstruction differs")
            if method == "AW_E04_SHORTLIST":
                base = read_json(root / "locked" / scene / "RV_A7_COS_REFIT.json")["labels"]
                if base != locked["labels"]:
                    raise ValueError("SHORTLIST failed its identical-decision control")
            for rank in RANKS:
                row_path = root / "rows" / scene / method / (rank + ".json")
                index.identity(row_path)
                row = read_json(row_path)
                if row["status"] != "COMPLETE" or row["prediction_identity"] != prediction.prediction_key or (row["scene"], row["method"], row["rank_mode"]) != (scene, method, rank):
                    raise ValueError("scene result provenance mismatch")
                if row["metrics"]["apall"] != row["metrics"]["uap"] or any(not math.isfinite(row["metrics"][k]) or not 0 <= row["metrics"][k] <= 1 for k in METRICS):
                    raise ValueError("invalid measured metric or APall alias")
                receipt_path = Path(row["evaluation_receipt"])
                index.identity(receipt_path)
                receipt = read_json(receipt_path)
                if receipt["metrics"] != row["metrics"] or receipt["trace_parity"]["ap_exact"] is not True or receipt["trace_parity"]["pr_and_fn_exact"] is not True:
                    raise ValueError("released evaluator/trace parity missing")
                current_overlaps = receipt["context"]["runtime_overlaps"]
                if overlaps is not None and overlaps != current_overlaps:
                    raise ValueError("runtime APall overlap thresholds changed across methods")
                overlaps = current_overlaps
                view = receipt["view"]
                maxima = {}
                if rank == "OFFICIAL_CURRENT_CLASS":
                    for value in view.values():
                        maxima[value["label"]] = max(maxima.get(value["label"], 0), value["area"])
                for owner, value in view.items():
                    expected_rank = frozen_ranks[int(owner)] if rank == "FROZEN_N0" else value["area"] / maxima[value["label"]]
                    if value["rank"] != f"{expected_rank:.6f}" or value["label"] != labels[int(owner)]:
                        raise ValueError("rank export differs from its declared mode")
                if method in spec["references"]:
                    historical = read_json(Path(parent["output_root"]) / "rows" / scene / method / (rank + ".json"))
                    if row["metrics"] != historical["metrics"]:
                        raise ValueError("same-mode historical control changed")
                rows_by_key[(scene, method, rank)] = row
                scene_rows.append(row)
    for split in ("cal", "replica"):
        scenes = spec["datasets"]["calibration" if split == "cal" else "replica"]
        for method in methods:
            for rank in RANKS:
                row_path = root / "pooled" / split / method / (rank + ".json")
                index.identity(row_path)
                row = read_json(row_path)
                expected_ids = [rows_by_key[(s, method, rank)]["evaluation_identity"] for s in scenes]
                if row["status"] != "COMPLETE" or row["scene_order"] != scenes or row["ordered_inputs"] != expected_ids or row["aggregation"] != "RELEASED_DATASET_POOL":
                    raise ValueError("pool is not bound to all ordered scene evaluations")
                if row["metrics"]["apall"] != row["metrics"]["uap"] or any(not math.isfinite(row["metrics"][k]) or not 0 <= row["metrics"][k] <= 1 for k in METRICS):
                    raise ValueError("invalid pooled metric or APall alias")
                pool_rows.append(row)
    first_replica_time = min((root / "rows" / s / m / (rank + ".json")).stat().st_mtime_ns
                             for s in spec["datasets"]["replica"] for m in methods for rank in RANKS)
    if (root / "transfer_lock.json").stat().st_mtime_ns >= first_replica_time:
        raise ValueError("transfer freeze did not precede new Replica scene metrics")
    result = {"status": "REAL_MATRIX_VERIFIED", "binding": binding["identity"], "methods": methods, "blocked_methods": blocked,
              "scene_rank_rows": len(scene_rows), "pool_rows": len(pool_rows), "owner_method_checks": owner_checks,
              "verified_DIRECT_fallbacks": int(fallbacks), "runtime_overlaps": overlaps,
              "new_Replica_metrics_after_transfer_freeze": True, "actual_prediction_arrays_verified": True,
              "same_mode_controls_match_parent": True, "inputs": index.entries()}
    result["calibration_checks"] = calibration_checks
    result["identity"] = canonical_digest(result)
    write_once(path, result)
    return result
