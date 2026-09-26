"""Two explicitly authorized confirmation captures under the new frozen contract."""

import fcntl
import hashlib
import subprocess
from pathlib import Path

import numpy as np

from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.scannet_download import verified_download
from src.static_ovmap.module_validation.scannet_frames import export_sensor
from src.static_ovmap.module_validation.scannet_ground_truth import (
    load_ground_truth,
    prepare_ground_truth,
)
from src.static_ovmap.module_validation.scannet_study import (
    build_native_prediction,
    freeze_projection,
    load_prediction,
    relabel_prediction,
    save_prediction,
)
from src.static_ovmap.module_validation.semantic_models import load_semantic_records
from src.static_ovmap.module_validation.semantic_study import (
    direct_readouts,
    prepare_semantic_manifest,
)
from src.static_ovmap.module_validation.study_scene import audit_native_export

from .binding import _bind_scene, confirmation_exposure
from .execution import require_access
from .io import ROOT, SourceIndex, read_json, write_once
from .object_evidence import owner_labels
from .selection import COMPOSITIONS, CONTROLS


def validate_contract(config, lock):
    if lock["status"] != "FROZEN" or lock["binding_key"] != config["binding_key"]:
        raise ValueError("new composition selection is not frozen for these inputs")
    for key in ("spec", "models", "checkpoint", "confirmation_rows"):
        if lock[key] != config[key]:
            raise ValueError(f"confirmation {key} differs from frozen selection")
    for key in ("runtime", "source", "gpu_lock"):
        if lock.get(key) != config.get(key):
            raise ValueError(f"confirmation {key} differs from frozen selection")
    nomination = lock["nomination"]
    nominee = nomination["nominee"]
    if nomination["status"] != "NOMINATED" or nominee not in COMPOSITIONS:
        raise ValueError(
            "confirmation requires one effective-change composition nominee"
        )
    expected = [
        *CONTROLS,
        nominee,
        *([COMPOSITIONS[5]] if nominee == COMPOSITIONS[6] else []),
    ]
    if nomination["confirmation_methods"] != expected:
        raise ValueError(
            "confirmation method set differs from nominee plus declared controls"
        )
    rows = config["confirmation_rows"]
    if (
        len(rows) != 2
        or [row["scene_id"] for row in rows] != config["spec"]["data"]["confirmation"]
    ):
        raise ValueError("exactly the two preselected confirmation rows are required")
    for row in rows:
        schedule = row["schedule"]
        if (
            row["role"] != "confirm"
            or len(schedule["frame_ids"]) != 200
            or schedule["frame_ids"]
            != list(range(schedule["start"], schedule["end"], schedule["step"]))
        ):
            raise ValueError("confirmation schedule or original role changed")
    return rows


def authorize_exports(config):
    root = Path(config["attempt_root"])
    lock = read_json(root / "selection.json")
    rows = validate_contract(config, lock)
    for row in rows:
        require_access(
            config, row["scene_id"], "confirm", lock["nomination"]["nominee"]
        )
    index = SourceIndex()
    index.identity(lock["source_manifest"]["path"], lock["source_manifest"])
    for entry in read_json(lock["source_manifest"]["path"])["entries"]:
        index.identity(entry["path"], entry)
    upstream = config["runtime"]["upstream"]
    pin = subprocess.check_output(
        ["git", "-C", upstream, "rev-parse", "HEAD"], text=True
    ).strip()
    patch = (
        ROOT
        / "third_party_patches/ovimap/module_validation_v1/ovimap_module_validation_v1.patch"
    )
    difference = subprocess.check_output(
        ["git", "-C", upstream, "diff", "--binary", "--unified=50", "HEAD"]
    )
    if (
        pin != config["spec"]["upstream_pin"]
        or hashlib.sha256(difference).hexdigest() != index.identity(patch)["sha256"]
    ):
        raise ValueError("native upstream recipe changed before confirmation export")
    build = read_json(config["runtime"]["native_build_receipt"])
    if (
        build["status"] != "COMPLETE"
        or index.identity(config["runtime"]["native_extension"]) != build["extension"]
    ):
        raise ValueError("native extension differs before confirmation export")
    path = root / "confirmation/access.json"
    if path.is_file():
        access = read_json(path)
        if access["selection_identity"] != lock["identity"] or access["rows"] != rows:
            raise ValueError(
                "confirmation access receipt belongs to different frozen inputs"
            )
        return access
    # Check other studies/attempts before any sensor export or annotation conversion.
    roots = [
        *config["confirmation_exposure"]["inspected_roots"],
        *[str(path) for path in root.parent.glob("attempt_*") if path != root],
    ]
    exposure = confirmation_exposure(roots, config["spec"]["data"]["confirmation"])
    if exposure["blocked_scenes"]:
        write_once(root / "confirmation/exposure_block.json", exposure)
        raise ValueError("BLOCKED_CONFIRMATION_EXPOSURE")
    raw_root = Path(config["runtime"]["data_root"])
    acquisition = read_json(raw_root / "acquisition_lock.json")
    if [row for row in acquisition["selected"] if row["role"] == "confirm"] != rows:
        raise ValueError("acquisition confirmation rows differ from frozen selection")
    downloads = read_json(raw_root / "download_receipt.json")
    acquisition_identity = index.identity(raw_root / "acquisition_lock.json")
    if (
        downloads["status"] != "RAW_DOWNLOADS_COMPLETE"
        or downloads["lock_sha256"] != acquisition_identity["sha256"]
    ):
        raise ValueError("already acquired raw-data receipt is incomplete")
    # Authorized now: verify all actual raw bytes against their download receipts.
    for row in rows:
        for suffix, remote in row["files"].items():
            raw = raw_root / "scans" / row["scene_id"] / (row["scene_id"] + suffix)
            if not verified_download(raw, remote):
                raise ValueError(
                    f"required acquired raw file is missing/changed: {raw}"
                )
            index.identity(raw, read_json(str(raw) + ".download.json"))
    result = {
        "status": "AUTHORIZED_FROZEN_COMPOSITION",
        "selection_identity": lock["identity"],
        "rows": rows,
        "methods": lock["nomination"]["confirmation_methods"],
        "exposure": exposure,
        "inputs": index.manifest()["entries"],
        "old_study_guards_unchanged": True,
        "no_new_download": True,
    }
    write_once(path, result)
    return result


def prepare_scene(config, row, lock):
    from .capture_recipe import capture_one

    parent = Path(config["attempt_root"])
    scene = row["scene_id"]
    root = parent / "confirmation" / scene
    runtime = config["runtime"]
    raw_root = Path(runtime["data_root"])
    export = root / "exported" / scene
    export_sensor(raw_root / "scans" / scene / f"{scene}.sens", export, row["schedule"])
    study_root = root / "study"
    annotation_path = prepare_ground_truth(
        Path(runtime["upstream"]),
        raw_root / "scans" / scene,
        scene,
        raw_root / "scannetv2-labels.combined.tsv",
        study_root / "annotations" / scene,
    )
    with Path(config["gpu_lock"]).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        mapping = capture_one(runtime, row, export, root / "native")
    capture_path = Path(mapping["capture_manifest"])
    capture = read_json(capture_path)
    base = study_root / "scenes" / scene
    targets = load_ground_truth(annotation_path)
    with np.load(
        capture_path.parent / capture["surface"]["path"], allow_pickle=False
    ) as arrays:
        projection = freeze_projection(
            arrays["surface_xyz"], targets["xyz"], base / "projection"
        )
    native_path = build_native_prediction(
        capture_path,
        Path(mapping["native_features"]),
        Path(config["models"]["native"]["text"]["path"]),
        projection,
        base / "baseline",
    )
    native = load_prediction(native_path)
    targets.update(nearest=projection["nearest"], matched=projection["matched"])
    parity = audit_native_export(
        native, targets, Path(runtime["upstream"]), base / "native_export_parity"
    )
    write_once(base / "native_export_parity.json", parity)
    manifest_path = base / "semantic_requests.json"
    manifest = prepare_semantic_manifest(capture_path, native, manifest_path)
    # The existing S2 worker sees the original runtime lock location. Its request
    # manifest itself identifies the new capture, so no old study path is mutated.
    source = {
        **config["source"],
        "study_root": str(study_root),
        "runtime_config": str(root / "source_runtime.json"),
    }
    write_once(root / "source_runtime.json", runtime)
    write_once(root / "semantic_config.json", source)
    semantic_root = base / "semantic_models/siglip2"
    log_path = root / "static_siglip2.log"
    command = [
        runtime["semantic_python"],
        str(ROOT / "scripts/evaluation/run_ovimap_semantic_model.py"),
        "--config",
        str(root / "semantic_config.json"),
        "--request-manifest",
        str(manifest_path),
        "--model",
        "siglip2",
        "--output",
        str(semantic_root),
    ]
    with log_path.open("a") as log:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode:
        raise RuntimeError(f"confirmation static S2 failed: {log_path}")
    records = load_semantic_records(semantic_root)
    with np.load(
        config["models"]["siglip2"]["text"]["path"], allow_pickle=False
    ) as arrays:
        suggestions = direct_readouts(
            manifest,
            records,
            arrays["text_embeddings"],
            tuple(map(int, arrays["valid_ids"])),
            owner_labels(native),
            model="siglip2",
        )["S_SIGLIP2_AREA"]
    labels = {owner: int(value["label_id"]) for owner, value in suggestions.items()}
    semantic_receipt = read_json(semantic_root / "receipt.json")
    costs = {
        **dict(native.logical_cost),
        "added_attempts": len(records),
        "added_crop_inputs": 6 * len(records),
        "added_background_inputs": 0,
        "generations": 0,
        "added_seconds": semantic_receipt["request_seconds"],
    }
    payload = relabel_prediction(
        native,
        "S_SIGLIP2_AREA",
        "S",
        labels,
        costs,
        {"request_manifest": manifest["identity"], "readout_id": "S_SIGLIP2_AREA"},
    )
    save_prediction(payload, base / "semantic/direct/S_SIGLIP2_AREA")
    index = SourceIndex()
    for entry in read_json(parent / "source_manifest.json")["entries"]:
        index.identity(entry["path"], entry)
    bound = _bind_scene(
        index,
        scene,
        "confirmation",
        {**runtime, "output_root": str(root / "native")},
        source,
    )
    child = {
        **config,
        "attempt_root": str(root),
        "source": source,
        "scenes": {**config["scenes"], scene: bound},
        "parent_attempt_root": str(parent),
    }
    write_once(root / "source_manifest.json", index.manifest())
    write_once(root / "selection.json", lock)
    write_once(root / "resolved_config.json", child)
    return child, root / "resolved_config.json"


def run_confirmation(config, config_path):
    from .pipeline import compose_static, invoke_leaf, rows_for_role

    root = Path(config["attempt_root"])
    lock = read_json(root / "selection.json")
    if not lock["nomination"]["confirmation_required"]:
        result = {"status": "NOT_REQUIRED_NO_EFFECTIVE_INTERVENTION", "rows": []}
        write_once(root / "confirmation/receipt.json", result)
        return result
    access = authorize_exports(config)
    results = []
    methods = lock["nomination"]["confirmation_methods"]
    for row in access["rows"]:
        child, child_path = prepare_scene(config, row, lock)
        scene = row["scene_id"]
        invoke_leaf(child, child_path, "confirm", "static", scene)
        for method in ("Q_COMBINE", "Q_GAIN"):
            invoke_leaf(child, child_path, "confirm", "query", scene, method)
        for method in COMPOSITIONS[3:]:
            if method in methods:
                # For M6 its M5 control must run first, even though frozen output order lists the nominee first.
                invoke_leaf(child, child_path, "confirm", "query", scene, method)
        if any(method in methods for method in COMPOSITIONS[:3]):
            compose_static(child, scene, "confirm")
        for method in methods:
            invoke_leaf(child, child_path, "confirm", "evaluate", scene, method)
        for measured in rows_for_role(child, "confirmation"):
            write_once(
                root / "rows/confirmation" / scene / (measured["method_id"] + ".json"),
                measured,
            )
            results.append(measured)
    result = {
        "status": "COMPLETE",
        "selection_identity": lock["identity"],
        "rows": results,
        "capture_count": len(access["rows"]),
        "configurations_per_scene": len(methods),
        "authorization_identity": canonical_digest(access),
    }
    write_once(root / "confirmation/receipt.json", result)
    return result
