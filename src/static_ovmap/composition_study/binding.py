"""Read-only native-v10 binding and separate composition attempt allocation."""

import os
import re
import subprocess
from pathlib import Path

import numpy as np

from src.static_ovmap.module_validation.contracts import canonical_digest

from .io import ROOT, SourceIndex, read_json, write_once


def select_attempt(output_root, binding_key):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    attempts = sorted(output_root.glob("attempt_[0-9][0-9][0-9]"))
    for path in reversed(attempts):
        config = path / "resolved_config.json"
        if config.is_file() and read_json(config).get("binding_key") == binding_key:
            return path
    number = max((int(path.name.split("_")[1]) for path in attempts), default=0) + 1
    result = output_root / f"attempt_{number:03d}"
    result.mkdir(exist_ok=False)
    return result


def confirmation_exposure(roots, scenes):
    """One bounded directory-name scan; never open held-out images or labels."""
    prepared = set()
    inspected = []
    for root in dict.fromkeys(str(Path(p).resolve()) for p in roots):
        path = Path(root)
        if not path.is_dir():
            continue
        inspected.append(root)
        for parent, directories, _ in os.walk(path):
            for name in directories:
                if name in scenes:
                    prepared.add(str(Path(parent) / name))
            # Once a scene is identified, descendants cannot constitute a new
            # scene. Avoid walking model trees, raw scans and per-frame files.
            directories[:] = [
                name
                for name in directories
                if name
                not in {
                    "scans",
                    "models",
                    "tooling",
                    ".git",
                    "requests",
                    "native_request_cache",
                }
                and not re.fullmatch(r"scene\d{4}_\d{2}", name)
                and not name.isdigit()
            ]
    return {
        "status": "BLOCKED_CONFIRMATION_EXPOSURE"
        if prepared
        else "NO_PRIOR_PREPARED_CONFIRMATION_FOUND",
        "inspected_roots": inspected,
        "prepared_scene_paths": sorted(prepared),
        "blocked_scenes": list(scenes) if prepared else [],
        "scope": "Known project output roots, metadata-only; acquired raw scans excluded; no claim about unrelated external projects",
    }


def _bind_model(index, source, name):
    model_root = Path(source[name + "_model"]).resolve()
    text_path = Path(source[name + "_text_cache"])
    receipt_path = text_path.with_name("receipt.json")
    index.identity(receipt_path)
    receipt = read_json(receipt_path)
    if receipt["status"] != "COMPLETE" or receipt["dtype"] != "float32":
        raise ValueError(f"{name}: incomplete or non-FP32 source text cache")
    text_identity = index.expected_output(text_path, receipt)
    model_files = [
        index.identity(row["path"], row)
        for row in receipt["inputs"]
        if Path(row["path"]).resolve().is_relative_to(model_root)
    ]
    if not model_files or not any(
        Path(row["path"]).suffix in {".bin", ".safetensors"} for row in model_files
    ):
        raise ValueError(f"{name}: no receipt-bound model weights")
    with np.load(text_path, allow_pickle=False) as arrays:
        valid_ids = arrays["valid_ids"].tolist()
        names = arrays["class_names"].tolist()
        if len(valid_ids) != 200 or len(set(valid_ids)) != 200 or 0 in valid_ids:
            raise ValueError("invalid official vocabulary")
    model_identity = canonical_digest(
        {
            "files": [
                {
                    "path": str(Path(row["path"]).relative_to(model_root)),
                    "sha256": row["sha256"],
                }
                for row in model_files
            ],
            "dtype": "float32",
            "torch": receipt["torch_version"],
            "transformers": receipt["transformers_version"],
            "crop_rule": "native_global_bbox_union_exclusive_upper_v1",
            "crop_count": 6,
            "backend": index.identity(
                ROOT / "src/static_ovmap/module_validation/rgb_siglip.py"
            )["sha256"],
        }
    )
    return {
        "status": "BOUND",
        "name": name,
        "path": str(model_root),
        "identity": model_identity,
        "files": model_files,
        "text": text_identity,
        "text_receipt": str(receipt_path),
        "valid_ids": valid_ids,
        "class_names": names,
        "torch_version": receipt["torch_version"],
        "transformers_version": receipt["transformers_version"],
        "dtype": "float32",
    }


def _bind_scene(index, scene, role, runtime, source):
    base = Path(source["study_root"]) / "scenes" / scene
    mapping_path = Path(runtime["output_root"]) / scene / "mapping_job/receipt.json"
    index.identity(mapping_path)
    mapping = read_json(mapping_path)
    if mapping["status"] != "COMPLETE":
        raise ValueError(f"incomplete native capture: {scene}")
    capture_path = Path(mapping["capture_manifest"])
    index.expected_output(capture_path, mapping)
    capture = read_json(capture_path)
    if len(capture["scheduled_frame_ids"]) != 200:
        raise ValueError(f"native schedule is not 200 slots: {scene}")
    surface = capture_path.parent / capture["surface"]["path"]
    index.identity(surface, capture["surface"])
    feature_path = Path(mapping["native_features"])
    # The saved native pickle is an input of the already frozen baseline.
    baseline_receipt = read_json(base / "baseline/receipt.json")
    expected_feature = next(
        row for row in baseline_receipt["inputs"] if Path(row["path"]) == feature_path
    )
    index.identity(feature_path, expected_feature)
    native_path = base / "baseline/N0/manifest.json"
    index.expected_output(native_path, baseline_receipt)
    native = read_json(native_path)
    index.identity(native_path.parent / native["arrays"]["path"], native["arrays"])
    projection = base / "projection/manifest.json"
    index.identity(projection)
    index.identity(projection.with_name("projection.npz"), read_json(projection))
    index.identity(base / "baseline/native_readout.json")
    index.expected_output(base / "baseline/native_readout.npz", baseline_receipt)
    annotations = Path(source["study_root"]) / "annotations" / scene / "receipt.json"
    index.identity(annotations)
    controls = {"N0": {"status": "BOUND", "prediction": str(native_path)}}
    for method, prediction in {
        "Q_COMBINE": base / "query/B200/Q_COMBINE/prediction/manifest.json",
        "Q_GAIN": base / "query/B200/Q_GAIN/prediction/manifest.json",
        "S_SIGLIP2_AREA": base / "semantic/direct/S_SIGLIP2_AREA/manifest.json",
    }.items():
        if prediction.is_file():
            index.identity(prediction)
            value = read_json(prediction)
            index.identity(prediction.parent / value["arrays"]["path"], value["arrays"])
            if (
                value["geometry"] != native["geometry"]
                or value["instance_ranks"] != native["instance_ranks"]
            ):
                raise ValueError(f"source geometry/ranks differ: {scene}/{method}")
            controls[method] = {"status": "BOUND", "prediction": str(prediction)}
        else:
            controls[method] = {"status": "GENERATE_IN_NEW_STUDY", "prediction": None}
    return {
        "status": "BOUND",
        "role": role,
        "source_directory": str(base),
        "mapping_receipt": str(mapping_path),
        "capture": str(capture_path),
        "surface": str(surface),
        "native_features": str(feature_path),
        "native_prediction": str(native_path),
        "projection": str(projection),
        "annotations": str(annotations),
        "schedule": capture["scheduled_frame_ids"],
        "controls": controls,
        "native_geometry": native["geometry"],
    }


def bind(spec_path, output_root, overrides=None):
    overrides = {} if overrides is None else dict(overrides)
    spec_path = Path(spec_path).resolve()
    spec = read_json(spec_path)
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", spec["reviewed_commit"], "HEAD"],
        cwd=ROOT,
        check=True,
    )
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if branch != spec["task_branch"]:
        raise ValueError("composition work must use the declared task branch")
    index = SourceIndex()
    index.identity(spec_path)
    source_config_path = Path(
        overrides.get("source_study_config", ROOT / spec["source_study_config"])
    ).resolve()
    runtime_config_path = Path(
        overrides.get("source_runtime_config", ROOT / spec["source_runtime_config"])
    ).resolve()
    index.identity(source_config_path)
    index.identity(runtime_config_path)
    source, runtime = read_json(source_config_path), read_json(runtime_config_path)
    lock_path = Path(runtime["data_root"]) / "acquisition_lock.json"
    index.identity(lock_path)
    lock = read_json(lock_path)
    for new_role, old_role in (
        ("compose_cal", "cal"),
        ("regression_only", "select"),
        ("confirmation", "confirm"),
    ):
        if [
            row["scene_id"] for row in lock["selected"] if row["role"] == old_role
        ] != spec["data"][new_role]:
            raise ValueError(f"original role/schedule binding differs: {new_role}")
    calibration_path = Path(source["study_root"]) / "query/calibration_receipt.json"
    index.identity(calibration_path)
    calibration = read_json(calibration_path)
    checkpoint = index.identity(
        calibration["checkpoint"]["path"], calibration["checkpoint"]
    )
    index.identity(runtime["native_extension"])
    build = read_json(runtime["native_build_receipt"])
    index.identity(runtime["native_build_receipt"])
    upstream = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=runtime["upstream"], text=True
    ).strip()
    if upstream != spec["upstream_pin"]:
        raise ValueError("native upstream pin changed")
    for name in (
        "eval_utils.py",
        "eval_sem_seg.py",
        "utils/semantic_const.py",
        "utils/instance_utils.py",
    ):
        path = Path(runtime["upstream"]) / "scripts" / name
        if path.is_file():
            index.identity(path)
    models = {}
    for name in ("native", "siglip2"):
        try:
            models[name] = _bind_model(index, source, name)
        except (OSError, ValueError, KeyError) as error:
            models[name] = {"status": "UNAVAILABLE", "error": str(error)}
    if all(row["status"] == "BOUND" for row in models.values()) and any(
        models["native"][key] != models["siglip2"][key]
        for key in ("valid_ids", "class_names")
    ):
        raise ValueError("native and SigLIP2 official vocabularies differ")
    scenes = {}
    for role in ("compose_cal", "regression_only"):
        for scene in spec["data"][role]:
            try:
                scenes[scene] = _bind_scene(index, scene, role, runtime, source)
            except (OSError, ValueError, KeyError) as error:
                scenes[scene] = {
                    "status": "UNAVAILABLE",
                    "role": role,
                    "error": str(error),
                }
    known_roots = sorted(Path(source["study_root"]).parent.parent.glob("ovimap-*"))
    known_roots = [
        path for path in known_roots if path.resolve() != Path(output_root).resolve()
    ]
    exposure = confirmation_exposure(known_roots, spec["data"]["confirmation"])
    settings = {
        "spec_sha256": index.identity(spec_path)["sha256"],
        "source": source,
        "runtime": runtime,
        "models": models,
        "checkpoint": checkpoint,
        "scenes": scenes,
        "confirmation_exposure": exposure,
        "source_manifest": index.manifest(),
    }
    binding_key = canonical_digest(settings)
    attempt = select_attempt(output_root, binding_key)
    config = {
        "schema_version": 1,
        "study": spec["study"],
        "task_branch": branch,
        "reviewed_commit": spec["reviewed_commit"],
        "repository_root": str(ROOT),
        "spec_path": str(spec_path),
        "spec": spec,
        "binding_key": binding_key,
        "attempt_root": str(attempt),
        "output_root": str(Path(output_root).resolve()),
        "source": source,
        "runtime": runtime,
        "models": models,
        "checkpoint": checkpoint,
        "scenes": scenes,
        "confirmation_exposure": exposure,
        "confirmation_rows": [
            row for row in lock["selected"] if row["role"] == "confirm"
        ],
        "gpu_lock": str(
            Path(runtime["output_root"]).parent
            / f".visual-gpu-{runtime['cuda_device']}.lock"
        ),
        "cpu_threads": 8,
        "source_lock": str(lock_path),
        "native_build_status": build.get("status"),
    }
    write_once(attempt / "source_manifest.json", index.manifest())
    write_once(
        attempt / "binding.json",
        {
            "binding_key": binding_key,
            "models": models,
            "scenes": scenes,
            "confirmation_exposure": exposure,
        },
    )
    return write_once(attempt / "resolved_config.json", config)
