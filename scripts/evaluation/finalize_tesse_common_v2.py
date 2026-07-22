#!/usr/bin/env python3
"""Finalize two deterministic TESSE-CD common-v2 scene summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from statistics import fmean
import tempfile
from typing import Any, Mapping


METHODS = {
    "OVIMAP_FROZEN": {
        "display_label": "OVI-MAP (frozen)",
        "mode": "frozen",
        "summary_mode": "frozen",
        "eligible_for_ranking": True,
    },
    "CONCEPTGRAPHS_FROZEN": {
        "display_label": "ConceptGraphs (frozen)",
        "mode": "frozen",
        "summary_mode": "frozen",
        "eligible_for_ranking": True,
    },
    "DUALMAP": {
        "display_label": "DualMap",
        "mode": "native",
        "summary_mode": "causal_checkpoints",
        "eligible_for_ranking": True,
    },
    "PANOPTIC_SHARED": {
        "display_label": "Panoptic Mapping + shared masks",
        "mode": "composed",
        "summary_mode": "causal_checkpoints",
        "eligible_for_ranking": True,
    },
    "KHRONOS_OPEN": {
        "display_label": "Khronos (open-set)",
        "mode": "online",
        "summary_mode": "causal_checkpoints",
        "eligible_for_ranking": True,
    },
    "KHRONOS_ORACLE": {
        "display_label": "Khronos (GT semantics)",
        "mode": "oracle",
        "summary_mode": "causal_checkpoints",
        "eligible_for_ranking": False,
    },
    "OVIV2": {
        "display_label": "OVIV2",
        "mode": "online",
        "summary_mode": "causal_checkpoints",
        "eligible_for_ranking": True,
    },
}
METRICS = {
    "current_miou": "CURRENT_MIOU",
    "ghost_rate": "GHOST_RATE",
    "background_f5": "BG_F5",
    "recovery_frames": "RECOVERY_FRAMES",
}
REQUIRED_SUMMARY_SOURCES = {
    "target_manifest",
    "target_arrays",
    "schedule",
    "aliases",
    "label_space",
    "evaluator",
}
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"source is not a file: {path}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "byte_count": resolved.stat().st_size,
    }


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _declared_record(
    value: object, *, label: str, base: Path | None = None
) -> dict[str, object]:
    declaration = _mapping(value, f"{label} declaration")
    raw = Path(str(declaration.get("path", "")))
    path = raw if raw.is_absolute() else (base / raw if base is not None else raw)
    record = _file_record(path)
    if (
        declaration.get("sha256") != record["sha256"]
        or declaration.get("byte_count") != record["byte_count"]
    ):
        raise ValueError(f"{label} hash mismatch")
    return record


def _nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _validate_metric_details(metrics: Mapping[str, Any], *, scene: str) -> None:
    expected_parameters = {
        "recovery_background_f5": 0.9,
        "recovery_consecutive": 3,
        "checkpoint_step_frames": 50,
        "recovery_horizon_frames": 450,
    }
    if any(metrics.get(name) != expected for name, expected in expected_parameters.items()):
        raise ValueError(f"{scene} recovery/censor protocol mismatch")
    events = _mapping(metrics.get("events"), f"{scene} event metrics")
    event_count = _nonnegative_int(metrics.get("event_count"), label=f"{scene} event_count")
    observable_count = _nonnegative_int(
        metrics.get("background_observable_event_count"),
        label=f"{scene} background_observable_event_count",
    )
    unobservable_count = _nonnegative_int(
        metrics.get("unobservable_revealed_target_event_count"),
        label=f"{scene} unobservable_revealed_target_event_count",
    )
    recovered_count = _nonnegative_int(
        metrics.get("recovered_event_count"), label=f"{scene} recovered_event_count"
    )
    censored_count = _nonnegative_int(
        metrics.get("censored_event_count"), label=f"{scene} censored_event_count"
    )
    if (
        event_count == 0
        or len(events) != event_count
        or observable_count == 0
        or observable_count + unobservable_count != event_count
        or recovered_count + censored_count != observable_count
    ):
        raise ValueError(f"{scene} event/censor counts are inconsistent")

    observed_recovered = 0
    observed_censored = 0
    observed_unobservable = 0
    for event_id, raw in events.items():
        if not str(event_id).startswith(f"{scene}_event_"):
            raise ValueError(f"{scene} event identity mismatch")
        event = _mapping(raw, f"{scene} event {event_id}")
        intervention = _nonnegative_int(
            event.get("intervention_frame_id"), label=f"{event_id} intervention"
        )
        if event.get("frame_ids") != list(range(intervention, intervention + 451, 50)):
            raise ValueError(f"{event_id} does not use the fixed censor grid")
        observable = event.get("background_observable")
        overlapping = event.get("overlapping_intervention")
        if type(observable) is not bool or type(overlapping) is not bool:
            raise ValueError(f"{event_id} observability/censor flags must be booleans")
        if not observable:
            observed_unobservable += 1
            if (
                event.get("recovered") is not None
                or event.get("recovery_frames") is not None
                or event.get("right_censored") is not False
                or event.get("censor_frame") is not None
                or event.get("censor_reason") != "unobservable_revealed_target"
            ):
                raise ValueError(f"{event_id} unobservable recovery fields must be null")
            continue

        recovered = event.get("recovered")
        right_censored = event.get("right_censored")
        recovery_frames = event.get("recovery_frames")
        censor_frame = event.get("censor_frame")
        if (
            type(recovered) is not bool
            or type(right_censored) is not bool
            or right_censored is recovered
            or type(recovery_frames) is not int
            or not 0 <= recovery_frames <= 450
            or recovery_frames % 50
            or type(censor_frame) is not int
            or not intervention < censor_frame <= intervention + 450
        ):
            raise ValueError(f"{event_id} observable recovery/censor fields are invalid")
        if recovered:
            observed_recovered += 1
            if event.get("censor_reason") is not None:
                raise ValueError(f"{event_id} recovered event cannot have a censor reason")
        else:
            observed_censored += 1
            expected_reason = (
                "overlapping_intervention" if overlapping else "administrative_horizon"
            )
            if event.get("censor_reason") != expected_reason:
                raise ValueError(f"{event_id} censor reason mismatch")
    if (
        observed_recovered != recovered_count
        or observed_censored != censored_count
        or observed_unobservable != unobservable_count
    ):
        raise ValueError(f"{scene} event/censor detail counts are inconsistent")


def _validate_target_package(
    manifest_record: Mapping[str, object],
    arrays_record: Mapping[str, object],
    schedule_record: Mapping[str, object],
) -> dict[str, Any]:
    manifest_path = Path(str(manifest_record["path"]))
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read common-v2 target package") from error
    target = _mapping(payload, "target package")
    metadata = _mapping(target.get("metadata"), "target metadata")
    if not (
        target.get("schema_version") == 1
        and target.get("manifest_id") == "tesse_cd_common_v2_targets"
        and target.get("dataset") == "TESSE-CD"
        and target.get("status") == "GENERATED"
        and target.get("targets_generated") is True
        and target.get("prediction_inputs_used") is False
        and metadata.get("prediction_inputs_used") is False
        and metadata.get("protocol_complete") is True
        and metadata.get("window_frames") == 450
        and metadata.get("voxel_size_m") == 0.05
        and set(metadata.get("scenes", ())) == {"apartment", "office"}
    ):
        raise ValueError("target package is not complete and prediction-independent")

    observed_arrays = _declared_record(
        target.get("target_arrays"),
        label="target arrays",
        base=manifest_path.parent,
    )
    if observed_arrays != dict(arrays_record):
        raise ValueError("scene summaries do not bind the target arrays")
    target_sources = target.get("sources")
    if not isinstance(target_sources, list) or not target_sources:
        raise ValueError("target package sources must be non-empty")
    validated_sources = []
    for declaration in target_sources:
        record = _declared_record(declaration, label="target source")
        validated_sources.append(record)

    source_manifest = _declared_record(
        metadata.get("source_manifest"), label="target source manifest"
    )
    declared_schedule = _declared_record(
        metadata.get("schedule"), label="target schedule"
    )
    if declared_schedule != dict(schedule_record):
        raise ValueError("target package schedule binding mismatch")
    declared_sources = _mapping(
        metadata.get("declared_source_records"), "target declared source records"
    )
    if not declared_sources:
        raise ValueError("target declared source records must be non-empty")
    validated_declared_sources = {}
    for label, declaration in declared_sources.items():
        record = _declared_record(declaration, label="target source")
        validated_declared_sources[str(label)] = record

    observability = _mapping(
        metadata.get("background_observable_by_event"), "target observability"
    )
    if not observability or any(type(value) is not bool for value in observability.values()):
        raise ValueError("target observability must contain prediction-independent booleans")
    arrays = _mapping(
        _mapping(target.get("target_arrays"), "target arrays").get("arrays"),
        "target array declarations",
    )
    revealed = {
        str(name).removesuffix(".revealed_background"): _mapping(
            declaration, f"target array {name}"
        )
        for name, declaration in arrays.items()
        if str(name).endswith(".revealed_background")
    }
    if set(revealed) != set(observability) or any(
        bool(
            _nonnegative_int(
                declaration.get("element_count"), label=f"{event_id} target elements"
            )
        )
        is not observability[event_id]
        for event_id, declaration in revealed.items()
    ):
        raise ValueError("target observability disagrees with target arrays")
    observable_count = sum(observability.values())
    if (
        metadata.get("background_observable_event_count") != observable_count
        or metadata.get("unobservable_revealed_target_event_count")
        != len(observability) - observable_count
        or any(
            not any(
                value
                for event_id, value in observability.items()
                if str(event_id).startswith(f"{scene}_event_")
            )
            for scene in ("apartment", "office")
        )
    ):
        raise ValueError("target observability counts are inconsistent")
    return {
        "manifest": dict(manifest_record),
        "target_arrays": dict(arrays_record),
        "prediction_inputs_used": False,
        "background_observable_by_event": dict(observability),
        "sources": validated_sources,
        "declared_source_records": validated_declared_sources,
    }


def _validate_scene_observability(
    summary: Mapping[str, Any], *, scene: str, observability: Mapping[str, bool]
) -> None:
    events = _mapping(
        _mapping(summary.get("metrics"), f"{scene} metrics").get("events"),
        f"{scene} events",
    )
    expected = {
        event_id: observable
        for event_id, observable in observability.items()
        if str(event_id).startswith(f"{scene}_event_")
    }
    if set(events) != set(expected) or any(
        _mapping(events[event_id], f"{scene} event {event_id}").get(
            "background_observable"
        )
        is not observable
        for event_id, observable in expected.items()
    ):
        raise ValueError(f"{scene} summary observability disagrees with target package")


def _load_summary(path: Path, *, scene: str, method: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read scene summary: {path}") from error
    summary = dict(_mapping(payload, "scene summary"))
    if not (
        summary.get("schema_version") == 1
        and summary.get("manifest_id") == "tesse_cd_common_v2_scene_summary"
        and summary.get("dataset") == "TESSE-CD"
        and summary.get("protocol") == "tesse_cd_common_v2"
        and summary.get("status") == "PASS"
        and summary.get("method") == method
        and summary.get("mode") == METHODS[method]["summary_mode"]
        and summary.get("scene") == scene
    ):
        raise ValueError(f"{scene} scene summary identity mismatch")
    metrics = _mapping(summary.get("metrics"), f"{scene} metrics")
    for name in METRICS:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{scene}.{name} must be numeric")
        number = float(value)
        upper = 450.0 if name == "recovery_frames" else 1.0
        if not math.isfinite(number) or not 0.0 <= number <= upper:
            raise ValueError(f"{scene}.{name} is outside its valid range")
    _validate_metric_details(metrics, scene=scene)
    sources = _mapping(summary.get("sources"), f"{scene} sources")
    if not REQUIRED_SUMMARY_SOURCES <= set(sources):
        raise ValueError(f"{scene} summary source coverage is incomplete")
    for label, raw in sources.items():
        source = _mapping(raw, f"{scene} source {label}")
        path_value = Path(str(source.get("path", "")))
        record = _file_record(path_value)
        if (
            source.get("sha256") != record["sha256"]
            or source.get("byte_count") != record["byte_count"]
        ):
            raise ValueError(f"{scene} source hash mismatch: {label}")
    return summary


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def finalize_common_v2(
    apartment_summary: Path,
    apartment_repeat: Path,
    office_summary: Path,
    office_repeat: Path,
    *,
    method: str,
    run_id: str,
    output: Path,
) -> Path:
    if method not in METHODS:
        raise ValueError(f"unsupported common-v2 method: {method}")
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    pairs = {
        "apartment": (apartment_summary, apartment_repeat),
        "office": (office_summary, office_repeat),
    }
    summaries: dict[str, dict[str, Any]] = {}
    summary_records: dict[str, dict[str, dict[str, object]]] = {}
    for scene, (primary, repeat) in pairs.items():
        for label, path in (("primary", primary), ("repeat", repeat)):
            try:
                file_stat = path.stat(follow_symlinks=False)
            except OSError as error:
                raise ValueError(
                    f"{scene} {label} must be a regular file"
                ) from error
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"{scene} {label} must be a regular file")
        if os.path.samefile(primary, repeat):
            raise ValueError(f"{scene} primary and repeat must be independent files")
        if primary.read_bytes() != repeat.read_bytes():
            raise ValueError(f"{scene} primary and repeat summaries must be byte-identical")
        summaries[scene] = _load_summary(primary, scene=scene, method=method)
        summary_records[scene] = {
            "primary": _file_record(primary),
            "repeat": _file_record(repeat),
        }

    scene_sources = {
        scene: {
            label: _declared_record(
                declaration,
                label=f"{scene} source {label}",
            )
            for label, declaration in _mapping(
                summary["sources"], f"{scene} sources"
            ).items()
        }
        for scene, summary in summaries.items()
    }
    for label in ("target_manifest", "target_arrays", "schedule", "aliases", "evaluator"):
        if scene_sources["apartment"][label] != scene_sources["office"][label]:
            raise ValueError(
                "Apartment and Office must use the same target package, schedule, aliases, and evaluator"
            )
    target_evidence = _validate_target_package(
        scene_sources["apartment"]["target_manifest"],
        scene_sources["apartment"]["target_arrays"],
        scene_sources["apartment"]["schedule"],
    )
    for scene, summary in summaries.items():
        _validate_scene_observability(
            summary,
            scene=scene,
            observability=target_evidence["background_observable_by_event"],
        )

    metrics = {
        name: fmean(
            float(summaries[scene]["metrics"][name])
            for scene in ("apartment", "office")
        )
        for name in METRICS
    }
    token_bindings = [
        {
            "token": f"T2_{method}_{token_suffix}",
            "json_pointer": f"/metrics/{name}",
            "precision": 3,
        }
        for name, token_suffix in METRICS.items()
    ]
    method_config = METHODS[method]
    protocol_deviations: list[str] = []
    if method_config["summary_mode"] == "frozen":
        protocol_deviations.extend(
            [
                "The method is frozen at the first intervention and the same pre-intervention map is queried at every common-v2 checkpoint.",
                "No post-freeze mapper update or future-state backfill is permitted.",
            ]
        )
    elif method == "PANOPTIC_SHARED":
        protocol_deviations.append(
            "Panoptic Mapping uses the predeclared shared-mask composition."
        )
    elif method == "KHRONOS_OPEN":
        protocol_deviations.append(
            "Open-set features use the frozen source-identical CLIP text head; instance-mask IDs are never treated as class IDs."
        )
    elif method == "KHRONOS_ORACLE":
        protocol_deviations.extend(
            [
                "This row uses ground-truth semantic input and is excluded from ranking.",
                "A native state may persist beyond the last internal map change only within hash-bound processed-input timestamp coverage.",
            ]
        )
    result = {
        "schema_version": 1,
        "status": "VERIFIED",
        "run_id": run_id,
        "method": {
            "key": method,
            "display_label": method_config["display_label"],
            "mode": method_config["mode"],
            "eligible_for_ranking": method_config["eligible_for_ranking"],
        },
        "dataset": {"name": "TESSE-CD", "splits": ["macro_test"]},
        "protocol": {
            "name": "tesse_cd_common_v2",
            "aggregation": "macro mean over Apartment and Office scene summaries",
            "deterministic_repeat": "byte-identical per scene",
            "recovery": "F@5cm >= 0.9 for three checkpoints; 450-frame restricted censor",
        },
        "metrics": metrics,
        "scene_metrics": {
            scene: {
                name: float(summary["metrics"][name]) for name in METRICS
            }
            for scene, summary in summaries.items()
        },
        "scene_details": {
            scene: dict(_mapping(summary["metrics"], f"{scene} metrics"))
            for scene, summary in summaries.items()
        },
        "scene_summaries": summary_records,
        "scene_sources": scene_sources,
        "target_package": target_evidence,
        "token_bindings": token_bindings,
        "unavailable_bindings": [],
        "protocol_deviations": protocol_deviations,
        "finalizer": _file_record(Path(__file__)),
    }
    _atomic_json(output, result)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apartment-summary", type=Path, required=True)
    parser.add_argument("--apartment-repeat", type=Path, required=True)
    parser.add_argument("--office-summary", type=Path, required=True)
    parser.add_argument("--office-repeat", type=Path, required=True)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    finalize_common_v2(
        args.apartment_summary,
        args.apartment_repeat,
        args.office_summary,
        args.office_repeat,
        method=args.method,
        run_id=args.run_id,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
