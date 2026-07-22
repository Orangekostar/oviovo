#!/usr/bin/env python3
"""Finalize official TESSE-CD T2 metrics with explicit partial availability."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCENE_METRICS = ("object_f1", "dynamic_f1", "change_f1")
METHOD_MODES = {
    "OVIMAP_FROZEN": "frozen",
    "CONCEPTGRAPHS_FROZEN": "frozen",
    "DUALMAP": "native",
    "PANOPTIC_SHARED": "composed",
    "KHRONOS_OPEN": "open-set",
    "OVIV2": "causal_checkpoints",
}
TABLE_MODES = {
    **METHOD_MODES,
    "KHRONOS_OPEN": "online",
    "OVIV2": "online",
}
METHOD_DISPLAY_LABELS = {"OVIV2": "OVIV2"}
SOURCE_METHOD_KEYS = {"KHRONOS_OPEN": "KHRONOS"}
METRIC_SOURCE_FILES = {
    "object_f1": "static_objects.csv",
    "dynamic_f1": "dynamic_objects.csv",
    "change_f1": "static_objects.csv",
}


def _scene_metrics(
    payload: Mapping[str, Any],
    *,
    scene: str,
    method_key: str,
    mode: str,
) -> tuple[dict[str, float | None], dict[str, str]]:
    if payload.get("status") not in {None, "PASS", "PARTIAL"}:
        raise ValueError(f"{scene} official metric source is not usable")
    if payload.get("dataset") != "TESSE-CD":
        raise ValueError(f"{scene} source is not TESSE-CD")
    if payload.get("scene") != scene or payload.get("split") != f"{scene}_test":
        raise ValueError(f"expected {scene} official metric source")
    source_method = SOURCE_METHOD_KEYS.get(method_key, method_key)
    if payload.get("method") != source_method or payload.get("mode") != mode:
        raise ValueError(f"{scene} method or mode mismatch")
    source = payload.get("metrics")
    if not isinstance(source, Mapping):
        raise ValueError(f"{scene} source has no metrics")
    unavailable_source = payload.get("unavailable", {})
    if not isinstance(unavailable_source, Mapping):
        raise ValueError(f"{scene} unavailable reasons must be a mapping")

    metrics: dict[str, float | None] = {}
    unavailable: dict[str, str] = {}
    for name in SCENE_METRICS:
        value = source.get(name)
        if value is None:
            reason = str(unavailable_source.get(name, "")).strip()
            if not reason:
                raise ValueError(f"{scene}.{name} requires an unavailable reason")
            metrics[name] = None
            unavailable[name] = reason
            continue
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{scene}.{name} must be finite and within [0, 1]")
        metrics[name] = number
    return metrics, unavailable


def build_partial_official_metrics(
    apartment: Mapping[str, Any],
    office: Mapping[str, Any],
    *,
    method_key: str,
    mode: str,
) -> tuple[dict[str, dict[str, float | None]], dict[str, dict[str, str]]]:
    if METHOD_MODES.get(method_key) != mode:
        raise ValueError("unsupported TESSE-CD method or mode")
    metrics: dict[str, dict[str, float | None]] = {}
    unavailable: dict[str, dict[str, str]] = {}
    for scene, payload in (("apartment", apartment), ("office", office)):
        scene_metrics, scene_unavailable = _scene_metrics(
            payload,
            scene=scene,
            method_key=method_key,
            mode=mode,
        )
        metrics[scene] = scene_metrics
        unavailable[scene] = scene_unavailable
    return metrics, unavailable


def partial_official_token_bindings(
    method_key: str,
    metrics: Mapping[str, Mapping[str, float | None]],
) -> list[dict[str, object]]:
    if method_key not in METHOD_MODES:
        raise ValueError("unsupported TESSE-CD method")
    return [
        {
            "token": f"T2_{method_key}_{scene.upper()}_{metric.upper()}",
            "json_pointer": f"/metrics/{scene}/{metric}",
            "precision": 3,
        }
        for scene in ("apartment", "office")
        for metric in SCENE_METRICS
        if metrics[scene][metric] is not None
    ]


def partial_official_unavailable_bindings(
    method_key: str,
    metrics: Mapping[str, Mapping[str, float | None]],
) -> list[dict[str, str]]:
    if method_key not in METHOD_MODES:
        raise ValueError("unsupported TESSE-CD method")
    return [
        {
            "token": f"T2_{method_key}_{scene.upper()}_{metric.upper()}",
            "reason_pointer": f"/unavailable/{scene}/{metric}",
            "evidence_pointer": f"/unavailable_evidence/{scene}/{metric}",
        }
        for scene in ("apartment", "office")
        for metric in SCENE_METRICS
        if metrics[scene][metric] is None
    ]


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required provenance file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_declared_source(
    entry: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    path = Path(str(entry.get("path", ""))).resolve()
    if not path.is_file():
        raise ValueError(f"{label} source is missing: {path}")
    byte_count = entry.get("byte_count")
    if type(byte_count) is not int or byte_count != path.stat().st_size:
        raise ValueError(f"{label} source byte count mismatch")
    observed = _sha256(path)
    if entry.get("sha256") != observed:
        raise ValueError(f"{label} source SHA256 mismatch")
    return {
        "path": str(path),
        "sha256": observed,
        "byte_count": byte_count,
    }


def _source_bound_unavailable(
    payload: Mapping[str, Any],
    unavailable: Mapping[str, str],
    *,
    scene: str,
) -> dict[str, dict[str, Any]]:
    if not unavailable:
        return {}
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise ValueError(f"{scene} unavailable metrics require source records")
    by_name: dict[str, Mapping[str, Any]] = {}
    for raw in sources:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{scene} metric source record must be a mapping")
        name = Path(str(raw.get("path", ""))).name
        if not name or name in by_name:
            raise ValueError(f"{scene} metric source names must be unique")
        by_name[name] = raw

    evidence: dict[str, dict[str, Any]] = {}
    for metric, reason in unavailable.items():
        filename = METRIC_SOURCE_FILES[metric]
        if filename not in by_name:
            raise ValueError(f"{scene}.{metric} source is missing: {filename}")
        evidence[metric] = {
            "reason": reason,
            "source": _validated_declared_source(
                by_name[filename], label=f"{scene}.{metric}"
            ),
        }
    return evidence


def _hashed_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(entry["path"]))
    result = dict(entry)
    result.update(
        path=str(path.resolve()),
        sha256=_sha256(path),
        byte_count=path.stat().st_size,
    )
    return result


def _hashed_entries(
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"provenance {label} must be a list")
    return [_hashed_entry(entry) for entry in value]


def _validate_run_status(
    status: Mapping[str, Any],
    *,
    scene: str,
    method_key: str,
    mode: str,
) -> None:
    if status.get("status") != "PASS":
        raise ValueError(f"{scene} run status must be PASS")
    if status.get("scene") != scene or status.get("mode") != mode:
        raise ValueError(f"{scene} run identity mismatch")
    source_method = SOURCE_METHOD_KEYS.get(method_key, method_key)
    if status.get("method") not in {None, method_key, source_method}:
        raise ValueError(f"{scene} run method mismatch")
    if mode == "frozen" and int(status.get("updates_after_freeze", -1)) != 0:
        raise ValueError(f"{scene} updates_after_freeze must be zero")


def build_scene_evidence(
    metrics_path: Path,
    status_path: Path,
    *,
    method_key: str,
    mode: str,
) -> dict[str, Any]:
    if METHOD_MODES.get(method_key) != mode:
        raise ValueError("unsupported TESSE-CD method or mode")
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    if not isinstance(metrics_payload, Mapping) or not isinstance(
        status_payload, Mapping
    ):
        raise ValueError("scene evidence inputs must contain JSON objects")
    scene = str(metrics_payload.get("scene", ""))
    if scene not in {"apartment", "office"}:
        raise ValueError("scene evidence requires apartment or office metrics")
    metrics, unavailable = _scene_metrics(
        metrics_payload,
        scene=scene,
        method_key=method_key,
        mode=mode,
    )
    _validate_run_status(
        status_payload,
        scene=scene,
        method_key=method_key,
        mode=mode,
    )
    unavailable_evidence = _source_bound_unavailable(
        metrics_payload, unavailable, scene=scene
    )
    return {
        "schema_version": 1,
        "manifest_id": "tesse_cd_t2_scene_evidence",
        "status": "PASS",
        "dataset": "TESSE-CD",
        "scene": scene,
        "method_key": method_key,
        "mode": mode,
        "metrics": metrics,
        "unavailable": unavailable,
        "unavailable_evidence": unavailable_evidence,
        "official_metrics_source": _hashed_entry({"path": str(metrics_path)}),
        "run_status_source": _hashed_entry({"path": str(status_path)}),
        "official_metrics": dict(metrics_payload),
        "run_status": dict(status_payload),
    }


def _build_result_payload(
    apartment: Mapping[str, Any],
    office: Mapping[str, Any],
    apartment_status: Mapping[str, Any],
    office_status: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    method_key: str,
    mode: str,
) -> dict[str, Any]:
    metrics, unavailable = build_partial_official_metrics(
        apartment,
        office,
        method_key=method_key,
        mode=mode,
    )
    unavailable_evidence = {
        "apartment": _source_bound_unavailable(
            apartment, unavailable["apartment"], scene="apartment"
        ),
        "office": _source_bound_unavailable(
            office, unavailable["office"], scene="office"
        ),
    }
    _validate_run_status(
        apartment_status,
        scene="apartment",
        method_key=method_key,
        mode=mode,
    )
    _validate_run_status(
        office_status,
        scene="office",
        method_key=method_key,
        mode=mode,
    )
    dirty_digest = str(provenance.get("dirty_state_digest", ""))
    if len(dirty_digest) != 64 or any(
        char not in "0123456789abcdef" for char in dirty_digest
    ):
        raise ValueError("dirty_state_digest must be a lowercase SHA256")
    commands = provenance.get("commands")
    if not isinstance(commands, list) or not commands or not all(commands):
        raise ValueError("provenance commands must be a non-empty list")

    return {
        "run_id": str(provenance["run_id"]),
        "method": {
            "key": method_key,
            "display_label": METHOD_DISPLAY_LABELS.get(
                method_key, method_key.replace("_", " ").title()
            ),
            "mode": TABLE_MODES[method_key],
            "eligible_for_ranking": True,
        },
        "upstream_commit": str(provenance["upstream_commit"]),
        "adapter_commit": str(provenance["adapter_commit"]),
        "dirty_state_digest": dirty_digest,
        "dataset": {
            "name": "TESSE-CD",
            "splits": ["apartment_test", "office_test"],
            "manifest": _hashed_entry(provenance["dataset_manifest"]),
        },
        "metrics": metrics,
        "unavailable": unavailable,
        "unavailable_evidence": unavailable_evidence,
        "protocol": {
            "name": "Khronos upstream TESSE-CD evaluator via neutral-map bridge",
            "aggregation": "macro over unique online state/query rows per sequence",
            "binding_policy": "only finite official metrics are importable",
        },
        "run_status": {
            "apartment": dict(apartment_status),
            "office": dict(office_status),
        },
        "commands": [str(command) for command in commands],
        "environment": dict(provenance["environment"]),
        "hardware": dict(provenance["hardware"]),
        "seed": int(provenance.get("seed", 0)),
        "configs": _hashed_entries(provenance["configs"], "configs"),
        "weights": _hashed_entries(
            provenance.get("weights", []), "weights", allow_empty=True
        ),
        "raw_outputs": _hashed_entries(provenance["raw_outputs"], "raw_outputs"),
        "protocol_deviations": [
            str(value) for value in provenance.get("protocol_deviations", ())
        ],
        "token_bindings": partial_official_token_bindings(method_key, metrics),
        "unavailable_bindings": partial_official_unavailable_bindings(
            method_key, metrics
        ),
    }


def _json_source(
    path: Path, *, label: str
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError
        content = resolved.read_bytes()
        payload = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"{label} must be a readable JSON object: {path}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return (
        dict(payload),
        {
            "path": str(resolved),
            "sha256": hashlib.sha256(content).hexdigest(),
            "byte_count": len(content),
        },
        content,
    )


def _json_repeat_pair(
    primary: Path, repeat: Path, *, label: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if primary.resolve() == repeat.resolve():
        raise ValueError(f"{label} repeat must use independent files")
    payload, primary_record, primary_bytes = _json_source(primary, label=label)
    _, repeat_record, repeat_bytes = _json_source(repeat, label=f"{label} repeat")
    if primary_bytes != repeat_bytes:
        raise ValueError(f"{label} repeat must be byte-identical")
    return payload, {"primary": primary_record, "repeat": repeat_record}


def build_result(
    apartment_metrics: Path,
    apartment_metrics_repeat: Path,
    office_metrics: Path,
    office_metrics_repeat: Path,
    apartment_status: Path,
    apartment_status_repeat: Path,
    office_status: Path,
    office_status_repeat: Path,
    provenance: Path,
    *,
    method_key: str,
    mode: str,
) -> dict[str, Any]:
    apartment_payload, apartment_metrics_sources = _json_repeat_pair(
        apartment_metrics,
        apartment_metrics_repeat,
        label="apartment metrics",
    )
    office_payload, office_metrics_sources = _json_repeat_pair(
        office_metrics,
        office_metrics_repeat,
        label="office metrics",
    )
    apartment_status_payload, apartment_status_sources = _json_repeat_pair(
        apartment_status,
        apartment_status_repeat,
        label="apartment status",
    )
    office_status_payload, office_status_sources = _json_repeat_pair(
        office_status,
        office_status_repeat,
        label="office status",
    )
    provenance_payload, provenance_source, _ = _json_source(
        provenance, label="provenance"
    )
    result = _build_result_payload(
        apartment_payload,
        office_payload,
        apartment_status_payload,
        office_status_payload,
        provenance_payload,
        method_key=method_key,
        mode=mode,
    )
    result["status"] = "VERIFIED"
    result["protocol"]["deterministic_repeat"] = "byte-identical"
    result["evidence_sources"] = {
        "apartment": {
            "metrics": apartment_metrics_sources,
            "status": apartment_status_sources,
        },
        "office": {
            "metrics": office_metrics_sources,
            "status": office_status_sources,
        },
        "provenance": provenance_source,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apartment-metrics", type=Path)
    parser.add_argument("--apartment-metrics-repeat", type=Path)
    parser.add_argument("--office-metrics", type=Path)
    parser.add_argument("--office-metrics-repeat", type=Path)
    parser.add_argument("--apartment-status", type=Path)
    parser.add_argument("--apartment-status-repeat", type=Path)
    parser.add_argument("--office-status", type=Path)
    parser.add_argument("--office-status-repeat", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--scene-metrics", type=Path)
    parser.add_argument("--scene-status", type=Path)
    parser.add_argument("--method", choices=tuple(METHOD_MODES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.scene_metrics is not None or args.scene_status is not None:
        if args.scene_metrics is None or args.scene_status is None:
            parser.error("scene evidence requires --scene-metrics and --scene-status")
        result = build_scene_evidence(
            args.scene_metrics,
            args.scene_status,
            method_key=args.method,
            mode=METHOD_MODES[args.method],
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
        return 0

    required = {
        "--apartment-metrics": args.apartment_metrics,
        "--apartment-metrics-repeat": args.apartment_metrics_repeat,
        "--office-metrics": args.office_metrics,
        "--office-metrics-repeat": args.office_metrics_repeat,
        "--apartment-status": args.apartment_status,
        "--apartment-status-repeat": args.apartment_status_repeat,
        "--office-status": args.office_status,
        "--office-status-repeat": args.office_status_repeat,
        "--provenance": args.provenance,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("full result requires " + ", ".join(missing))

    result = build_result(
        args.apartment_metrics,
        args.apartment_metrics_repeat,
        args.office_metrics,
        args.office_metrics_repeat,
        args.apartment_status,
        args.apartment_status_repeat,
        args.office_status,
        args.office_status_repeat,
        args.provenance,
        method_key=args.method,
        mode=METHOD_MODES[args.method],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
