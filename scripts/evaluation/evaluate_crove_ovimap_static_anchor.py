#!/usr/bin/env python3
"""Evaluate and gate the Apartment CROVE OVI-MAP static-anchor candidate."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OBJECT_F1_FLOOR = 0.372762
CHANGE_F1_BASELINE = 0.060853
CURRENT_MIOU_FLOOR = 0.142897
GHOST_RATE_BASELINE = 0.646883
APARTMENT_FRAME_COUNT = 1745
APARTMENT_OFFICIAL_STATE_COUNT = 43


def _metric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        return None
    return normalized


@dataclass(frozen=True)
class ApartmentGateDecision:
    status: str
    office_authorized: bool
    object_f1: float | None
    dynamic_f1: float | None
    change_f1: float | None
    current_miou: float | None
    ghost_rate: float | None
    processed_frames: int | None
    official_state_count: int | None
    failed_gates: tuple[str, ...]

    def to_json_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": self.status,
            "office_authorized": self.office_authorized,
            "metrics": {
                "object_f1": self.object_f1,
                "dynamic_f1": self.dynamic_f1,
                "change_f1": self.change_f1,
                "current_miou": self.current_miou,
                "ghost_rate": self.ghost_rate,
                "processed_frames": self.processed_frames,
                "official_state_count": self.official_state_count,
            },
            "baseline": {
                "object_f1": OBJECT_F1_FLOOR,
                "change_f1": CHANGE_F1_BASELINE,
                "current_miou": CURRENT_MIOU_FLOOR,
                "ghost_rate": GHOST_RATE_BASELINE,
            },
            "deltas": {
                "object_f1": (
                    None
                    if self.object_f1 is None
                    else self.object_f1 - OBJECT_F1_FLOOR
                ),
                "change_f1": (
                    None
                    if self.change_f1 is None
                    else self.change_f1 - CHANGE_F1_BASELINE
                ),
                "current_miou": (
                    None
                    if self.current_miou is None
                    else self.current_miou - CURRENT_MIOU_FLOOR
                ),
                "ghost_rate": (
                    None
                    if self.ghost_rate is None
                    else self.ghost_rate - GHOST_RATE_BASELINE
                ),
            },
            "failed_gates": list(self.failed_gates),
        }


def decide_apartment_gate(
    *,
    obj_f1: object,
    dyn_f1: object,
    chg_f1: object,
    current_miou: object,
    ghost_rate: object,
    processed_frames: object,
    official_state_count: object,
) -> ApartmentGateDecision:
    """Apply the preregistered all-or-nothing Apartment promotion gate."""

    object_value = _metric(obj_f1)
    dynamic_value = _metric(dyn_f1)
    change_value = _metric(chg_f1)
    miou_value = _metric(current_miou)
    ghost_value = _metric(ghost_rate)
    frame_count = (
        int(processed_frames)
        if type(processed_frames) is int and processed_frames >= 0
        else None
    )
    state_count = (
        int(official_state_count)
        if type(official_state_count) is int and official_state_count >= 0
        else None
    )
    gates = (
        ("object_f1_floor", object_value is not None and object_value >= OBJECT_F1_FLOOR),
        ("dynamic_f1_finite", dynamic_value is not None),
        (
            "change_f1_strict_gain",
            change_value is not None and change_value > CHANGE_F1_BASELINE,
        ),
        (
            "current_miou_floor",
            miou_value is not None and miou_value >= CURRENT_MIOU_FLOOR,
        ),
        (
            "ghost_rate_strict_reduction",
            ghost_value is not None and ghost_value < GHOST_RATE_BASELINE,
        ),
        ("complete_frame_coverage", frame_count == APARTMENT_FRAME_COUNT),
        (
            "complete_official_states",
            state_count == APARTMENT_OFFICIAL_STATE_COUNT,
        ),
    )
    failed = tuple(name for name, passed in gates if not passed)
    passed = not failed
    return ApartmentGateDecision(
        status="PASS_APARTMENT" if passed else "REJECTED_RETAIN_A6",
        office_authorized=passed,
        object_f1=object_value,
        dynamic_f1=dynamic_value,
        change_f1=change_value,
        current_miou=miou_value,
        ghost_rate=ghost_value,
        processed_frames=frame_count,
        official_state_count=state_count,
        failed_gates=failed,
    )


@dataclass(frozen=True)
class GateEvaluationDependencies:
    common_evaluator: Callable[[Path, Path, Path, Path, Path], Path]
    official_summarizer: Callable[[Path], Mapping[str, object]]


def _default_common_evaluator(
    temporal_index: Path,
    target_manifest: Path,
    aliases: Path,
    label_space: Path,
    output: Path,
) -> Path:
    from scripts.evaluation.evaluate_tesse_cd_common_v2 import evaluate_common_v2

    return evaluate_common_v2(
        temporal_index,
        target_manifest,
        aliases,
        label_space,
        output,
    )


def _default_official_summarizer(results_dir: Path) -> Mapping[str, object]:
    from src.evaluation.baselines.tesse_cd import summarize_khronos_official_metrics

    return summarize_khronos_official_metrics(results_dir)


DEFAULT_DEPENDENCIES = GateEvaluationDependencies(
    common_evaluator=_default_common_evaluator,
    official_summarizer=_default_official_summarizer,
)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _regular_bytes(path: Path, *, label: str) -> bytes:
    path = Path(os.path.abspath(os.fspath(path)))
    before = path.stat(follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a direct regular file")
    data = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(data) != after.st_size
    ):
        raise RuntimeError(f"{label} changed while reading")
    return data


def _record(path: Path, *, root: Path | None = None) -> dict[str, object]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    data = _regular_bytes(absolute, label="gate source")
    return {
        "path": (
            str(absolute)
            if root is None
            else absolute.relative_to(root).as_posix()
        ),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    data = _regular_bytes(path, label=label)
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value, data


def _bound_source_index(
    composition_manifest: Path, composition: Mapping[str, Any]
) -> tuple[Path, dict[str, object]]:
    declared = composition.get("source_index")
    if not isinstance(declared, Mapping) or set(declared) != {
        "path",
        "sha256",
        "byte_count",
    }:
        raise ValueError("composition source index binding is invalid")
    raw_path = declared.get("path")
    if not isinstance(raw_path, str):
        raise ValueError("composition source index path is invalid")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("composition source index must stay within the run")
    path = composition_manifest.parent / relative
    observed = _record(path)
    compared = dict(observed)
    compared["path"] = relative.as_posix()
    if compared != dict(declared):
        raise ValueError("composition source index binding mismatch")
    return path, observed


def _atomic_json(path: Path, value: object) -> None:
    data = _canonical_json(value)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _revalidate(records: Sequence[tuple[Path, Mapping[str, object]]]) -> None:
    for path, expected in records:
        if _record(path) != dict(expected):
            raise RuntimeError(f"gate input changed during evaluation: {path}")


def evaluate_apartment_candidate(
    *,
    composition_manifest: Path,
    official_results_dir: Path,
    target_manifest: Path,
    aliases: Path,
    label_space: Path,
    output_root: Path,
    dependencies: GateEvaluationDependencies = DEFAULT_DEPENDENCIES,
) -> Path:
    """Repeat both metric paths and publish one immutable gate receipt."""

    if not isinstance(dependencies, GateEvaluationDependencies):
        raise TypeError("dependencies must be GateEvaluationDependencies")
    composition_manifest = Path(
        os.path.abspath(os.fspath(composition_manifest))
    )
    official_results_dir = Path(
        os.path.abspath(os.fspath(official_results_dir))
    )
    target_manifest = Path(os.path.abspath(os.fspath(target_manifest)))
    aliases = Path(os.path.abspath(os.fspath(aliases)))
    label_space = Path(os.path.abspath(os.fspath(label_space)))
    output_root = Path(os.path.abspath(os.fspath(output_root)))
    if output_root.exists():
        raise ValueError(f"gate output already exists: {output_root}")
    composition, composition_bytes = _json(
        composition_manifest, label="composition manifest"
    )
    checkpoints = composition.get("checkpoints")
    if (
        composition.get("schema_version") != 1
        or composition.get("status") != "PASS"
        or composition.get("dataset") != "TESSE-CD"
        or composition.get("scene") != "apartment"
        or composition.get("integration") != "composed"
        or composition.get("processed_frame_count") != APARTMENT_FRAME_COUNT
        or not isinstance(checkpoints, list)
        or len(checkpoints) != APARTMENT_OFFICIAL_STATE_COUNT
    ):
        raise ValueError("composition is not a complete Apartment candidate")
    temporal_index, temporal_record = _bound_source_index(
        composition_manifest, composition
    )
    input_records: list[tuple[Path, dict[str, object]]] = [
        (
            composition_manifest,
            {
                "path": str(composition_manifest),
                "sha256": hashlib.sha256(composition_bytes).hexdigest(),
                "byte_count": len(composition_bytes),
            },
        ),
        (temporal_index, temporal_record),
        (target_manifest, _record(target_manifest)),
        (aliases, _record(aliases)),
        (label_space, _record(label_space)),
    ]
    official_records: dict[str, dict[str, object]] = {}
    for name in ("static_objects.csv", "dynamic_objects.csv", "background_mesh.csv"):
        path = official_results_dir / name
        record = _record(path)
        official_records[name] = record
        input_records.append((path, record))

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        common_paths = [
            dependencies.common_evaluator(
                temporal_index,
                target_manifest,
                aliases,
                label_space,
                staging / f"common-repeat-{index}",
            )
            for index in (1, 2)
        ]
        common_payloads = [_json(path, label="common-v2 summary") for path in common_paths]
        if common_payloads[0][1] != common_payloads[1][1]:
            raise RuntimeError("common-v2 metric repeat is not byte-identical")
        common = common_payloads[0][0]
        common_metrics = common.get("metrics")
        if (
            common.get("status") != "PASS"
            or common.get("dataset") != "TESSE-CD"
            or common.get("scene") != "apartment"
            or not isinstance(common_metrics, Mapping)
        ):
            raise ValueError("common-v2 summary identity is invalid")

        official_values = [
            dict(dependencies.official_summarizer(official_results_dir))
            for _ in range(2)
        ]
        if _canonical_json(official_values[0]) != _canonical_json(official_values[1]):
            raise RuntimeError("official metric repeat is not byte-identical")
        official = official_values[0]
        decision = decide_apartment_gate(
            obj_f1=official.get("object_f1"),
            dyn_f1=official.get("dynamic_f1"),
            chg_f1=official.get("change_f1"),
            current_miou=common_metrics.get("current_miou"),
            ghost_rate=common_metrics.get("ghost_rate"),
            processed_frames=composition["processed_frame_count"],
            official_state_count=official.get("state_count"),
        )
        _revalidate(input_records)
        receipt = {
            "schema_version": 1,
            "manifest_id": "crove_ovimap_static_anchor_apartment_gate_v1",
            "dataset": "TESSE-CD",
            "scene": "apartment",
            "decision": decision.to_json_record(),
            "metrics": {
                "official": official,
                "common_v2": dict(common_metrics),
            },
            "sources": {
                "composition_manifest": input_records[0][1],
                "temporal_index": temporal_record,
                "target_manifest": input_records[2][1],
                "aliases": input_records[3][1],
                "label_space": input_records[4][1],
                "official_results": official_records,
                "common_repeat_1": _record(common_paths[0], root=staging),
                "common_repeat_2": _record(common_paths[1], root=staging),
                "evaluator": _record(Path(__file__)),
            },
        }
        receipt_path = staging / "gate_decision.json"
        _atomic_json(receipt_path, receipt)
        _revalidate(input_records)
        os.replace(staging, output_root)
        return output_root / "gate_decision.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-manifest", type=Path, required=True)
    parser.add_argument("--official-results-dir", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--label-space", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    evaluate_apartment_candidate(
        composition_manifest=args.composition_manifest,
        official_results_dir=args.official_results_dir,
        target_manifest=args.target_manifest,
        aliases=args.aliases,
        label_space=args.label_space,
        output_root=args.output_root,
    )
    return 0


__all__ = [
    "ApartmentGateDecision",
    "GateEvaluationDependencies",
    "decide_apartment_gate",
    "evaluate_apartment_candidate",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
