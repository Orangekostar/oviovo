#!/usr/bin/env python3
"""Finalize OVIV2 common-v2 metrics from canonical independent repeats."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import fmean
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation import finalize_tesse_common_v2 as pinned_common
from scripts.evaluation.canonicalize_tesse_common_v2_summary import (
    EXTERNAL_SOURCE_ROLES,
    canonical_summary_bytes,
    capture_and_canonicalize_summary,
)
from scripts.evaluation.finalize_tesse_t2 import (
    _absolute_lexical,
    _atomic_json_no_replace,
    _open_directory_no_symlinks,
    _stable_regular_file,
)
from scripts.evaluation.freeze_oviv2_tesse_cd import (
    STAGE3_LINEAGE_COMMIT,
    _validate_repository,
)
from src.evaluation.json_contracts import loads_strict


SCENES = ("apartment", "office")
REPEATS = (1, 2)
_REQUIRED_SHARED_BINDINGS = frozenset(
    {
        "common_target_manifest",
        "common_target_arrays",
        "schedule",
        "alias_map",
        "evaluator",
        "finalizers",
    }
)


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, object]]:
    digest, byte_count, content = _stable_regular_file(path, label=label, capture=True)
    assert content is not None
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    payload = loads_strict(text, label=label)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a mapping")
    return dict(payload), {
        "role": "freeze_manifest",
        "sha256": digest,
        "byte_count": byte_count,
    }


def _verified_binding(
    value: object,
    *,
    label: str,
    expected_path: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    declaration = _mapping(value, label=f"{label} binding")
    if set(declaration) != {"path", "sha256", "byte_count"}:
        raise ValueError(f"{label} binding fields are invalid")
    raw_path = declaration.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"{label} path must be absolute")
    path = _absolute_lexical(Path(raw_path))
    if raw_path != os.fspath(path):
        raise ValueError(f"{label} path must be canonical")
    if expected_path is not None and path != _absolute_lexical(expected_path):
        raise ValueError(f"{label} path mismatch")
    digest, byte_count, _ = _stable_regular_file(
        path, label=f"{label} file", capture=False
    )
    if (
        not isinstance(declaration.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(declaration.get("sha256")))
        or type(declaration.get("byte_count")) is not int
        or declaration.get("sha256") != digest
        or declaration.get("byte_count") != byte_count
    ):
        raise ValueError(f"{label} content mismatch")
    return path, {
        "role": label,
        "sha256": digest,
        "byte_count": byte_count,
    }


def _direct_root(path: Path, *, label: str) -> Path:
    absolute = _absolute_lexical(path)
    descriptor, _ = _open_directory_no_symlinks(absolute, label=label)
    os.close(descriptor)
    return absolute


def _canonical_record(payload: Mapping[str, Any], *, role: str) -> dict[str, object]:
    data = canonical_summary_bytes(payload)
    return {
        "role": role,
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _validate_captured_summary(
    payload: Mapping[str, Any], *, scene: str
) -> None:
    if not (
        payload.get("schema_version") == 1
        and payload.get("manifest_id") == "tesse_cd_common_v2_scene_summary"
        and payload.get("dataset") == "TESSE-CD"
        and payload.get("protocol") == "tesse_cd_common_v2"
        and payload.get("status") == "PASS"
        and payload.get("method") == "OVIV2"
        and payload.get("mode") == pinned_common.METHODS["OVIV2"]["summary_mode"]
        and payload.get("scene") == scene
    ):
        raise ValueError(f"{scene} scene summary identity mismatch")
    metrics = _mapping(payload.get("metrics"), label=f"{scene} metrics")
    for name in pinned_common.METRICS:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{scene}.{name} must be numeric")
        number = float(value)
        upper = 450.0 if name == "recovery_frames" else 1.0
        if not math.isfinite(number) or not 0.0 <= number <= upper:
            raise ValueError(f"{scene}.{name} is outside its valid range")
    pinned_common._validate_metric_details(metrics, scene=scene)
    sources = _mapping(payload.get("sources"), label=f"{scene} sources")
    if not pinned_common.REQUIRED_SUMMARY_SOURCES <= set(sources):
        raise ValueError(f"{scene} summary source coverage is incomplete")


def finalize_oviv2_common_v2_release(
    freeze_manifest: Path,
    *,
    run_id: str,
    output: Path,
) -> Path:
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    freeze, freeze_record = _read_json(freeze_manifest, label="freeze manifest")
    if not (
        freeze.get("schema_version") == 1
        and freeze.get("freeze_id") == "oviv2-tessecd-v1"
        and freeze.get("status") == "FROZEN"
        and freeze.get("method") == "OVIV2"
        and freeze.get("dataset") == "TESSE-CD"
    ):
        raise ValueError("freeze manifest identity mismatch")
    repository = _validate_repository(
        _mapping(freeze.get("repository"), label="freeze repository")
    )
    if repository["stage3_lineage_commit"] != STAGE3_LINEAGE_COMMIT:
        raise ValueError("freeze manifest Stage3 lineage mismatch")
    algorithm = _mapping(freeze.get("algorithm"), label="freeze algorithm")
    if not re.fullmatch(r"[0-9a-f]{64}", str(algorithm.get("sha256", ""))):
        raise ValueError("freeze algorithm hash is invalid")

    shared = _mapping(freeze.get("shared_bindings"), label="shared bindings")
    if not _REQUIRED_SHARED_BINDINGS <= set(shared):
        raise ValueError("freeze shared bindings are incomplete")
    target_manifest_path, target_manifest_record = _verified_binding(
        shared["common_target_manifest"], label="common_target_manifest"
    )
    target_arrays_path, target_arrays_record = _verified_binding(
        shared["common_target_arrays"], label="common_target_arrays"
    )
    schedule_path, schedule_record = _verified_binding(
        shared["schedule"], label="schedule"
    )
    aliases_path, aliases_record = _verified_binding(
        shared["alias_map"], label="alias_map"
    )
    evaluator_path, evaluator_record = _verified_binding(
        shared["evaluator"],
        label="evaluator",
        expected_path=ROOT / "scripts/evaluation/evaluate_tesse_cd_common_v2.py",
    )
    finalizers = _mapping(shared["finalizers"], label="frozen finalizers")
    if set(finalizers) != {"common_v2", "official_t2"}:
        raise ValueError("frozen finalizer bindings are incomplete")
    _, common_finalizer_record = _verified_binding(
        finalizers["common_v2"],
        label="common_v2_finalizer",
        expected_path=ROOT / "scripts/evaluation/finalize_tesse_common_v2.py",
    )
    _, official_finalizer_record = _verified_binding(
        finalizers["official_t2"],
        label="official_t2_finalizer",
        expected_path=ROOT / "scripts/evaluation/finalize_tesse_t2.py",
    )

    release_bindings = _mapping(
        freeze.get("release_bindings"), label="release bindings"
    )
    if set(release_bindings) != {
        "canonical_summary_generator",
        "release_finalizer",
        "label_spaces",
    }:
        raise ValueError("freeze release bindings are incomplete")
    _, canonicalizer_record = _verified_binding(
        release_bindings["canonical_summary_generator"],
        label="canonical_summary_generator",
        expected_path=ROOT
        / "scripts/evaluation/canonicalize_tesse_common_v2_summary.py",
    )
    _, release_finalizer_record = _verified_binding(
        release_bindings["release_finalizer"],
        label="release_finalizer",
        expected_path=Path(__file__),
    )
    label_bindings = _mapping(
        release_bindings["label_spaces"], label="label space bindings"
    )
    if set(label_bindings) != set(SCENES):
        raise ValueError("label space bindings must cover Apartment and Office")
    label_spaces: dict[str, Path] = {}
    label_records: dict[str, dict[str, object]] = {}
    for scene in SCENES:
        label_spaces[scene], label_records[scene] = _verified_binding(
            label_bindings[scene], label=f"{scene}_label_space"
        )

    raw_roots = _mapping(freeze.get("output_roots"), label="output roots")
    expected_root_roles = {
        f"{scene}_run{repeat}" for scene in SCENES for repeat in REPEATS
    }
    if set(raw_roots) != expected_root_roles:
        raise ValueError("freeze output roots do not cover exact independent runs")
    roots: dict[tuple[str, int], Path] = {}
    for scene in SCENES:
        for repeat in REPEATS:
            role = f"{scene}_run{repeat}"
            raw = raw_roots[role]
            if not isinstance(raw, str) or not Path(raw).is_absolute():
                raise ValueError(f"{role} root must be absolute")
            if raw != os.fspath(_absolute_lexical(Path(raw))):
                raise ValueError(f"{role} root must be canonical")
            roots[(scene, repeat)] = _direct_root(Path(raw), label=f"{role} root")
    root_values = list(roots.values())
    if len(set(root_values)) != len(root_values) or any(
        os.path.samefile(first, second)
        for index, first in enumerate(root_values)
        for second in root_values[index + 1 :]
    ):
        raise ValueError("release runs must use physically independent artifact roots")

    external_common = {
        "target_manifest": target_manifest_path,
        "target_arrays": target_arrays_path,
        "aliases": aliases_path,
        "evaluator": evaluator_path,
    }
    raw_summaries: dict[tuple[str, int], dict[str, Any]] = {}
    canonical: dict[tuple[str, int], dict[str, Any]] = {}
    summary_paths: dict[tuple[str, int], Path] = {}
    raw_records: dict[tuple[str, int], dict[str, object]] = {}
    for scene in SCENES:
        for repeat in REPEATS:
            key = (scene, repeat)
            summary_path = roots[key] / "evaluation/summary.json"
            summary_paths[key] = summary_path
            raw_payload, canonical_payload, raw_content_record = (
                capture_and_canonicalize_summary(
                    summary_path,
                    artifact_root=roots[key],
                    external_sources={
                        **external_common,
                        "label_space": label_spaces[scene],
                    },
                )
            )
            _validate_captured_summary(raw_payload, scene=scene)
            raw_summaries[key] = raw_payload
            canonical[key] = canonical_payload
            raw_records[key] = {
                "role": f"{scene}.run{repeat}.raw_summary",
                **raw_content_record,
            }

    for scene in SCENES:
        if os.path.samefile(summary_paths[(scene, 1)], summary_paths[(scene, 2)]):
            raise ValueError(f"{scene} summaries must be independent files")
        if canonical_summary_bytes(canonical[(scene, 1)]) != canonical_summary_bytes(
            canonical[(scene, 2)]
        ):
            raise ValueError(
                f"{scene} canonical summaries must be byte-identical"
            )
        primary_sources = _mapping(
            raw_summaries[(scene, 1)].get("sources"),
            label=f"{scene} primary sources",
        )
        repeat_sources = _mapping(
            raw_summaries[(scene, 2)].get("sources"),
            label=f"{scene} repeat sources",
        )
        for role in set(primary_sources) - EXTERNAL_SOURCE_ROLES:
            primary_source = _mapping(
                primary_sources[role], label=f"{scene} primary {role} source"
            )
            repeat_source = _mapping(
                repeat_sources[role], label=f"{scene} repeat {role} source"
            )
            if os.path.samefile(
                Path(str(primary_source["path"])),
                Path(str(repeat_source["path"])),
            ):
                raise ValueError(
                    f"{scene} run-local sources must be independent files: {role}"
                )

    shared_source_roles = (
        "target_manifest",
        "target_arrays",
        "schedule",
        "aliases",
        "evaluator",
    )
    for role in shared_source_roles:
        apartment_source = canonical[("apartment", 1)]["sources"][role]
        office_source = canonical[("office", 1)]["sources"][role]
        if apartment_source != office_source:
            raise ValueError(f"Apartment and Office {role} bindings differ")
    if canonical[("apartment", 1)]["sources"]["schedule"] != {
        "role": "schedule",
        "sha256": schedule_record["sha256"],
        "byte_count": schedule_record["byte_count"],
    }:
        raise ValueError("scene summaries disagree with frozen schedule")

    target_evidence = pinned_common._validate_target_package(
        {
            "path": str(target_manifest_path),
            "sha256": target_manifest_record["sha256"],
            "byte_count": target_manifest_record["byte_count"],
        },
        {
            "path": str(target_arrays_path),
            "sha256": target_arrays_record["sha256"],
            "byte_count": target_arrays_record["byte_count"],
        },
        {
            "path": str(schedule_path),
            "sha256": schedule_record["sha256"],
            "byte_count": schedule_record["byte_count"],
        },
    )
    observability = _mapping(
        target_evidence.get("background_observable_by_event"),
        label="target observability",
    )
    for scene in SCENES:
        pinned_common._validate_scene_observability(
            raw_summaries[(scene, 1)],
            scene=scene,
            observability=observability,
        )

    scene_metrics = {
        scene: {
            name: float(raw_summaries[(scene, 1)]["metrics"][name])
            for name in pinned_common.METRICS
        }
        for scene in SCENES
    }
    metrics = {
        name: fmean(scene_metrics[scene][name] for scene in SCENES)
        for name in pinned_common.METRICS
    }
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("release metrics must be finite")
    token_bindings = [
        {
            "token": f"T2_OVIV2_{suffix}",
            "json_pointer": f"/metrics/{name}",
            "precision": 3,
        }
        for name, suffix in pinned_common.METRICS.items()
    ]
    method_config = pinned_common.METHODS["OVIV2"]
    result = {
        "schema_version": 1,
        "manifest_id": "oviv2_tesse_cd_common_v2_release_v1",
        "status": "VERIFIED",
        "run_id": run_id,
        "method": {
            "key": "OVIV2",
            "display_label": method_config["display_label"],
            "mode": method_config["mode"],
            "eligible_for_ranking": method_config["eligible_for_ranking"],
        },
        "dataset": {"name": "TESSE-CD", "splits": ["macro_test"]},
        "protocol": {
            "name": "tesse_cd_common_v2",
            "aggregation": "macro mean over Apartment and Office scene summaries",
            "deterministic_repeat": "canonical-role-content-byte-identical",
            "recovery": "F@5cm >= 0.9 for three checkpoints; 450-frame restricted censor",
        },
        "metrics": metrics,
        "scene_metrics": scene_metrics,
        "scene_details": {
            scene: dict(raw_summaries[(scene, 1)]["metrics"])
            for scene in SCENES
        },
        "scene_summaries": {
            scene: {
                "primary": _canonical_record(
                    canonical[(scene, 1)], role=f"{scene}.run1.canonical_summary"
                ),
                "repeat": _canonical_record(
                    canonical[(scene, 2)], role=f"{scene}.run2.canonical_summary"
                ),
                "raw_primary": raw_records[(scene, 1)],
                "raw_repeat": raw_records[(scene, 2)],
            }
            for scene in SCENES
        },
        "scene_sources": {
            scene: dict(canonical[(scene, 1)]["sources"]) for scene in SCENES
        },
        "target_package": {
            "manifest": target_manifest_record,
            "target_arrays": target_arrays_record,
            "schedule": schedule_record,
            "prediction_inputs_used": False,
            "background_observable_by_event": dict(observability),
        },
        "frozen_identity": {
            **freeze_record,
            "freeze_id": "oviv2-tessecd-v1",
            "repository_commit": repository["commit"],
            "algorithm_sha256": algorithm["sha256"],
        },
        "validation_tools": {
            "evaluator": evaluator_record,
            "common_v2_finalizer": common_finalizer_record,
            "official_t2_finalizer": official_finalizer_record,
            "canonical_summary_generator": canonicalizer_record,
            "release_finalizer": release_finalizer_record,
            "aliases": aliases_record,
            "label_spaces": label_records,
        },
        "token_bindings": token_bindings,
        "unavailable_bindings": [],
        "protocol_deviations": [],
    }
    _atomic_json_no_replace(output, result)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    finalize_oviv2_common_v2_release(
        args.freeze_manifest,
        run_id=args.run_id,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
