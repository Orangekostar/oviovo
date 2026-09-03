#!/usr/bin/env python3
"""Plan, compose, and collect CROVE P6-A anchor counterfactuals."""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import io
import json
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.run_crove_ovimap_static_anchor import compose_run
from src.evaluation.crove_anchor_counterfactual import (
    AnchorCounterfactualFeature,
    CounterfactualVariant,
    build_anchor_counterfactual_features,
    build_counterfactual_metric_rows,
    derive_counterfactual_variants,
)
from src.evaluation.exporters.oviovo import read_map_snapshot

_PLAN_ID = "crove_anchor_counterfactual_plan_v1"
_RECORD_FIELDS = {"path", "sha256", "byte_count"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _regular_bytes(path: Path, *, label: str) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    before = absolute.stat(follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a direct regular file")
    data = absolute.read_bytes()
    after = absolute.stat(follow_symlinks=False)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(data) != after.st_size
    ):
        raise RuntimeError(f"{label} changed while reading")
    return data


def _json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    data = _regular_bytes(path, label=label)
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value, data


def _record(path: Path, data: bytes | None = None) -> dict[str, object]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    content = _regular_bytes(absolute, label="counterfactual source") if data is None else data
    return {
        "path": str(absolute),
        "sha256": _sha256(content),
        "byte_count": len(content),
    }


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: Path, content: bytes) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.exists() or absolute.is_symlink():
        raise FileExistsError(absolute)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{absolute.name}.", dir=absolute.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, absolute)
    finally:
        temporary.unlink(missing_ok=True)


def _diagnostic_transitions(payload: Mapping[str, Any]) -> list[Mapping[str, object]]:
    transitions = payload.get("transitions")
    if (
        payload.get("manifest_id")
        != "crove_ovimap_unbound_visibility_diagnostics_v1"
        or not isinstance(transitions, list)
        or payload.get("transition_count") != len(transitions)
        or any(not isinstance(item, Mapping) for item in transitions)
    ):
        raise ValueError("P5 runtime diagnostics identity is invalid")
    return transitions


def create_counterfactual_plan(
    *,
    p5_runtime_diagnostics: str | Path,
    p2_attribution: str | Path,
    output: str | Path,
) -> Path:
    """Publish the immutable 24-variant P6-A diagnostic plan."""

    diagnostics_path = Path(os.path.abspath(os.fspath(p5_runtime_diagnostics)))
    attribution_path = Path(os.path.abspath(os.fspath(p2_attribution)))
    output_path = Path(os.path.abspath(os.fspath(output)))
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    diagnostics, diagnostics_bytes = _json_object(
        diagnostics_path, label="P5 runtime diagnostics"
    )
    attribution, attribution_bytes = _json_object(
        attribution_path, label="P2 attribution"
    )
    variants = derive_counterfactual_variants(
        _diagnostic_transitions(diagnostics), attribution
    )
    manifest = {
        "schema_version": 1,
        "manifest_id": _PLAN_ID,
        "status": "PASS",
        "dataset": "TESSE-CD",
        "scene": "apartment",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "sources": {
            "p5_runtime_diagnostics": _record(
                diagnostics_path, diagnostics_bytes
            ),
            "p2_attribution": _record(attribution_path, attribution_bytes),
        },
        "variants": [variant.to_json_record() for variant in variants],
    }
    _atomic_bytes(output_path, _canonical_json(manifest))
    return output_path


def _bound_source(record: object, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != _RECORD_FIELDS:
        raise ValueError(f"{label} source binding is invalid")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} source path is invalid")
    path = Path(raw_path)
    if not path.is_absolute():
        raise ValueError(f"{label} source path must be absolute")
    if _record(path) != dict(record):
        raise ValueError(f"{label} source binding mismatch")
    return path


def _bound_relative_file(
    record: object, *, root: Path, label: str
) -> Path:
    if not isinstance(record, Mapping) or set(record) != _RECORD_FIELDS:
        raise ValueError(f"{label} binding is invalid")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path is invalid")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path must remain under its artifact root")
    path = root / relative
    actual = _record(path)
    expected = {**dict(record), "path": str(path.resolve())}
    if actual != expected:
        raise ValueError(f"{label} binding mismatch")
    return path


def _variant_from_record(record: object) -> CounterfactualVariant:
    expected = {
        "variant_id",
        "family",
        "selected_anchor_ids",
        "focal_anchor_id",
        "diagnostic_only",
    }
    if not isinstance(record, Mapping) or set(record) != expected:
        raise ValueError("counterfactual variant record is invalid")
    selected = record.get("selected_anchor_ids")
    if not isinstance(selected, list):
        raise ValueError("counterfactual selected anchors are invalid")
    return CounterfactualVariant(
        variant_id=record.get("variant_id"),  # type: ignore[arg-type]
        family=record.get("family"),  # type: ignore[arg-type]
        selected_anchor_ids=tuple(selected),
        focal_anchor_id=record.get("focal_anchor_id"),  # type: ignore[arg-type]
        diagnostic_only=record.get("diagnostic_only"),  # type: ignore[arg-type]
    )


def _load_plan(path: Path) -> tuple[dict[str, Any], tuple[CounterfactualVariant, ...]]:
    payload, _ = _json_object(path, label="counterfactual plan")
    sources = payload.get("sources")
    raw_variants = payload.get("variants")
    if (
        payload.get("schema_version") != 1
        or payload.get("manifest_id") != _PLAN_ID
        or payload.get("status") != "PASS"
        or payload.get("dataset") != "TESSE-CD"
        or payload.get("scene") != "apartment"
        or payload.get("diagnostic_only") is not True
        or payload.get("promotion_eligible") is not False
        or not isinstance(sources, Mapping)
        or set(sources) != {"p5_runtime_diagnostics", "p2_attribution"}
        or not isinstance(raw_variants, list)
    ):
        raise ValueError("counterfactual plan identity is invalid")
    diagnostics_path = _bound_source(
        sources["p5_runtime_diagnostics"], label="P5 runtime diagnostics"
    )
    attribution_path = _bound_source(
        sources["p2_attribution"], label="P2 attribution"
    )
    diagnostics, _ = _json_object(
        diagnostics_path, label="P5 runtime diagnostics"
    )
    attribution, _ = _json_object(attribution_path, label="P2 attribution")
    expected = derive_counterfactual_variants(
        _diagnostic_transitions(diagnostics), attribution
    )
    observed = tuple(_variant_from_record(record) for record in raw_variants)
    if observed != expected:
        raise ValueError("counterfactual plan variants do not match bound evidence")
    return payload, observed


def load_counterfactual_plan_variant(
    plan: str | Path, variant_id: str
) -> CounterfactualVariant:
    """Load one variant after revalidating the full evidence-bound plan."""

    _, variants = _load_plan(Path(os.path.abspath(os.fspath(plan))))
    matches = [variant for variant in variants if variant.variant_id == variant_id]
    if len(matches) != 1:
        raise ValueError(f"counterfactual variant is absent: {variant_id}")
    return matches[0]


def compose_counterfactual_variant(
    *,
    plan: str | Path,
    variant_id: str,
    source_run_manifest: str | Path,
    anchor_manifest: str | Path,
    visibility_policy: str | Path,
    output: str | Path,
    visibility_diagnostics_cache: str | Path | None = None,
) -> Path:
    """Recompose one diagnostic variant from the existing causal source run."""

    variant = load_counterfactual_plan_variant(plan, variant_id)
    return compose_run(
        source_run_manifest=Path(source_run_manifest),
        anchor_manifest=Path(anchor_manifest),
        output_root=Path(output),
        moved_geometry_mode="temporal_compact",
        readout_role="counterfactual_diagnostic",
        visibility_policy=Path(visibility_policy),
        counterfactual_variant=variant,
        visibility_diagnostics_cache=(
            None
            if visibility_diagnostics_cache is None
            else Path(visibility_diagnostics_cache)
        ),
    )


def load_variant_gate_metrics(
    *,
    gate: str | Path,
    composition: str | Path,
    variant: CounterfactualVariant,
) -> dict[str, object]:
    """Load one measured metric row after validating its composition witness."""

    gate_path = Path(os.path.abspath(os.fspath(gate)))
    composition_path = Path(os.path.abspath(os.fspath(composition)))
    gate_payload, _ = _json_object(gate_path, label="counterfactual gate receipt")
    composition_payload, _ = _json_object(
        composition_path, label="counterfactual composition"
    )
    sources = gate_payload.get("sources")
    if (
        gate_payload.get("schema_version") != 1
        or gate_payload.get("manifest_id")
        != "crove_ovimap_static_anchor_apartment_gate_v1"
        or gate_payload.get("dataset") != "TESSE-CD"
        or gate_payload.get("scene") != "apartment"
        or not isinstance(sources, Mapping)
        or sources.get("composition_manifest") != _record(composition_path)
    ):
        raise ValueError("counterfactual gate composition binding is invalid")
    expected_contract = {
        "diagnostic_only": True,
        "moved_geometry_mode": "temporal_compact",
        "promotion_eligible": False,
        "readout_role": "counterfactual_diagnostic",
        "unbound_anchor_mode": "causal_visibility_filtered",
    }
    if (
        composition_payload.get("schema_version") != 1
        or composition_payload.get("manifest_id")
        != "crove_ovimap_static_anchor_composition_v1"
        or composition_payload.get("status") != "PASS"
        or composition_payload.get("dataset") != "TESSE-CD"
        or composition_payload.get("scene") != "apartment"
        or composition_payload.get("readout_contract") != expected_contract
        or composition_payload.get("counterfactual_variant")
        != variant.to_json_record()
    ):
        raise ValueError("counterfactual composition identity is invalid")
    decision = gate_payload.get("decision")
    metrics = decision.get("metrics") if isinstance(decision, Mapping) else None
    if (
        not isinstance(metrics, Mapping)
        or metrics.get("processed_frames") != 1745
        or metrics.get("official_state_count") != 43
    ):
        raise ValueError("counterfactual gate metric coverage is invalid")
    return {
        "object_f1": metrics.get("object_f1"),
        "dynamic_f1": metrics.get("dynamic_f1"),
        "change_f1": metrics.get("change_f1"),
        "current_miou": metrics.get("current_miou"),
        "ghost_rate": metrics.get("ghost_rate"),
    }


def _load_anchor_snapshot(anchor_manifest: Path):
    payload, _ = _json_object(anchor_manifest, label="anchor manifest")
    outputs = payload.get("outputs")
    if (
        payload.get("schema_version") != 1
        or payload.get("manifest_id") != "crove_ovimap_static_anchor_v1"
        or payload.get("status") != "PASS"
        or payload.get("scene") != "apartment"
        or not isinstance(outputs, Mapping)
    ):
        raise ValueError("counterfactual anchor manifest identity is invalid")
    snapshot = _bound_relative_file(
        outputs.get("snapshot"), root=anchor_manifest.parent, label="anchor snapshot"
    )
    entities = _bound_relative_file(
        outputs.get("entities"), root=anchor_manifest.parent, label="anchor entities"
    )
    anchor = read_map_snapshot(snapshot, entities)
    if anchor.scene_id != "apartment" or anchor.scope != "current":
        raise ValueError("counterfactual anchor snapshot identity is invalid")
    return anchor


def _load_cf1_diagnostics(composition: Path) -> Mapping[str, object]:
    payload, _ = _json_object(composition, label="CF1 composition")
    source_index = _bound_relative_file(
        payload.get("source_index"),
        root=composition.parent,
        label="CF1 source index",
    )
    index, _ = _json_object(source_index, label="CF1 source index")
    if (
        index.get("schema_version") != 1
        or index.get("dataset") != "TESSE-CD"
        or index.get("scene") != "apartment"
        or index.get("method") != "OVIV2"
    ):
        raise ValueError("CF1 source index identity is invalid")
    diagnostics = _bound_relative_file(
        index.get("runtime_diagnostics"),
        root=composition.parent,
        label="CF1 runtime diagnostics",
    )
    value, _ = _json_object(diagnostics, label="CF1 runtime diagnostics")
    return value


def collect_counterfactual_results(
    *,
    plan: str | Path,
    variant_results_root: str | Path,
    anchor_manifest: str | Path,
    output: str | Path,
) -> Path:
    """Collect the fixed 24-run directory layout into the P6-A artifact."""

    plan_path = Path(os.path.abspath(os.fspath(plan)))
    _, variants = _load_plan(plan_path)
    results_root = Path(os.path.abspath(os.fspath(variant_results_root)))
    anchor_path = Path(os.path.abspath(os.fspath(anchor_manifest)))
    anchor_record = _record(anchor_path)
    metrics: dict[str, Mapping[str, object]] = {}
    receipts: dict[str, Path] = {}
    cf1_composition: Path | None = None
    for variant in variants:
        root = results_root / variant.variant_id
        composition = root / "composition" / "run_manifest.json"
        gate = root / "gate" / "gate_decision.json"
        composition_payload, _ = _json_object(
            composition, label=f"{variant.variant_id} composition"
        )
        inputs = composition_payload.get("inputs")
        if (
            not isinstance(inputs, Mapping)
            or inputs.get("anchor_manifest") != anchor_record
        ):
            raise ValueError(
                f"{variant.variant_id} composition anchor binding is invalid"
            )
        metrics[variant.variant_id] = load_variant_gate_metrics(
            gate=gate, composition=composition, variant=variant
        )
        receipts[variant.variant_id] = gate
        if variant.variant_id == "CF1":
            cf1_composition = composition
    if cf1_composition is None:
        raise ValueError("CF1 composition is absent")
    diagnostics = _load_cf1_diagnostics(cf1_composition)
    anchor = _load_anchor_snapshot(anchor_path)
    cf1 = next(item for item in variants if item.variant_id == "CF1")
    features = build_anchor_counterfactual_features(
        anchor,
        diagnostics,
        cf1.selected_anchor_ids,
        voxel_size_m=0.05,
    )
    return publish_counterfactual_collection(
        plan=plan_path,
        metrics_by_variant=metrics,
        features=features,
        metric_receipts=receipts,
        output=output,
    )


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise ValueError("counterfactual CSV rows must be nonempty")
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise ValueError("counterfactual CSV rows have inconsistent fields")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _relative_output_record(path: Path, *, root: Path) -> dict[str, object]:
    content = _regular_bytes(path, label="counterfactual output")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(content),
        "byte_count": len(content),
    }


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory_no_replace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise RuntimeError(
            "atomic no-clobber directory publication is unavailable"
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(target)
        raise OSError(error_number, os.strerror(error_number), target)
    _fsync_directory(target.parent)


def publish_counterfactual_collection(
    *,
    plan: str | Path,
    metrics_by_variant: Mapping[str, Mapping[str, object]],
    features: Sequence[AnchorCounterfactualFeature],
    metric_receipts: Mapping[str, str | Path],
    output: str | Path,
) -> Path:
    """Atomically publish measured P6-A rows and their evidence bindings."""

    plan_path = Path(os.path.abspath(os.fspath(plan)))
    _, variants = _load_plan(plan_path)
    variant_ids = tuple(item.variant_id for item in variants)
    if not isinstance(metric_receipts, Mapping) or set(metric_receipts) != set(
        variant_ids
    ):
        raise ValueError("metric receipts do not match the counterfactual matrix")
    receipt_records = {
        variant_id: _record(Path(metric_receipts[variant_id]))
        for variant_id in variant_ids
    }
    if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
        raise TypeError("counterfactual features must be a sequence")
    feature_rows = []
    feature_ids = []
    for feature in features:
        if not isinstance(feature, AnchorCounterfactualFeature):
            raise TypeError("counterfactual feature type is invalid")
        feature_rows.append(asdict(feature))
        feature_ids.append(feature.anchor_entity_id)
    if not feature_rows or feature_ids != sorted(set(feature_ids)):
        raise ValueError("counterfactual feature IDs must be sorted and unique")
    metric_rows = build_counterfactual_metric_rows(variants, metrics_by_variant)
    metrics_bytes = _csv_bytes(metric_rows)
    features_bytes = _csv_bytes(feature_rows)
    destination = Path(os.path.abspath(os.fspath(output)))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-", dir=destination.parent
        )
    )
    try:
        metrics_path = staging / "counterfactual_metrics.csv"
        features_path = staging / "counterfactual_anchor_features.csv"
        metrics_path.write_bytes(metrics_bytes)
        features_path.write_bytes(features_bytes)
        manifest = {
            "schema_version": 1,
            "manifest_id": "crove_anchor_counterfactual_v1",
            "status": (
                "COMPLETE"
                if all(row["metric_status"] == "COMPLETE" for row in metric_rows)
                else "PARTIAL"
            ),
            "dataset": "TESSE-CD",
            "scene": "apartment",
            "diagnostic_only": True,
            "promotion_eligible": False,
            "variant_count": len(variants),
            "feature_count": len(feature_rows),
            "sources": {"counterfactual_plan": _record(plan_path)},
            "metric_receipts": receipt_records,
            "outputs": {
                "counterfactual_metrics": _relative_output_record(
                    metrics_path, root=staging
                ),
                "counterfactual_anchor_features": _relative_output_record(
                    features_path, root=staging
                ),
            },
        }
        manifest_path = staging / "counterfactual_manifest.json"
        manifest_path.write_bytes(_canonical_json(manifest))
        for path in (metrics_path, features_path, manifest_path):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        _fsync_directory(staging)
        _publish_directory_no_replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "counterfactual_manifest.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="freeze the 24 diagnostic variants")
    plan.add_argument("--p5-runtime-diagnostics", type=Path, required=True)
    plan.add_argument("--p2-attribution", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)

    compose = subparsers.add_parser("compose", help="compose one planned variant")
    compose.add_argument("--plan", type=Path, required=True)
    compose.add_argument("--variant-id", required=True)
    compose.add_argument("--source-run-manifest", type=Path, required=True)
    compose.add_argument("--anchor-manifest", type=Path, required=True)
    compose.add_argument("--visibility-policy", type=Path, required=True)
    compose.add_argument("--visibility-diagnostics-cache", type=Path)
    compose.add_argument("--output", type=Path, required=True)

    collect = subparsers.add_parser("collect", help="collect measured variant metrics")
    collect.add_argument("--plan", type=Path, required=True)
    collect.add_argument("--variant-results-root", type=Path, required=True)
    collect.add_argument("--anchor-manifest", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        output = create_counterfactual_plan(
            p5_runtime_diagnostics=args.p5_runtime_diagnostics,
            p2_attribution=args.p2_attribution,
            output=args.output,
        )
    elif args.command == "compose":
        output = compose_counterfactual_variant(
            plan=args.plan,
            variant_id=args.variant_id,
            source_run_manifest=args.source_run_manifest,
            anchor_manifest=args.anchor_manifest,
            visibility_policy=args.visibility_policy,
            output=args.output,
            visibility_diagnostics_cache=args.visibility_diagnostics_cache,
        )
    else:
        output = collect_counterfactual_results(
            plan=args.plan,
            variant_results_root=args.variant_results_root,
            anchor_manifest=args.anchor_manifest,
            output=args.output,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
