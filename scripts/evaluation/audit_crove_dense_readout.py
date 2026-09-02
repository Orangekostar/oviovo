#!/usr/bin/env python3
"""Audit one Apartment compact/dense moved-anchor checkpoint pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.crove_dense_readout_audit import audit_dense_moved_entities
from src.evaluation.exporters.oviovo import read_map_snapshot

_RECORD_KEYS = {"path", "sha256", "byte_count"}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _regular_bytes(path: Path, *, label: str) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    before = absolute.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file")
    content = absolute.read_bytes()
    after = absolute.stat(follow_symlinks=False)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(content) != after.st_size
    ):
        raise RuntimeError(f"{label} changed while reading")
    return content


def _input_record(path: Path, content: bytes | None = None) -> dict[str, object]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    data = _regular_bytes(absolute, label="input") if content is None else content
    return {
        "path": str(absolute),
        "sha256": _sha256(data),
        "byte_count": len(data),
    }


def _revalidate_input(path: Path, expected: Mapping[str, object]) -> None:
    if _input_record(path) != dict(expected):
        raise RuntimeError("audit input changed before publication")


def _json_manifest(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    content = _regular_bytes(path, label=label)
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain an object")
    return value, content


def _bound_artifact(
    root: Path, record: object, *, label: str
) -> tuple[Path, dict[str, object]]:
    if not isinstance(record, Mapping) or set(record) != _RECORD_KEYS:
        raise ValueError(f"{label} record schema is invalid")
    relative = record.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path is invalid")
    declared = Path(relative)
    if declared.is_absolute() or ".." in declared.parts:
        raise ValueError(f"{label} path escapes its manifest root")
    path = root / declared
    content = _regular_bytes(path, label=label)
    observed = {
        "path": declared.as_posix(),
        "sha256": _sha256(content),
        "byte_count": len(content),
    }
    if observed != dict(record):
        raise ValueError(f"{label} binding mismatch")
    return path, _input_record(path, content)


def _checkpoint(manifest: Mapping[str, Any], frame_index: int, *, label: str):
    values = manifest.get("checkpoints")
    if not isinstance(values, list):
        raise TypeError(f"{label} checkpoints must be a list")
    matches = [
        item
        for item in values
        if isinstance(item, Mapping) and item.get("frame_index") == frame_index
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} must contain exactly one requested checkpoint")
    return matches[0]


def _moved_ids(checkpoint: Mapping[str, Any], *, label: str) -> tuple[str, ...]:
    diagnostics = checkpoint.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise TypeError(f"{label} diagnostics must be an object")
    values = diagnostics.get("moved_anchor_ids")
    if not isinstance(values, list):
        raise TypeError(f"{label} moved_anchor_ids must be a list")
    result = tuple(values)
    if (
        not result
        or any(not isinstance(item, str) or not item for item in result)
        or result != tuple(sorted(set(result)))
    ):
        raise ValueError(f"{label} moved_anchor_ids are invalid")
    return result


def _validate_manifest_identities(
    anchor: Mapping[str, Any],
    compact: Mapping[str, Any],
    dense: Mapping[str, Any],
) -> None:
    if (
        anchor.get("manifest_id") != "crove_ovimap_static_anchor_v1"
        or compact.get("manifest_id")
        != "crove_ovimap_static_anchor_composition_v1"
        or dense.get("manifest_id")
        != "crove_ovimap_static_anchor_composition_v1"
        or any(item.get("status") != "PASS" for item in (anchor, compact, dense))
        or any(item.get("scene") != "apartment" for item in (anchor, compact, dense))
    ):
        raise ValueError("audit manifest identities are invalid")
    if compact.get("readout_contract") != {
        "moved_geometry_mode": "temporal_compact",
        "readout_role": "formal_baseline",
    }:
        raise ValueError("compact readout contract is invalid")
    dense_contract = dense.get("readout_contract")
    if dense_contract not in (
        {
            "moved_geometry_mode": "anchor_centroid_translation",
            "readout_role": "visualization_shadow",
        },
        {
            "moved_geometry_mode": "anchor_centroid_translation",
            "readout_role": "evaluation_candidate",
        },
    ):
        raise ValueError("dense readout contract is invalid")


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _markdown(result: Mapping[str, Any]) -> bytes:
    geometry = result["geometry"]
    assert isinstance(geometry, Mapping)
    rows = geometry["entities"]
    assert isinstance(rows, list)
    lines = [
        "# CROVE Dense Moved-Readout Audit",
        "",
        f"- Status: `{result['status']}`",
        f"- Scene: `{result['scene']}`",
        f"- Frame: `{result['frame_index']}`",
        "",
        "| Entity | Anchor pts | Compact pts | Dense pts | NN median m | NN p90 m | Coverage @ 5 cm |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        assert isinstance(row, Mapping)
        lines.append(
            "| {anchor_entity_id} | {anchor_point_count} | {compact_point_count} | "
            "{dense_point_count} | {dense_to_compact_nn_median_m} | "
            "{dense_to_compact_nn_p90_m} | "
            "{dense_to_compact_coverage_at_threshold} |".format(**row)
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _publish_pair(
    output_json: Path,
    output_markdown: Path,
    *,
    json_content: bytes,
    markdown_content: bytes,
) -> None:
    outputs = tuple(
        Path(os.path.abspath(os.fspath(path)))
        for path in (output_json, output_markdown)
    )
    if outputs[0] == outputs[1]:
        raise ValueError("JSON and Markdown outputs must be different")
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise ValueError("audit output already exists")
    if outputs[0].parent != outputs[1].parent:
        raise ValueError("audit outputs must share one parent")
    parent = outputs[0].parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dense-readout-audit-", dir=parent))
    try:
        staged_json = staging / outputs[0].name
        staged_markdown = staging / outputs[1].name
        staged_json.write_bytes(json_content)
        staged_markdown.write_bytes(markdown_content)
        os.replace(staged_json, outputs[0])
        try:
            os.replace(staged_markdown, outputs[1])
        except BaseException:
            outputs[0].unlink(missing_ok=True)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def audit_run_checkpoint(
    *,
    anchor_manifest_path: Path,
    compact_manifest_path: Path,
    dense_manifest_path: Path,
    frame_index: int,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, object]:
    """Load, revalidate, compare, and publish one P4A checkpoint audit."""

    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError("frame_index must be a nonnegative integer")
    manifest_paths = tuple(
        Path(os.path.abspath(os.fspath(path)))
        for path in (
            anchor_manifest_path,
            compact_manifest_path,
            dense_manifest_path,
        )
    )
    anchor, anchor_bytes = _json_manifest(manifest_paths[0], label="anchor manifest")
    compact, compact_bytes = _json_manifest(
        manifest_paths[1], label="compact manifest"
    )
    dense, dense_bytes = _json_manifest(manifest_paths[2], label="dense manifest")
    _validate_manifest_identities(anchor, compact, dense)
    anchor_outputs = anchor.get("outputs")
    if not isinstance(anchor_outputs, Mapping):
        raise TypeError("anchor outputs must be an object")
    compact_checkpoint = _checkpoint(compact, frame_index, label="compact")
    dense_checkpoint = _checkpoint(dense, frame_index, label="dense")
    compact_moved = _moved_ids(compact_checkpoint, label="compact")
    dense_moved = _moved_ids(dense_checkpoint, label="dense")
    if compact_moved != dense_moved:
        raise ValueError("compact and dense moved identities differ")

    bound_paths = []
    for root, values, label in (
        (manifest_paths[0].parent, anchor_outputs, "anchor"),
        (manifest_paths[1].parent, compact_checkpoint, "compact"),
        (manifest_paths[2].parent, dense_checkpoint, "dense"),
    ):
        snapshot_path, snapshot_record = _bound_artifact(
            root, values.get("snapshot"), label=f"{label} snapshot"
        )
        entities_path, entities_record = _bound_artifact(
            root, values.get("entities"), label=f"{label} entities"
        )
        bound_paths.append(
            (snapshot_path, entities_path, snapshot_record, entities_record)
        )

    snapshots = tuple(
        read_map_snapshot(snapshot_path, entities_path)
        for snapshot_path, entities_path, _, _ in bound_paths
    )
    geometry = audit_dense_moved_entities(
        snapshots[0], snapshots[1], snapshots[2], compact_moved
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "scene": "apartment",
        "frame_index": frame_index,
        "sources": {
            "anchor_manifest": _input_record(manifest_paths[0], anchor_bytes),
            "compact_manifest": _input_record(manifest_paths[1], compact_bytes),
            "dense_manifest": _input_record(manifest_paths[2], dense_bytes),
            "artifacts": {
                label: {"snapshot": item[2], "entities": item[3]}
                for label, item in zip(
                    ("anchor", "compact", "dense"), bound_paths, strict=True
                )
            },
        },
        "serialized_byte_count": {
            label: int(item[2]["byte_count"]) + int(item[3]["byte_count"])
            for label, item in zip(
                ("anchor", "compact", "dense"), bound_paths, strict=True
            )
        },
        "geometry": geometry,
    }
    for path, content in zip(
        manifest_paths,
        (anchor_bytes, compact_bytes, dense_bytes),
        strict=True,
    ):
        _revalidate_input(path, _input_record(path, content))
    for snapshot_path, entities_path, snapshot_record, entities_record in bound_paths:
        _revalidate_input(snapshot_path, snapshot_record)
        _revalidate_input(entities_path, entities_record)
    _publish_pair(
        output_json,
        output_markdown,
        json_content=_canonical_json(result),
        markdown_content=_markdown(result),
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--compact-run-manifest", type=Path, required=True)
    parser.add_argument("--dense-run-manifest", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_run_checkpoint(
        anchor_manifest_path=args.anchor_manifest,
        compact_manifest_path=args.compact_run_manifest,
        dense_manifest_path=args.dense_run_manifest,
        frame_index=args.frame_index,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    print(
        json.dumps(
            {
                "frame_index": result["frame_index"],
                "output_json": str(args.output_json.resolve()),
                "output_markdown": str(args.output_markdown.resolve()),
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
