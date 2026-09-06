#!/usr/bin/env python3
"""Publish and audit a compact supported-domain OVI/ReScene input."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.prepare_ovi_rescene_input_v2 import (
    audit_recovered_input,
    load_static_input_artifact,
)
from src.oviv2.ovi_rescene_adapter import (
    load_neural_sample_artifact,
    write_neural_sample_artifact,
)
from src.oviv2.rescene_input_bridge import (
    RecoveredModelSupport,
    load_model_input_artifact,
    write_model_input_artifact,
)
from src.oviv2.rescene_supported_view import (
    EntityCoverage,
    SupportedInferenceView,
    build_supported_inference_view,
)

_RECOVERY_ARRAYS = (
    "support_valid",
    "representative_source_point_indices",
    "rgb_uint8",
    "local_frame_indices",
    "global_frame_indices",
    "rows",
    "columns",
    "camera_depth_m",
    "observed_depth_m",
    "depth_residual_m",
)
_MAPPING_ARRAYS = (
    "new_to_old_model_indices",
    "old_to_new_model_indices",
    "new_to_old_adapter_indices",
    "new_to_old_source_point_indices",
)
_RECEIPT_KEYS = {
    "schema_version",
    "artifact_id",
    "status",
    "parent_input_contract",
    "full_domains",
    "supported_domains",
    "coverage",
    "model_input_manifest",
    "adapter_pair_manifest",
    "mappings",
    "entity_coverage",
}
_CONFIG_KEYS = {"schema_version", "config_id", "input_v2_root", "output_root"}


class SupportedPreparationError(ValueError):
    """Raised when a supported V3 artifact is incomplete or inconsistent."""


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = absolute.stat(follow_symlinks=False)
    except OSError as error:
        raise SupportedPreparationError(f"{label} is unavailable") from error
    if not stat.S_ISREG(before.st_mode):
        raise SupportedPreparationError(f"{label} must be a regular non-symlink file")
    content = absolute.read_bytes()
    after = absolute.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or len(content) != before.st_size:
        raise SupportedPreparationError(f"{label} changed while being read")
    return content


def _file_record(
    path: Path, *, label: str, relative_path: str | None = None
) -> dict[str, object]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    content = _regular_file_bytes(absolute, label=label)
    return {
        "path": relative_path if relative_path is not None else str(absolute),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_regular_file_bytes(path, label=label).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupportedPreparationError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise SupportedPreparationError(f"{label} must contain a JSON object")
    return value


def _parent_paths(input_v2_root: Path) -> tuple[Path, Path, dict[str, Any]]:
    root = Path(os.path.abspath(os.fspath(input_v2_root)))
    receipt_path = root / "input_contract_v2.json"
    receipt = _read_json(receipt_path, label="parent V2 input contract")
    try:
        static_manifest = Path(receipt["static_input_manifest"]["path"])
        calibration_manifest = Path(receipt["calibration_manifest"]["path"])
    except (KeyError, TypeError) as error:
        raise SupportedPreparationError("parent V2 bindings are invalid") from error
    if static_manifest.name != "manifest.json":
        raise SupportedPreparationError("parent V2 static manifest path is invalid")
    try:
        audited = audit_recovered_input(
            output_root=root,
            static_input_root=static_manifest.parent,
            calibration_manifest_path=calibration_manifest,
        )
    except (OSError, TypeError, ValueError) as error:
        raise SupportedPreparationError("parent V2 input audit failed") from error
    if audited != receipt:
        raise SupportedPreparationError("parent V2 audit result changed")
    return static_manifest.parent, calibration_manifest, receipt


def _load_support(input_v2_root: Path, receipt: Mapping[str, Any]) -> RecoveredModelSupport:
    root = Path(os.path.abspath(os.fspath(input_v2_root)))
    record = receipt.get("recovered_support")
    if not isinstance(record, Mapping) or record.get("path") != "recovered_support.npz":
        raise SupportedPreparationError("parent recovered support binding is invalid")
    path = root / "recovered_support.npz"
    content = _regular_file_bytes(path, label="parent recovered support")
    observed = {
        "path": "recovered_support.npz",
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }
    if dict(record) != observed:
        raise SupportedPreparationError("parent recovered support binding mismatch")
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            if set(archive.files) != {*_RECOVERY_ARRAYS, "unsupported_model_indices"}:
                raise SupportedPreparationError("parent recovered support schema is invalid")
            arrays = {name: archive[name] for name in _RECOVERY_ARRAYS}
    except (OSError, ValueError) as error:
        if isinstance(error, SupportedPreparationError):
            raise
        raise SupportedPreparationError("parent recovered support cannot be decoded") from error
    return RecoveredModelSupport(**arrays)


def _coverage_payload(view: SupportedInferenceView) -> dict[str, object]:
    def row_payload(row: EntityCoverage) -> dict[str, object]:
        return {
            "visit_id": row.visit_id,
            "entity_id": row.entity_id,
            "full_adapter_token_count": row.full_adapter_token_count,
            "supported_adapter_token_count": row.supported_adapter_token_count,
            "full_source_point_count": row.full_source_point_count,
            "supported_source_point_count": row.supported_source_point_count,
            "full_model_token_count": row.full_model_token_count,
            "supported_model_token_count": row.supported_model_token_count,
        }

    return {
        "schema_version": 1,
        "artifact_id": "OVI_RESCENE_ENTITY_COVERAGE_V3",
        "status": "PASS",
        "unsupported_entity_keys": [list(key) for key in view.unsupported_entity_keys],
        "entities": [row_payload(row) for row in view.entity_coverage],
    }


def _domain_payload(view: SupportedInferenceView) -> tuple[dict[str, int], dict[str, int]]:
    return (
        {
            "source": view.full_source_point_count,
            "adapter": view.full_adapter_count,
            "model": view.full_model_count,
        },
        {
            "source": len(view.new_to_old_source_point_indices),
            "adapter": len(view.new_to_old_adapter_indices),
            "model": len(view.new_to_old_model_indices),
        },
    )


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_view(
    *, view: SupportedInferenceView, parent_receipt: Path, output_root: Path
) -> dict[str, Any]:
    output = Path(os.path.abspath(os.fspath(output_root)))
    if output.exists() or output.is_symlink():
        raise SupportedPreparationError(f"supported input artifact already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        model_paths = write_model_input_artifact(view.model_input, staging / "model_input")
        pair_paths = write_neural_sample_artifact(view.pair, staging / "adapter_pair")
        mappings_path = staging / "mappings.npz"
        _write_npz(
            mappings_path,
            {name: getattr(view, name) for name in _MAPPING_ARRAYS},
        )
        coverage_path = staging / "entity_coverage.json"
        with coverage_path.open("xb") as stream:
            stream.write(_json_bytes(_coverage_payload(view)))
            stream.flush()
            os.fsync(stream.fileno())
        full, supported = _domain_payload(view)
        receipt = {
            "schema_version": 3,
            "artifact_id": "OVI_RESCENE_SUPPORTED_INPUT_V3",
            "status": "SUPPORTED_INFERENCE_PASS",
            "parent_input_contract": _file_record(
                parent_receipt, label="parent V2 input contract"
            ),
            "full_domains": full,
            "supported_domains": supported,
            "coverage": {
                f"{name}_fraction": supported[name] / full[name]
                for name in ("source", "adapter", "model")
            },
            "model_input_manifest": _file_record(
                model_paths.manifest,
                label="model input manifest",
                relative_path="model_input/manifest.json",
            ),
            "adapter_pair_manifest": _file_record(
                pair_paths.manifest,
                label="adapter pair manifest",
                relative_path="adapter_pair/manifest.json",
            ),
            "mappings": _file_record(
                mappings_path,
                label="supported mappings",
                relative_path="mappings.npz",
            ),
            "entity_coverage": _file_record(
                coverage_path,
                label="entity coverage",
                relative_path="entity_coverage.json",
            ),
        }
        receipt_path = staging / "input_contract_v3.json"
        with receipt_path.open("xb") as stream:
            stream.write(_json_bytes(receipt))
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return receipt


def build_supported_input(
    *, input_v2_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Audit one V2 partial input and publish its executable supported subset."""

    input_root = Path(os.path.abspath(os.fspath(input_v2_root)))
    static_root, _, receipt = _parent_paths(input_root)
    prepared = load_static_input_artifact(static_root)
    support = _load_support(input_root, receipt)
    try:
        view = build_supported_inference_view(prepared, support)
    except (TypeError, ValueError) as error:
        raise SupportedPreparationError("supported view construction failed") from error
    return _publish_view(
        view=view,
        parent_receipt=input_root / "input_contract_v2.json",
        output_root=Path(output_root),
    )


def _bound_relative(
    root: Path,
    record: object,
    *,
    expected_path: str,
    label: str,
) -> Path:
    if not isinstance(record, Mapping) or record.get("path") != expected_path:
        raise SupportedPreparationError(f"{label} binding is invalid")
    path = root / expected_path
    observed = _file_record(path, label=label, relative_path=expected_path)
    if dict(record) != observed:
        raise SupportedPreparationError(f"{label} binding mismatch")
    return path


def _load_mappings(path: Path) -> dict[str, np.ndarray]:
    content = _regular_file_bytes(path, label="supported mappings")
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            if set(archive.files) != set(_MAPPING_ARRAYS):
                raise SupportedPreparationError("supported mappings schema is invalid")
            return {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, SupportedPreparationError):
            raise
        raise SupportedPreparationError("supported mappings cannot be decoded") from error


def audit_supported_input(output_root: str | Path) -> dict[str, Any]:
    """Rebuild and verify every V3 mapping, denominator, and child artifact."""

    root = Path(os.path.abspath(os.fspath(output_root)))
    if root.is_symlink() or not root.is_dir():
        raise SupportedPreparationError("supported input root is missing or a symlink")
    receipt = _read_json(root / "input_contract_v3.json", label="V3 input contract")
    if (
        set(receipt) != _RECEIPT_KEYS
        or receipt.get("schema_version") != 3
        or receipt.get("artifact_id") != "OVI_RESCENE_SUPPORTED_INPUT_V3"
        or receipt.get("status") != "SUPPORTED_INFERENCE_PASS"
    ):
        raise SupportedPreparationError("V3 input contract identity is invalid")
    parent_record = receipt.get("parent_input_contract")
    if not isinstance(parent_record, Mapping):
        raise SupportedPreparationError("parent V2 binding is invalid")
    try:
        parent_path = Path(parent_record["path"])
    except (KeyError, TypeError) as error:
        raise SupportedPreparationError("parent V2 binding is invalid") from error
    if dict(parent_record) != _file_record(
        parent_path, label="parent V2 input contract"
    ):
        raise SupportedPreparationError("parent V2 binding mismatch")
    input_root = parent_path.parent
    static_root, _, parent_receipt = _parent_paths(input_root)
    prepared = load_static_input_artifact(static_root)
    support = _load_support(input_root, parent_receipt)
    try:
        expected = build_supported_inference_view(prepared, support)
    except (TypeError, ValueError) as error:
        raise SupportedPreparationError("supported view reconstruction failed") from error

    model_manifest = _bound_relative(
        root,
        receipt.get("model_input_manifest"),
        expected_path="model_input/manifest.json",
        label="model input manifest",
    )
    pair_manifest = _bound_relative(
        root,
        receipt.get("adapter_pair_manifest"),
        expected_path="adapter_pair/manifest.json",
        label="adapter pair manifest",
    )
    mappings_path = _bound_relative(
        root,
        receipt.get("mappings"),
        expected_path="mappings.npz",
        label="mappings",
    )
    coverage_path = _bound_relative(
        root,
        receipt.get("entity_coverage"),
        expected_path="entity_coverage.json",
        label="entity coverage",
    )
    try:
        model_input = load_model_input_artifact(model_manifest.parent)
        pair = load_neural_sample_artifact(pair_manifest.parent)
    except (OSError, TypeError, ValueError) as error:
        raise SupportedPreparationError("V3 child artifact audit failed") from error
    if (
        model_input.content_sha256() != expected.model_input.content_sha256()
        or pair.content_sha256() != expected.pair.content_sha256()
    ):
        raise SupportedPreparationError("V3 child artifact content mismatch")
    mappings = _load_mappings(mappings_path)
    if any(
        not np.array_equal(mappings[name], getattr(expected, name))
        for name in _MAPPING_ARRAYS
    ):
        raise SupportedPreparationError("supported mappings content mismatch")
    if _read_json(coverage_path, label="entity coverage") != _coverage_payload(expected):
        raise SupportedPreparationError("entity coverage content mismatch")
    full, supported = _domain_payload(expected)
    expected_coverage = {
        f"{name}_fraction": supported[name] / full[name]
        for name in ("source", "adapter", "model")
    }
    if (
        receipt.get("full_domains") != full
        or receipt.get("supported_domains") != supported
        or receipt.get("coverage") != expected_coverage
    ):
        raise SupportedPreparationError("V3 domain counts or coverage mismatch")
    return receipt


def _load_config(config_path: str | Path) -> tuple[Path, Path]:
    config = _read_json(Path(config_path), label="supported V3 config")
    if (
        set(config) != _CONFIG_KEYS
        or config.get("schema_version") != 1
        or config.get("config_id") != "OVI_RESCENE_SUPPORTED_V3_CONFIG_V1"
    ):
        raise SupportedPreparationError("supported V3 config identity is invalid")
    values = (config.get("input_v2_root"), config.get("output_root"))
    if any(not isinstance(value, str) or not value for value in values):
        raise SupportedPreparationError("supported V3 config paths are invalid")
    input_root, output_root = (Path(value) for value in values)
    if not input_root.is_absolute() or not output_root.is_absolute():
        raise SupportedPreparationError("supported V3 config paths must be absolute")
    return input_root, output_root


def run_from_config(command: str, config_path: str | Path) -> dict[str, Any]:
    """Build or audit the exact roots named by one strict formal config."""

    input_root, output_root = _load_config(config_path)
    if command == "build":
        return build_supported_input(
            input_v2_root=input_root,
            output_root=output_root,
        )
    if command != "audit":
        raise SupportedPreparationError("supported V3 command is invalid")
    receipt = audit_supported_input(output_root)
    parent = receipt.get("parent_input_contract")
    expected_parent = input_root / "input_contract_v2.json"
    if not isinstance(parent, Mapping) or parent.get("path") != str(expected_parent):
        raise SupportedPreparationError("configured parent V2 path mismatch")
    return receipt


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "audit"))
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    try:
        receipt = run_from_config(arguments.command, arguments.config)
        _, output_root = _load_config(arguments.config)
    except (OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(output_root),
                "status": receipt["status"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "SupportedPreparationError",
    "audit_supported_input",
    "build_supported_input",
    "main",
    "run_from_config",
]


if __name__ == "__main__":
    raise SystemExit(main())
