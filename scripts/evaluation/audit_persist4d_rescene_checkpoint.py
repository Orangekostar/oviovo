#!/usr/bin/env python3
"""Audit a source-bound Persist4D ReScene checkpoint without model inference."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import inspect
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")


class CheckpointAuditError(ValueError):
    """Raised when checkpoint or source identity cannot be proven."""


def _require_exact_keys(
    value: object,
    keys: set[str],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise CheckpointAuditError(f"{label} schema is invalid")
    return value


def _require_bound_file(value: object, *, label: str) -> dict[str, object]:
    record = _require_exact_keys(
        value,
        {"path", "sha256", "byte_count"},
        label=label,
    )
    path_value = record.get("path")
    expected_sha256 = record.get("sha256")
    expected_byte_count = record.get("byte_count")
    if (
        not isinstance(path_value, str)
        or not Path(path_value).is_absolute()
        or not isinstance(expected_sha256, str)
        or _SHA256.fullmatch(expected_sha256) is None
        or type(expected_byte_count) is not int
        or expected_byte_count <= 0
    ):
        raise CheckpointAuditError(f"{label} binding values are invalid")
    path, before = _stable_regular_file(path_value, label=label)
    observed_sha256 = _sha256_file(path, before, label=label)
    if before.st_size != expected_byte_count or observed_sha256 != expected_sha256:
        raise CheckpointAuditError(f"{label} binding mismatch")
    return dict(record)


def load_checkpoint_provenance(config_path: str | Path) -> dict[str, object]:
    """Load and validate the source-bound local checkpoint provenance contract."""

    path, before = _stable_regular_file(config_path, label="provenance config")
    raw = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise CheckpointAuditError("provenance config changed while being read")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointAuditError("provenance config is not valid JSON") from error

    record = _require_exact_keys(
        payload,
        {
            "schema_version",
            "artifact_id",
            "status",
            "classification",
            "persist4d_repository",
            "persist4d_evidence_commit",
            "canonical_checkpoint",
            "training_source",
            "training_config",
            "reproduction_metrics",
            "concerto_initialization",
            "official_author_checkpoint",
            "source_audit",
        },
        label="provenance",
    )
    if record.get("schema_version") != 1:
        raise CheckpointAuditError("provenance schema_version must be 1")
    if record.get("artifact_id") != "PERSIST4D_RESCENE4D_C_T2_REPRO_V1":
        raise CheckpointAuditError("provenance artifact_id is invalid")
    if record.get("status") != "PASS":
        raise CheckpointAuditError("provenance status must be PASS")
    if record.get("classification") != "SOURCE_BOUND_RESCENE_REPRODUCTION":
        raise CheckpointAuditError("provenance classification is invalid")
    repository = record.get("persist4d_repository")
    if not isinstance(repository, str) or not repository:
        raise CheckpointAuditError("Persist4D repository is invalid")
    evidence_commit = record.get("persist4d_evidence_commit")
    if not isinstance(evidence_commit, str) or _GIT_SHA.fullmatch(evidence_commit) is None:
        raise CheckpointAuditError("Persist4D evidence commit is invalid")
    if record.get("official_author_checkpoint") is not False:
        raise CheckpointAuditError("checkpoint must not be classified as official author")

    checkpoint = _require_exact_keys(
        record.get("canonical_checkpoint"),
        {"path", "sha256", "byte_count", "regular_file", "symlink"},
        label="canonical checkpoint",
    )
    if checkpoint.get("regular_file") is not True or checkpoint.get("symlink") is not False:
        raise CheckpointAuditError("canonical checkpoint file classification is invalid")
    _require_bound_file(
        {key: checkpoint[key] for key in ("path", "sha256", "byte_count")},
        label="canonical checkpoint",
    )

    training_source = _require_exact_keys(
        record.get("training_source"),
        {"official_upstream_base_commit", "persist4d_runtime_commit"},
        label="training source",
    )
    for key in ("official_upstream_base_commit", "persist4d_runtime_commit"):
        value = training_source.get(key)
        if not isinstance(value, str) or _GIT_SHA.fullmatch(value) is None:
            raise CheckpointAuditError(f"training source {key} is invalid")
    if training_source.get("persist4d_runtime_commit") != evidence_commit:
        raise CheckpointAuditError("Persist4D runtime/evidence commit mismatch")

    _require_bound_file(record.get("training_config"), label="training config")

    metrics = _require_exact_keys(
        record.get("reproduction_metrics"),
        {"validation_sequence_count", "t_mAP", "t_REC", "overall_mAP", "paper_target_t_mAP"},
        label="reproduction metrics",
    )
    if type(metrics.get("validation_sequence_count")) is not int or metrics.get(
        "validation_sequence_count"
    ) != 154:
        raise CheckpointAuditError("reproduction validation count is invalid")
    for key in ("t_mAP", "t_REC", "overall_mAP", "paper_target_t_mAP"):
        value = metrics.get(key)
        if type(value) not in (int, float) or not float("-inf") < float(value) < float("inf"):
            raise CheckpointAuditError(f"reproduction metric {key} is invalid")

    concerto = _require_exact_keys(
        record.get("concerto_initialization"),
        {"path", "revision", "sha256", "byte_count", "required_for_runtime"},
        label="Concerto initialization",
    )
    revision = concerto.get("revision")
    if not isinstance(revision, str) or _GIT_SHA.fullmatch(revision) is None:
        raise CheckpointAuditError("Concerto revision is invalid")
    if concerto.get("required_for_runtime") is not True:
        raise CheckpointAuditError("Concerto runtime requirement is invalid")
    _require_bound_file(
        {key: concerto[key] for key in ("path", "sha256", "byte_count")},
        label="Concerto initialization",
    )

    source_audit = _require_exact_keys(
        record.get("source_audit"),
        {"persist4d", "pristine_upstream_rescene4d"},
        label="source audit",
    )
    expected_source_commits = {
        "persist4d": evidence_commit,
        "pristine_upstream_rescene4d": training_source[
            "official_upstream_base_commit"
        ],
    }
    for source_name, expected_commit in expected_source_commits.items():
        source = _require_exact_keys(
            source_audit.get(source_name),
            {"checkout_path", "commit", "required_files"},
            label=f"{source_name} source",
        )
        checkout_path = source.get("checkout_path")
        commit = source.get("commit")
        required_files = source.get("required_files")
        if (
            not isinstance(checkout_path, str)
            or not Path(checkout_path).is_absolute()
            or commit != expected_commit
            or not isinstance(required_files, Sequence)
            or isinstance(required_files, (str, bytes))
            or not required_files
        ):
            raise CheckpointAuditError(f"{source_name} source binding is invalid")
        for file_binding in required_files:
            binding = _require_exact_keys(
                file_binding,
                {"path", "sha256", "byte_count"},
                label=f"{source_name} required file",
            )
            relative = binding.get("path")
            digest = binding.get("sha256")
            byte_count = binding.get("byte_count")
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
                or type(byte_count) is not int
                or byte_count <= 0
            ):
                raise CheckpointAuditError(
                    f"{source_name} required file binding is invalid"
                )
    return dict(record)


def _stable_regular_file(path: str | Path, *, label: str) -> tuple[Path, os.stat_result]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = absolute.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise CheckpointAuditError(f"{label} is missing") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CheckpointAuditError(f"{label} must be a regular non-symlink file")
    return absolute, before


def _sha256_file(path: Path, before: os.stat_result, *, label: str) -> str:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    after = path.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after) or byte_count != before.st_size:
        raise CheckpointAuditError(f"{label} changed while being read")
    return digest.hexdigest()


def audit_checkpoint_structure(
    checkpoint_path: str | Path,
    expected_sha256: str,
) -> tuple[dict[str, object], Mapping[str, Any]]:
    """Verify and compactly inventory one Lightning checkpoint on CPU."""

    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise CheckpointAuditError("expected checkpoint SHA-256 is invalid")
    path, before = _stable_regular_file(checkpoint_path, label="checkpoint")
    observed_sha256 = _sha256_file(path, before, label="checkpoint")
    if observed_sha256 != expected_sha256:
        raise CheckpointAuditError("checkpoint SHA-256 mismatch")

    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise CheckpointAuditError("checkpoint deserialization failed") from error
    if not isinstance(payload, Mapping):
        raise CheckpointAuditError("checkpoint root must be a mapping")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise CheckpointAuditError("checkpoint state_dict must be a non-empty mapping")

    prefix_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    element_count = 0
    nonfinite_count = 0
    for key, value in state_dict.items():
        if not isinstance(key, str) or not key or not isinstance(value, torch.Tensor):
            raise CheckpointAuditError("checkpoint state_dict entries must be named tensors")
        prefix_counts[key.split(".", 1)[0]] += 1
        dtype_counts[str(value.dtype)] += 1
        element_count += value.numel()
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            nonfinite_count += 1
    if nonfinite_count:
        raise CheckpointAuditError("checkpoint contains non-finite tensors")

    result = {
        "byte_count": before.st_size,
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "nonfinite_tensor_count": nonfinite_count,
        "prefix_counts": dict(sorted(prefix_counts.items())),
        "sha256": observed_sha256,
        "state_dict_key_count": len(state_dict),
        "status": "CHECKPOINT_STRUCTURE_PASS",
        "tensor_dtype_counts": dict(sorted(dtype_counts.items())),
        "tensor_element_count": element_count,
        "top_level_keys": sorted(str(key) for key in payload),
    }
    return result, state_dict


def compare_model_topology(
    checkpoint_state_dict: Mapping[str, Any],
    model_state_dict: Mapping[str, Any],
) -> dict[str, object]:
    """Compare every `model.*` checkpoint tensor with an unprefixed model state."""

    checkpoint_model = {
        key.removeprefix("model."): value
        for key, value in checkpoint_state_dict.items()
        if isinstance(key, str) and key.startswith("model.")
    }
    model_keys = set(model_state_dict)
    checkpoint_keys = set(checkpoint_model)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    shape_mismatches = []
    for key in sorted(model_keys & checkpoint_keys):
        checkpoint_shape = tuple(getattr(checkpoint_model[key], "shape", ()))
        model_shape = tuple(getattr(model_state_dict[key], "shape", ()))
        if checkpoint_shape != model_shape:
            shape_mismatches.append(
                {
                    "checkpoint": list(checkpoint_shape),
                    "key": key,
                    "model": list(model_shape),
                }
            )
    compatible = not missing and not unexpected and not shape_mismatches
    return {
        "checkpoint_model_key_count": len(checkpoint_model),
        "ignored_or_dropped_model_keys": [],
        "missing_keys": missing,
        "model_key_count": len(model_state_dict),
        "shape_mismatches": shape_mismatches,
        "status": (
            "MODEL_TOPOLOGY_STRICT_COMPATIBLE"
            if compatible
            else "UPSTREAM_TOPOLOGY_COMPATIBILITY_FAIL"
        ),
        "unexpected_keys": unexpected,
    }


def _git(checkout: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(checkout), *args],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise CheckpointAuditError("source Git identity check failed") from error


def verify_bound_git_source(
    checkout_path: str | Path,
    expected_commit: str,
    required_files: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Verify one clean checkout and every required file against its Git object."""

    checkout = Path(os.path.abspath(os.fspath(checkout_path)))
    if checkout.is_symlink() or not checkout.is_dir():
        raise CheckpointAuditError("source checkout is missing or a symlink")
    if not isinstance(expected_commit, str) or _GIT_SHA.fullmatch(expected_commit) is None:
        raise CheckpointAuditError("source expected commit is invalid")
    head = _git(checkout, "rev-parse", "HEAD").decode().strip()
    if head != expected_commit:
        raise CheckpointAuditError("source commit mismatch")
    if _git(checkout, "status", "--porcelain=v1", "--untracked-files=all"):
        raise CheckpointAuditError("source working tree is dirty")
    if not isinstance(required_files, Sequence) or not required_files:
        raise CheckpointAuditError("source required_files must be non-empty")

    seen: set[str] = set()
    for record in required_files:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "byte_count",
        }:
            raise CheckpointAuditError("source file binding schema is invalid")
        relative = record.get("path")
        digest = record.get("sha256")
        byte_count = record.get("byte_count")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in seen
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or type(byte_count) is not int
            or byte_count <= 0
        ):
            raise CheckpointAuditError("source file binding values are invalid")
        seen.add(relative)
        working_path, before = _stable_regular_file(
            checkout / relative, label=f"source file {relative}"
        )
        working_digest = _sha256_file(
            working_path, before, label=f"source file {relative}"
        )
        frozen = _git(checkout, "show", f"{expected_commit}:{relative}")
        if (
            working_digest != digest
            or len(frozen) != byte_count
            or hashlib.sha256(frozen).hexdigest() != digest
        ):
            raise CheckpointAuditError(f"source file binding mismatch: {relative}")
    return {
        "commit": expected_commit,
        "required_file_count": len(seen),
        "status": "SOURCE_IDENTITY_PASS",
    }


def _instantiate_pristine_upstream_model(
    hyperparameters: Mapping[str, object],
    concerto_path: Path,
    upstream_checkout: Path,
) -> Any:
    model_config = hyperparameters.get("model")
    if model_config is None:
        raise CheckpointAuditError("checkpoint is missing saved model hyperparameters")
    try:
        model_config = copy.deepcopy(model_config)
        model_config.config.backbone.name = str(concerto_path)
    except (AttributeError, KeyError, TypeError) as error:
        raise CheckpointAuditError("saved model hyperparameters are incompatible") from error
    if any(name == "models" or name.startswith("models.") for name in sys.modules):
        raise CheckpointAuditError("models package was imported before source binding")

    import hydra

    sys.path.insert(0, str(upstream_checkout))
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(
            sink
        ):
            model = hydra.utils.instantiate(model_config)
        source_file = Path(inspect.getfile(type(model))).resolve()
        try:
            source_file.relative_to(upstream_checkout.resolve())
        except ValueError as error:
            raise CheckpointAuditError(
                "instantiated model did not come from pristine upstream checkout"
            ) from error
        return model
    except CheckpointAuditError:
        raise
    except Exception as error:
        raise CheckpointAuditError("pristine upstream model construction failed") from error
    finally:
        if sys.path and sys.path[0] == str(upstream_checkout):
            sys.path.pop(0)


def _write_canonical_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def audit_checkpoint(
    config_path: str | Path,
    output_path: str | Path,
    *,
    _model_factory: Callable[[Mapping[str, object], Path, Path], Any] | None = None,
) -> dict[str, object]:
    """Run the real C0/C1 checkpoint, source, and strict-topology audit."""

    provenance = load_checkpoint_provenance(config_path)
    source_config = provenance["source_audit"]
    assert isinstance(source_config, Mapping)
    source_identity: dict[str, object] = {}
    for source_name in ("persist4d", "pristine_upstream_rescene4d"):
        source = source_config[source_name]
        assert isinstance(source, Mapping)
        source_identity[source_name] = verify_bound_git_source(
            str(source["checkout_path"]),
            str(source["commit"]),
            source["required_files"],
        )

    checkpoint = provenance["canonical_checkpoint"]
    concerto = provenance["concerto_initialization"]
    assert isinstance(checkpoint, Mapping)
    assert isinstance(concerto, Mapping)
    checkpoint_structure, state_dict = audit_checkpoint_structure(
        str(checkpoint["path"]),
        str(checkpoint["sha256"]),
    )

    import torch

    payload = torch.load(
        str(checkpoint["path"]),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("hyper_parameters"), Mapping
    ):
        raise CheckpointAuditError("checkpoint is missing saved hyperparameters")
    hyperparameters = payload["hyper_parameters"]
    del payload

    upstream = source_config["pristine_upstream_rescene4d"]
    assert isinstance(upstream, Mapping)
    factory = _model_factory or _instantiate_pristine_upstream_model
    model = factory(
        hyperparameters,
        Path(str(concerto["path"])),
        Path(str(upstream["checkout_path"])),
    )
    model_state = model.state_dict()
    model_topology = compare_model_topology(state_dict, model_state)
    if model_topology["status"] != "MODEL_TOPOLOGY_STRICT_COMPATIBLE":
        raise CheckpointAuditError("pristine upstream model topology is incompatible")
    checkpoint_model = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if isinstance(key, str) and key.startswith("model.")
    }
    try:
        incompatible = model.load_state_dict(checkpoint_model, strict=True)
    except Exception as error:
        raise CheckpointAuditError("strict model state load failed") from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise CheckpointAuditError("strict model state load returned incompatible keys")
    model_topology["strict_load"] = True

    config_file, config_stat = _stable_regular_file(
        config_path, label="provenance config"
    )
    config_record = {
        "byte_count": config_stat.st_size,
        "path": str(config_file),
        "sha256": _sha256_file(
            config_file, config_stat, label="provenance config"
        ),
    }
    repository_root = Path(__file__).resolve().parents[2]
    producer_commit = _git(repository_root, "rev-parse", "HEAD").decode().strip()

    result: dict[str, object] = {
        "artifact_id": provenance["artifact_id"],
        "canonical_checkpoint": dict(checkpoint),
        "checkpoint_structure": checkpoint_structure,
        "classification": provenance["classification"],
        "concerto_initialization": {
            "byte_count": concerto["byte_count"],
            "required_for_model_construction": True,
            "revision": concerto["revision"],
            "sha256": concerto["sha256"],
        },
        "model_topology": model_topology,
        "official_author_checkpoint": False,
        "producer_command": [sys.executable, *sys.argv],
        "producer_git_commit": producer_commit,
        "provenance_config": config_record,
        "schema_version": 1,
        "source_identity": source_identity,
        "status": "C0_C1_PASS",
    }
    _write_canonical_json(Path(output_path), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    audit_checkpoint(arguments.config, arguments.output)
    return 0


__all__ = [
    "CheckpointAuditError",
    "audit_checkpoint",
    "audit_checkpoint_structure",
    "compare_model_topology",
    "load_checkpoint_provenance",
    "verify_bound_git_source",
]


if __name__ == "__main__":
    raise SystemExit(main())
