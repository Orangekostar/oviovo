"""Build auditable result summaries from evaluator-produced aggregate JSON."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ResultManifestError(ValueError):
    """A result summary lacks required provenance or violates scope."""


REQUIRED_FIELDS = (
    "run_id",
    "method",
    "upstream_commit",
    "adapter_commit",
    "environment",
    "command",
    "config",
    "weights",
    "dataset",
    "hardware",
    "seed",
    "raw_outputs",
    "protocol_deviations",
    "token_bindings",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hashed_file(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    output = dict(record)
    path = Path(str(output.get("path", "")))
    if not path.is_file():
        raise ResultManifestError(f"{label} file does not exist: {path}")
    output["sha256"] = _sha256(path)
    return output


def finalize_static_result(
    aggregate_path: str | Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Inject aggregate metrics and compute every declared file hash."""
    aggregate_path = Path(aggregate_path)
    if not aggregate_path.is_file():
        raise ResultManifestError(f"aggregate file does not exist: {aggregate_path}")
    if provenance.get("status") != "VERIFIED":
        raise ResultManifestError("result provenance status must be VERIFIED")
    missing = [field for field in REQUIRED_FIELDS if field not in provenance]
    if missing:
        raise ResultManifestError(f"result provenance missing fields: {', '.join(missing)}")

    result = dict(provenance)
    method_key = str(result.get("method", {}).get("key", ""))
    tokens = [str(binding.get("token", "")) for binding in result["token_bindings"]]
    if "OVIOVO" in method_key.upper() or any("OVIOVO" in token.upper() for token in tokens):
        raise ResultManifestError("OVIOVO method and tokens are outside this result scope")

    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if not isinstance(aggregate.get("metrics"), Mapping):
        raise ResultManifestError("aggregate JSON must contain a metrics object")
    result["metrics"] = aggregate["metrics"]
    result["aggregate_source"] = {
        "path": str(aggregate_path.resolve()),
        "sha256": _sha256(aggregate_path),
    }
    result["config"] = _hashed_file(result["config"], "config")
    dataset = dict(result["dataset"])
    dataset["manifest"] = _hashed_file(dataset.get("manifest", {}), "dataset manifest")
    result["dataset"] = dataset
    result["weights"] = [
        _hashed_file(record, f"weight {index}")
        for index, record in enumerate(result["weights"])
    ]
    result["raw_outputs"] = [
        _hashed_file(record, f"raw output {index}")
        for index, record in enumerate(result["raw_outputs"])
    ]
    return result
