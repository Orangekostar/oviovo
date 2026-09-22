"""Bounded publication of verified evidence; tensors remain external."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .assets import sha256_file

MAX_HEAD_BYTES = 10 * 1024 * 1024
MAX_RELEASE_BYTES = 100 * 1024 * 1024
LEDGER_NAMES = frozenset({
    "receipt.json", "manifest.json", "metrics.json", "decisions.json", "events.json", "targets.json",
    "object_events.json", "select_object_events.json", "semantic_requests.json", "request_index.json",
    "training.json", "calibration.json", "event_support.json", "target_support.json",
    "teacher_selection.json", "margin_selection.json", "selection.json", "module_decision.json",
    "frozen_config.json", "budget_curve_gate.json", "confirmation.json", "native_export_parity.json",
})


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _pool_summary(value):
    result = {key: item for key, item in value.items() if key not in {"groups", "hypotheses"}}
    result["groups"] = [{**{key: item for key, item in group.items() if key != "leaf_ids"},
                         "leaf_count": len(group["leaf_ids"])} for group in value["groups"]]
    result["hypotheses"] = {group: [{
        **{key: item for key, item in row.items() if key not in {"leaf_ids", "components"}},
        "leaf_count": len(row["leaf_ids"]), "component_count": len(set(row["components"])),
        "component_leaf_counts": dict(sorted(Counter(map(str, row["components"])).items())),
    } for row in rows] for group, rows in value["hypotheses"].items()}
    return result


def export_release(attempt, destination, cache_key, resolved_config):
    """Validate and size the full package before creating its new directory."""
    attempt, destination = Path(attempt).resolve(), Path(destination).resolve()
    if destination.is_relative_to(attempt) or attempt.is_relative_to(destination):
        raise ValueError("export directory must be separate from the attempt")
    if destination.exists():
        raise FileExistsError(destination)
    payloads, sources = {}, {}
    for path in sorted(attempt.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".md"}:
            payloads[str(path.relative_to(attempt))] = path.read_bytes()

    config_path = resolved_config.get("scannet_study_config")
    external_path = attempt / "external_artifacts.json"
    if config_path and external_path.is_file():
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = Path(resolved_config["repository_root"]) / config_path
        study = Path(json.loads(config_path.read_text())["study_root"]).resolve()
        runtime = resolved_config["scannet_runtime"]
        capture = Path(runtime["output_root"]).resolve()
        if destination.is_relative_to(study) or destination.is_relative_to(capture):
            raise ValueError("release must be outside scientific runtime roots")
        for identity in json.loads(external_path.read_text())["entries"]:
            path = Path(identity["path"]).resolve()
            if path.is_relative_to(study):
                relative = path.relative_to(study)
                target = Path("study") / relative
                head = path.name == "checkpoint.pt" and relative.parts[0] in {"semantic", "geometry", "query"}
                request = path.suffix == ".json" and (path.parent.name == "requests" or "native_request_cache" in relative.parts)
                pool = path.name == "hypotheses.json" and path.parent.name == "pool"
                ledger = path.suffix == ".json" and (path.name in LEDGER_NAMES or path.name.endswith("_receipt.json"))
                if not (head or request or pool or ledger):
                    continue
            elif path.is_relative_to(capture) and path.name == "receipt.json" and path.parent.name in {"mapping_job", "frontend_job"}:
                target = Path("capture") / path.relative_to(capture)
                head = request = pool = False
            elif str(path) == runtime.get("native_build_receipt"):
                target = Path("native_build/receipt.json")
                head = request = pool = False
            else:
                continue
            if not path.is_file() or path.stat().st_size != identity["bytes"] or sha256_file(path) != identity["sha256"]:
                raise ValueError(f"release evidence changed: {path}")
            if head and path.stat().st_size > MAX_HEAD_BYTES:
                raise ValueError(f"small head exceeds 10 MiB: {path}")
            blob = path.read_bytes()
            transformation = "exact_copy"
            if pool:
                blob = _json_bytes(_pool_summary(json.loads(blob)))
                target = target.with_name("hypotheses_summary.json")
                transformation = "partition_counts_without_leaf_arrays"
            elif request:
                value = json.loads(blob)
                if "mapping" in value:
                    value["mapping"] = {key: item for key, item in value["mapping"].items() if key != "similarities"}
                blob = _json_bytes(value)
                transformation = "request_ledger_without_class_similarity_vector"
            key = str(target)
            if key in payloads and payloads[key] != blob:
                raise ValueError(f"conflicting release artifact: {key}")
            payloads[key] = blob
            sources[key] = {"source": identity, "transformation": transformation}

    manifest = {"schema_version": 2, "source_attempt": str(attempt), "cache_key": cache_key,
        "files": {key: hashlib.sha256(blob).hexdigest() for key, blob in sorted(payloads.items())},
        "source_files": sources, "total_bytes": 0, "maximum_bytes": MAX_RELEASE_BYTES,
        "large_arrays_and_backbone_weights": "external_artifacts.json; not included"}
    body_size = sum(map(len, payloads.values()))
    while manifest["total_bytes"] != body_size + len(_json_bytes(manifest)):
        manifest["total_bytes"] = body_size + len(_json_bytes(manifest))
    if manifest["total_bytes"] > MAX_RELEASE_BYTES:
        raise ValueError(f"release exceeds 100 MiB: {manifest['total_bytes']} bytes")
    payloads["export_manifest.json"] = _json_bytes(manifest)
    destination.mkdir(parents=True, exist_ok=False)
    for relative, blob in payloads.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    return manifest
