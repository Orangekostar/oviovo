"""Losslessly compact an exported evidence package, without changing frozen code."""

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path

MAXIMUM_BYTES = 100 * 1024 * 1024
PLAIN_NAMES = {
    "external_artifacts.json",
    "scientific_evidence.json",
    "method_matrix.json",
    "selection.json",
    "frozen_config.json",
    "resolved_config.json",
    "completed_phase_verification.json",
}


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def json_bytes(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def compact_release(source, destination, maximum_bytes=MAXIMUM_BYTES):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not 0 < maximum_bytes <= MAXIMUM_BYTES:
        raise ValueError("size limit must be positive and at most 100 MiB")
    if destination.exists():
        raise FileExistsError(destination)
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("source and destination must be separate")
    manifest_path = source / "export_manifest.json"
    original_manifest = manifest_path.read_bytes()
    manifest = copy.deepcopy(json.loads(original_manifest))
    if manifest["schema_version"] != 2:
        raise ValueError("unsupported export schema")
    expected = set(manifest["files"]) | {"export_manifest.json"}
    actual = {
        str(path.relative_to(source)) for path in source.rglob("*") if path.is_file()
    }
    if actual != expected:
        raise ValueError("export file set mismatch")
    payloads = {}
    for name, expected_sha in manifest["files"].items():
        path = source / name
        if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink():
            raise ValueError(f"unsafe export path: {name}")
        blob = path.read_bytes()
        if sha(blob) != expected_sha:
            raise ValueError(f"export hash mismatch: {name}")
        payloads[name] = blob
    original_bytes = sum(map(len, payloads.values())) + len(original_manifest)
    if original_bytes != manifest["total_bytes"]:
        raise ValueError("export total bytes mismatch")

    compressed_count = saved_bytes = 0
    for name, blob in list(payloads.items()):
        if not name.endswith(".json") or Path(name).name in PLAIN_NAMES:
            continue
        compressed = gzip.compress(blob, mtime=0)
        if len(blob) - len(compressed) <= 512:
            continue
        compressed_name = name + ".gz"
        if compressed_name in payloads:
            raise ValueError(f"compressed path collision: {compressed_name}")
        provenance = manifest["source_files"].pop(name, None)
        if provenance is None:
            provenance = {
                "source": {
                    "path": str(Path(manifest["source_attempt"]) / name),
                    "bytes": len(blob),
                    "sha256": sha(blob),
                },
                "transformation": "exact_copy",
            }
        manifest["source_files"][compressed_name] = {
            **provenance,
            "encoding": "gzip",
            "decoded_bytes": len(blob),
            "decoded_sha256": sha(blob),
        }
        payloads[compressed_name] = compressed
        del payloads[name]
        compressed_count += 1
        saved_bytes += len(blob) - len(compressed)

    manifest["files"] = {name: sha(blob) for name, blob in sorted(payloads.items())}
    manifest["maximum_bytes"] = maximum_bytes
    manifest["lossless_compaction"] = {
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha(Path(__file__).read_bytes()),
        },
        "input_manifest": {
            "path": str(manifest_path),
            "sha256": sha(original_manifest),
            "bytes": len(original_manifest),
        },
        "input_total_bytes": original_bytes,
        "input_maximum_bytes": json.loads(original_manifest)["maximum_bytes"],
        "policy": "gzip JSON saving over 512 bytes; keep principal documents and all non-JSON bytes exact",
        "newly_compressed_files": compressed_count,
        "payload_bytes_saved": saved_bytes,
    }
    body_size = sum(map(len, payloads.values()))
    while manifest["total_bytes"] != body_size + len(json_bytes(manifest)):
        manifest["total_bytes"] = body_size + len(json_bytes(manifest))
    if manifest["total_bytes"] > maximum_bytes:
        raise ValueError(f"release exceeds size limit: {manifest['total_bytes']} bytes")
    payloads["export_manifest.json"] = json_bytes(manifest)
    destination.mkdir(parents=True, exist_ok=False)
    for name, blob in payloads.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = compact_release(args.source, args.destination)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "total_bytes": result["total_bytes"],
                "files": len(result["files"]),
                "lossless_compaction": result["lossless_compaction"],
            }
        )
    )
