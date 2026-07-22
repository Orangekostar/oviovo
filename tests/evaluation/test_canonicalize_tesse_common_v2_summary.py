from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.evaluation.canonicalize_tesse_common_v2_summary import (
    canonical_summary_bytes,
    canonicalize_summary,
)


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fixture(root: Path, external: Path) -> tuple[Path, dict[str, Path]]:
    (root / "evaluation").mkdir(parents=True)
    checkpoint = root / "temporal/checkpoints/00000010"
    checkpoint.mkdir(parents=True)
    sidecars = root / "temporal/sidecars"
    sidecars.mkdir()
    temporal = root / "temporal/temporal_manifest.json"
    temporal.write_text('{"mode":"causal_checkpoints"}\n', encoding="utf-8")
    schedule = sidecars / "schedule.json"
    schedule.write_text('{"dataset":"TESSE-CD"}\n', encoding="utf-8")
    snapshot = checkpoint / "snapshot.npz"
    snapshot.write_bytes(b"snapshot")
    entities = checkpoint / "entities.jsonl"
    entities.write_text('{"entity_id":"one"}\n', encoding="utf-8")

    external.mkdir(exist_ok=True)
    external_sources = {
        "target_manifest": external / "target-manifest.json",
        "target_arrays": external / "targets.npz",
        "aliases": external / "aliases.yaml",
        "label_space": external / "labels.yaml",
        "evaluator": external / "evaluator.py",
    }
    for role, path in external_sources.items():
        path.write_text(f"{role}\n", encoding="utf-8")

    summary = root / "evaluation/summary.json"
    _write_json(
        summary,
        {
            "schema_version": 1,
            "manifest_id": "tesse_cd_common_v2_scene_summary",
            "dataset": "TESSE-CD",
            "protocol": "tesse_cd_common_v2",
            "status": "PASS",
            "method": "OVIV2",
            "mode": "causal_checkpoints",
            "scene": "apartment",
            "metrics": {
                "background_f5": 0.5,
                "current_miou": 0.6,
                "ghost_rate": 0.1,
                "recovery_frames": 100.0,
            },
            "frames": [{"frame_id": 10, "current_miou": 0.6}],
            "sources": {
                "temporal_index": _record(temporal),
                "schedule": _record(schedule),
                "snapshot.000010": _record(snapshot),
                "entities.000010": _record(entities),
                **{
                    role: _record(path)
                    for role, path in external_sources.items()
                },
            },
        },
    )
    return summary, external_sources


def test_canonical_summary_is_identical_across_independent_run_roots(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    first, expected = _fixture(tmp_path / "run-a", external)
    second, _ = _fixture(tmp_path / "run-b", external)

    first_payload = canonicalize_summary(
        first,
        artifact_root=tmp_path / "run-a",
        external_sources=expected,
    )
    second_payload = canonicalize_summary(
        second,
        artifact_root=tmp_path / "run-b",
        external_sources=expected,
    )

    assert canonical_summary_bytes(first_payload) == canonical_summary_bytes(
        second_payload
    )
    assert first_payload["manifest_id"] == (
        "tesse_cd_common_v2_canonical_scene_summary_v1"
    )
    assert first_payload["sources"]["temporal_index"] == {
        "role": "temporal_index",
        "sha256": hashlib.sha256(
            (tmp_path / "run-a/temporal/temporal_manifest.json").read_bytes()
        ).hexdigest(),
        "byte_count": 30,
    }
    assert str(tmp_path / "run-a") not in canonical_summary_bytes(
        first_payload
    ).decode("utf-8")


def test_canonical_summary_rejects_internal_role_path_escape(tmp_path: Path) -> None:
    summary, expected = _fixture(tmp_path / "run", tmp_path / "external")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["sources"]["temporal_index"] = _record(expected["evaluator"])
    _write_json(summary, payload)

    with pytest.raises(ValueError, match="temporal_index path mismatch"):
        canonicalize_summary(
            summary,
            artifact_root=tmp_path / "run",
            external_sources=expected,
        )


def test_canonical_summary_rejects_noncanonical_parent_path(tmp_path: Path) -> None:
    summary, expected = _fixture(tmp_path / "run", tmp_path / "external")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["sources"]["schedule"]["path"] = str(
        tmp_path / "run/temporal/sidecars/../sidecars/schedule.json"
    )
    _write_json(summary, payload)

    with pytest.raises(ValueError, match="schedule source path must be canonical"):
        canonicalize_summary(
            summary,
            artifact_root=tmp_path / "run",
            external_sources=expected,
        )


def test_canonical_summary_rejects_reserved_transform_fields(tmp_path: Path) -> None:
    summary, expected = _fixture(tmp_path / "run", tmp_path / "external")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["canonicalization"] = {"physical_paths_serialized": True}
    _write_json(summary, payload)

    with pytest.raises(ValueError, match="reserved canonicalization fields"):
        canonicalize_summary(
            summary,
            artifact_root=tmp_path / "run",
            external_sources=expected,
        )


def test_canonical_summary_rejects_mutated_hashed_source(tmp_path: Path) -> None:
    summary, expected = _fixture(tmp_path / "run", tmp_path / "external")
    (tmp_path / "run/temporal/sidecars/schedule.json").write_text(
        "mutated\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="schedule content mismatch"):
        canonicalize_summary(
            summary,
            artifact_root=tmp_path / "run",
            external_sources=expected,
        )


def test_canonical_summary_changes_when_bound_content_changes(tmp_path: Path) -> None:
    external = tmp_path / "external"
    first, expected = _fixture(tmp_path / "run-a", external)
    second, _ = _fixture(tmp_path / "run-b", external)
    changed = tmp_path / "run-b/temporal/checkpoints/00000010/entities.jsonl"
    changed.write_text('{"entity_id":"two"}\n', encoding="utf-8")
    second_payload = json.loads(second.read_text(encoding="utf-8"))
    second_payload["sources"]["entities.000010"] = _record(changed)
    _write_json(second, second_payload)

    first_bytes = canonical_summary_bytes(
        canonicalize_summary(
            first,
            artifact_root=tmp_path / "run-a",
            external_sources=expected,
        )
    )
    second_bytes = canonical_summary_bytes(
        canonicalize_summary(
            second,
            artifact_root=tmp_path / "run-b",
            external_sources=expected,
        )
    )

    assert first_bytes != second_bytes


def test_canonicalizer_cli_writes_one_no_replace_summary(tmp_path: Path) -> None:
    summary, expected = _fixture(tmp_path / "run", tmp_path / "external")
    output = tmp_path / "canonical.json"
    command = [
        sys.executable,
        str(
            Path(__file__).resolve().parents[2]
            / "scripts/evaluation/canonicalize_tesse_common_v2_summary.py"
        ),
        "--summary",
        str(summary),
        "--artifact-root",
        str(tmp_path / "run"),
    ]
    for role, path in sorted(expected.items()):
        command.extend(("--external-source", f"{role}={path}"))
    command.extend(("--output", str(output)))

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["manifest_id"] == (
        "tesse_cd_common_v2_canonical_scene_summary_v1"
    )
    repeated = subprocess.run(command, capture_output=True, text=True, check=False)
    assert repeated.returncode != 0
    assert "FileExistsError" in repeated.stderr
