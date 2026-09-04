from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.evaluation.audit_ovi_rescene_sources import (
    SourceAuditError,
    audit_sources,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/external/ovi_rescene_sources.json"


def _run_git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(root: Path, files: dict[str, str]) -> str:
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    _run_git(root, "config", "user.email", "source-audit@example.invalid")
    _run_git(root, "config", "user.name", "Source Audit")
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _run_git(root, "add", ".")
    _run_git(root, "commit", "-q", "-m", "fixture")
    return _run_git(root, "rev-parse", "HEAD")


def _binding(root: Path, relative: str) -> dict[str, object]:
    content = (root / relative).read_bytes()
    return {
        "path": relative,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


@pytest.fixture
def source_fixture(tmp_path: Path) -> tuple[dict[str, object], dict[str, Path]]:
    ovimap = tmp_path / "ovimap"
    rescene = tmp_path / "rescene"
    current = tmp_path / "current"
    ovimap_files = {
        "README.md": "instance mapping voxel size: 0.01 m\n",
        "LICENSE.txt": "MIT License\n",
        "label_voxel.h": "struct LabelVoxel {};\n",
        "label_tsdf_map.h": "TSDF and label layers\n",
        "panoptic_mapping_.py": "# mesh export\n",
    }
    rescene_files = {
        "README.md": "## Checkpoints\nComing soon.\n",
        "LICENSE": "MIT License\n",
        "rescene.yaml": "D: 4\ntemporal_window: 2\n",
        "indoor.yaml": "voxel_size: 0.02\n",
        "concerto.yaml": "name: concerto_base\n",
        "semseg.py": "# concatenate [x, y, z, t]\n",
        "minkowski_utils.py": "# spatial voxelization only\n",
        "pointcept_utils.py": "# temporal stage remains integer\n",
        "models_rescene.py": "# shared temporal decoder\n",
        "models_pointcept.py": "# learned hierarchy\n",
    }
    current_files = {
        "legacy.yaml": "voxel_size_m: 0.05\n",
        "evaluator.json": '{"voxel_size_m": 0.05}\n',
    }
    commits = {
        "ovimap": _repository(ovimap, ovimap_files),
        "rescene": _repository(rescene, rescene_files),
        "current": _repository(current, current_files),
    }
    checkouts = {"ovimap": ovimap, "rescene": rescene, "current": current}
    files_by_source = {
        "ovimap": tuple(ovimap_files),
        "rescene": tuple(rescene_files),
        "current": tuple(current_files),
    }
    sources: dict[str, object] = {}
    for source_id in ("ovimap", "rescene", "current"):
        source_root = checkouts[source_id]
        sources[source_id] = {
            "repository_url": f"https://example.invalid/{source_id}",
            "commit": commits[source_id],
            "license": "MIT" if source_id != "current" else "PROJECT",
            "license_path": (
                "LICENSE.txt"
                if source_id == "ovimap"
                else "LICENSE"
                if source_id == "rescene"
                else None
            ),
            "required_files": [
                _binding(source_root, relative)
                for relative in files_by_source[source_id]
            ],
        }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "EXTERNAL_SOURCE_PASS",
        "voxel_contract_status": "VOXEL_CONTRACT_PASS",
        "sources": sources,
        "checkpoint": {
            "status": "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT",
            "source_id": "rescene",
            "evidence": {
                "path": "README.md",
                "contains": "Coming soon.",
            },
        },
        "representations": {
            "ovi_mapping": {
                "resolution_m": 0.01,
                "source_id": "ovimap",
                "evidence": {
                    "path": "README.md",
                    "contains": "voxel size: 0.01 m",
                },
                "persistent": True,
                "input_only": False,
                "final_map_authoritative": True,
            },
            "rescene_neural": {
                "resolution_m": 0.02,
                "source_id": "rescene",
                "evidence": {
                    "path": "indoor.yaml",
                    "contains": "voxel_size: 0.02",
                },
                "persistent": False,
                "input_only": True,
                "final_map_authoritative": False,
            },
            "crove_legacy_temporal": {
                "resolution_m": 0.05,
                "source_id": "current",
                "evidence": {
                    "path": "legacy.yaml",
                    "contains": "voxel_size_m: 0.05",
                },
                "persistent": True,
                "input_only": False,
                "final_map_authoritative": False,
            },
            "evaluator": {
                "resolution_m": 0.05,
                "source_id": "current",
                "evidence": {
                    "path": "evaluator.json",
                    "contains": '"voxel_size_m": 0.05',
                },
                "persistent": False,
                "input_only": True,
                "final_map_authoritative": False,
            },
        },
        "adapter_contract": {
            "spatial_quantization_axes": ["x", "y", "z"],
            "temporal_coordinate": "exact_visit_index_0_or_1",
            "reverse_index": "csr_all_contributing_ovi_samples",
            "final_geometry_sources": ["ovi_t0", "ovi_t1"],
        },
    }
    return manifest, checkouts


def test_audit_accepts_exact_sources_and_four_distinct_voxel_roles(
    source_fixture: tuple[dict[str, object], dict[str, Path]],
) -> None:
    manifest, checkouts = source_fixture

    result = audit_sources(manifest, checkouts)

    assert result.status == "EXTERNAL_SOURCE_PASS"
    assert result.voxel_status == "VOXEL_CONTRACT_PASS"
    assert result.checkpoint_status == "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT"
    assert result.voxel_sizes_m == {
        "ovi_mapping": 0.01,
        "rescene_neural": 0.02,
        "crove_legacy_temporal": 0.05,
        "evaluator": 0.05,
    }
    assert len(result.verified_files) == 17


def test_audit_rejects_required_working_file_different_from_commit(
    source_fixture: tuple[dict[str, object], dict[str, Path]],
) -> None:
    manifest, checkouts = source_fixture
    (checkouts["ovimap"] / "README.md").write_text("changed\n", encoding="utf-8")

    with pytest.raises(SourceAuditError, match="differs from pinned commit"):
        audit_sources(manifest, checkouts)


def test_audit_rejects_symlink_checkout_root(
    source_fixture: tuple[dict[str, object], dict[str, Path]],
    tmp_path: Path,
) -> None:
    manifest, checkouts = source_fixture
    symlink = tmp_path / "ovimap-link"
    symlink.symlink_to(checkouts["ovimap"], target_is_directory=True)
    checkouts["ovimap"] = symlink

    with pytest.raises(SourceAuditError, match="checkout must not be a symlink"):
        audit_sources(manifest, checkouts)


def test_audit_rejects_declared_file_hash_mismatch(
    source_fixture: tuple[dict[str, object], dict[str, Path]],
) -> None:
    manifest, checkouts = source_fixture
    manifest["sources"]["rescene"]["required_files"][0]["sha256"] = "0" * 64

    with pytest.raises(SourceAuditError, match="SHA-256"):
        audit_sources(manifest, checkouts)


def test_audit_rejects_neural_tokens_as_final_geometry(
    source_fixture: tuple[dict[str, object], dict[str, Path]],
) -> None:
    manifest, checkouts = source_fixture
    manifest["representations"]["rescene_neural"]["final_map_authoritative"] = True

    with pytest.raises(SourceAuditError, match="ReScene neural tokens"):
        audit_sources(manifest, checkouts)


def test_audit_rejects_spatially_quantized_time_coordinate(
    source_fixture: tuple[dict[str, object], dict[str, Path]],
) -> None:
    manifest, checkouts = source_fixture
    manifest["adapter_contract"]["spatial_quantization_axes"] = ["x", "y", "z", "t"]

    with pytest.raises(SourceAuditError, match="spatial quantization"):
        audit_sources(manifest, checkouts)


def test_checked_manifest_is_public_and_source_bound() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    text = MANIFEST.read_text(encoding="utf-8")

    assert payload["status"] == "EXTERNAL_SOURCE_PASS"
    assert payload["voxel_contract_status"] == "VOXEL_CONTRACT_PASS"
    assert payload["sources"]["ovimap"]["commit"] == (
        "f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424"
    )
    assert payload["sources"]["rescene"]["commit"] == (
        "fb2fe42eb8f1e926567c48eea9acb874e608ee10"
    )
    assert payload["checkpoint"]["status"] == (
        "BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT"
    )
    assert "/home/" not in text
    assert "file://" not in text
