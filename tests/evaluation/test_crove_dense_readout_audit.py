from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation import audit_crove_dense_readout as audit_cli
from scripts.evaluation.audit_crove_dense_readout import audit_run_checkpoint, main
from src.evaluation.contracts import EntityPrediction, MapSnapshot
from src.evaluation.crove_dense_readout_audit import audit_dense_moved_entities
from src.evaluation.exporters.oviovo import write_map_snapshot


def _entity(
    entity_id: str,
    points: tuple[tuple[float, float, float], ...],
    *,
    dense: bool = False,
) -> EntityPrediction:
    metadata: dict[str, object] = {
        "authority": "crove_temporal",
        "anchor_entity_id": entity_id,
        "anchor_manifest_sha256": "a" * 64,
        "overlay_state": "moved",
        "temporal_entity_id": 7,
    }
    if dense:
        metadata.update(
            {
                "geometry_authority": "ovimap_anchor_template",
                "geometry_source": "causal_ovimap_anchor",
                "state_authority": "crove_temporal",
                "template_anchor_id": entity_id,
                "transform_source": "current_export_centroid_translation",
                "readout_resolution_m": None,
                "readout_resolution_source": "native_ovimap_mesh_not_declared",
            }
        )
    return EntityPrediction(
        entity_id=entity_id,
        points_xyz=np.asarray(points, dtype=np.float32),
        semantic_embedding=np.asarray((1.0, 0.0), dtype=np.float32),
        semantic_label="Chair",
        semantic_score=0.9,
        lifecycle_state="active",
        first_seen=0.0,
        last_seen=10.0,
        metadata=metadata,
    )


def _snapshot(entity: EntityPrediction) -> MapSnapshot:
    return MapSnapshot(
        method="OVIV2",
        scene_id="apartment",
        timestamp=10.0,
        entities=[entity],
        background_xyz=None,
        scope="current",
    )


def _fixtures() -> tuple[MapSnapshot, MapSnapshot, MapSnapshot]:
    anchor_points = (
        (-0.2, 0.0, 0.0),
        (-0.1, 0.0, 0.0),
        (0.1, 0.0, 0.0),
        (0.2, 0.0, 0.0),
    )
    compact_points = ((0.95, 0.0, 0.0), (1.05, 0.0, 0.0))
    dense_points = tuple((x + 1.0, y, z) for x, y, z in anchor_points)
    anchor = _snapshot(
        replace(
            _entity("ovimap:1", anchor_points),
            metadata={"authority": "ovimap_anchor"},
        )
    )
    compact = _snapshot(_entity("ovimap:1", compact_points))
    dense = _snapshot(_entity("ovimap:1", dense_points, dense=True))
    return anchor, compact, dense


def test_audit_dense_moved_entities_measures_translation_and_coverage() -> None:
    anchor, compact, dense = _fixtures()

    result = audit_dense_moved_entities(
        anchor,
        compact,
        dense,
        ("ovimap:1",),
        distance_threshold_m=0.05,
    )

    assert result["status"] == "PASS"
    assert result["moved_entity_count"] == 1
    assert result["anchor_point_count"] == 4
    assert result["compact_point_count"] == 2
    assert result["dense_point_count"] == 4
    entity = result["entities"][0]
    assert entity["anchor_entity_id"] == "ovimap:1"
    assert entity["translation_xyz"] == pytest.approx([1.0, 0.0, 0.0])
    assert entity["rigid_translation_residual_max_m"] <= 1e-7
    assert entity["dense_to_compact_nn_median_m"] == pytest.approx(0.10)
    assert entity["dense_to_compact_nn_p90_m"] == pytest.approx(0.15)
    assert entity["dense_to_compact_coverage_at_threshold"] == 0.5
    assert entity["compact_to_dense_nn_median_m"] == pytest.approx(0.05)
    assert entity["compact_to_dense_nn_p90_m"] == pytest.approx(0.05, abs=1e-6)
    assert entity["compact_to_dense_coverage_at_threshold"] == 1.0
    assert entity["geometry_gate"]["accepted"] is False
    assert "template_to_compact_coverage" in entity["geometry_gate"][
        "rejection_reasons"
    ]
    assert entity["anchor_bbox_min_xyz"] == pytest.approx([-0.2, 0.0, 0.0])
    assert entity["anchor_bbox_max_xyz"] == pytest.approx([0.2, 0.0, 0.0])
    assert entity["dense_bbox_min_xyz"] == pytest.approx([0.8, 0.0, 0.0])
    assert entity["dense_bbox_max_xyz"] == pytest.approx([1.2, 0.0, 0.0])
    assert entity["state_fields_equal"] is True


def test_audit_dense_moved_entities_rejects_non_translation_deformation() -> None:
    anchor, compact, dense = _fixtures()
    points = np.array(dense.entities[0].points_xyz, copy=True)
    points[-1, 0] += 0.1
    dense.entities[0] = _entity(
        "ovimap:1",
        tuple(tuple(float(value) for value in row) for row in points),
        dense=True,
    )

    with pytest.raises(ValueError, match="constant translation"):
        audit_dense_moved_entities(anchor, compact, dense, ("ovimap:1",))


def test_audit_dense_moved_entities_rejects_changed_state() -> None:
    anchor, compact, dense = _fixtures()
    dense.entities[0].semantic_label = "Table"

    with pytest.raises(ValueError, match="state fields"):
        audit_dense_moved_entities(anchor, compact, dense, ("ovimap:1",))


def test_audit_dense_moved_entities_rejects_missing_identity() -> None:
    anchor, compact, dense = _fixtures()

    with pytest.raises(ValueError, match="missing moved entity"):
        audit_dense_moved_entities(anchor, compact, dense, ("ovimap:2",))


@pytest.mark.parametrize("threshold", (0.0, -0.1, float("inf"), float("nan")))
def test_audit_dense_moved_entities_rejects_invalid_threshold(
    threshold: float,
) -> None:
    anchor, compact, dense = _fixtures()

    with pytest.raises(ValueError, match="distance_threshold_m"):
        audit_dense_moved_entities(
            anchor,
            compact,
            dense,
            ("ovimap:1",),
            distance_threshold_m=threshold,
        )


def test_audit_dense_moved_entities_rejects_wrong_dense_authority() -> None:
    anchor, compact, dense = _fixtures()
    dense.entities[0].metadata["geometry_authority"] = "crove_temporal"

    with pytest.raises(ValueError, match="geometry authority"):
        audit_dense_moved_entities(anchor, compact, dense, ("ovimap:1",))


def _record(path: Path, *, root: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_audit_packages(tmp_path: Path) -> tuple[Path, Path, Path]:
    anchor, compact, dense = _fixtures()
    anchor_root = tmp_path / "anchor"
    compact_root = tmp_path / "compact"
    dense_root = tmp_path / "dense"
    anchor_files = write_map_snapshot(anchor, anchor_root / "current")
    compact_files = write_map_snapshot(compact, compact_root / "checkpoint")
    dense_files = write_map_snapshot(dense, dense_root / "checkpoint")
    anchor_manifest = anchor_root / "anchor_manifest.json"
    _write_json(
        anchor_manifest,
        {
            "schema_version": 1,
            "manifest_id": "crove_ovimap_static_anchor_v1",
            "status": "PASS",
            "scene": "apartment",
            "outputs": {
                key: _record(path, root=anchor_root)
                for key, path in anchor_files.items()
            },
        },
    )

    def write_run(
        root: Path,
        files: dict[str, Path],
        *,
        mode: str,
        role: str,
    ) -> Path:
        manifest = root / "run_manifest.json"
        _write_json(
            manifest,
            {
                "schema_version": 1,
                "manifest_id": "crove_ovimap_static_anchor_composition_v1",
                "status": "PASS",
                "scene": "apartment",
                "readout_contract": {
                    "moved_geometry_mode": mode,
                    "readout_role": role,
                },
                "checkpoints": [
                    {
                        "frame_index": 10,
                        "snapshot": _record(files["snapshot"], root=root),
                        "entities": _record(files["entities"], root=root),
                        "diagnostics": {"moved_anchor_ids": ["ovimap:1"]},
                    }
                ],
            },
        )
        return manifest

    return (
        anchor_manifest,
        write_run(
            compact_root,
            compact_files,
            mode="temporal_compact",
            role="formal_baseline",
        ),
        write_run(
            dense_root,
            dense_files,
            mode="anchor_centroid_translation",
            role="visualization_shadow",
        ),
    )


def test_audit_run_checkpoint_revalidates_manifests_and_publishes(
    tmp_path: Path,
) -> None:
    anchor, compact, dense = _write_audit_packages(tmp_path)
    output_json = tmp_path / "audit" / "result.json"
    output_markdown = tmp_path / "audit" / "result.md"

    result = audit_run_checkpoint(
        anchor_manifest_path=anchor,
        compact_manifest_path=compact,
        dense_manifest_path=dense,
        frame_index=10,
        output_json=output_json,
        output_markdown=output_markdown,
    )

    assert result["status"] == "PASS"
    assert result["frame_index"] == 10
    assert result["geometry"]["dense_point_count"] == 4
    assert output_json.is_file()
    assert output_markdown.is_file()
    assert json.loads(output_json.read_text(encoding="utf-8")) == result


def test_audit_run_checkpoint_rejects_tampered_dense_artifact(
    tmp_path: Path,
) -> None:
    anchor, compact, dense = _write_audit_packages(tmp_path)
    dense_payload = json.loads(dense.read_text(encoding="utf-8"))
    snapshot = dense.parent / dense_payload["checkpoints"][0]["snapshot"]["path"]
    snapshot.write_bytes(snapshot.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="binding mismatch"):
        audit_run_checkpoint(
            anchor_manifest_path=anchor,
            compact_manifest_path=compact,
            dense_manifest_path=dense,
            frame_index=10,
            output_json=tmp_path / "audit.json",
            output_markdown=tmp_path / "audit.md",
        )


def test_audit_run_checkpoint_rejects_manifest_changed_during_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor, compact, dense = _write_audit_packages(tmp_path)
    original = audit_cli.audit_dense_moved_entities

    def mutate_manifest(*args: object, **kwargs: object) -> dict[str, object]:
        result = original(*args, **kwargs)
        dense.write_bytes(dense.read_bytes() + b" ")
        return result

    monkeypatch.setattr(audit_cli, "audit_dense_moved_entities", mutate_manifest)

    with pytest.raises(RuntimeError, match="changed before publication"):
        audit_run_checkpoint(
            anchor_manifest_path=anchor,
            compact_manifest_path=compact,
            dense_manifest_path=dense,
            frame_index=10,
            output_json=tmp_path / "audit.json",
            output_markdown=tmp_path / "audit.md",
        )


def test_audit_cli_help_lists_checkpoint_contract(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--compact-run-manifest" in output
    assert "--dense-run-manifest" in output
    assert "--frame-index" in output
