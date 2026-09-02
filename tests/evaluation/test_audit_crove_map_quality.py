from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluation.audit_crove_map_quality import (
    audit_ascii_instance_ply,
    audit_map_quality,
    bind_producer_chain,
    load_pass_manifest,
    main,
    parse_instance_color_log,
    summarize_geometry_authority,
    verify_file_record,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def _file_record(path: Path, *, recorded_path: str | None = None) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": recorded_path if recorded_path is not None else str(path.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
    }


def test_verify_file_record_accepts_bound_relative_regular_file(tmp_path: Path) -> None:
    artifact = tmp_path / "inputs" / "artifact.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"bound-input")

    observed = verify_file_record(
        _file_record(artifact, recorded_path="inputs/artifact.bin"),
        base_dir=tmp_path,
        label="fixture artifact",
    )

    assert observed == {
        "path": str(artifact.resolve()),
        "sha256": hashlib.sha256(b"bound-input").hexdigest(),
        "byte_count": len(b"bound-input"),
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("sha256", "f" * 64, "SHA-256 mismatch"),
        ("byte_count", 999, "byte-count mismatch"),
    ),
)
def test_verify_file_record_rejects_tampered_binding(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"original")
    record = _file_record(artifact)
    record[field] = value

    with pytest.raises(ValueError, match=message):
        verify_file_record(record, base_dir=tmp_path, label="fixture artifact")


def test_verify_file_record_rejects_relative_escape(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")

    with pytest.raises(ValueError, match="escapes manifest directory"):
        verify_file_record(
            _file_record(outside, recorded_path="../outside.bin"),
            base_dir=manifest_dir,
            label="fixture artifact",
        )


def test_verify_file_record_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"target")
    alias = tmp_path / "alias.bin"
    alias.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        verify_file_record(
            _file_record(target, recorded_path="alias.bin"),
            base_dir=tmp_path,
            label="fixture artifact",
        )


def test_load_pass_manifest_returns_payload_and_bound_source(tmp_path: Path) -> None:
    manifest = _write_json(
        tmp_path / "manifest.json",
        {"schema_version": 1, "status": "PASS", "scene": "apartment"},
    )

    payload, source = load_pass_manifest(manifest, label="native manifest")

    assert payload["scene"] == "apartment"
    assert source == _file_record(manifest)


def test_load_pass_manifest_rejects_non_pass_status(tmp_path: Path) -> None:
    manifest = _write_json(
        tmp_path / "manifest.json",
        {"schema_version": 1, "status": "FAIL"},
    )

    with pytest.raises(ValueError, match="status PASS"):
        load_pass_manifest(manifest, label="native manifest")


def _write_ascii_ply(
    path: Path,
    vertices: tuple[tuple[float, float, float, int, int, int], ...],
    *,
    face_count: int = 1,
    format_line: str = "format ascii 1.0",
    extra_vertex_property: str | None = None,
) -> Path:
    properties = [
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    if extra_vertex_property is not None:
        properties.append(extra_vertex_property)
    header = [
        "ply",
        format_line,
        f"element vertex {len(vertices)}",
        *properties,
        f"element face {face_count}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    rows = [" ".join(str(value) for value in vertex) for vertex in vertices]
    path.write_text("\n".join((*header, *rows)) + "\n", encoding="ascii")
    return path


def _write_color_log(
    path: Path, entries: tuple[tuple[int, int, int, int], ...]
) -> Path:
    path.write_text(
        "\n".join(
            f"Instance: {instance_id} Color: ({red},{green},{blue})"
            for instance_id, red, green, blue in entries
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_ascii_ply_audit_proves_instance_palette_and_counts_background(
    tmp_path: Path,
) -> None:
    mesh = _write_ascii_ply(
        tmp_path / "mesh.ply",
        (
            (0.0, 0.0, 0.0, 10, 20, 30),
            (1.0, 0.0, 0.0, 10, 20, 30),
            (0.0, 1.0, 0.0, 40, 50, 60),
            (0.0, 0.0, 1.0, 0, 0, 0),
        ),
        face_count=12,
    )
    colors = parse_instance_color_log(
        _write_color_log(
            tmp_path / "mapping.log",
            ((2, 40, 50, 60), (1, 10, 20, 30)),
        )
    )

    result = audit_ascii_instance_ply(mesh, colors)

    assert result["format"] == "ascii 1.0"
    assert result["vertex_count"] == 4
    assert result["face_count"] == 12
    assert result["visualization_mode"] == "instance_palette"
    assert result["registered_instance_count"] == 2
    assert result["present_instance_count"] == 2
    assert result["instance_vertex_count"] == 3
    assert result["background_or_unregistered_vertex_count"] == 1
    assert result["missing_instance_ids"] == []
    assert result["instances"] == [
        {"instance_id": 1, "rgb": [10, 20, 30], "vertex_count": 2},
        {"instance_id": 2, "rgb": [40, 50, 60], "vertex_count": 1},
    ]


def test_instance_color_log_rejects_reused_color(tmp_path: Path) -> None:
    log = _write_color_log(
        tmp_path / "mapping.log",
        ((1, 10, 20, 30), (2, 10, 20, 30)),
    )

    with pytest.raises(ValueError, match="reused"):
        parse_instance_color_log(log)


def test_ascii_ply_audit_rejects_truncated_vertices(tmp_path: Path) -> None:
    mesh = _write_ascii_ply(
        tmp_path / "mesh.ply",
        ((0.0, 0.0, 0.0, 10, 20, 30),),
    )
    content = mesh.read_text(encoding="ascii").replace(
        "element vertex 1", "element vertex 2"
    )
    mesh.write_text(content, encoding="ascii")

    with pytest.raises(ValueError, match="truncated vertex data"):
        audit_ascii_instance_ply(mesh, {1: (10, 20, 30)})


def test_ascii_ply_audit_rejects_invalid_rgb(tmp_path: Path) -> None:
    mesh = _write_ascii_ply(
        tmp_path / "mesh.ply",
        ((0.0, 0.0, 0.0, 256, 20, 30),),
    )

    with pytest.raises(ValueError, match="RGB values"):
        audit_ascii_instance_ply(mesh, {1: (10, 20, 30)})


@pytest.mark.parametrize(
    ("format_line", "extra_property", "message"),
    (
        ("format binary_little_endian 1.0", None, "ASCII PLY"),
        (
            "format ascii 1.0",
            "property list uchar int samples",
            "scalar vertex properties",
        ),
    ),
)
def test_ascii_ply_audit_rejects_unsupported_vertex_encoding(
    tmp_path: Path,
    format_line: str,
    extra_property: str | None,
    message: str,
) -> None:
    mesh = _write_ascii_ply(
        tmp_path / "mesh.ply",
        ((0.0, 0.0, 0.0, 10, 20, 30),),
        format_line=format_line,
        extra_vertex_property=extra_property,
    )

    with pytest.raises(ValueError, match=message):
        audit_ascii_instance_ply(mesh, {1: (10, 20, 30)})


def _write_producer_tree(root: Path) -> dict[str, Path]:
    sources = {
        "python_entrypoint": root / "scripts/panoptic_mapping_.py",
        "instance_mesh_writer": root
        / "mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm/src/global_segment_map_py.cpp",
        "instance_colorizer": root
        / "mapping_ros_ws/src/consistent_panoptic_mapping/global_segment_map/src/meshing/label_tsdf_mesh_integrator.cc",
    }
    contents = {
        "python_entrypoint": "gsm_node.generateMesh(output, frame, False, False, True)\n",
        "instance_mesh_writer": (
            "mesh_instance_integrator_->generateMesh(false, true);\n"
            'outputMeshLayerAsPly(folder + "/instance_mesh_" + frame, false, layer);\n'
        ),
        "instance_colorizer": (
            "case kInstance: {\n"
            "instance_color_map_.getColor(instance_label, &mesh->colors[i]);\n"
            "}\n"
        ),
    }
    for role, path in sources.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents[role], encoding="utf-8")
    return sources


def _native_payload(root: Path, entrypoint: Path) -> dict[str, object]:
    extension = root / "mapping_ros_ws/devel/lib/consistent_gsm.fixture.so"
    extension.parent.mkdir(parents=True, exist_ok=True)
    if not extension.exists():
        extension.write_bytes(b"compiled-consistent-gsm")
    source_hashes = root.parent / "environment/source-hashes.json"
    if not source_hashes.exists():
        _write_json(
            source_hashes,
            {"OVI-MAP": {"commit": "58a804e2d7c82ba05a489eb071aba3367301fed8"}},
        )
    return {
        "commands": {
            "mapping": {
                "argv": ["python", str(entrypoint), "--dataset", "replica"],
                "cwd": str(root),
                "exit_code": 0,
            }
        },
        "preflight": {
            "ovimap_commit": "58a804e2d7c82ba05a489eb071aba3367301fed8",
            "sources": {
                "consistent_gsm_extension": _file_record(extension),
                "mapper": _file_record(entrypoint),
                "source_hashes": _file_record(source_hashes),
            },
        },
        "scene": "apartment",
        "status": "PASS",
    }


def test_bind_producer_chain_records_exact_sources(tmp_path: Path) -> None:
    root = tmp_path / "OVI-MAP"
    sources = _write_producer_tree(root)

    result = bind_producer_chain(_native_payload(root, sources["python_entrypoint"]))

    assert result["ovimap_root"] == str(root.resolve())
    assert result["ovimap_commit"] == "58a804e2d7c82ba05a489eb071aba3367301fed8"
    assert list(result["sources"]) == [
        "python_entrypoint",
        "instance_mesh_writer",
        "instance_colorizer",
    ]
    for role, source in sources.items():
        assert result["sources"][role] == _file_record(source)
    assert result["chain"] == [
        "panoptic_mapping.generateMesh(instance_only)",
        "GlobalSegmentMap_py.generateMesh",
        "MeshLabelIntegrator.kInstance",
        "instance_color_map",
        "outputMeshLayerAsPly",
    ]


def test_bind_producer_chain_rejects_entrypoint_outside_mapping_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "OVI-MAP"
    _write_producer_tree(root)
    outside = tmp_path / "outside.py"
    outside.write_text(
        "gsm_node.generateMesh(output, frame, False, False, True)\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside mapping cwd"):
        bind_producer_chain(_native_payload(root, outside))


@pytest.mark.parametrize(
    ("role", "removed", "message"),
    (
        ("python_entrypoint", "False, False, True", "instance-only generateMesh"),
        ("instance_mesh_writer", "outputMeshLayerAsPly", "instance mesh writer"),
        ("instance_colorizer", "instance_color_map_", "instance colorizer"),
    ),
)
def test_bind_producer_chain_rejects_missing_required_marker(
    tmp_path: Path,
    role: str,
    removed: str,
    message: str,
) -> None:
    root = tmp_path / "OVI-MAP"
    sources = _write_producer_tree(root)
    source = sources[role]
    source.write_text(
        source.read_text(encoding="utf-8").replace(removed, "removed_marker"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        bind_producer_chain(_native_payload(root, sources["python_entrypoint"]))


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    return path


def _anchor_entity(entity_id: str, point_count: int) -> dict[str, object]:
    return {
        "entity_id": entity_id,
        "point_count": point_count,
        "metadata": {"authority": "ovimap_anchor"},
    }


def _current_anchor(
    entity_id: str,
    point_count: int,
    *,
    authority: str,
    overlay_state: str,
    temporal_entity_id: int | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "anchor_entity_id": entity_id,
        "authority": authority,
        "overlay_state": overlay_state,
    }
    if temporal_entity_id is not None:
        metadata["temporal_entity_id"] = temporal_entity_id
    return {
        "entity_id": entity_id,
        "point_count": point_count,
        "metadata": metadata,
    }


def _current_new(entity_id: int, point_count: int) -> dict[str, object]:
    return {
        "entity_id": f"temporal:{entity_id}",
        "point_count": point_count,
        "metadata": {
            "anchor_entity_id": None,
            "authority": "crove_temporal",
            "overlay_state": "new",
            "temporal_entity_id": entity_id,
        },
    }


def _geometry_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    anchor = _write_jsonl(
        tmp_path / "anchor.jsonl",
        [
            _anchor_entity("ovimap:1", 100),
            _anchor_entity("ovimap:2", 200),
            _anchor_entity("ovimap:3", 300),
            _anchor_entity("ovimap:4", 400),
        ],
    )
    current = _write_jsonl(
        tmp_path / "current.jsonl",
        [
            _current_anchor(
                "ovimap:1",
                100,
                authority="ovimap_anchor",
                overlay_state="unchanged",
            ),
            _current_anchor(
                "ovimap:2",
                20,
                authority="crove_temporal",
                overlay_state="moved",
                temporal_entity_id=20,
            ),
            _current_anchor(
                "ovimap:3",
                30,
                authority="crove_temporal",
                overlay_state="moved",
                temporal_entity_id=30,
            ),
            _current_new(10, 5),
            _current_new(11, 15),
        ],
    )
    diagnostics = {
        "moved_anchor_ids": ["ovimap:2", "ovimap:3"],
        "new_temporal_ids": [10, 11],
        "occluded_anchor_ids": [],
        "removed_anchor_ids": ["ovimap:4"],
        "unchanged_anchor_ids": ["ovimap:1"],
    }
    return anchor, current, diagnostics


def test_geometry_authority_summary_quantifies_dense_to_sparse_switch(
    tmp_path: Path,
) -> None:
    anchor, current, diagnostics = _geometry_fixture(tmp_path)

    result = summarize_geometry_authority(anchor, current, diagnostics)

    assert result["anchor"] == {"entity_count": 4, "total_points": 1000}
    assert result["current"]["entity_count"] == 5
    assert result["current"]["total_points"] == 170
    assert result["current"]["by_authority"] == [
        {"authority": "crove_temporal", "entity_count": 4, "total_points": 70},
        {"authority": "ovimap_anchor", "entity_count": 1, "total_points": 100},
    ]
    assert result["moved"] == {
        "entity_count": 2,
        "anchor_points": 500,
        "current_points": 50,
        "retained_point_ratio": pytest.approx(0.1),
        "entities": [
            {
                "anchor_entity_id": "ovimap:2",
                "anchor_points": 200,
                "current_points": 20,
                "retained_point_ratio": pytest.approx(0.1),
                "temporal_entity_id": 20,
            },
            {
                "anchor_entity_id": "ovimap:3",
                "anchor_points": 300,
                "current_points": 30,
                "retained_point_ratio": pytest.approx(0.1),
                "temporal_entity_id": 30,
            },
        ],
    }
    assert result["new_temporal"] == {
        "entity_count": 2,
        "total_points": 20,
        "minimum_points": 5,
        "median_points": 10.0,
        "maximum_points": 15,
    }
    assert result["removed"] == {"entity_count": 1, "anchor_points": 400}
    assert result["conclusion"] == "dense_anchor_to_sparse_temporal_switch_measured"


def test_geometry_authority_rejects_missing_moved_replacement(tmp_path: Path) -> None:
    anchor, current, diagnostics = _geometry_fixture(tmp_path)
    records = [
        json.loads(line)
        for line in current.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["entity_id"] != "ovimap:3"
    ]
    _write_jsonl(current, records)

    with pytest.raises(ValueError, match="moved anchor replacement is missing"):
        summarize_geometry_authority(anchor, current, diagnostics)


def test_geometry_authority_rejects_duplicate_current_entity(tmp_path: Path) -> None:
    anchor, current, diagnostics = _geometry_fixture(tmp_path)
    records = [
        json.loads(line) for line in current.read_text(encoding="utf-8").splitlines()
    ]
    records.append(dict(records[0]))
    _write_jsonl(current, records)

    with pytest.raises(ValueError, match="duplicate current entity ID"):
        summarize_geometry_authority(anchor, current, diagnostics)


def test_geometry_authority_rejects_moved_anchor_authority_mismatch(
    tmp_path: Path,
) -> None:
    anchor, current, diagnostics = _geometry_fixture(tmp_path)
    records = [
        json.loads(line) for line in current.read_text(encoding="utf-8").splitlines()
    ]
    records[1]["metadata"]["authority"] = "ovimap_anchor"
    _write_jsonl(current, records)

    with pytest.raises(
        ValueError, match="moved anchor must use crove_temporal authority"
    ):
        summarize_geometry_authority(anchor, current, diagnostics)


def test_geometry_authority_rejects_diagnostics_metadata_disagreement(
    tmp_path: Path,
) -> None:
    anchor, current, diagnostics = _geometry_fixture(tmp_path)
    records = [
        json.loads(line) for line in current.read_text(encoding="utf-8").splitlines()
    ]
    records[-1]["metadata"]["temporal_entity_id"] = 99
    _write_jsonl(current, records)

    with pytest.raises(ValueError, match="new temporal metadata disagrees"):
        summarize_geometry_authority(anchor, current, diagnostics)


def _relative_record(path: Path, base_dir: Path) -> dict[str, object]:
    return _file_record(path, recorded_path=str(path.relative_to(base_dir)))


def _audit_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    ovimap_root = tmp_path / "OVI-MAP"
    producer_sources = _write_producer_tree(ovimap_root)
    native_root = tmp_path / "native"
    native_root.mkdir()
    color_log = _write_color_log(
        native_root / "mapping.log",
        (
            (1, 10, 20, 30),
            (2, 40, 50, 60),
            (3, 70, 80, 90),
            (4, 100, 110, 120),
        ),
    )
    vertices = tuple(
        (float(index), 0.0, 0.0, *color)
        for color, count in (
            ((10, 20, 30), 100),
            ((40, 50, 60), 200),
            ((70, 80, 90), 300),
            ((100, 110, 120), 400),
        )
        for index in range(count)
    ) + ((0.0, 0.0, 0.0, 0, 0, 0),)
    mesh = _write_ascii_ply(native_root / "instance_mesh_4.ply", vertices)
    native_payload = _native_payload(ovimap_root, producer_sources["python_entrypoint"])
    native_manifest = _write_json(
        native_root / "native_mapping_manifest.json",
        {
            **native_payload,
            "artifacts": {
                "instance_color_log": _file_record(color_log),
                "instance_mesh": _file_record(mesh),
            },
        },
    )

    anchor_root = tmp_path / "anchor"
    anchor_root.mkdir()
    anchor_entities = _write_jsonl(
        anchor_root / "entities.jsonl",
        [
            _anchor_entity("ovimap:1", 100),
            _anchor_entity("ovimap:2", 200),
            _anchor_entity("ovimap:3", 300),
            _anchor_entity("ovimap:4", 400),
        ],
    )
    anchor_snapshot = anchor_root / "snapshot.npz"
    anchor_snapshot.write_bytes(b"anchor-snapshot")
    anchor_manifest = _write_json(
        anchor_root / "anchor_manifest.json",
        {
            "outputs": {
                "entities": _relative_record(anchor_entities, anchor_root),
                "snapshot": _relative_record(anchor_snapshot, anchor_root),
            },
            "scene": "apartment",
            "sources": {
                "instance_color_log": _file_record(color_log),
                "instance_mesh": _file_record(mesh),
                "native_manifest": _file_record(native_manifest),
            },
            "status": "PASS",
        },
    )

    composed_root = tmp_path / "composed"
    checkpoint_root = composed_root / "checkpoints/final"
    checkpoint_root.mkdir(parents=True)
    current_entities = _write_jsonl(
        checkpoint_root / "entities.jsonl",
        [
            _current_anchor(
                "ovimap:1",
                100,
                authority="ovimap_anchor",
                overlay_state="unchanged",
            ),
            _current_anchor(
                "ovimap:2",
                20,
                authority="crove_temporal",
                overlay_state="moved",
                temporal_entity_id=20,
            ),
            _current_anchor(
                "ovimap:3",
                30,
                authority="crove_temporal",
                overlay_state="moved",
                temporal_entity_id=30,
            ),
            _current_new(10, 5),
            _current_new(11, 15),
        ],
    )
    current_snapshot = checkpoint_root / "snapshot.npz"
    current_snapshot.write_bytes(b"current-snapshot")
    diagnostics = {
        "frame_index": 9,
        "moved_anchor_ids": ["ovimap:2", "ovimap:3"],
        "new_temporal_ids": [10, 11],
        "occluded_anchor_ids": [],
        "removed_anchor_ids": ["ovimap:4"],
        "unchanged_anchor_ids": ["ovimap:1"],
    }
    diagnostics_path = _write_json(checkpoint_root / "diagnostics.json", diagnostics)
    composed_manifest = _write_json(
        composed_root / "run_manifest.json",
        {
            "checkpoints": [
                {
                    "diagnostics": diagnostics,
                    "diagnostics_file": _relative_record(
                        diagnostics_path, composed_root
                    ),
                    "entities": _relative_record(current_entities, composed_root),
                    "frame_index": 9,
                    "snapshot": _relative_record(current_snapshot, composed_root),
                }
            ],
            "inputs": {
                "anchor_entities": _file_record(anchor_entities),
                "anchor_manifest": _file_record(anchor_manifest),
                "anchor_snapshot": _file_record(anchor_snapshot),
            },
            "scene": "apartment",
            "status": "PASS",
        },
    )
    return native_manifest, anchor_manifest, composed_manifest


def test_audit_map_quality_verifies_complete_manifest_chain(tmp_path: Path) -> None:
    native, anchor, composed = _audit_fixture(tmp_path)

    result = audit_map_quality(
        native_manifest_path=native,
        anchor_manifest_path=anchor,
        composed_manifest_path=composed,
    )

    assert result["schema_version"] == 1
    assert result["status"] == "PASS"
    assert result["scene"] == "apartment"
    assert result["evidence"]["producer_chain"]["label"] == "CODE_EVIDENCE"
    assert result["evidence"]["ply_visualization"]["label"] == "MEASURED_EVIDENCE"
    assert result["evidence"]["geometry_authority"]["label"] == "MEASURED_EVIDENCE"
    assert result["evidence"]["hypothesis"]["label"] == "HYPOTHESIS"
    assert (
        result["evidence"]["ply_visualization"]["result"]["visualization_mode"]
        == "instance_palette"
    )
    assert result["evidence"]["geometry_authority"]["result"]["moved"][
        "retained_point_ratio"
    ] == pytest.approx(0.1)
    assert "metrics" not in json.dumps(result, sort_keys=True).lower()


def test_cli_writes_deterministic_json_and_markdown(tmp_path: Path) -> None:
    native, anchor, composed = _audit_fixture(tmp_path)
    first_json = tmp_path / "outputs-a/audit.json"
    first_markdown = tmp_path / "outputs-a/audit.md"
    second_json = tmp_path / "outputs-b/audit.json"
    second_markdown = tmp_path / "outputs-b/audit.md"
    common = [
        "--native-manifest",
        str(native),
        "--anchor-manifest",
        str(anchor),
        "--composed-manifest",
        str(composed),
    ]

    assert (
        main(
            [
                *common,
                "--output-json",
                str(first_json),
                "--output-markdown",
                str(first_markdown),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                *common,
                "--output-json",
                str(second_json),
                "--output-markdown",
                str(second_markdown),
            ]
        )
        == 0
    )

    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_markdown.read_bytes() == second_markdown.read_bytes()
    payload = json.loads(first_json.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    report = first_markdown.read_text(encoding="utf-8")
    assert "# CROVE Map-Quality Audit" in report
    assert "instance_palette" in report
    assert "dense_anchor_to_sparse_temporal_switch_measured" in report


def test_cli_rejects_output_inside_input_artifact_root(tmp_path: Path) -> None:
    native, anchor, composed = _audit_fixture(tmp_path)

    with pytest.raises(ValueError, match="inside an input artifact root"):
        main(
            [
                "--native-manifest",
                str(native),
                "--anchor-manifest",
                str(anchor),
                "--composed-manifest",
                str(composed),
                "--output-json",
                str(composed.parent / "audit.json"),
                "--output-markdown",
                str(tmp_path / "outside/audit.md"),
            ]
        )


def test_audit_rejects_tampered_bound_native_extension(tmp_path: Path) -> None:
    native, anchor, composed = _audit_fixture(tmp_path)
    native_payload = json.loads(native.read_text(encoding="utf-8"))
    extension = Path(
        native_payload["preflight"]["sources"]["consistent_gsm_extension"]["path"]
    )
    extension.write_bytes(b"tampered-extension")

    with pytest.raises(ValueError, match="consistent_gsm extension.*mismatch"):
        audit_map_quality(
            native_manifest_path=native,
            anchor_manifest_path=anchor,
            composed_manifest_path=composed,
        )
