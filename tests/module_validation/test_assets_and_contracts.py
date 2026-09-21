"""Contracts for the module-validation study bootstrap."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

SPEC_PATH = (
    Path(__file__).parents[2]
    / "docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json"
)


def _contracts():
    try:
        return importlib.import_module("src.static_ovmap.module_validation.contracts")
    except ModuleNotFoundError:
        pytest.fail("module-validation contracts are not implemented")


def _assets():
    try:
        return importlib.import_module("src.static_ovmap.module_validation.assets")
    except ModuleNotFoundError:
        pytest.fail("module-validation asset binding is not implemented")


def test_study_spec_loads_authoritative_metadata() -> None:
    contracts = _contracts()

    spec = contracts.StudySpec.load(SPEC_PATH)

    assert spec.specification_only is True
    assert spec.phases[-1] == "all"
    assert spec.output_root == Path("/mnt/shared/ww/ovimap-module-validation-v1")
    assert spec.project_commit == "b6455520c758a3413988e0827b3e1f34667bdfd1"
    assert len(spec.digest) == 64

    with pytest.raises(TypeError):
        spec.values["data"]["fit"] = 7


def test_study_spec_rejects_changed_contract(tmp_path: Path) -> None:
    contracts = _contracts()
    raw = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    raw["phases"] = ["bind", "all"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="phases"):
        contracts.StudySpec.load(path)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_phase_receipt_rejects_nonfinite_metrics(bad_value: float) -> None:
    contracts = _contracts()

    with pytest.raises(ValueError, match="finite"):
        contracts.PhaseReceipt(
            phase="bind",
            status=contracts.ReceiptStatus.COMPLETE,
            cache_key="a" * 64,
            metrics={"x": bad_value},
        )


def test_phase_receipt_writes_strict_json_atomically(tmp_path: Path) -> None:
    contracts = _contracts()
    receipt = contracts.PhaseReceipt(
        phase="bind",
        status=contracts.ReceiptStatus.COMPLETE,
        cache_key="b" * 64,
        outputs={"resolved_config": "resolved_config.json"},
        metrics={"asset_count": 3, "undefined": None},
    )
    target = tmp_path / "receipts" / "bind.json"

    receipt.write(target)

    assert json.loads(target.read_text(encoding="utf-8")) == receipt.to_dict()
    assert not list(target.parent.glob("*.tmp"))
    with pytest.raises(TypeError):
        receipt.metrics["asset_count"] = 4


def test_asset_resolution_precedence_env_config_then_receipt(tmp_path: Path) -> None:
    assets = _assets()
    spec = _contracts().StudySpec.load(SPEC_PATH)
    explicit_root = tmp_path / "explicit"
    explicit = explicit_root / "weights/model.bin"
    configured = tmp_path / "configured/model.bin"
    received = tmp_path / "received/model.bin"
    for path, payload in ((explicit, b"env"), (configured, b"config"), (received, b"receipt")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"model_path": str(received)}), encoding="utf-8")
    requirement = assets.AssetRequirement(
        key="test_model",
        kind="file",
        relative_paths=("weights/model.bin",),
        configured_paths=(configured,),
    )

    resolved = assets.resolve_assets(
        spec,
        {"OVIMAP_MODEL_ROOTS": str(explicit_root)},
        requirements=(requirement,),
        receipt_paths=(receipt,),
    )

    assert resolved.status.value == "COMPLETE"
    assert resolved.bindings["test_model"].path == explicit.resolve()
    assert resolved.bindings["test_model"].source == "environment"
    assert resolved.bindings["test_model"].identity.sha256 == assets.sha256_file(explicit)


def test_asset_resolution_uses_config_before_receipt(tmp_path: Path) -> None:
    assets = _assets()
    spec = _contracts().StudySpec.load(SPEC_PATH)
    configured = tmp_path / "configured.bin"
    received = tmp_path / "received.bin"
    configured.write_bytes(b"configured")
    received.write_bytes(b"receipt")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"candidate": str(received)}), encoding="utf-8")

    resolved = assets.resolve_assets(
        spec,
        {},
        requirements=(
            assets.AssetRequirement(
                key="test_data",
                kind="file",
                relative_paths=("received.bin",),
                configured_paths=(configured,),
            ),
        ),
        receipt_paths=(receipt,),
    )

    assert resolved.bindings["test_data"].path == configured.resolve()
    assert resolved.bindings["test_data"].source == "configuration"


def test_asset_resolution_blocks_differing_explicit_candidates(tmp_path: Path) -> None:
    assets = _assets()
    spec = _contracts().StudySpec.load(SPEC_PATH)
    roots = [tmp_path / "root_a", tmp_path / "root_b"]
    for root, payload in zip(roots, (b"a", b"b"), strict=True):
        root.mkdir()
        (root / "asset.bin").write_bytes(payload)

    resolved = assets.resolve_assets(
        spec,
        {"OVIMAP_DATA_ROOTS": "\n".join(map(str, roots))},
        requirements=(
            assets.AssetRequirement(
                key="ambiguous",
                kind="file",
                relative_paths=("asset.bin",),
            ),
        ),
    )

    assert resolved.status.value == "BLOCKED_ASSET_IDENTITY"
    assert "ambiguous" not in resolved.bindings
    assert len(resolved.ambiguities["ambiguous"]) == 2


def test_environment_discovery_does_not_follow_path_outside_root(tmp_path: Path) -> None:
    assets = _assets()
    spec = _contracts().StudySpec.load(SPEC_PATH)
    root = tmp_path / "authorized"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escaped").symlink_to(outside, target_is_directory=True)

    resolved = assets.resolve_assets(
        spec,
        {"OVIMAP_DATA_ROOTS": str(root)},
        requirements=(
            assets.AssetRequirement(
                key="escaped",
                kind="directory",
                relative_paths=("escaped",),
            ),
        ),
    )

    assert resolved.status.value == "BLOCKED_PREREQUISITE"
    assert resolved.missing == ("escaped",)


def _scene(scene_id: str, source_split: str, *, complete: bool = True) -> dict:
    return {
        "scene_id": scene_id,
        "source_split": source_split,
        "complete": complete,
    }


def test_scene_splits_use_physical_families_and_literal_hash_order() -> None:
    assets = _assets()
    spec = _contracts().StudySpec.load(SPEC_PATH)
    inventory = [_scene(f"scene{index:04d}_00", "train") for index in range(12)]
    inventory.append(_scene("scene0004_01", "train"))
    inventory.extend(_scene(f"scene{index:04d}_00", "validation") for index in range(100, 105))

    splits = assets.build_scene_splits(inventory, exclusions=(), spec=spec)

    assert splits.status.value == "COMPLETE"
    assert splits.fit == (
        "scene0000_00",
        "scene0002_00",
        "scene0004_00",
        "scene0011_00",
        "scene0006_00",
        "scene0008_00",
        "scene0001_00",
        "scene0010_00",
    )
    assert splits.cal == ("scene0005_00", "scene0007_00")
    assert splits.select == ("scene0009_00", "scene0003_00")
    assert splits.confirm == ("scene0100_00", "scene0103_00")
    assert "scene0004_01" not in splits.all_selected


def test_scene_splits_do_not_shrink_or_reuse_exposed_families() -> None:
    assets = _assets()
    spec = _contracts().StudySpec.load(SPEC_PATH)
    inventory = [_scene(f"scene{index:04d}_00", "train") for index in range(12)]
    inventory.extend(_scene(f"scene{index:04d}_00", "validation") for index in range(100, 102))

    splits = assets.build_scene_splits(
        inventory,
        exclusions=("scene0002", "scene0100"),
        spec=spec,
    )

    assert splits.status.value == "BLOCKED_INDEPENDENT_SCENES"
    assert splits.fit == ()
    assert splits.cal == ()
    assert splits.select == ()
    assert splits.confirm == ()
    assert splits.eligible_development_count == 11
    assert splits.eligible_confirmation_count == 1


def test_scannet_inventory_requires_complete_modalities_and_200_frames(tmp_path: Path) -> None:
    assets = _assets()
    scans = tmp_path / "scans"
    scans.mkdir()
    (tmp_path / "scannetv2_train.txt").write_text("scene0000_00\n", encoding="utf-8")
    (tmp_path / "scannetv2_val.txt").write_text("scene0100_00\n", encoding="utf-8")
    suffixes = (
        ".sens",
        ".txt",
        "_vh_clean_2.ply",
        "_vh_clean_2.labels.ply",
        "_vh_clean_2.0.010000.segs.json",
        ".aggregation.json",
    )
    for scene_id, frame_count in (("scene0000_00", 200), ("scene0100_00", 199)):
        scene_root = scans / scene_id
        scene_root.mkdir()
        for suffix in suffixes:
            payload = f"numDepthFrames = {frame_count}\n" if suffix == ".txt" else "x"
            (scene_root / f"{scene_id}{suffix}").write_text(payload, encoding="utf-8")

    inventory = assets.build_scannet_inventory(tmp_path)

    assert inventory["status"] == "COMPLETE"
    assert inventory["scenes"][0]["complete"] is True
    assert inventory["scenes"][1]["complete"] is False
    assert inventory["scenes"][1]["missing"] == ["metadata:numDepthFrames>=200"]
