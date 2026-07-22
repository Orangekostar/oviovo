from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import pytest

from src.datasets.tesse_cd import TesseCdRgbdDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/evaluation/manifests/oviv2_tesse_cd_cache.json"
VOCABULARY_ROOT = REPO_ROOT / "configs/evaluation/vocabularies"
CHECKED_SCHEDULE_PATH = (
    REPO_ROOT / "configs/evaluation/manifests/tesse_cd_causal_schedule_v2.json"
)
OFFICIAL_RGBD_ROOT = Path(
    "/home/ww/oviovo_benchmark_assets/tesse_cd/derived/rgbd_v1"
)

EXPECTED = {
    "apartment": {
        "frame_count": 1745,
        "classes": [
            "Fridge",
            "Books",
            "Chair",
            "Vase",
            "Couch",
            "Drawer",
            "Objects",
            "Table",
            "Bin",
            "Humans",
        ],
        "object_semantic_ids": [1, 2, 5, 6, 7, 9, 10, 16, 18, 20],
        "label_space_sha256": "f7bacfb3c7bafc674bd3a34540e3f9b5d98b0ae11389f91768c70ecc65a79f0f",
    },
    "office": {
        "frame_count": 4346,
        "classes": [
            "Small office objects",
            "Large static wall furniture",
            "Large office objects",
            "Bathroom",
            "Bedroom",
            "Chairs",
            "Signs",
        ],
        "object_semantic_ids": [2, 3, 4, 8, 9, 11, 15],
        "label_space_sha256": "91a7b359ee678dd67871653959a50c477ba7f371473b9c2bf1b4f41f63691765",
    },
}
OFFICIAL_TIMELINES = {
    "apartment": (1745, 4_204_107_999, 91_404_110_000),
    "office": (4346, 7_243_018_000, 224_492_999_999),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_values(value: Any, key: str = "") -> Iterator[str]:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            yield from _path_values(child_value, str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _path_values(child, key)
    elif isinstance(value, str) and (key.endswith("path") or key.endswith("root")):
        yield value


def test_cache_config_freezes_input_only_scene_contracts() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["schema_version"] == 1
    assert config["manifest_id"] == "oviv2_tesse_cd_cache_v1"
    assert config["dataset"] == "TESSE-CD"
    assert config["stage3_lineage_commit"] == (
        "47962fbd9f363c0696cc5016f8ab42f83a3bf7e5"
    )
    camera = config["camera"]
    assert {key: value for key, value in camera.items() if key != "sha256"} == {
        "path": "/home/ww/oviovo_benchmark_assets/tesse_cd/derived/rgbd_v1/cam_params.json",
        "width": 720,
        "height": 480,
        "fx": 415.69219381653056,
        "fy": 415.69219381653056,
        "cx": 360.0,
        "cy": 240.0,
        "depth_scale": 1000.0,
    }
    assert len(camera["sha256"]) == 64

    for scene, expected in EXPECTED.items():
        record = config["scenes"][scene]
        assert record["frame_count"] == expected["frame_count"]
        assert record["source_frame_ids"] == {
            "start": 0,
            "stop_exclusive": expected["frame_count"],
            "stride": 1,
        }
        assert record["image_shape"] == [480, 720]
        assert record["source_depth_dtype"] == "uint16"
        assert record["depth_unit"] == "millimeter"
        assert record["pose_convention"] == "camera_to_world"
        assert record["export_manifest"]["file_hash_count"] == (
            2 * expected["frame_count"] + 3
        )


def test_checked_protocol_hashes_match_cache_declarations() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    for role in ("source_manifest", "schedule_manifest"):
        binding = config[role]
        assert binding["sha256"] == _sha256(REPO_ROOT / binding["path"])


@pytest.mark.skipif(
    os.environ.get("TESSE_CD_RUN_FORMAL_ASSET_TESTS") != "1",
    reason="set TESSE_CD_RUN_FORMAL_ASSET_TESTS=1 to validate formal assets",
)
@pytest.mark.parametrize("scene", ["apartment", "office"])
def test_official_scene_constructs_with_checked_input_hashes(scene: str) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    scene_config = config["scenes"][scene]
    frame_count, first_timestamp_ns, last_timestamp_ns = OFFICIAL_TIMELINES[scene]
    export_path = Path(scene_config["export_manifest"]["path"])

    assert _sha256(export_path) == scene_config["export_manifest"]["sha256"]
    assert _sha256(CHECKED_SCHEDULE_PATH) == config["schedule_manifest"]["sha256"]
    assert _sha256(Path(config["camera"]["path"])) == config["camera"]["sha256"]
    assert _sha256(Path(scene_config["timestamps"]["path"])) == (
        scene_config["timestamps"]["sha256"]
    )
    assert _sha256(Path(scene_config["trajectory"]["path"])) == (
        scene_config["trajectory"]["sha256"]
    )
    export = json.loads(export_path.read_text(encoding="utf-8"))
    assert export["combined_output_sha256"] == (
        scene_config["export_manifest"]["combined_output_sha256"]
    )
    assert export["file_hash_count"] == (
        scene_config["export_manifest"]["file_hash_count"]
    )

    dataset = TesseCdRgbdDataset(
        OFFICIAL_RGBD_ROOT,
        scene,
        export_path,
        CHECKED_SCHEDULE_PATH,
    )

    assert len(dataset) == frame_count
    assert dataset.timestamp_ns(0) == first_timestamp_ns
    assert dataset.timestamp_ns(-1) == last_timestamp_ns


def test_scene_vocabularies_match_ordered_json_txt_and_hash_bindings() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    for scene, expected in EXPECTED.items():
        json_path = VOCABULARY_ROOT / f"tesse_cd_{scene}.json"
        txt_path = VOCABULARY_ROOT / f"tesse_cd_{scene}.txt"
        vocabulary = json.loads(json_path.read_text(encoding="utf-8"))
        txt_classes = txt_path.read_text(encoding="utf-8").splitlines()

        assert vocabulary["scene"] == scene
        assert vocabulary["classes"] == expected["classes"]
        assert vocabulary["object_semantic_ids"] == expected["object_semantic_ids"]
        assert txt_classes == expected["classes"]
        assert vocabulary["label_space"]["sha256"] == expected["label_space_sha256"]
        alias_path = REPO_ROOT / vocabulary["alias_map"]["path"]
        assert vocabulary["alias_map"]["sha256"] == _sha256(alias_path)

        binding = config["scenes"][scene]["vocabulary"]
        assert binding["json_path"] == (
            f"configs/evaluation/vocabularies/tesse_cd_{scene}.json"
        )
        assert binding["txt_path"] == (
            f"configs/evaluation/vocabularies/tesse_cd_{scene}.txt"
        )
        assert binding["json_sha256"] == _sha256(json_path)
        assert binding["txt_sha256"] == _sha256(txt_path)


def test_cache_config_contains_no_evaluator_or_method_outputs() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    forbidden = {"gt", "ground_truth", "target", "targets", "prediction", "predictions"}
    for raw_path in _path_values(config):
        path_parts = {part.lower() for part in Path(raw_path).parts}
        assert path_parts.isdisjoint(forbidden), raw_path

    serialized = json.dumps(config, sort_keys=True).lower()
    assert "route3" not in serialized
    assert "scannet200" not in serialized
    assert "stage4" not in serialized
