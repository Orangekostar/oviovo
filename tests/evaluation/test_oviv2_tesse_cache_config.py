from __future__ import annotations

import hashlib
import json
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
ALIAS_MAP_SHA256 = "217b3bcd730a6485159e23e56ee9e97ac3c1c56007a016cdc8a20e9fc07abdbc"
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
    assert config["camera"] == {
        "path": "/home/ww/oviovo_benchmark_assets/tesse_cd/derived/rgbd_v1/cam_params.json",
        "sha256": "73fe7357e25e4708bc6554bb2a0fc925de00e8368d06a18f9eb3eadd9cb808eb",
        "width": 720,
        "height": 480,
        "fx": 415.69219381653056,
        "fy": 415.69219381653056,
        "cx": 360.0,
        "cy": 240.0,
        "depth_scale": 1000.0,
    }
    assert config["source_manifest"]["sha256"] == (
        "be63826267109a67fe4109c1419eb02dbad99a9e3856d44ac8f848e32e00b907"
    )
    assert config["schedule_manifest"]["sha256"] == (
        "fb97bacee377f9fd67ee9dae8064dc6f33ac32d129ee633629fec4164d5003e0"
    )

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


def test_checked_schedule_hash_matches_adapter_and_cache_declarations() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    checked_sha256 = _sha256(CHECKED_SCHEDULE_PATH)

    assert TesseCdRgbdDataset.SCHEDULE_SHA256 == checked_sha256
    assert config["schedule_manifest"]["sha256"] == checked_sha256


@pytest.mark.parametrize("scene", ["apartment", "office"])
def test_official_scene_constructs_with_checked_input_hashes(scene: str) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    scene_config = config["scenes"][scene]
    frame_count, first_timestamp_ns, last_timestamp_ns = OFFICIAL_TIMELINES[scene]
    export_path = Path(scene_config["export_manifest"]["path"])

    assert _sha256(export_path) == TesseCdRgbdDataset.EXPORT_SHA256[scene]
    assert _sha256(CHECKED_SCHEDULE_PATH) == TesseCdRgbdDataset.SCHEDULE_SHA256

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
        assert vocabulary["alias_map"]["sha256"] == ALIAS_MAP_SHA256

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
