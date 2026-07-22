from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/evaluation/manifests/tesse_cd.json"


def test_tesse_cd_manifest_freezes_official_assets_and_causal_cutoffs() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert payload["manifest_id"] == "tesse_cd_dynamic_v1"
    assert payload["dataset"] == "TESSE-CD"
    assert payload["source"]["owner"] == "MIT-SPARK Khronos official release"
    assert set(payload["sequences"]) == {"apartment", "office"}

    expected = {
        "apartment": {
            "archive_sha256": "6a6bae7d6baec526349127713c817b1872f63b3b68ec99b3728adeacbca0c3d0",
            "database_sha256": "43a30e694b8deabf4861b2b20a8f32a5c18c6288f37a886289783af0f26dab07",
            "frame_count": 1745,
            "first_change_ns": 13_142_347_000,
            "frozen_stop_exclusive": 263,
        },
        "office": {
            "archive_sha256": "c89d4b9f008bb3bf707e293563391531a3c7087f4c049b4a2b40535d2461faa0",
            "database_sha256": "4305b56149378799bfa468da2ac0acd84716bef59aba004f8e4feb34ec0d3b51",
            "frame_count": 4346,
            "first_change_ns": 100_000_000_000,
            "frozen_stop_exclusive": 2001,
        },
    }
    for name, values in expected.items():
        sequence = payload["sequences"][name]
        assert sequence["archive"]["sha256"] == values["archive_sha256"]
        assert sequence["bag"]["database"]["sha256"] == values["database_sha256"]
        assert sequence["timeline"]["depth_frame_count"] == values["frame_count"]
        assert sequence["timeline"]["first_change_relative_ns"] == values["first_change_ns"]
        assert (
            sequence["frozen_prechange_frames"]["stop_exclusive"]
            == values["frozen_stop_exclusive"]
        )
        assert sequence["frozen_prechange_frames"]["updates_after_freeze"] == 0


def test_tesse_cd_manifest_keeps_ground_truth_out_of_runtime() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    protocol = payload["protocol"]

    assert protocol["runtime_ground_truth_access"] is False
    assert protocol["future_frames_allowed"] is False
    assert protocol["ground_truth_evaluator_only"] is True
    assert protocol["oracle_exception"] == {
        "method": "KHRONOS_ORACLE",
        "allowed_topic": "/tesse/seg_cam/converted/image_raw",
        "eligible_for_ranking": False,
    }
    for sequence in payload["sequences"].values():
        assert sequence["ground_truth"]["runtime_access_allowed"] is False
        assert set(sequence["ground_truth"]["files"]) == {
            "background_mesh",
            "changes",
            "dsg",
            "dsg_with_mesh",
        }
