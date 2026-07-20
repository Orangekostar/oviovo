from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation.finalize_oviv2_scannet200_result import _token_bindings
from src.evaluation.oviv2_scannet200_result import (
    EVALUATION_FILES,
    SCANNET200_5_SCENES,
    aggregate_oviv2_scannet200,
    validate_scene_run_manifests,
    verify_byte_identical_evaluation_dirs,
)


def _metrics(value: float) -> dict[str, float]:
    return {
        "miou": value,
        "macc": value + 0.01,
        "f_miou": value + 0.02,
        "ap25": value + 0.03,
        "ap50": value + 0.04,
        "f5": value + 0.05,
    }


def test_scannet200_result_aggregates_exact_five_scene_macro() -> None:
    scene_metrics = {
        scene: _metrics(index / 10.0)
        for index, scene in enumerate(SCANNET200_5_SCENES)
    }

    result = aggregate_oviv2_scannet200(scene_metrics)

    aggregate = result["scannet200_5_heldout"]
    assert aggregate["scene_ids"] == list(SCANNET200_5_SCENES)
    assert aggregate["scene_count"] == 5
    assert aggregate["semantic"]["miou"] == pytest.approx(0.2)
    assert aggregate["semantic"]["macc"] == pytest.approx(0.21)
    assert aggregate["instance"]["ap25"] == pytest.approx(0.23)
    assert aggregate["instance"]["ap50"] == pytest.approx(0.24)
    assert aggregate["geometry"]["f5"] == pytest.approx(0.25)


def test_scannet200_result_rejects_incomplete_scene_set() -> None:
    scene_metrics = {
        scene: _metrics(0.1) for scene in SCANNET200_5_SCENES[:-1]
    }

    with pytest.raises(ValueError, match="exactly match"):
        aggregate_oviv2_scannet200(scene_metrics)


def _evaluation(directory: Path) -> None:
    directory.mkdir(parents=True)
    for name in EVALUATION_FILES:
        if name.endswith(".json"):
            (directory / name).write_text(json.dumps({"value": name}), encoding="utf-8")
        else:
            (directory / name).write_bytes(name.encode("ascii"))


def test_scannet200_repeat_evaluation_requires_exact_byte_identity(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _evaluation(first)
    _evaluation(second)

    hashes = verify_byte_identical_evaluation_dirs(first, second)

    assert set(hashes) == set(EVALUATION_FILES)
    (second / "metrics.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="not byte-identical"):
        verify_byte_identical_evaluation_dirs(first, second)


def test_scannet200_run_manifests_bind_variable_revisions_and_provenance() -> None:
    frame_counts = {
        scene: count
        for scene, count in zip(
            SCANNET200_5_SCENES,
            (238, 465, 444, 190, 147),
            strict=True,
        )
    }
    runs = {
        scene: {
            "method": "OVIV2",
            "dataset_name": "ScanNet200",
            "scene": scene,
            "final_revision": count,
            "frame_selection": {"sampled_frame_count": count},
            "algorithm_hash": "a" * 64,
            "frontend_algorithm_hash": "b" * 64,
            "vocabulary_hash": "c" * 64,
            "benchmark_manifest_hash": "d" * 64,
        }
        for scene, count in frame_counts.items()
    }

    contract = validate_scene_run_manifests(runs, frame_counts)

    assert contract["algorithm_hash"] == "a" * 64
    runs[SCANNET200_5_SCENES[-1]]["final_revision"] -= 1
    with pytest.raises(ValueError, match="revision"):
        validate_scene_run_manifests(runs, frame_counts)


def test_scannet200_finalizer_binds_exact_four_table_tokens() -> None:
    bindings = _token_bindings()

    assert [value["token"] for value in bindings] == [
        "T1_OVIV2_SCANNET5_MIOU",
        "T1_OVIV2_SCANNET5_AP25",
        "T1_OVIV2_SCANNET5_AP50",
        "T1_OVIV2_SCANNET5_F5",
    ]
    assert all(
        value["json_pointer"].startswith("/metrics/scannet200_5_heldout/")
        for value in bindings
    )
