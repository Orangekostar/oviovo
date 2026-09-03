from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.evaluation.evaluate_panoptic_flat_current import main
from src.evaluation.datasets.panoptic_flat import (
    FlatDatasetError,
    load_flat_dataset,
)


def _write_png(path: Path, value: int = 0) -> None:
    Image.fromarray(np.full((2, 3, 3), value, dtype=np.uint8), mode="RGB").save(path)


def _write_run(root: Path, run_id: str, start: int) -> None:
    directory = root / run_id
    directory.mkdir(parents=True)
    with (directory / "timestamps.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("ImageID", "TimestampNs"))
        writer.writerow(("000001", start))
        writer.writerow(("000002", start + 1))
    for frame_id in ("000001", "000002"):
        prefix = directory / frame_id
        _write_png(Path(f"{prefix}_color.png"), 20)
        Image.fromarray(np.ones((2, 3), dtype=np.float32), mode="F").save(
            Path(f"{prefix}_depth.tiff")
        )
        _write_png(Path(f"{prefix}_segmentation.png"), 1)
        _write_png(Path(f"{prefix}_predicted.png"), 2)
        Path(f"{prefix}_labels.json").write_text(
            json.dumps(
                [
                    {
                        "id": 1,
                        "instance_id": 1,
                        "isthing": True,
                        "category_id": 2,
                        "score": 0.9,
                    }
                ]
            ),
            encoding="utf-8",
        )
        Path(f"{prefix}_pose.txt").write_text(
            " ".join(str(value) for value in np.eye(4).reshape(-1)) + "\n",
            encoding="utf-8",
        )


def flat_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "flat_dataset"
    root.mkdir(parents=True)
    _write_run(root, "run1", 1)
    _write_run(root, "run2", 3)
    (root / "labels.csv").write_text(
        "InstanceID,ClassID,PanopticID,Name\n1,2,1,chair\n",
        encoding="utf-8",
    )
    (root / "changes.csv").write_text(
        "object_id,change_type,change_frame\n1,moved,2\n",
        encoding="utf-8",
    )
    ground_truth = root / "ground_truth"
    (ground_truth / "run1").mkdir(parents=True)
    (ground_truth / "run2").mkdir(parents=True)
    (ground_truth / "run1" / "flat_1_gt_10000.ply").write_text(
        "ply\nformat ascii 1.0\nend_header\n", encoding="ascii"
    )
    (ground_truth / "run2" / "flat_2_gt_10000.ply").write_text(
        "ply\nformat ascii 1.0\nend_header\n", encoding="ascii"
    )
    return root


def test_flat_adapter_requires_run1_before_run2(tmp_path: Path) -> None:
    dataset = load_flat_dataset(flat_fixture(tmp_path), condition="Flat-GT-Panoptic")

    assert [frame.run_id for frame in dataset.frames] == [
        "run1",
        "run1",
        "run2",
        "run2",
    ]
    assert [frame.global_index for frame in dataset.frames] == [0, 1, 2, 3]
    assert dataset.condition.oracle is True
    assert dataset.source_bindings


def test_predicted_condition_requires_prediction_labels(tmp_path: Path) -> None:
    root = flat_fixture(tmp_path)
    (root / "run2" / "000002_labels.json").unlink()

    with pytest.raises(FlatDatasetError, match="required Flat member"):
        load_flat_dataset(root, condition="Flat-Predicted-Panoptic")


def test_predicted_condition_accepts_upstream_optional_label_fields(
    tmp_path: Path,
) -> None:
    root = flat_fixture(tmp_path)
    for path in root.glob("run?/*_labels.json"):
        path.write_text(
            json.dumps([{"id": 1, "isthing": True, "category_id": 2}]),
            encoding="utf-8",
        )

    dataset = load_flat_dataset(root, condition="Flat-Predicted-Panoptic")

    assert len(dataset.frames) == 4


def test_flat_change_log_rejects_unknown_objects(tmp_path: Path) -> None:
    root = flat_fixture(tmp_path)
    (root / "changes.csv").write_text(
        "object_id,change_type,change_frame\n99,moved,2\n", encoding="utf-8"
    )

    with pytest.raises(FlatDatasetError, match="unknown label"):
        load_flat_dataset(root, condition="Flat-GT-Panoptic")


def test_flat_adapter_rejects_invalid_prediction_label_values(tmp_path: Path) -> None:
    root = flat_fixture(tmp_path)
    (root / "run1" / "000001_labels.json").write_text(
        '[{"id":1,"isthing":"yes","category_id":2}]', encoding="utf-8"
    )

    with pytest.raises(FlatDatasetError, match="prediction label"):
        load_flat_dataset(root, condition="Flat-Predicted-Panoptic")


def test_flat_adapter_rejects_change_frame_outside_run2(tmp_path: Path) -> None:
    root = flat_fixture(tmp_path)
    (root / "changes.csv").write_text(
        "object_id,change_type,change_frame\n1,moved,1\n", encoding="utf-8"
    )

    with pytest.raises(FlatDatasetError, match="run2 frame"):
        load_flat_dataset(root, condition="Flat-GT-Panoptic")


def test_flat_adapter_rejects_non_float_depth_and_timestamp_regression(
    tmp_path: Path,
) -> None:
    root = flat_fixture(tmp_path)
    Image.fromarray(np.ones((2, 3), dtype=np.uint16)).save(
        root / "run1" / "000001_depth.tiff"
    )
    with pytest.raises(FlatDatasetError, match="32-bit floating"):
        load_flat_dataset(root, condition="Flat-GT-Panoptic")

    root = flat_fixture(tmp_path / "second")
    (root / "run2" / "timestamps.csv").write_text(
        "ImageID,TimestampNs\n000001,4\n000002,3\n", encoding="utf-8"
    )
    with pytest.raises(FlatDatasetError, match="strictly increasing"):
        load_flat_dataset(root, condition="Flat-GT-Panoptic")


def test_flat_inspection_cli_emits_relative_content_bindings(tmp_path: Path) -> None:
    root = flat_fixture(tmp_path)
    output = tmp_path / "inspection.json"

    assert main(
        (
            "inspect",
            "--dataset-root",
            str(root),
            "--condition",
            "Flat-GT-Panoptic",
            "--output",
            str(output),
        )
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["evidence_class"] == "ORACLE_DIAGNOSTIC"
    assert payload["run_frame_counts"] == {"run1": 2, "run2": 2}
    assert all(not row["relative_path"].startswith("/") for row in payload["source_bindings"])
