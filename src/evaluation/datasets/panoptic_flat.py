"""Strict adapter for the two-run Panoptic Mapping Flat dataset."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from PIL import Image

from src.evaluation.json_contracts import loads_strict

RUN_IDS = ("run1", "run2")
CHANGE_TYPES = ("moved", "added", "removed")
_FRAME_ID = re.compile(r"[A-Za-z0-9_.-]+")


class FlatDatasetError(ValueError):
    """Raised when Flat inputs are incomplete or ambiguous."""


@dataclass(frozen=True, slots=True)
class FlatInputCondition:
    condition_id: str
    oracle: bool
    panoptic_suffix: str | None
    labels_suffix: str | None


CONDITIONS = {
    "Flat-GT-Panoptic": FlatInputCondition(
        "Flat-GT-Panoptic", True, "_segmentation.png", None
    ),
    "Flat-Predicted-Panoptic": FlatInputCondition(
        "Flat-Predicted-Panoptic", False, "_predicted.png", "_labels.json"
    ),
    "CROVE-Open-Vocabulary": FlatInputCondition(
        "CROVE-Open-Vocabulary", False, None, None
    ),
}


@dataclass(frozen=True, slots=True)
class FlatSourceBinding:
    relative_path: str
    sha256: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class FlatLabel:
    instance_id: int
    class_id: int
    panoptic_id: int
    name: str


@dataclass(frozen=True, slots=True)
class FlatChange:
    object_id: int
    change_type: str
    change_frame: int


@dataclass(frozen=True, slots=True)
class FlatFrame:
    run_id: str
    run_index: int
    frame_id: str
    global_index: int
    timestamp_ns: int
    color_path: Path
    depth_path: Path
    panoptic_path: Path | None
    prediction_labels_path: Path | None
    pose: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FlatDataset:
    condition: FlatInputCondition
    frames: tuple[FlatFrame, ...]
    labels: tuple[FlatLabel, ...]
    changes: tuple[FlatChange, ...]
    structural_ground_truth: tuple[Path, Path]
    source_bindings: tuple[FlatSourceBinding, ...]


def input_condition(condition: str) -> FlatInputCondition:
    try:
        return CONDITIONS[condition]
    except KeyError as exc:
        raise FlatDatasetError(f"unknown Flat input condition: {condition}") from exc


def _required(root: Path, relative: str) -> Path:
    path = root / relative
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FlatDatasetError(f"required Flat member is missing: {relative}") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file() or path.is_symlink():
        raise FlatDatasetError(f"required Flat member is invalid: {relative}")
    return resolved


def _binding(root: Path, path: Path) -> FlatSourceBinding:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return FlatSourceBinding(
        path.relative_to(root).as_posix(), digest.hexdigest(), path.stat().st_size
    )


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise FlatDatasetError(f"{label} must be an integer")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise FlatDatasetError(f"{label} must be an integer") from exc
    if str(parsed) != str(value) or parsed < minimum:
        raise FlatDatasetError(f"{label} must be an integer >= {minimum}")
    return parsed


def _load_labels(path: Path) -> tuple[FlatLabel, ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"InstanceID", "ClassID", "PanopticID", "Name"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise FlatDatasetError("Flat labels.csv has an invalid header")
            labels = tuple(
                FlatLabel(
                    _integer(row["InstanceID"], label="InstanceID"),
                    _integer(row["ClassID"], label="ClassID"),
                    _integer(row["PanopticID"], label="PanopticID"),
                    row["Name"],
                )
                for row in reader
            )
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise FlatDatasetError("Flat labels.csv is unreadable") from exc
    if not labels or any(not label.name for label in labels):
        raise FlatDatasetError("Flat labels.csv contains an invalid label")
    ids = [label.instance_id for label in labels]
    if len(ids) != len(set(ids)):
        raise FlatDatasetError("Flat labels.csv contains duplicate instance IDs")
    return labels


def _load_changes(path: Path) -> tuple[FlatChange, ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            expected = {"object_id", "change_type", "change_frame"}
            if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
                raise FlatDatasetError("Flat changes.csv has an invalid header")
            changes = tuple(
                FlatChange(
                    _integer(row["object_id"], label="change object_id"),
                    row["change_type"],
                    _integer(row["change_frame"], label="change_frame"),
                )
                for row in reader
            )
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise FlatDatasetError("Flat changes.csv is unreadable") from exc
    if any(change.change_type not in CHANGE_TYPES for change in changes):
        raise FlatDatasetError("Flat changes.csv contains an invalid change type")
    keys = [(change.object_id, change.change_type, change.change_frame) for change in changes]
    if len(keys) != len(set(keys)):
        raise FlatDatasetError("Flat changes.csv contains duplicate changes")
    return changes


def _timestamp_rows(path: Path) -> tuple[tuple[str, int], ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise FlatDatasetError("Flat timestamps.csv is unreadable") from exc
    if not rows or len(rows[0]) != 2 or rows[0][0] != "ImageID":
        raise FlatDatasetError("Flat timestamps.csv has an invalid header")
    if rows[0][1] not in {"Time", "Timestamp", "TimestampNs"}:
        raise FlatDatasetError("Flat timestamps.csv has an invalid timestamp field")
    parsed: list[tuple[str, int]] = []
    for row in rows[1:]:
        if len(row) != 2 or not _FRAME_ID.fullmatch(row[0]):
            raise FlatDatasetError("Flat timestamps.csv contains an invalid frame ID")
        parsed.append((row[0], _integer(row[1], label="timestamp")))
    if not parsed or len({frame_id for frame_id, _ in parsed}) != len(parsed):
        raise FlatDatasetError("Flat timestamps.csv is empty or duplicated")
    timestamps = [timestamp for _, timestamp in parsed]
    if any(right <= left for left, right in pairwise(timestamps)):
        raise FlatDatasetError("Flat timestamps must be strictly increasing within each run")
    return tuple(parsed)


def _pose(path: Path) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in path.read_text(encoding="utf-8").split())
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise FlatDatasetError("Flat pose is unreadable") from exc
    if len(values) != 16 or not all(math.isfinite(value) for value in values):
        raise FlatDatasetError("Flat pose must be a finite 4x4 matrix")
    return values


def _validate_images(
    color: Path, depth: Path, panoptic: Path | None, labels: Path | None
) -> None:
    try:
        with Image.open(color) as image:
            image.verify()
        with Image.open(color) as image:
            color_size = image.size
            if image.format != "PNG" or image.mode not in {"RGB", "RGBA"}:
                raise FlatDatasetError("Flat color image must be RGB PNG")
        with Image.open(depth) as image:
            image.verify()
        with Image.open(depth) as image:
            if image.format != "TIFF" or image.mode != "F":
                raise FlatDatasetError("Flat depth image must be 32-bit floating TIFF")
            if image.size != color_size:
                raise FlatDatasetError("Flat color and depth dimensions differ")
        if panoptic is not None:
            with Image.open(panoptic) as image:
                image.verify()
            with Image.open(panoptic) as image:
                if image.format != "PNG" or image.size != color_size:
                    raise FlatDatasetError("Flat panoptic image is invalid or unpaired")
        if labels is not None:
            value = loads_strict(
                labels.read_text(encoding="utf-8"), label=labels.name
            )
            if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
                raise FlatDatasetError("Flat prediction labels must contain a JSON list")
            required = {"id", "isthing", "category_id"}
            allowed = required | {"instance_id", "score"}
            if any(
                not required.issubset(row)
                or not set(row).issubset(allowed)
                or type(row["id"]) is not int
                or row["id"] < 0
                or type(row["category_id"]) is not int
                or row["category_id"] < 0
                or type(row["isthing"]) is not bool
                or (
                    "instance_id" in row
                    and (type(row["instance_id"]) is not int or row["instance_id"] < 0)
                )
                or (
                    "score" in row
                    and (
                        isinstance(row["score"], bool)
                        or not isinstance(row["score"], (int, float))
                        or not math.isfinite(row["score"])
                        or not 0 <= row["score"] <= 1
                    )
                )
                for row in value
            ):
                raise FlatDatasetError("Flat prediction label schema is invalid")
    except FlatDatasetError:
        raise
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError) as exc:
        raise FlatDatasetError("Flat image or prediction label is unreadable") from exc


def load_flat_dataset(root: str | Path, *, condition: str) -> FlatDataset:
    """Load and content-bind both Flat visits in fixed run1-then-run2 order."""

    dataset_root = Path(root).resolve()
    if not dataset_root.is_dir():
        raise FlatDatasetError("Flat dataset root is missing")
    selected_condition = input_condition(condition)
    labels_path = _required(dataset_root, "labels.csv")
    changes_path = _required(dataset_root, "changes.csv")
    structural = (
        _required(dataset_root, "ground_truth/run1/flat_1_gt_10000.ply"),
        _required(dataset_root, "ground_truth/run2/flat_2_gt_10000.ply"),
    )
    labels = _load_labels(labels_path)
    changes = _load_changes(changes_path)
    label_ids = {label.instance_id for label in labels}
    if any(change.object_id not in label_ids for change in changes):
        raise FlatDatasetError("Flat change log references an unknown label")
    paths: list[Path] = [labels_path, changes_path, *structural]
    frames: list[FlatFrame] = []
    for run_index, run_id in enumerate(RUN_IDS):
        timestamp_path = _required(dataset_root, f"{run_id}/timestamps.csv")
        paths.append(timestamp_path)
        for frame_id, timestamp_ns in _timestamp_rows(timestamp_path):
            prefix = f"{run_id}/{frame_id}"
            color = _required(dataset_root, f"{prefix}_color.png")
            depth = _required(dataset_root, f"{prefix}_depth.tiff")
            panoptic = (
                _required(dataset_root, f"{prefix}{selected_condition.panoptic_suffix}")
                if selected_condition.panoptic_suffix is not None
                else None
            )
            prediction_labels = (
                _required(dataset_root, f"{prefix}{selected_condition.labels_suffix}")
                if selected_condition.labels_suffix is not None
                else None
            )
            pose_path = _required(dataset_root, f"{prefix}_pose.txt")
            _validate_images(color, depth, panoptic, prediction_labels)
            paths.extend(
                path
                for path in (color, depth, panoptic, prediction_labels, pose_path)
                if path is not None
            )
            frames.append(
                FlatFrame(
                    run_id,
                    run_index,
                    frame_id,
                    len(frames),
                    timestamp_ns,
                    color,
                    depth,
                    panoptic,
                    prediction_labels,
                    _pose(pose_path),
                )
            )
    run2_start = next(frame.global_index for frame in frames if frame.run_id == "run2")
    if any(
        change.change_frame < run2_start or change.change_frame >= len(frames)
        for change in changes
    ):
        raise FlatDatasetError("Flat change_frame must identify a run2 frame")
    bindings = tuple(
        _binding(dataset_root, path)
        for path in sorted(set(paths), key=lambda item: item.relative_to(dataset_root).as_posix())
    )
    return FlatDataset(
        selected_condition,
        tuple(frames),
        labels,
        changes,
        structural,
        bindings,
    )
