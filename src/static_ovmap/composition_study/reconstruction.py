"""Reconstruct an exact released prediction from compact labels and bound N0."""

import argparse
from pathlib import Path

from src.static_ovmap.module_validation.scannet_study import (
    load_prediction,
    relabel_prediction,
    save_prediction,
)

from .io import SourceIndex, read_json
from .object_evidence import owner_labels


def reconstruct(label_path, output):
    row = read_json(label_path)
    entry = row["native_manifest"]
    SourceIndex().identity(entry["path"], entry)
    native = load_prediction(entry["path"])
    labels = {int(owner): int(label) for owner, label in row["labels"].items()}
    if set(labels) != set(owner_labels(native)):
        raise ValueError("released compact labels omit native owners")
    payload = (
        native
        if row["method_id"] == "N0"
        else relabel_prediction(
            native,
            row["method_id"],
            row["branch"],
            labels,
            row["logical_cost"],
            row["metadata"],
        )
    )
    if (
        owner_labels(payload) != labels
        or payload.prediction_key != row["prediction_key"]
        or payload.record_key != row["record_key"]
    ):
        raise ValueError("compact labels do not reconstruct the locked prediction")
    return save_prediction(payload, Path(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(reconstruct(args.labels, args.output))


if __name__ == "__main__":
    main()
