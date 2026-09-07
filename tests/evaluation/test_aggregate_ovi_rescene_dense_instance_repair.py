from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.evaluation.aggregate_ovi_rescene_dense_instance_repair import (
    DenseResultAggregationError,
    aggregate_dense_results,
)

_TABLES = (
    "endpoint_diagnosis.csv",
    "instance_metrics.csv",
    "identity_metrics.csv",
    "runs.csv",
)


def _source(root: Path, pair_id: str, value: int) -> None:
    root.mkdir(parents=True)
    for name in _TABLES:
        pair_column = "pair_id" if name == "endpoint_diagnosis.csv" else "pair"
        (root / name).write_text(
            f"{pair_column},method,value\n{pair_id},P2,{value}\n", encoding="utf-8"
        )
    association = root / "fixed_p2_association"
    association.mkdir()
    (association / "association_metrics.csv").write_text(
        f"pair,method,value\n{pair_id},R_obj,{value}\n", encoding="utf-8"
    )


def test_aggregates_pair_rows_without_overwriting_sources(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "aggregate"
    _source(first, "pair-a", 1)
    _source(second, "pair-b", 2)

    manifest = aggregate_dense_results(
        (("pair-a", first), ("pair-b", second)), output_root=output
    )

    with (output / "instance_metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (output / "association_metrics.csv").open(newline="") as stream:
        association = list(csv.DictReader(stream))
    assert [row["pair"] for row in rows] == ["pair-a", "pair-b"]
    assert [row["method"] for row in association] == ["R_obj", "R_obj"]
    assert manifest["pair_ids"] == ["pair-a", "pair-b"]
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert (first / "instance_metrics.csv").read_text().endswith("pair-a,P2,1\n")


def test_rejects_header_or_pair_identity_mismatch(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _source(first, "pair-a", 1)
    _source(second, "wrong-pair", 2)
    (second / "runs.csv").write_text(
        "pair,different\nwrong-pair,2\n", encoding="utf-8"
    )

    with pytest.raises(DenseResultAggregationError):
        aggregate_dense_results(
            (("pair-a", first), ("pair-b", second)),
            output_root=tmp_path / "aggregate",
        )

    assert not (tmp_path / "aggregate").exists()
