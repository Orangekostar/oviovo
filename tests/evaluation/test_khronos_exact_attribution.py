from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from src.evaluation.khronos_attribution import (
    AttributionContractError,
    assert_official_outputs_identical,
    assert_source_patch_identity,
    read_exact_associations,
    summarize_status,
    validate_attribution_mass,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return path


def _row(
    *,
    status: str,
    pred: str | None,
    gt: str | None,
    trajectory: int,
    metric_type: str = "dynamic_object",
    ordinal: int = 0,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "metric_type": metric_type,
        "map_name": "0",
        "metric_row_ordinal": ordinal,
        "query_time_ns": 20,
        "trajectory_timestamp_ns": trajectory,
        "pred_node_id": pred,
        "gt_node_id": gt,
        "distance_m": 0.1 if pred is not None and gt is not None else None,
        "status": status,
    }


def _write_official_csvs(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "static_objects.csv").write_text(
        "Name,Query,AppearedTP,DisappearedTP,AppearedFP,DisappearedFP,"
        "AppearedHallucinatedP,DisappearedHallucinatedP,AppearedFN,"
        "DisappearedFN,AppearedMissedP,DisappearedMissedP,NumObjDetected,"
        "NumObjMissed,NumObjHallucinated\n"
        "0,20,1,0,0,0,0,0,0,0,0,0,1,1,1\n",
        encoding="utf-8",
    )
    (root / "dynamic_objects.csv").write_text(
        "Name,Query,NumObjDetected,NumObjMissed,NumObjHallucinated\n"
        "0,20,3,1,2\n",
        encoding="utf-8",
    )
    (root / "background_mesh.csv").write_text(
        "Name,Accuracy@0.2,Completeness@0.2\n0,1,1\n",
        encoding="utf-8",
    )
    return root


def test_exact_sidecar_conserves_metric_mass_and_has_node_identity(
    tmp_path: Path,
) -> None:
    path = _write_jsonl(
        tmp_path / "dynamic_associations.jsonl",
        [
            _row(status="TP", pred="O1", gt="O9", trajectory=10),
            _row(status="TP", pred="O2", gt="O10", trajectory=11),
            _row(status="TP", pred="O3", gt="O11", trajectory=12),
            _row(status="FP", pred="O4", gt=None, trajectory=13),
            _row(status="FP", pred="O5", gt=None, trajectory=14),
            _row(status="FN", pred=None, gt="O12", trajectory=15),
        ],
    )

    rows = read_exact_associations(path)

    assert {row.status for row in rows} == {"TP", "FP", "FN"}
    assert all(
        row.map_name and row.query_time_ns >= row.trajectory_timestamp_ns
        for row in rows
    )
    assert summarize_status(rows) == {"TP": 3, "FP": 2, "FN": 1}


def test_sidecar_rejects_duplicate_identity_and_future_time(tmp_path: Path) -> None:
    duplicate = _row(status="TP", pred="O1", gt="O9", trajectory=10)
    path = _write_jsonl(tmp_path / "duplicate.jsonl", [duplicate, duplicate])
    with pytest.raises(AttributionContractError, match="duplicate event identity"):
        read_exact_associations(path)

    path = _write_jsonl(
        tmp_path / "future.jsonl",
        [_row(status="TP", pred="O1", gt="O9", trajectory=21)],
    )
    with pytest.raises(AttributionContractError, match="future trajectory"):
        read_exact_associations(path)


def test_attribution_mass_matches_every_official_metric_row(tmp_path: Path) -> None:
    results = _write_official_csvs(tmp_path / "results")
    dynamic_rows = read_exact_associations(
        _write_jsonl(
            tmp_path / "dynamic.jsonl",
            [
                _row(status="TP", pred=f"O{i}", gt=f"G{i}", trajectory=i)
                for i in range(3)
            ]
            + [
                _row(status="FP", pred=f"F{i}", gt=None, trajectory=10 + i)
                for i in range(2)
            ]
            + [_row(status="FN", pred=None, gt="M0", trajectory=12)],
        )
    )
    object_rows = read_exact_associations(
        _write_jsonl(
            tmp_path / "object.jsonl",
            [
                _row(
                    status="TP",
                    pred="O1",
                    gt="G1",
                    trajectory=20,
                    metric_type="object",
                ),
                _row(
                    status="FP",
                    pred="O2",
                    gt=None,
                    trajectory=20,
                    metric_type="object",
                ),
                _row(
                    status="FN",
                    pred=None,
                    gt="G2",
                    trajectory=20,
                    metric_type="object",
                ),
                _row(
                    status="TP",
                    pred="O1",
                    gt="G1",
                    trajectory=20,
                    metric_type="change_appeared",
                ),
            ],
        )
    )

    summary = validate_attribution_mass(
        object_rows=object_rows,
        dynamic_rows=dynamic_rows,
        official_results=results,
    )

    assert summary["dynamic_object"] == {"TP": 3, "FP": 2, "FN": 1}
    assert summary["object"] == {"TP": 1, "FP": 1, "FN": 1}
    assert summary["change_appeared"] == {"TP": 1, "FP": 0, "FN": 0}
    assert summary["change_disappeared"] == {"TP": 0, "FP": 0, "FN": 0}

    with (results / "dynamic_objects.csv").open("a", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("1", 30, 1, 0, 0))
    with pytest.raises(AttributionContractError, match="metric mass mismatch"):
        validate_attribution_mass(
            object_rows=object_rows,
            dynamic_rows=dynamic_rows,
            official_results=results,
        )


def test_noninterference_requires_byte_identical_official_csvs(
    tmp_path: Path,
) -> None:
    unpatched = _write_official_csvs(tmp_path / "unpatched")
    patched = _write_official_csvs(tmp_path / "patched")

    receipt = assert_official_outputs_identical(unpatched, patched)

    assert set(receipt) == {
        "background_mesh.csv",
        "dynamic_objects.csv",
        "static_objects.csv",
    }
    assert all(
        record["unpatched_sha256"] == record["patched_sha256"]
        for record in receipt.values()
    )
    expected = hashlib.sha256((patched / "static_objects.csv").read_bytes()).hexdigest()
    assert receipt["static_objects.csv"]["patched_sha256"] == expected

    (patched / "dynamic_objects.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(AttributionContractError, match="not byte-identical"):
        assert_official_outputs_identical(unpatched, patched)


def test_source_patch_identity_rejects_source_and_patch_tampering(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "source"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", checkout], check=True)
    subprocess.run(
        ["git", "-C", checkout, "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", checkout, "config", "user.name", "Test"], check=True
    )
    source = checkout / "source.cc"
    source.write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "-C", checkout, "add", "source.cc"], check=True)
    subprocess.run(
        ["git", "-C", checkout, "commit", "-qm", "fixture"], check=True
    )
    commit = subprocess.run(
        ["git", "-C", checkout, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source.write_text("after\n", encoding="utf-8")
    patch_bytes = subprocess.run(
        ["git", "-C", checkout, "diff", "--binary"],
        check=True,
        capture_output=True,
    ).stdout
    source.write_text("before\n", encoding="utf-8")
    patch = tmp_path / "change.patch"
    patch.write_bytes(patch_bytes)
    patch_sha256 = hashlib.sha256(patch_bytes).hexdigest()

    identity = assert_source_patch_identity(
        source_checkout=checkout,
        source_commit=commit,
        patch_file=patch,
        patch_sha256=patch_sha256,
    )
    assert identity["commit"] == commit
    assert identity["patch_sha256"] == patch_sha256

    source.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(AttributionContractError, match="not clean"):
        assert_source_patch_identity(
            source_checkout=checkout,
            source_commit=commit,
            patch_file=patch,
            patch_sha256=patch_sha256,
        )
    source.write_text("before\n", encoding="utf-8")
    with pytest.raises(AttributionContractError, match="patch SHA-256 mismatch"):
        assert_source_patch_identity(
            source_checkout=checkout,
            source_commit=commit,
            patch_file=patch,
            patch_sha256="0" * 64,
        )
