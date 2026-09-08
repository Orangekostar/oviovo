from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/evaluation/run_observation_query_multienv.py"

FIELDS = (
    "run_id",
    "method_id",
    "pair_id",
    "environment_uuid",
    "visit_id",
    "split_role",
    "evaluation_domain",
    "checkpoint_id",
    "training_updates",
    "status",
    "f1_50",
    "tp50",
    "fp50",
    "fn50",
    "raw_best_iou_mean",
    "raw_ar50",
    "raw_ar25",
    "identity_recall",
)


def _write_results(
    root: Path,
    *,
    run_id: str,
    pair_id: str,
    environment_uuid: str,
    common_f1: str,
    counts: tuple[str, str, str],
    raw_common: tuple[str, str, str],
    checkpoint_id: str = "checkpoint-500",
    training_updates: str = "500",
) -> None:
    output = root / "evaluation_runs" / run_id / "results.csv"
    output.parent.mkdir(parents=True)
    common_tp, common_fp, common_fn = counts
    rows = (
        {
            "evaluation_domain": "COMMON_INPUT_SUPPORT_V2",
            "f1_50": common_f1,
            "tp50": common_tp,
            "fp50": common_fp,
            "fn50": common_fn,
            "identity_recall": "0.25",
        },
        {
            "evaluation_domain": "FULL_GT_V2",
            "f1_50": "0.2",
            "tp50": "1",
            "fp50": "4",
            "fn50": "4",
            "identity_recall": "0.1",
        },
        {
            "evaluation_domain": "RAW_COMMON_INPUT_SUPPORT_V2",
            "raw_best_iou_mean": raw_common[0],
            "raw_ar50": raw_common[1],
            "raw_ar25": raw_common[2],
        },
        {
            "evaluation_domain": "RAW_FULL_GT_V2",
            "raw_best_iou_mean": "0.1",
            "raw_ar50": "0.0",
            "raw_ar25": "0.25",
        },
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for values in rows:
            writer.writerow(
                {
                    "run_id": run_id,
                    "method_id": "OBS_FULL",
                    "pair_id": pair_id,
                    "environment_uuid": environment_uuid,
                    "visit_id": "1",
                    "split_role": "DEV",
                    "checkpoint_id": checkpoint_id,
                    "training_updates": training_updates,
                    "status": "PASS",
                    **values,
                }
            )


def test_aggregate_only_pools_counts_and_macro_averages_environments(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps({"cache_root": str(cache_root)}), encoding="utf-8"
    )
    runs = []
    for run_id, pair_id, environment, f1, counts, raw in (
        ("run-a", "pair-a", "env-a", "0.5", ("1", "1", "1"), ("0.6", "0.5", "0.75")),
        ("run-b", "pair-b", "env-b", "0.25", ("1", "3", "3"), ("0.2", "0.0", "0.25")),
    ):
        _write_results(
            cache_root,
            run_id=run_id,
            pair_id=pair_id,
            environment_uuid=environment,
            common_f1=f1,
            counts=counts,
            raw_common=raw,
        )
        runs.append(
            {
                "config": str(tmp_path / "config.json"),
                "runtime": str(runtime),
                "method": "OBS_FULL",
                "checkpoint": str(tmp_path / "checkpoint"),
                "role": "DEV",
                "pair_id": pair_id,
                "run_id": run_id,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "runs": runs}), encoding="utf-8"
    )
    output = tmp_path / "aggregate"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--output-root",
            str(output),
            "--aggregate-only",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    with (output / "macro_micro_comparison.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        aggregate = list(csv.DictReader(handle))
    assert aggregate == [
        {
            "method_id": "OBS_FULL",
            "checkpoint_id": "checkpoint-500",
            "training_updates": "500",
            "environment_count": "2",
            "valid_common_environment_count": "2",
            "common_macro_f1_50": "0.375",
            "common_micro_tp50": "2",
            "common_micro_fp50": "4",
            "common_micro_fn50": "4",
            "common_micro_f1_50": "0.3333333333333333",
            "common_raw_best_iou_macro": "0.4",
            "common_raw_ar50_macro": "0.25",
            "common_raw_ar25_macro": "0.5",
            "full_macro_f1_50": "0.2",
            "full_micro_tp50": "2",
            "full_micro_fp50": "8",
            "full_micro_fn50": "8",
            "full_micro_f1_50": "0.2",
            "full_raw_best_iou_macro": "0.1",
            "full_raw_ar50_macro": "0.0",
            "full_raw_ar25_macro": "0.25",
            "identity_recall_macro": "0.25",
            "status": "PASS",
        }
    ]
    with (output / "checkpoint_selection.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        selection = list(csv.DictReader(handle))
    assert selection[0]["selected"] == "true"
    assert selection[0]["selection_reason"] == "MAX_COMMON_MACRO_THEN_RAW_THEN_EARLIER"


def test_aggregate_rejects_checkpoint_with_incomplete_environment_coverage(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps({"cache_root": str(cache_root)}), encoding="utf-8"
    )
    runs = []
    for run_id, pair_id, environment, checkpoint, updates in (
        ("run-a-200", "pair-a", "env-a", "checkpoint-200", "200"),
        ("run-b-200", "pair-b", "env-b", "checkpoint-200", "200"),
        ("run-a-500", "pair-a", "env-a", "checkpoint-500", "500"),
    ):
        _write_results(
            cache_root,
            run_id=run_id,
            pair_id=pair_id,
            environment_uuid=environment,
            common_f1="0.5",
            counts=("1", "1", "1"),
            raw_common=("0.6", "0.5", "0.75"),
            checkpoint_id=checkpoint,
            training_updates=updates,
        )
        runs.append(
            {
                "config": str(tmp_path / "config.json"),
                "runtime": str(runtime),
                "method": "OBS_FULL",
                "checkpoint": str(tmp_path / checkpoint),
                "role": "DEV",
                "pair_id": pair_id,
                "run_id": run_id,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "runs": runs}), encoding="utf-8"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--output-root",
            str(tmp_path / "aggregate"),
            "--aggregate-only",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "identical environment coverage" in completed.stderr
