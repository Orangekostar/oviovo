from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.evaluation.baselines.ovimap_diagnostics import (
    parse_official_instance_output,
    parse_official_semantic_output,
)


INSTANCE_OUTPUT = """mIoU\twIoU\tmP@75\tmR@75\tmP@50\tmR@50\tmP@25\tmR@25
0.363\t0.500\t0.220\t0.180\t0.508\t0.410\t0.767\t0.620
"""


def test_parser_preserves_released_instance_metric_names() -> None:
    payload = parse_official_instance_output(INSTANCE_OUTPUT, scene_id="room0")

    assert payload["schema_version"] == 1
    assert payload["scene_id"] == "room0"
    assert payload["source_protocol"] == "released_ovimap"
    assert payload["source_header"] == [
        "mIoU",
        "wIoU",
        "mP@75",
        "mR@75",
        "mP@50",
        "mR@50",
        "mP@25",
        "mR@25",
    ]
    assert payload["metrics"]["mean_precision_at_25"] == pytest.approx(0.767)
    assert payload["metrics"]["mean_recall_at_25"] == pytest.approx(0.620)
    assert "ap25" not in payload["metrics"]


def test_parser_reads_released_semantic_metrics() -> None:
    payload = parse_official_semantic_output(
        "mIoU\tmAcc\n0.265\t0.322\n",
        scene_id="room0",
    )

    assert payload["source_header"] == ["mIoU", "mAcc"]
    assert payload["metrics"] == {
        "mean_iou": pytest.approx(0.265),
        "mean_accuracy": pytest.approx(0.322),
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("wrong\theader\n0.1\t0.2\n", "header"),
        (INSTANCE_OUTPUT + "0.1\t0.2\t0.3\t0.4\t0.5\t0.6\t0.7\t0.8\n", "one data row"),
        (INSTANCE_OUTPUT.replace("0.767", "nan"), "finite"),
    ],
)
def test_instance_parser_rejects_malformed_output(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_official_instance_output(text, scene_id="room0")


def test_capture_cli_hashes_inputs_and_marks_diagnostics_non_headline(tmp_path) -> None:
    instance = tmp_path / "instance.txt"
    semantic = tmp_path / "semantic.txt"
    neutral = tmp_path / "metrics.json"
    output = tmp_path / "diagnostics.json"
    instance.write_text(INSTANCE_OUTPUT, encoding="utf-8")
    semantic.write_text("mIoU\tmAcc\n0.265\t0.322\n", encoding="utf-8")
    neutral.write_text('{"metrics": {"semantic": {"miou": 0.03}}}\n', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/capture_ovimap_diagnostics.py",
            "--scene-id",
            "room0",
            "--instance-stdout",
            str(instance),
            "--semantic-stdout",
            str(semantic),
            "--neutral-metrics",
            str(neutral),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["headline_token_eligible"] is False
    assert "token_bindings" not in payload
    assert payload["instance"]["metrics"]["mean_precision_at_25"] == pytest.approx(0.767)
    assert payload["sources"]["neutral_metrics"]["sha256"] == hashlib.sha256(
        neutral.read_bytes()
    ).hexdigest()
    assert Path(payload["sources"]["neutral_metrics"]["path"]) == neutral.resolve()
