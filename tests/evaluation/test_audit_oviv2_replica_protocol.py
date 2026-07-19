from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path("scripts/evaluation/audit_oviv2_replica_protocol.py")


def _metrics(*, ap25: float, ap50: float) -> dict:
    return {
        "ap25": ap25,
        "ap50": ap50,
        "miou": 0.3,
        "f5": 0.9,
        "instance": {
            "class_agnostic": {"ap25": ap25, "ap50": ap50},
            "semantic_class_constrained": {"ap25": 0.1, "ap50": 0.05},
        },
        "protocol": {
            "distance_threshold_m": 0.05,
            "projection_comparator": "strict_less_than",
            "min_instance_vertices": 100,
            "headline_instance_protocol": "class_agnostic",
            "semantic_instance_protocol": "diagnostic_only",
        },
    }


def _run(tmp_path: Path, old: dict, new: dict) -> tuple[subprocess.CompletedProcess[str], Path]:
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    output = tmp_path / "audit.json"
    old_path.write_text(json.dumps(old), encoding="utf-8")
    new_path.write_text(json.dumps(new), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--old-metrics",
            str(old_path),
            "--new-metrics",
            str(new_path),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return result, output


def test_audit_records_protocol_deltas_and_hashes(tmp_path: Path) -> None:
    old = _metrics(ap25=0.1, ap50=0.05)
    new = _metrics(ap25=0.4, ap50=0.2)

    result, output = _run(tmp_path, old, new)

    assert result.returncode == 0, result.stderr
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["headline_compatible"] is True
    assert audit["reasons"] == []
    assert audit["protocol"] == {
        "distance_threshold_m": pytest.approx(0.05),
        "headline_accepted_view_filter": False,
        "headline_instance_protocol": "class_agnostic",
        "min_gt_instance_vertices": 100,
        "min_predicted_instance_vertices": 100,
        "projection_comparator": "strict_less_than",
        "semantic_diagnostic_accepted_view_filter": True,
    }
    assert audit["headline_metrics"]["delta"] == {
        "ap25": pytest.approx(0.3),
        "ap50": pytest.approx(0.15),
    }
    old_path = tmp_path / "old.json"
    assert audit["sources"]["old_metrics"]["sha256"] == hashlib.sha256(
        old_path.read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_protocol", "headline_instance_protocol"),
        ("nonfinite", "finite"),
        ("headline_mismatch", "class_agnostic"),
    ],
)
def test_audit_rejects_incompatible_headline_contract(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    old = _metrics(ap25=0.1, ap50=0.05)
    new = _metrics(ap25=0.4, ap50=0.2)
    if mutation == "missing_protocol":
        new["protocol"].pop("headline_instance_protocol")
    elif mutation == "nonfinite":
        new["miou"] = float("nan")
    else:
        new["ap25"] = 0.9

    result, output = _run(tmp_path, old, new)

    assert result.returncode != 0
    assert message in result.stderr
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["headline_compatible"] is False
    assert audit["reasons"]
