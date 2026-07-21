from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.evaluation.compare_oviv2_pareto import (
    _publish_audit,
    compare_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "evaluation" / "compare_oviv2_pareto.py"
METRIC_PATHS = {
    "miou": ("semantic", "miou"),
    "macc": ("semantic", "macc"),
    "f_miou": ("semantic", "f_miou"),
    "ap25": ("instance", "class_agnostic", "ap25"),
    "ap50": ("instance", "class_agnostic", "ap50"),
    "f5": ("geometry", "f5"),
}


def metrics(
    *,
    miou: float = 0.38,
    macc: float = 0.43,
    f_miou: float = 0.66,
    ap25: float = 0.28,
    ap50: float = 0.05,
    f5: float = 0.916,
    scene: str = "room0",
    snapshot_checksum: str = "a" * 64,
) -> dict:
    return {
        "miou": miou,
        "macc": macc,
        "f_miou": f_miou,
        "ap25": ap25,
        "ap50": ap50,
        "f5": f5,
        "semantic": {"miou": miou, "macc": macc, "f_miou": f_miou},
        "instance": {
            "class_agnostic": {"ap25": ap25, "ap50": ap50},
            "semantic_class_constrained": {"ap25": 0.0, "ap50": 0.0},
        },
        "geometry": {"f5": f5, "precision": 0.9, "recall": 0.9},
        "protocol": {
            "distance_threshold_m": 0.05,
            "headline_instance_protocol": "class_agnostic",
            "manifest_id": "replica8_static_v1",
            "min_instance_vertices": 100,
            "scene_id": scene,
            "snapshot_checksums": {"geometry.npz": snapshot_checksum},
            "snapshot_revision": 200,
        },
    }


def _set_metric(payload: dict, name: str, value: object) -> None:
    target = payload
    path = METRIC_PATHS[name]
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    if name in payload:
        payload[name] = value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True), encoding="utf-8"
    )


def _run_cli(
    baseline: Path,
    candidate: Path,
    output: Path,
    mode: str = "route2",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--mode",
            mode,
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_route2_gate_requires_semantic_ap_gain_and_identical_f5() -> None:
    baseline = metrics()
    candidate = metrics(
        miou=0.39,
        macc=0.44,
        f_miou=0.67,
        ap25=0.29,
        ap50=0.06,
        f5=0.916,
        snapshot_checksum="b" * 64,
    )

    audit = compare_metrics(baseline, candidate, mode="route2")

    assert audit["status"] == "PASS"
    assert audit["scene"] == "room0"
    assert audit["checks"]["f5"]["relation"] == "byte_identical"
    assert audit["checks"]["f5"]["passed"] is True


def test_instance_gate_requires_ap_gain_and_identical_other_heads() -> None:
    baseline = metrics()
    candidate = metrics(ap25=0.38, ap50=0.13)
    candidate["protocol"].update(
        {
            "headline_instance_protocol": "independent_gt_projection_hypotheses",
            "instance_head_config": {"minimum_component_vertices": 20},
            "instance_head_config_hash": "b" * 64,
            "instance_head_source_hashes": {"instance_head": "c" * 64},
            "instance_head_algorithm_hash": "d" * 64,
        }
    )

    audit = compare_metrics(baseline, candidate, mode="instance")

    assert audit["status"] == "PASS"
    assert audit["checks"]["ap25"]["relation"] == "strictly_greater"
    assert audit["checks"]["ap50"]["relation"] == "strictly_greater"
    for name in ("miou", "macc", "f_miou", "f5"):
        assert audit["checks"][name]["relation"] == "byte_identical"
    assert set(audit["protocol"]["comparison_excludes"]) == {
        "headline_instance_protocol",
        "instance_head_config",
        "instance_head_config_hash",
        "instance_head_source_hashes",
        "instance_head_algorithm_hash",
    }


def test_semantic_gate_requires_three_gains_and_identical_ap_geometry() -> None:
    baseline = metrics()
    candidate = metrics(miou=0.39, macc=0.44, f_miou=0.67)
    candidate["protocol"]["semantic_replay"] = {
        "algorithm_hash": "e" * 64,
    }

    audit = compare_metrics(baseline, candidate, mode="semantic")

    assert audit["status"] == "PASS"
    for name in ("miou", "macc", "f_miou"):
        assert audit["checks"][name]["relation"] == "strictly_greater"
    for name in ("ap25", "ap50", "f5"):
        assert audit["checks"][name]["relation"] == "byte_identical"
    assert "semantic_replay" in audit["protocol"]["comparison_excludes"]


def test_composed_gate_requires_all_gains_on_the_frozen_snapshot() -> None:
    baseline = metrics(snapshot_checksum="a" * 64)
    candidate = metrics(
        miou=0.39,
        macc=0.44,
        f_miou=0.67,
        ap25=0.29,
        ap50=0.06,
        f5=0.92,
        snapshot_checksum="a" * 64,
    )
    candidate["protocol"].update(
        {
            "headline_instance_protocol": "independent_gt_projection_hypotheses",
            "instance_head_algorithm_hash": "a" * 64,
            "semantic_replay": {"algorithm_hash": "b" * 64},
            "mesh_weight_threshold": 0.5,
            "geometry_semantic_stabilization": {"algorithm_hash": "c" * 64},
        }
    )

    audit = compare_metrics(baseline, candidate, mode="composed")

    assert audit["status"] == "PASS"
    assert all(
        check["relation"] == "strictly_greater"
        for check in audit["checks"].values()
    )
    assert "snapshot_checksums" not in audit["protocol"]["comparison_excludes"]
    assert {
        "semantic_replay",
        "mesh_weight_threshold",
        "geometry_semantic_stabilization",
    }.issubset(audit["protocol"]["comparison_excludes"])


def test_composed_gate_rejects_snapshot_change() -> None:
    candidate = metrics(
        miou=0.39,
        macc=0.44,
        f_miou=0.67,
        ap25=0.29,
        ap50=0.06,
        f5=0.92,
        snapshot_checksum="b" * 64,
    )
    candidate["protocol"].update(
        {
            "instance_head_algorithm_hash": "a" * 64,
            "semantic_replay": {"algorithm_hash": "b" * 64},
            "mesh_weight_threshold": 0.5,
            "geometry_semantic_stabilization": {"algorithm_hash": "c" * 64},
        }
    )
    with pytest.raises(ValueError, match="protocol mismatch"):
        compare_metrics(metrics(snapshot_checksum="a" * 64), candidate, mode="composed")


@pytest.mark.parametrize(
    "missing",
    ("instance_head_algorithm_hash", "semantic_replay", "geometry_semantic_stabilization"),
)
def test_composed_gate_requires_all_head_provenance(missing: str) -> None:
    candidate = metrics(
        miou=0.39,
        macc=0.44,
        f_miou=0.67,
        ap25=0.29,
        ap50=0.06,
        f5=0.92,
    )
    candidate["protocol"].update(
        {
            "instance_head_algorithm_hash": "a" * 64,
            "semantic_replay": {"algorithm_hash": "b" * 64},
            "mesh_weight_threshold": 0.5,
            "geometry_semantic_stabilization": {"algorithm_hash": "c" * 64},
        }
    )
    candidate["protocol"].pop(missing)

    with pytest.raises(ValueError, match="requires.*provenance"):
        compare_metrics(metrics(), candidate, mode="composed")


@pytest.mark.parametrize("name", ("miou", "macc", "f_miou"))
def test_semantic_gate_rejects_semantic_tie(name: str) -> None:
    candidate = metrics(miou=0.39, macc=0.44, f_miou=0.67)
    _set_metric(candidate, name, metrics()[name])
    assert compare_metrics(metrics(), candidate, mode="semantic")["status"] == "FAIL"


@pytest.mark.parametrize("name", ("ap25", "ap50", "f5"))
def test_semantic_gate_rejects_other_head_change(name: str) -> None:
    candidate = metrics(miou=0.39, macc=0.44, f_miou=0.67)
    _set_metric(candidate, name, metrics()[name] + 0.001)
    assert compare_metrics(metrics(), candidate, mode="semantic")["status"] == "FAIL"


@pytest.mark.parametrize("mode", ("instance", "semantic"))
def test_frozen_snapshot_stage_rejects_snapshot_change(mode: str) -> None:
    baseline = metrics(snapshot_checksum="a" * 64)
    if mode == "instance":
        candidate = metrics(ap25=0.29, ap50=0.06, snapshot_checksum="b" * 64)
    else:
        candidate = metrics(
            miou=0.39,
            macc=0.44,
            f_miou=0.67,
            snapshot_checksum="b" * 64,
        )
    with pytest.raises(ValueError, match="protocol mismatch"):
        compare_metrics(baseline, candidate, mode=mode)


@pytest.mark.parametrize("name", ("ap25", "ap50"))
def test_instance_gate_rejects_ap_tie(name: str) -> None:
    candidate = metrics(ap25=0.29, ap50=0.06)
    _set_metric(candidate, name, metrics()[name])
    assert compare_metrics(metrics(), candidate, mode="instance")["status"] == "FAIL"


@pytest.mark.parametrize("name", ("miou", "macc", "f_miou", "f5"))
def test_instance_gate_rejects_any_other_head_change(name: str) -> None:
    candidate = metrics(ap25=0.29, ap50=0.06)
    _set_metric(candidate, name, metrics()[name] + 0.001)
    assert compare_metrics(metrics(), candidate, mode="instance")["status"] == "FAIL"


@pytest.mark.parametrize("name", tuple(METRIC_PATHS))
def test_route2_gate_fails_each_individual_regression(name: str) -> None:
    baseline = metrics()
    candidate = metrics(
        miou=0.39,
        macc=0.44,
        f_miou=0.67,
        ap25=0.29,
        ap50=0.06,
    )
    _set_metric(candidate, name, baseline[name] - 0.001)

    audit = compare_metrics(baseline, candidate, mode="route2")

    assert audit["status"] == "FAIL"
    assert audit["checks"][name]["passed"] is False


@pytest.mark.parametrize("name", ("miou", "macc", "f_miou", "ap25", "ap50"))
def test_route2_gate_rejects_non_strict_ties(name: str) -> None:
    baseline = metrics()
    candidate = metrics(
        miou=0.39,
        macc=0.44,
        f_miou=0.67,
        ap25=0.29,
        ap50=0.06,
    )
    _set_metric(candidate, name, baseline[name])

    assert compare_metrics(baseline, candidate, mode="route2")["status"] == "FAIL"


def test_final_gate_requires_strict_improvement_of_all_six_metrics() -> None:
    baseline = metrics()
    candidate = metrics(
        miou=0.39,
        macc=0.44,
        f_miou=0.67,
        ap25=0.29,
        ap50=0.06,
        f5=0.917,
    )

    audit = compare_metrics(baseline, candidate, mode="final")

    assert audit["status"] == "PASS"
    assert all(check["relation"] == "strictly_greater" for check in audit["checks"].values())


@pytest.mark.parametrize("name", tuple(METRIC_PATHS))
def test_final_gate_rejects_each_exact_tie(name: str) -> None:
    baseline = metrics()
    candidate = metrics(
        miou=0.39,
        macc=0.44,
        f_miou=0.67,
        ap25=0.29,
        ap50=0.06,
        f5=0.917,
    )
    _set_metric(candidate, name, baseline[name])

    assert compare_metrics(baseline, candidate, mode="final")["status"] == "FAIL"


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf"), True])
@pytest.mark.parametrize("name", tuple(METRIC_PATHS))
def test_gate_rejects_non_finite_and_boolean_metrics(
    name: str, bad_value: object
) -> None:
    candidate = metrics()
    _set_metric(candidate, name, bad_value)

    with pytest.raises(ValueError, match=name):
        compare_metrics(metrics(), candidate, mode="route2")


@pytest.mark.parametrize("name,path", tuple(METRIC_PATHS.items()))
def test_gate_rejects_each_missing_metric(name: str, path: tuple[str, ...]) -> None:
    candidate = metrics()
    target = candidate
    for component in path[:-1]:
        target = target[component]
    del target[path[-1]]

    with pytest.raises(ValueError, match=name):
        compare_metrics(metrics(), candidate, mode="route2")


def test_gate_rejects_ambiguous_headline_metric() -> None:
    candidate = metrics(miou=0.39)
    candidate["miou"] = 0.99

    with pytest.raises(ValueError, match="ambiguous.*miou"):
        compare_metrics(metrics(), candidate, mode="route2")


def test_gate_rejects_ambiguous_instance_metric() -> None:
    candidate = metrics(ap25=0.29)
    candidate["instance"]["ap25"] = 0.99

    with pytest.raises(ValueError, match="ambiguous.*ap25"):
        compare_metrics(metrics(), candidate, mode="route2")


def test_gate_rejects_protocol_mismatch_except_snapshot_checksums() -> None:
    baseline = metrics(snapshot_checksum="a" * 64)
    candidate = metrics(snapshot_checksum="b" * 64)
    candidate["protocol"]["distance_threshold_m"] = 0.04

    with pytest.raises(ValueError, match="protocol mismatch"):
        compare_metrics(baseline, candidate, mode="route2")


def test_gate_rejects_scene_mismatch() -> None:
    with pytest.raises(ValueError, match="scene mismatch"):
        compare_metrics(metrics(scene="room0"), metrics(scene="office0"), mode="route2")


def test_compatible_top_level_scene_must_agree_with_protocol() -> None:
    baseline = metrics()
    candidate = metrics(
        miou=0.39, macc=0.44, f_miou=0.67, ap25=0.29, ap50=0.06
    )
    baseline["scene"] = "room0"
    candidate["scene"] = "room0"
    assert compare_metrics(baseline, candidate, mode="route2")["status"] == "PASS"

    candidate["scene"] = "office0"
    with pytest.raises(ValueError, match="scene"):
        compare_metrics(baseline, candidate, mode="route2")


def test_route2_f5_uses_exact_ieee_value_not_json_token_spelling(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "audit.json"
    baseline.write_text(
        json.dumps(metrics()).replace("0.916", "0.9160"), encoding="utf-8"
    )
    _write_json(
        candidate,
        metrics(miou=0.39, macc=0.44, f_miou=0.67, ap25=0.29, ap50=0.06),
    )

    completed = _run_cli(baseline, candidate, output)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"


def test_cli_writes_complete_pass_and_fail_audits(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    passing = tmp_path / "passing.json"
    failing = tmp_path / "failing.json"
    pass_output = tmp_path / "pass-audit.json"
    fail_output = tmp_path / "fail-audit.json"
    _write_json(baseline, metrics())
    _write_json(
        passing,
        metrics(miou=0.39, macc=0.44, f_miou=0.67, ap25=0.29, ap50=0.06),
    )
    _write_json(failing, metrics())

    passed = _run_cli(baseline, passing, pass_output)
    failed = _run_cli(baseline, failing, fail_output)

    assert passed.returncode == 0, passed.stderr
    assert failed.returncode == 1, failed.stderr
    pass_audit = json.loads(pass_output.read_text(encoding="utf-8"))
    fail_audit = json.loads(fail_output.read_text(encoding="utf-8"))
    assert pass_audit["status"] == "PASS"
    assert fail_audit["status"] == "FAIL"
    for audit in (pass_audit, fail_audit):
        assert audit["mode"] == "route2"
        assert set(audit["checks"]) == set(METRIC_PATHS)
        assert set(audit["inputs"]) == {"baseline", "candidate"}
        assert all(len(source["sha256"]) == 64 for source in audit["inputs"].values())


def test_cli_structure_error_leaves_no_output(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "nested" / "audit.json"
    _write_json(baseline, metrics())
    candidate.write_text('{"semantic":{"miou":NaN}}', encoding="utf-8")

    completed = _run_cli(baseline, candidate, output)

    assert completed.returncode == 2
    assert "finite" in completed.stderr.lower() or "invalid" in completed.stderr.lower()
    assert not output.exists()
    assert not output.parent.exists()


def test_cli_rejects_duplicate_json_keys_without_output(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "audit.json"
    _write_json(baseline, metrics())
    candidate.write_text('{"protocol":{},"protocol":{}}', encoding="utf-8")

    completed = _run_cli(baseline, candidate, output)

    assert completed.returncode == 2
    assert "duplicate" in completed.stderr.lower()
    assert not output.exists()


def test_publish_is_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    output.write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _publish_audit(output, {"status": "PASS"})

    assert output.read_text(encoding="utf-8") == "original"


def test_publish_rolls_back_when_directory_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "audit.json"
    real_fsync = os.fsync
    calls = 0

    def fail_directory_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)

    with pytest.raises(OSError, match="directory fsync failed"):
        _publish_audit(output, {"status": "PASS"})

    assert not output.exists()
    assert not tuple(tmp_path.glob(".audit.*.tmp"))


def test_publish_rolls_back_when_directory_open_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "audit.json"
    real_open = os.open
    failed = False

    def fail_directory_open(path, flags: int, *args, **kwargs):
        nonlocal failed
        if not failed and Path(path) == tmp_path and flags & os.O_DIRECTORY:
            failed = True
            raise OSError("directory open failed")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", fail_directory_open)

    with pytest.raises(OSError, match="directory open failed"):
        _publish_audit(output, {"status": "PASS"})

    assert not output.exists()
    assert not tuple(tmp_path.glob(".audit.*.tmp"))


def test_publish_does_not_remove_competing_destination_on_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "audit.json"
    real_fsync = os.fsync
    calls = 0

    def replace_then_fail(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            output.unlink()
            output.write_text("competitor", encoding="utf-8")
            raise OSError("directory fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", replace_then_fail)

    with pytest.raises(OSError, match="roll back"):
        _publish_audit(output, {"status": "PASS"})

    assert output.read_text(encoding="utf-8") == "competitor"
    assert not tuple(tmp_path.glob(".audit.*.tmp"))


def test_publish_rolls_back_when_success_cleanup_fails_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "audit.json"
    real_unlink = Path.unlink
    failed = False

    def fail_first_temporary_cleanup(path: Path, *args, **kwargs) -> None:
        nonlocal failed
        if not failed and path.name.startswith(".audit.json."):
            failed = True
            raise OSError("temporary cleanup failed")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_first_temporary_cleanup)

    with pytest.raises(OSError, match="temporary"):
        _publish_audit(output, {"status": "PASS"})

    assert not output.exists()
    assert not tuple(tmp_path.glob(".audit.*.tmp"))


def test_cli_existing_output_is_exit_two_and_preserved(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "audit.json"
    _write_json(baseline, metrics())
    _write_json(
        candidate,
        metrics(miou=0.39, macc=0.44, f_miou=0.67, ap25=0.29, ap50=0.06),
    )
    output.write_text("existing", encoding="utf-8")

    completed = _run_cli(baseline, candidate, output)

    assert completed.returncode == 2
    assert output.read_text(encoding="utf-8") == "existing"


def test_compare_does_not_mutate_inputs() -> None:
    baseline = metrics()
    candidate = metrics(
        miou=0.39, macc=0.44, f_miou=0.67, ap25=0.29, ap50=0.06
    )
    before = copy.deepcopy((baseline, candidate))

    compare_metrics(baseline, candidate, mode="route2")

    assert (baseline, candidate) == before
