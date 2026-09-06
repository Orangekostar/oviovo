from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation.prepare_ovi_rescene_input_v2 import (
    RecoveryResult,
    load_static_input_artifact,
    publish_recovered_input,
    write_static_input_artifact,
)
from scripts.evaluation.prepare_ovi_rescene_supported_v3 import (
    SupportedPreparationError,
    audit_supported_input,
    build_supported_input,
    main,
)
from src.oviv2.rescene_input_bridge import RecoveredModelSupport
from tests.evaluation.test_prepare_ovi_rescene_input_v2 import (
    _file_binding,
    _static_contract,
)


def _partial_v2(tmp_path: Path) -> tuple[Path, Path]:
    geometry, surface, sampling, candidates = _static_contract()
    parent = tmp_path / "parent.json"
    parent.write_text('{"source":true}\n', encoding="utf-8")
    static_root = tmp_path / "static"
    write_static_input_artifact(
        geometry=geometry,
        surface=surface,
        sampling=sampling,
        candidates=candidates,
        input_bindings={"parent": _file_binding(parent)},
        output_root=static_root,
    )
    prepared = load_static_input_artifact(static_root)
    support = RecoveredModelSupport(
        support_valid=np.asarray([True, True, False], dtype=np.bool_),
        representative_source_point_indices=np.asarray([0, 2, -1]),
        rgb_uint8=np.asarray(
            [[10, 20, 30], [40, 50, 60], [0, 0, 0]], dtype=np.uint8
        ),
        local_frame_indices=np.asarray([0, 1, -1]),
        global_frame_indices=np.asarray([766, 1218, -1]),
        rows=np.asarray([1, 2, -1]),
        columns=np.asarray([4, 5, -1]),
        camera_depth_m=np.asarray([2.0, 2.0, np.nan], dtype=np.float32),
        observed_depth_m=np.asarray([2.0, 2.0, np.nan], dtype=np.float32),
        depth_residual_m=np.asarray([0.0, 0.0, np.nan], dtype=np.float32),
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"status":"PASS"}\n', encoding="utf-8")
    input_root = tmp_path / "partial-v2"
    publish_recovered_input(
        static_input_root=static_root,
        calibration_manifest_path=calibration,
        prepared=prepared,
        recovery=RecoveryResult(
            status="PARTIAL_INPUT_SCOPE",
            support=support,
            unsupported_model_indices=np.asarray([2], dtype=np.int64),
        ),
        depth_tolerance_m=0.02,
        output_root=input_root,
    )
    return input_root, tmp_path / "supported-v3"


def test_supported_v3_round_trip_binds_compact_artifacts_and_denominators(
    tmp_path: Path,
) -> None:
    input_root, output_root = _partial_v2(tmp_path)

    receipt = build_supported_input(input_v2_root=input_root, output_root=output_root)
    audited = audit_supported_input(output_root)

    assert audited == receipt
    assert receipt["artifact_id"] == "OVI_RESCENE_SUPPORTED_INPUT_V3"
    assert receipt["status"] == "SUPPORTED_INFERENCE_PASS"
    assert receipt["full_domains"] == {"source": 4, "adapter": 4, "model": 3}
    assert receipt["supported_domains"] == {"source": 3, "adapter": 3, "model": 2}
    assert receipt["coverage"] == {
        "source_fraction": 0.75,
        "adapter_fraction": 0.75,
        "model_fraction": 2 / 3,
    }
    with np.load(output_root / "mappings.npz", allow_pickle=False) as archive:
        assert archive["new_to_old_model_indices"].tolist() == [0, 1]
        assert archive["old_to_new_model_indices"].tolist() == [0, 1, -1]
        assert archive["new_to_old_adapter_indices"].tolist() == [0, 1, 2]
        assert archive["new_to_old_source_point_indices"].tolist() == [0, 1, 2]
    coverage = json.loads(
        (output_root / "entity_coverage.json").read_text(encoding="utf-8")
    )
    assert coverage["unsupported_entity_keys"] == []
    assert coverage["entities"][1]["full_model_token_count"] == 2
    assert coverage["entities"][1]["supported_model_token_count"] == 1


def test_supported_v3_is_no_clobber_and_rejects_mapping_tamper(tmp_path: Path) -> None:
    input_root, output_root = _partial_v2(tmp_path)
    build_supported_input(input_v2_root=input_root, output_root=output_root)

    with pytest.raises(SupportedPreparationError, match="already exists"):
        build_supported_input(input_v2_root=input_root, output_root=output_root)

    mappings = output_root / "mappings.npz"
    mappings.write_bytes(mappings.read_bytes() + b"tampered")
    with pytest.raises(SupportedPreparationError, match="mappings binding mismatch"):
        audit_supported_input(output_root)


def test_supported_v3_rejects_parent_receipt_changed_after_publication(
    tmp_path: Path,
) -> None:
    input_root, output_root = _partial_v2(tmp_path)
    build_supported_input(input_v2_root=input_root, output_root=output_root)
    parent = input_root / "input_contract_v2.json"
    parent.write_bytes(parent.read_bytes() + b" ")

    with pytest.raises(SupportedPreparationError, match="parent V2 binding mismatch"):
        audit_supported_input(output_root)


def test_supported_v3_cli_builds_and_audits_from_strict_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_root, output_root = _partial_v2(tmp_path)
    config = tmp_path / "supported-v3.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_id": "OVI_RESCENE_SUPPORTED_V3_CONFIG_V1",
                "input_v2_root": str(input_root),
                "output_root": str(output_root),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["build", "--config", str(config)]) == 0
    assert main(["audit", "--config", str(config)]) == 0
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert messages == [
        {
            "output": str(output_root.absolute()),
            "status": "SUPPORTED_INFERENCE_PASS",
        },
        {
            "output": str(output_root.absolute()),
            "status": "SUPPORTED_INFERENCE_PASS",
        },
    ]


def test_supported_v3_cli_returns_two_for_invalid_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "invalid.json"
    config.write_text("{}\n", encoding="utf-8")

    assert main(["audit", "--config", str(config)]) == 2
    assert "config identity is invalid" in capsys.readouterr().err


def test_supported_v3_script_starts_directly_outside_pytest_import_path() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/evaluation/prepare_ovi_rescene_supported_v3.py"
    )

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "{build,audit}" in completed.stdout
