from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.evaluation.slice_3rscan_t2_selection as cli


def _source_manifest(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    pairs = [
        {
            "pair_id": "scene0001_00-scene0001_01",
            "role": "development_selection",
            "nested": {"values": [1, {"keep": True}]},
        },
        {
            "pair_id": "scene0002_00-scene0002_01",
            "role": "development_selection",
            "nested": {"values": [2, {"keep": False}]},
        },
    ]
    payload: dict[str, object] = {
        "artifact_id": "RSCAN_T2_DEV_V1",
        "status": "PASS",
        "pairs": pairs,
        "extra_source_field": {"must": "not be copied at top level"},
    }
    source = tmp_path / "selection.json"
    source.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return source, payload


def test_slice_preserves_requested_order_and_source_binding(tmp_path: Path) -> None:
    source, payload = _source_manifest(tmp_path)
    output = tmp_path / "transfer.json"
    pair_ids = [
        "scene0002_00-scene0002_01",
        "scene0001_00-scene0001_01",
    ]

    result = cli.slice_selection_manifest(
        source,
        pair_ids,
        "transfer_only",
        output,
    )

    source_bytes = source.read_bytes()
    assert set(result) == {
        "schema_version",
        "artifact_id",
        "status",
        "role",
        "source_manifest",
        "selected_pair_count",
        "pair_ids",
        "pairs",
    }
    assert result["schema_version"] == 1
    assert result["artifact_id"] == "RSCAN_T2_SELECTION_SLICE_V1"
    assert result["status"] == "PASS"
    assert result["role"] == "transfer_only"
    assert result["selected_pair_count"] == 2
    assert result["pair_ids"] == pair_ids
    assert result["pairs"] == [payload["pairs"][1], payload["pairs"][0]]
    assert result["source_manifest"] == {
        "path": str(source.absolute()),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "byte_count": len(source_bytes),
    }
    expected_bytes = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    assert output.read_bytes() == expected_bytes


def test_slice_rejects_duplicate_pair_ids(tmp_path: Path) -> None:
    source, _ = _source_manifest(tmp_path)

    with pytest.raises(ValueError, match="unique"):
        cli.slice_selection_manifest(
            source,
            ["scene0001_00-scene0001_01", "scene0001_00-scene0001_01"],
            "transfer_only",
            tmp_path / "transfer.json",
        )


def test_slice_rejects_unknown_pair_id(tmp_path: Path) -> None:
    source, _ = _source_manifest(tmp_path)

    with pytest.raises(ValueError, match="not found"):
        cli.slice_selection_manifest(
            source,
            ["scene9999_00-scene9999_01"],
            "transfer_only",
            tmp_path / "transfer.json",
        )


def test_slice_refuses_existing_output_without_clobbering(tmp_path: Path) -> None:
    source, _ = _source_manifest(tmp_path)
    output = tmp_path / "transfer.json"
    output.write_text("keep\n", encoding="utf-8")

    with pytest.raises((FileExistsError, ValueError), match="exist"):
        cli.slice_selection_manifest(source, ["scene0001_00-scene0001_01"], "transfer_only", output)

    assert output.read_text(encoding="utf-8") == "keep\n"


def test_cli_forwards_arguments_and_prints_success_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "output.json"
    observed: dict[str, object] = {}

    def fake_slice(
        source_path: str | Path,
        pair_ids: list[str],
        role: str,
        output_path: str | Path,
    ) -> dict[str, object]:
        observed.update(
            source=source_path,
            pair_ids=pair_ids,
            role=role,
            output=output_path,
        )
        return {"status": "PASS", "selected_pair_count": 2}

    monkeypatch.setattr(cli, "slice_selection_manifest", fake_slice)

    assert cli.main(
        [
            "--source",
            str(source),
            "--pair-id",
            "pair-b",
            "--pair-id",
            "pair-a",
            "--role",
            "transfer_only",
            "--output",
            str(output),
        ]
    ) == 0

    assert observed == {
        "source": source,
        "pair_ids": ["pair-b", "pair-a"],
        "role": "transfer_only",
        "output": output,
    }
    assert json.loads(capsys.readouterr().out) == {
        "output": str(output.absolute()),
        "selected_pair_count": 2,
        "status": "PASS",
    }


def test_cli_returns_two_and_reports_input_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, _ = _source_manifest(tmp_path)

    result = cli.main(
        [
            "--source",
            str(source),
            "--pair-id",
            "missing-pair",
            "--role",
            "transfer_only",
            "--output",
            str(tmp_path / "transfer.json"),
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.err.startswith("ERROR:")
    assert captured.out == ""
