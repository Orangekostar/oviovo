from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation import run_oviv2_t1_reference as worker


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path.resolve()


def _audit() -> dict[str, object]:
    return {
        "inventory": [{"path": "x", "sha256": "d" * 64, "byte_count": 1}],
        "checkpoint_frames": [2, 7],
        "root_sha256": "e" * 64,
    }


def _development_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    scene: str = "apartment",
    runner=None,
) -> tuple[dict[str, object], Path, Path, list[str]]:
    config = _write_json(
        tmp_path / "config.json",
        {"scene": scene, "algorithm_hash": "a" * 64},
    )
    source = _write_json(tmp_path / "source.json", {"files": {}})
    output = (tmp_path / "output").resolve()
    receipt = output / "t1_exact_receipt.json"
    argv = [
        "/env/bin/python",
        str(Path(worker.__file__).resolve()),
        "--config",
        str(config),
        "--output",
        str(output),
        "--receipt",
        str(receipt),
        "--source-manifest",
        str(source),
    ]
    monkeypatch.setattr(worker, "_commit", lambda repo: "b" * 40)
    monkeypatch.setattr(worker, "compare_cumulative_artifacts", lambda left, right: _audit())

    if runner is None:
        def runner(config_path: Path, output_path: Path, **kwargs: object) -> dict[str, object]:
            assert config_path == config
            assert output_path == output
            assert kwargs == {"freeze_manifest": None, "run_slot": None}
            output.mkdir()
            return {}

    payload = worker.run_development_reference(
        config=config,
        output=output,
        receipt=receipt,
        argv=argv,
        repo=tmp_path,
        source_manifest=source,
        runner=runner,
    )
    return payload, config, source, argv


def test_development_reference_runs_unfrozen_apartment_and_records_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, _, source, argv = _development_call(tmp_path, monkeypatch)

    assert payload["execution"] == {
        "profile": "reference",
        "mode": worker.DEVELOPMENT_MODE,
        "argv": argv,
        "pid": payload["execution"]["pid"],
        "code_commit": "b" * 40,
        "source_manifest_sha256": worker._sha256(source.read_bytes()),
        "input_fingerprints": payload["execution"]["input_fingerprints"],
        "output_root": str((tmp_path / "output").resolve()),
    }
    assert payload["artifact_inventory"] == _audit()["inventory"]
    assert json.loads((tmp_path / "output/t1_exact_receipt.json").read_text()) == payload


def test_development_reference_rejects_office_before_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def runner(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    with pytest.raises(ValueError, match="Apartment|apartment"):
        _development_call(tmp_path, monkeypatch, scene="office", runner=runner)
    assert not called


def test_development_reference_rejects_freeze_flag_in_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_json(tmp_path / "config.json", {"scene": "apartment"})
    source = _write_json(tmp_path / "source.json", {"files": {}})
    output = (tmp_path / "output").resolve()
    receipt = output / "t1_exact_receipt.json"
    argv = [
        "/env/bin/python", str(Path(worker.__file__).resolve()),
        "--config", str(config), "--output", str(output),
        "--freeze-manifest", str((tmp_path / "freeze.json").resolve()),
        "--receipt", str(receipt), "--source-manifest", str(source),
    ]
    monkeypatch.setattr(worker, "_commit", lambda repo: "b" * 40)
    with pytest.raises(ValueError, match="canonical"):
        worker.run_development_reference(
            config=config, output=output, receipt=receipt, argv=argv,
            repo=tmp_path, source_manifest=source, runner=lambda *args, **kwargs: {},
        )


def test_development_reference_rejects_config_changed_during_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def runner(config: Path, output: Path, **kwargs: object) -> dict[str, object]:
        del kwargs
        output.mkdir()
        config.write_text('{"scene":"apartment","changed":true}\n')
        return {}

    with pytest.raises(ValueError, match="config changed"):
        _development_call(tmp_path, monkeypatch, runner=runner)


def test_cli_is_development_only_and_rejects_freeze_options() -> None:
    args = worker.parse_args(
        [
            "--config", "/tmp/config.json", "--output", "/tmp/output",
            "--receipt", "/tmp/output/t1_exact_receipt.json",
            "--source-manifest", "/tmp/source.json",
        ]
    )
    assert vars(args) == {
        "config": Path("/tmp/config.json"),
        "output": Path("/tmp/output"),
        "receipt": Path("/tmp/output/t1_exact_receipt.json"),
        "source_manifest": Path("/tmp/source.json"),
    }
    with pytest.raises(SystemExit):
        worker.parse_args(
            [
                "--config", "/tmp/config.json", "--output", "/tmp/output",
                "--freeze-manifest", "/tmp/freeze.json", "--run-slot", "apartment_run1",
                "--receipt", "/tmp/output/t1_exact_receipt.json",
                "--source-manifest", "/tmp/source.json",
            ]
        )


def test_frozen_run_reference_contract_remains_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_json(tmp_path / "config.json", {"scene": "apartment"})
    source = _write_json(tmp_path / "source.json", {"files": {}})
    freeze = _write_json(tmp_path / "freeze.json", {"shared_bindings": {}})
    output = (tmp_path / "output").resolve()
    receipt = output / "t1_exact_receipt.json"
    argv = [
        "/env/bin/python", str(Path(worker.__file__).resolve()),
        "--config", str(config), "--output", str(output),
        "--freeze-manifest", str(freeze), "--run-slot", "apartment_run1",
        "--receipt", str(receipt), "--source-manifest", str(source),
    ]
    seen: list[dict[str, object]] = []

    def runner(config_path: Path, output_path: Path, **kwargs: object) -> dict[str, object]:
        del config_path
        output_path.mkdir()
        seen.append(kwargs)
        return {}

    monkeypatch.setattr(worker, "_commit", lambda repo: "b" * 40)
    monkeypatch.setattr(worker, "compare_cumulative_artifacts", lambda left, right: _audit())
    worker.run_reference(
        config=config, output=output, freeze_manifest=freeze,
        run_slot="apartment_run1", receipt=receipt, argv=argv,
        repo=tmp_path, source_manifest=source, runner=runner,
    )
    assert seen == [{"freeze_manifest": freeze, "run_slot": "apartment_run1"}]
