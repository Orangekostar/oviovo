from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluation.freeze_oviv2_tesse_baseline_evidence import (
    freeze_baseline_evidence,
)
from scripts.evaluation.package_oviv2_tesse_dual_readout_result import (
    _baseline_values,
    _snapshot,
)


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    return path


def _package(root: Path, name: str, metrics: dict[str, float]) -> Path:
    return _write(
        root / f"{name}.json",
        {
            "schema_version": 1,
            "manifest_id": "tesse-cd-frozen-baseline-scene-result-v1",
            "dataset": "TESSE-CD",
            "protocol_id": "oviv2-tessecd-v2",
            "scene": "apartment",
            "status": "PASS",
            "method_id": name,
            "oracle": False,
            "metrics": metrics,
        },
    )


def _sources(tmp_path: Path) -> list[Path]:
    return [
        _package(
            tmp_path,
            "release",
            {"current_miou": 0.3, "object_f1": 0.7},
        ),
        _package(
            tmp_path,
            "strongest",
            {
                "dynamic_f1": 0.7,
                "change_f1": 0.5,
                "ghost_rate": 0.2,
                "background_f5_cm": 0.2,
                "recovery_frames": 200.0,
            },
        ),
    ]


def test_freezes_exact_source_bound_evidence_consumable_by_packager(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path / "sources")
    output = tmp_path / "publication/baseline_evidence.json"

    result = freeze_baseline_evidence(packages=sources, output=output)

    snapshot = _snapshot(output, "baseline evidence")
    values = _baseline_values(snapshot, [])
    assert values == {
        "dynamic_f1": 0.7,
        "change_f1": 0.5,
        "ghost_rate": 0.2,
        "background_f5_cm": 0.2,
        "recovery_frames": 200.0,
        "current_miou": 0.3,
        "object_f1": 0.7,
    }
    for entry in result["baselines"].values():
        source = entry["source"]
        source_path = Path(source["path"])
        content = source_path.read_bytes()
        assert source == {
            "path": str(source_path.absolute()),
            "sha256": hashlib.sha256(content).hexdigest(),
            "byte_count": len(content),
        }


def test_rejects_duplicate_metric_across_packages(tmp_path: Path) -> None:
    sources = _sources(tmp_path / "sources")
    duplicate = _package(tmp_path / "sources", "duplicate", {"dynamic_f1": 0.8})

    with pytest.raises(ValueError, match="duplicate baseline metric: dynamic_f1"):
        freeze_baseline_evidence(
            packages=[*sources, duplicate], output=tmp_path / "evidence.json"
        )


def test_rejects_missing_or_non_frozen_package(tmp_path: Path) -> None:
    sources = _sources(tmp_path / "sources")
    payload = json.loads(sources[0].read_text())
    payload["oracle"] = True
    _write(sources[0], payload)

    with pytest.raises(ValueError, match="identity/schema"):
        freeze_baseline_evidence(
            packages=sources, output=tmp_path / "evidence.json"
        )


@pytest.mark.parametrize(("metric", "value"), [("ghost_rate", 1.1), ("recovery_frames", -1.0)])
def test_rejects_baseline_metric_outside_consumer_domain(
    tmp_path: Path, metric: str, value: float
) -> None:
    sources = _sources(tmp_path / "sources")
    target = sources[1]
    payload = json.loads(target.read_text())
    payload["metrics"][metric] = value
    _write(target, payload)
    with pytest.raises(ValueError, match="baseline metric domain"):
        freeze_baseline_evidence(packages=sources, output=tmp_path / "evidence.json")


def test_rejects_no_clobber_and_hash_drift(tmp_path: Path) -> None:
    sources = _sources(tmp_path / "sources")
    output = tmp_path / "evidence.json"
    freeze_baseline_evidence(packages=sources, output=output)
    with pytest.raises(FileExistsError):
        freeze_baseline_evidence(packages=sources, output=output)

    evidence = json.loads(output.read_text())
    source = output.parent / evidence["baselines"]["current_miou"]["source"]["path"]
    source.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _baseline_values(_snapshot(output, "baseline evidence"), [])
