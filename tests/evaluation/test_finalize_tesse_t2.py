from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.evaluation import finalize_tesse_t2


CONFIG_SHA256 = hashlib.sha256(b"config.yaml").hexdigest()


def _run_identity(run_id: str = "dualmap-temporal-test") -> dict[str, str]:
    return {"run_id": run_id, "config_sha256": CONFIG_SHA256}


def test_finalize_tesse_t2_module_exists() -> None:
    assert importlib.util.find_spec("scripts.evaluation.finalize_tesse_t2") is not None


def _scene(
    scene: str,
    method: str,
    mode: str,
    *,
    source_root: Path | None = None,
    run_id: str = "dualmap-temporal-test",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "PARTIAL",
        "dataset": "TESSE-CD",
        "scene": scene,
        "split": f"{scene}_test",
        "method": method,
        "mode": mode,
        "run_identity": _run_identity(run_id),
        "metrics": {
            "object_f1": 0.1,
            "dynamic_f1": 0.2,
            "change_f1": None,
        },
        "unavailable": {"change_f1": "no finite change states"},
    }
    if source_root is not None:
        source_root.mkdir(parents=True, exist_ok=True)
        sources = []
        for name in ("static_objects.csv", "dynamic_objects.csv", "background_mesh.csv"):
            path = source_root / name
            path.write_text(f"source={name}\n", encoding="utf-8")
            sources.append(
                {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "byte_count": path.stat().st_size,
                }
            )
        payload["sources"] = sources
    return payload


def test_partial_official_metrics_bind_only_finite_scene_values() -> None:
    metrics, unavailable = finalize_tesse_t2.build_partial_official_metrics(
        _scene("apartment", "OVIMAP_FROZEN", "frozen"),
        _scene("office", "OVIMAP_FROZEN", "frozen"),
        method_key="OVIMAP_FROZEN",
        mode="frozen",
    )

    assert metrics["apartment"] == {
        "object_f1": pytest.approx(0.1),
        "dynamic_f1": pytest.approx(0.2),
        "change_f1": None,
    }
    assert unavailable == {
        "apartment": {"change_f1": "no finite change states"},
        "office": {"change_f1": "no finite change states"},
    }
    bindings = finalize_tesse_t2.partial_official_token_bindings(
        "OVIMAP_FROZEN", metrics
    )
    assert {binding["token"] for binding in bindings} == {
        "T2_OVIMAP_FROZEN_APARTMENT_OBJECT_F1",
        "T2_OVIMAP_FROZEN_APARTMENT_DYNAMIC_F1",
        "T2_OVIMAP_FROZEN_OFFICE_OBJECT_F1",
        "T2_OVIMAP_FROZEN_OFFICE_DYNAMIC_F1",
    }


def test_partial_official_metrics_reject_null_without_reason() -> None:
    apartment = _scene("apartment", "OVIMAP_FROZEN", "frozen")
    apartment["unavailable"] = {}

    with pytest.raises(ValueError, match="unavailable reason"):
        finalize_tesse_t2.build_partial_official_metrics(
            apartment,
            _scene("office", "OVIMAP_FROZEN", "frozen"),
            method_key="OVIMAP_FROZEN",
            mode="frozen",
        )


@pytest.mark.parametrize("value", [True, "0.5", {"value": 0.5}])
def test_partial_official_metrics_rejects_non_numeric_f1_type(
    value: object,
) -> None:
    apartment = _scene("apartment", "OVIMAP_FROZEN", "frozen")
    apartment["metrics"]["object_f1"] = value

    with pytest.raises(ValueError, match="int or float"):
        finalize_tesse_t2.build_partial_official_metrics(
            apartment,
            _scene("office", "OVIMAP_FROZEN", "frozen"),
            method_key="OVIMAP_FROZEN",
            mode="frozen",
        )


def test_partial_official_metrics_accept_khronos_source_for_open_method() -> None:
    apartment = _scene("apartment", "KHRONOS", "open-set")
    apartment["status"] = "PASS"
    apartment["metrics"]["change_f1"] = 0.3
    apartment["unavailable"] = {}

    metrics, unavailable = finalize_tesse_t2.build_partial_official_metrics(
        apartment,
        _scene("office", "KHRONOS", "open-set"),
        method_key="KHRONOS_OPEN",
        mode="open-set",
    )

    assert metrics["apartment"]["change_f1"] == pytest.approx(0.3)
    assert unavailable["office"]["change_f1"] == "no finite change states"
    assert len(
        finalize_tesse_t2.partial_official_token_bindings("KHRONOS_OPEN", metrics)
    ) == 5


def test_partial_official_metrics_accept_legacy_strict_source_without_status() -> None:
    apartment = _scene("apartment", "KHRONOS", "open-set")
    apartment.pop("status")
    apartment["metrics"]["change_f1"] = 0.3
    apartment.pop("unavailable")

    metrics, _ = finalize_tesse_t2.build_partial_official_metrics(
        apartment,
        _scene("office", "KHRONOS", "open-set"),
        method_key="KHRONOS_OPEN",
        mode="open-set",
    )

    assert metrics["apartment"]["change_f1"] == pytest.approx(0.3)


def _status(
    scene: str, *, run_id: str = "dualmap-temporal-test"
) -> dict[str, object]:
    return {
        "status": "PASS",
        "scene": scene,
        "method": "OVIMAP_FROZEN",
        "mode": "frozen",
        "run_identity": _run_identity(run_id),
        "updates_after_freeze": 0,
    }


def _source(path: Path) -> dict[str, str]:
    path.write_text(path.name, encoding="utf-8")
    return {"name": path.stem, "path": str(path)}


def test_build_result_hashes_sources_and_omits_unavailable_bindings(tmp_path) -> None:
    manifest = _source(tmp_path / "manifest.json")
    config = _source(tmp_path / "config.yaml")
    raw = _source(tmp_path / "raw.json")
    provenance = {
        "run_id": "ovimap-frozen-test",
        "upstream_commit": "1" * 40,
        "adapter_commit": "2" * 40,
        "dirty_state_digest": "3" * 64,
        "dataset_manifest": manifest,
        "commands": ["run apartment", "run office"],
        "environment": {"name": "test"},
        "hardware": {"gpu": "test"},
        "configs": [config],
        "weights": [],
        "raw_outputs": [raw],
    }

    result = finalize_tesse_t2._build_result_payload(
        _scene(
            "apartment",
            "OVIMAP_FROZEN",
            "frozen",
            source_root=tmp_path / "apartment_sources",
            run_id="ovimap-frozen-test",
        ),
        _scene(
            "office",
            "OVIMAP_FROZEN",
            "frozen",
            source_root=tmp_path / "office_sources",
            run_id="ovimap-frozen-test",
        ),
        _status("apartment", run_id="ovimap-frozen-test"),
        _status("office", run_id="ovimap-frozen-test"),
        provenance,
        method_key="OVIMAP_FROZEN",
        mode="frozen",
    )

    assert "status" not in result
    assert result["unavailable"]["apartment"]["change_f1"] == "no finite change states"
    assert len(result["token_bindings"]) == 4
    assert result["unavailable_bindings"] == [
        {
            "token": "T2_OVIMAP_FROZEN_APARTMENT_CHANGE_F1",
            "reason_pointer": "/unavailable/apartment/change_f1",
            "evidence_pointer": "/unavailable_evidence/apartment/change_f1",
        },
        {
            "token": "T2_OVIMAP_FROZEN_OFFICE_CHANGE_F1",
            "reason_pointer": "/unavailable/office/change_f1",
            "evidence_pointer": "/unavailable_evidence/office/change_f1",
        },
    ]
    assert result["dataset"]["manifest"]["sha256"]
    assert result["configs"][0]["byte_count"] == len("config.yaml")
    assert result["run_identity"] == {
        "apartment": _run_identity("ovimap-frozen-test"),
        "office": _run_identity("ovimap-frozen-test"),
    }


def test_build_result_rejects_frozen_updates_after_freeze(tmp_path) -> None:
    office_status = _status("office", run_id="ovimap-frozen-test")
    office_status["updates_after_freeze"] = 1
    provenance = {
        "run_id": "ovimap-frozen-test",
        "upstream_commit": "1" * 40,
        "adapter_commit": "2" * 40,
        "dirty_state_digest": "3" * 64,
        "dataset_manifest": _source(tmp_path / "manifest.json"),
        "commands": ["run"],
        "environment": {},
        "hardware": {},
        "configs": [_source(tmp_path / "config.yaml")],
        "weights": [],
        "raw_outputs": [_source(tmp_path / "raw.json")],
    }

    with pytest.raises(ValueError, match="updates_after_freeze"):
        finalize_tesse_t2._build_result_payload(
            _scene(
                "apartment",
                "OVIMAP_FROZEN",
                "frozen",
                source_root=tmp_path / "apartment_sources",
                run_id="ovimap-frozen-test",
            ),
            _scene(
                "office",
                "OVIMAP_FROZEN",
                "frozen",
                source_root=tmp_path / "office_sources",
                run_id="ovimap-frozen-test",
            ),
            _status("apartment", run_id="ovimap-frozen-test"),
            office_status,
            provenance,
            method_key="OVIMAP_FROZEN",
            mode="frozen",
        )


def test_build_result_separates_khronos_execution_and_table_modes(tmp_path) -> None:
    provenance = {
        "run_id": "khronos-open-test",
        "upstream_commit": "1" * 40,
        "adapter_commit": "2" * 40,
        "dirty_state_digest": "3" * 64,
        "dataset_manifest": _source(tmp_path / "manifest.json"),
        "commands": ["run"],
        "environment": {},
        "hardware": {},
        "configs": [_source(tmp_path / "config.yaml")],
        "weights": [],
        "raw_outputs": [_source(tmp_path / "raw.json")],
    }
    statuses = {
        scene: {
            "status": "PASS",
            "scene": scene,
            "method": "KHRONOS",
            "mode": "open-set",
            "run_identity": _run_identity("khronos-open-test"),
        }
        for scene in ("apartment", "office")
    }

    result = finalize_tesse_t2._build_result_payload(
        _scene(
            "apartment",
            "KHRONOS",
            "open-set",
            source_root=tmp_path / "apartment_sources",
            run_id="khronos-open-test",
        ),
        _scene(
            "office",
            "KHRONOS",
            "open-set",
            source_root=tmp_path / "office_sources",
            run_id="khronos-open-test",
        ),
        statuses["apartment"],
        statuses["office"],
        provenance,
        method_key="KHRONOS_OPEN",
        mode="open-set",
    )

    assert result["method"]["mode"] == "online"
    assert result["run_status"]["apartment"]["mode"] == "open-set"


def test_oviv2_official_finalizer_uses_causal_execution_and_online_table_mode(
    tmp_path: Path,
) -> None:
    provenance = _provenance(tmp_path)
    scenes = {
        scene: _scene(
            scene,
            "OVIV2",
            "causal_checkpoints",
            source_root=tmp_path / f"{scene}_sources",
        )
        for scene in ("apartment", "office")
    }
    statuses = {
        scene: {
            "status": "PASS",
            "scene": scene,
            "method": "OVIV2",
            "mode": "causal_checkpoints",
            "run_identity": _run_identity(),
        }
        for scene in ("apartment", "office")
    }

    result = finalize_tesse_t2._build_result_payload(
        scenes["apartment"],
        scenes["office"],
        statuses["apartment"],
        statuses["office"],
        provenance,
        method_key="OVIV2",
        mode="causal_checkpoints",
    )

    assert finalize_tesse_t2.METHOD_MODES["OVIV2"] == "causal_checkpoints"
    assert finalize_tesse_t2.TABLE_MODES["OVIV2"] == "online"
    assert result["method"] == {
        "key": "OVIV2",
        "display_label": "OVIV2",
        "mode": "online",
        "eligible_for_ranking": True,
    }
    all_bindings = result["token_bindings"] + result["unavailable_bindings"]
    assert len(all_bindings) == 6
    assert all(binding["token"].startswith("T2_OVIV2_") for binding in all_bindings)


def test_official_finalizer_rejects_legacy_oviovo_method_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported .*method"):
        finalize_tesse_t2._build_result_payload(
            _scene("apartment", "OVIOVO", "causal_checkpoints"),
            _scene("office", "OVIOVO", "causal_checkpoints"),
            {
                "status": "PASS",
                "scene": "apartment",
                "method": "OVIOVO",
                "mode": "causal_checkpoints",
            },
            {
                "status": "PASS",
                "scene": "office",
                "method": "OVIOVO",
                "mode": "causal_checkpoints",
            },
            _provenance(tmp_path),
            method_key="OVIOVO",
            mode="causal_checkpoints",
        )


def _dualmap_scene(scene: str, source_root: Path) -> dict[str, object]:
    payload = _scene(
        scene,
        "DUALMAP",
        "native",
        source_root=source_root,
    )
    payload["metrics"] = {
        "object_f1": 0.0,
        "dynamic_f1": None,
        "change_f1": None,
    }
    payload["unavailable"] = {
        "dynamic_f1": "Khronos dynamic F1 has no finite states",
        "change_f1": "Khronos change F1 has no finite states",
    }
    return payload


def _dualmap_status(scene: str) -> dict[str, object]:
    return {
        "status": "PASS",
        "scene": scene,
        "method": "DUALMAP",
        "mode": "native",
        "run_identity": _run_identity(),
    }


def _write_relative_scene_inputs(
    root: Path, *, scene: str = "apartment"
) -> tuple[Path, Path]:
    source_root = root / "bridge"
    metrics = _dualmap_scene(scene, source_root)
    for source in metrics["sources"]:
        source["path"] = Path(source["path"]).relative_to(root).as_posix()
    metrics_path = root / "official_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    status_path = root / "run_status.json"
    status_path.write_text(
        json.dumps(_dualmap_status(scene), sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metrics_path, status_path


def test_scene_evidence_resolves_relative_sources_from_metrics_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics_path, status_path = _write_relative_scene_inputs(tmp_path / "run")
    unrelated = tmp_path / "unrelated-cwd"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    evidence = finalize_tesse_t2.build_scene_evidence(
        metrics_path,
        status_path,
        method_key="DUALMAP",
        mode="native",
    )

    sources = evidence["unavailable_evidence"]
    assert sources["dynamic_f1"]["source"]["path"] == "bridge/dynamic_objects.csv"
    assert sources["change_f1"]["source"]["path"] == "bridge/static_objects.csv"


def test_relative_source_bindings_are_identical_across_run_roots(tmp_path: Path) -> None:
    first_paths = _write_relative_scene_inputs(tmp_path / "run-a")
    second_paths = _write_relative_scene_inputs(tmp_path / "run-b")

    evidence = [
        finalize_tesse_t2.build_scene_evidence(
            metrics_path,
            status_path,
            method_key="DUALMAP",
            mode="native",
        )["unavailable_evidence"]
        for metrics_path, status_path in (first_paths, second_paths)
    ]

    assert json.dumps(evidence[0], sort_keys=True) == json.dumps(
        evidence[1], sort_keys=True
    )


def test_scene_evidence_rejects_relative_source_parent_escape(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    metrics_path, status_path = _write_relative_scene_inputs(run)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    outside = tmp_path / "outside"
    outside.mkdir()
    escaped = outside / "dynamic_objects.csv"
    escaped.write_text("escaped", encoding="utf-8")
    for source in payload["sources"]:
        if Path(source["path"]).name == escaped.name:
            source.update(
                path="../outside/dynamic_objects.csv",
                sha256=hashlib.sha256(escaped.read_bytes()).hexdigest(),
                byte_count=escaped.stat().st_size,
            )
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="canonical|escape|relative"):
        finalize_tesse_t2.build_scene_evidence(
            metrics_path,
            status_path,
            method_key="DUALMAP",
            mode="native",
        )


@pytest.mark.parametrize("link_kind", ["intermediate", "terminal"])
def test_scene_evidence_rejects_symlinked_relative_source(
    tmp_path: Path, link_kind: str
) -> None:
    run = tmp_path / "run"
    metrics_path, status_path = _write_relative_scene_inputs(run)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    dynamic = run / "bridge" / "dynamic_objects.csv"
    if link_kind == "intermediate":
        actual = run / "actual"
        (run / "bridge").rename(actual)
        (run / "bridge").symlink_to(actual, target_is_directory=True)
    else:
        target = run / "bridge" / "dynamic-target.csv"
        dynamic.rename(target)
        dynamic.symlink_to(target)
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="symbolic link|symlink"):
        finalize_tesse_t2.build_scene_evidence(
            metrics_path,
            status_path,
            method_key="DUALMAP",
            mode="native",
        )


def test_declared_source_rejects_path_replacement_while_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.csv"
    source.write_bytes(b"original")
    replacement = tmp_path / "replacement.csv"
    replacement.write_bytes(b"replaced")
    entry = {
        "path": source.name,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "byte_count": source.stat().st_size,
    }
    original_read = os.read
    replaced = False

    def replace_path(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        if not replaced:
            replaced = True
            replacement.replace(source)
        return original_read(descriptor, size)

    monkeypatch.setattr(finalize_tesse_t2.os, "read", replace_path)

    with pytest.raises(ValueError, match="changed"):
        finalize_tesse_t2._validated_declared_source(
            entry,
            label="test",
            artifact_base=tmp_path,
        )
    assert replaced


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ('{"scene":"apartment","scene":"apartment"}', "duplicate JSON key"),
        ('{"metric":NaN}', "non-finite JSON constant"),
    ],
)
def test_scene_evidence_rejects_noncanonical_metrics_json(
    tmp_path: Path, corruption: str, message: str
) -> None:
    metrics_path = tmp_path / "official_metrics.json"
    metrics_path.write_text(corruption, encoding="utf-8")
    status_path = tmp_path / "run_status.json"
    status_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        finalize_tesse_t2.build_scene_evidence(
            metrics_path,
            status_path,
            method_key="DUALMAP",
            mode="native",
        )


def _panoptic_scene(scene: str, source_root: Path) -> dict[str, object]:
    payload = _dualmap_scene(scene, source_root)
    payload["method"] = "PANOPTIC_SHARED"
    payload["mode"] = "composed"
    return payload


def _panoptic_status(scene: str) -> dict[str, object]:
    return {
        "status": "PASS",
        "scene": scene,
        "method": "PANOPTIC_SHARED",
        "mode": "composed",
        "run_identity": _run_identity(),
    }


def _provenance(tmp_path: Path) -> dict[str, object]:
    return {
        "run_id": "dualmap-temporal-test",
        "upstream_commit": "1" * 40,
        "adapter_commit": "2" * 40,
        "dirty_state_digest": "3" * 64,
        "dataset_manifest": _source(tmp_path / "manifest.json"),
        "commands": ["run apartment", "run office"],
        "environment": {},
        "hardware": {},
        "configs": [_source(tmp_path / "config.yaml")],
        "weights": [],
        "raw_outputs": [_source(tmp_path / "raw.json")],
    }


def test_dualmap_native_preserves_source_bound_unavailable_metrics(tmp_path) -> None:
    result = finalize_tesse_t2._build_result_payload(
        _dualmap_scene("apartment", tmp_path / "apartment_sources"),
        _dualmap_scene("office", tmp_path / "office_sources"),
        _dualmap_status("apartment"),
        _dualmap_status("office"),
        _provenance(tmp_path),
        method_key="DUALMAP",
        mode="native",
    )

    assert result["metrics"]["apartment"] == {
        "object_f1": 0.0,
        "dynamic_f1": None,
        "change_f1": None,
    }
    apartment = result["unavailable_evidence"]["apartment"]
    assert Path(apartment["dynamic_f1"]["source"]["path"]).name == "dynamic_objects.csv"
    assert Path(apartment["change_f1"]["source"]["path"]).name == "static_objects.csv"
    assert apartment["dynamic_f1"]["source"]["sha256"]
    assert apartment["change_f1"]["source"]["byte_count"] > 0
    assert {binding["token"] for binding in result["token_bindings"]} == {
        "T2_DUALMAP_APARTMENT_OBJECT_F1",
        "T2_DUALMAP_OFFICE_OBJECT_F1",
    }
    assert all(
        binding["evidence_pointer"].startswith("/unavailable_evidence/")
        for binding in result["unavailable_bindings"]
    )


def test_panoptic_composed_preserves_source_bound_unavailable_metrics(tmp_path) -> None:
    result = finalize_tesse_t2._build_result_payload(
        _panoptic_scene("apartment", tmp_path / "apartment_sources"),
        _panoptic_scene("office", tmp_path / "office_sources"),
        _panoptic_status("apartment"),
        _panoptic_status("office"),
        _provenance(tmp_path),
        method_key="PANOPTIC_SHARED",
        mode="composed",
    )

    assert result["method"]["mode"] == "composed"
    assert result["metrics"]["office"] == {
        "object_f1": 0.0,
        "dynamic_f1": None,
        "change_f1": None,
    }
    assert {binding["token"] for binding in result["token_bindings"]} == {
        "T2_PANOPTIC_SHARED_APARTMENT_OBJECT_F1",
        "T2_PANOPTIC_SHARED_OFFICE_OBJECT_F1",
    }
    office = result["unavailable_evidence"]["office"]
    assert Path(office["dynamic_f1"]["source"]["path"]).name == "dynamic_objects.csv"
    assert Path(office["change_f1"]["source"]["path"]).name == "static_objects.csv"


def test_dualmap_native_rejects_tampered_unavailable_metric_source(tmp_path) -> None:
    apartment = _dualmap_scene("apartment", tmp_path / "apartment_sources")
    dynamic = next(
        entry
        for entry in apartment["sources"]
        if Path(entry["path"]).name == "dynamic_objects.csv"
    )
    dynamic_path = Path(dynamic["path"])
    changed = bytearray(dynamic_path.read_bytes())
    changed[0] ^= 1
    dynamic_path.write_bytes(changed)

    with pytest.raises(ValueError, match="dynamic_f1.*SHA256"):
        finalize_tesse_t2._build_result_payload(
            apartment,
            _dualmap_scene("office", tmp_path / "office_sources"),
            _dualmap_status("apartment"),
            _dualmap_status("office"),
            _provenance(tmp_path),
            method_key="DUALMAP",
            mode="native",
        )


def test_dualmap_scene_evidence_is_hash_bound_and_mergeable(tmp_path) -> None:
    apartment = _dualmap_scene("apartment", tmp_path / "apartment_sources")
    metrics_path = tmp_path / "official_metrics.json"
    metrics_path.write_text(
        json.dumps(apartment, sort_keys=True) + "\n", encoding="utf-8"
    )
    status = _dualmap_status("apartment")
    status_path = tmp_path / "run_status.json"
    status_path.write_text(json.dumps(status, sort_keys=True) + "\n", encoding="utf-8")

    evidence = finalize_tesse_t2.build_scene_evidence(
        metrics_path,
        status_path,
        method_key="DUALMAP",
        mode="native",
    )

    assert evidence["status"] == "PASS"
    assert evidence["scene"] == "apartment"
    assert evidence["official_metrics"] == apartment
    assert evidence["run_status"] == status
    assert evidence["run_identity"] == _run_identity()
    assert evidence["metrics"]["dynamic_f1"] is None
    assert evidence["official_metrics_source"]["sha256"] == hashlib.sha256(
        metrics_path.read_bytes()
    ).hexdigest()
    assert Path(
        evidence["unavailable_evidence"]["dynamic_f1"]["source"]["path"]
    ).name == "dynamic_objects.csv"


def test_scene_evidence_rejects_metrics_status_run_identity_mismatch(
    tmp_path: Path,
) -> None:
    metrics = _dualmap_scene("apartment", tmp_path / "apartment_sources")
    status = _dualmap_status("apartment")
    status["run_identity"] = _run_identity("run-b")
    metrics_path = tmp_path / "official_metrics.json"
    metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
    status_path = tmp_path / "run_status.json"
    status_path.write_text(json.dumps(status) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="run identity mismatch"):
        finalize_tesse_t2.build_scene_evidence(
            metrics_path,
            status_path,
            method_key="DUALMAP",
            mode="native",
        )


def test_scene_evidence_normalizes_run_identity_extra_fields(
    tmp_path: Path,
) -> None:
    metrics = _dualmap_scene("apartment", tmp_path / "apartment_sources")
    status = _dualmap_status("apartment")
    metrics["run_identity"]["metrics_note"] = "must not propagate"
    status["run_identity"]["status_note"] = "must not propagate"
    metrics_path = tmp_path / "official_metrics.json"
    metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
    status_path = tmp_path / "run_status.json"
    status_path.write_text(json.dumps(status) + "\n", encoding="utf-8")

    evidence = finalize_tesse_t2.build_scene_evidence(
        metrics_path,
        status_path,
        method_key="DUALMAP",
        mode="native",
    )

    assert evidence["run_identity"] == _run_identity()


@pytest.mark.parametrize("missing_from", ["metrics", "status"])
def test_scene_evidence_rejects_missing_run_identity(
    tmp_path: Path, missing_from: str
) -> None:
    metrics = _dualmap_scene("apartment", tmp_path / "apartment_sources")
    status = _dualmap_status("apartment")
    (metrics if missing_from == "metrics" else status).pop("run_identity")
    metrics_path = tmp_path / "official_metrics.json"
    metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
    status_path = tmp_path / "run_status.json"
    status_path.write_text(json.dumps(status) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="run identity.*required"):
        finalize_tesse_t2.build_scene_evidence(
            metrics_path,
            status_path,
            method_key="DUALMAP",
            mode="native",
        )


def test_full_result_rejects_run_identity_not_bound_to_provenance(
    tmp_path: Path,
) -> None:
    apartment = _dualmap_scene("apartment", tmp_path / "apartment_sources")
    office = _dualmap_scene("office", tmp_path / "office_sources")
    apartment_status = _dualmap_status("apartment")
    office_status = _dualmap_status("office")
    for payload in (apartment, office, apartment_status, office_status):
        payload["run_identity"] = _run_identity("different-run")

    with pytest.raises(ValueError, match="provenance run_id"):
        finalize_tesse_t2._build_result_payload(
            apartment,
            office,
            apartment_status,
            office_status,
            _provenance(tmp_path),
            method_key="DUALMAP",
            mode="native",
        )


def test_full_result_rejects_config_identity_not_bound_to_provenance(
    tmp_path: Path,
) -> None:
    apartment = _dualmap_scene("apartment", tmp_path / "apartment_sources")
    office = _dualmap_scene("office", tmp_path / "office_sources")
    apartment_status = _dualmap_status("apartment")
    office_status = _dualmap_status("office")
    wrong_identity = {
        "run_id": "dualmap-temporal-test",
        "config_sha256": "f" * 64,
    }
    apartment["run_identity"] = wrong_identity
    apartment_status["run_identity"] = wrong_identity

    with pytest.raises(ValueError, match="provenance config"):
        finalize_tesse_t2._build_result_payload(
            apartment,
            office,
            apartment_status,
            office_status,
            _provenance(tmp_path),
            method_key="DUALMAP",
            mode="native",
        )


def test_scene_evidence_cli_writes_mergeable_packet(tmp_path) -> None:
    apartment = _dualmap_scene("apartment", tmp_path / "apartment_sources")
    metrics_path = tmp_path / "official_metrics.json"
    metrics_path.write_text(json.dumps(apartment) + "\n", encoding="utf-8")
    status_path = tmp_path / "run_status.json"
    status_path.write_text(
        json.dumps(_dualmap_status("apartment")) + "\n", encoding="utf-8"
    )
    output = tmp_path / "apartment_evidence.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(finalize_tesse_t2.__file__)),
            "--method",
            "DUALMAP",
            "--scene-metrics",
            str(metrics_path),
            "--scene-status",
            str(status_path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["scene"] == "apartment"


@pytest.mark.parametrize("cli_mode", ["scene", "full"])
def test_cli_output_is_atomic_no_clobber(
    tmp_path: Path, cli_mode: str
) -> None:
    output = tmp_path / "result.json"
    output.write_bytes(b"sentinel")
    if cli_mode == "full":
        command = _full_cli_command(_full_cli_inputs(tmp_path), output)
    else:
        metrics = _dualmap_scene("apartment", tmp_path / "apartment_sources")
        metrics_path = tmp_path / "official_metrics.json"
        metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
        status_path = tmp_path / "run_status.json"
        status_path.write_text(
            json.dumps(_dualmap_status("apartment")) + "\n",
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(Path(finalize_tesse_t2.__file__)),
            "--method",
            "DUALMAP",
            "--scene-metrics",
            str(metrics_path),
            "--scene-status",
            str(status_path),
            "--output",
            str(output),
        ]

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode != 0
    assert "FileExistsError" in completed.stderr
    assert output.read_bytes() == b"sentinel"


def test_scene_cli_publication_failure_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics = _dualmap_scene("apartment", tmp_path / "apartment_sources")
    metrics_path = tmp_path / "official_metrics.json"
    metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
    status_path = tmp_path / "run_status.json"
    status_path.write_text(
        json.dumps(_dualmap_status("apartment")) + "\n", encoding="utf-8"
    )
    output = tmp_path / "result.json"

    def fail_link(source: object, target: object) -> None:
        raise OSError("injected publication failure")

    monkeypatch.setattr(finalize_tesse_t2.os, "link", fail_link)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(Path(finalize_tesse_t2.__file__)),
            "--method",
            "DUALMAP",
            "--scene-metrics",
            str(metrics_path),
            "--scene-status",
            str(status_path),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(OSError, match="injected publication failure"):
        finalize_tesse_t2.main()

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*"))


def _write_json_pair(
    root: Path, name: str, payload: dict[str, object]
) -> tuple[Path, Path]:
    primary = root / f"{name}.json"
    repeat = root / f"{name}-repeat.json"
    primary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    repeat.write_bytes(primary.read_bytes())
    return primary, repeat


def _full_cli_inputs(tmp_path: Path) -> dict[str, Path]:
    apartment_metrics, apartment_metrics_repeat = _write_json_pair(
        tmp_path,
        "apartment-metrics",
        _scene(
            "apartment",
            "OVIMAP_FROZEN",
            "frozen",
            source_root=tmp_path / "apartment_sources",
        ),
    )
    office_metrics, office_metrics_repeat = _write_json_pair(
        tmp_path,
        "office-metrics",
        _scene(
            "office",
            "OVIMAP_FROZEN",
            "frozen",
            source_root=tmp_path / "office_sources",
        ),
    )
    apartment_status, apartment_status_repeat = _write_json_pair(
        tmp_path, "apartment-status", _status("apartment")
    )
    office_status, office_status_repeat = _write_json_pair(
        tmp_path, "office-status", _status("office")
    )
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(_provenance(tmp_path), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "apartment_metrics": apartment_metrics,
        "apartment_metrics_repeat": apartment_metrics_repeat,
        "office_metrics": office_metrics,
        "office_metrics_repeat": office_metrics_repeat,
        "apartment_status": apartment_status,
        "apartment_status_repeat": apartment_status_repeat,
        "office_status": office_status,
        "office_status_repeat": office_status_repeat,
        "provenance": provenance,
    }


def _full_cli_command(inputs: dict[str, Path], output: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(finalize_tesse_t2.__file__)),
        "--method",
        "OVIMAP_FROZEN",
        "--apartment-metrics",
        str(inputs["apartment_metrics"]),
        "--apartment-metrics-repeat",
        str(inputs["apartment_metrics_repeat"]),
        "--office-metrics",
        str(inputs["office_metrics"]),
        "--office-metrics-repeat",
        str(inputs["office_metrics_repeat"]),
        "--apartment-status",
        str(inputs["apartment_status"]),
        "--apartment-status-repeat",
        str(inputs["apartment_status_repeat"]),
        "--office-status",
        str(inputs["office_status"]),
        "--office-status-repeat",
        str(inputs["office_status_repeat"]),
        "--provenance",
        str(inputs["provenance"]),
        "--output",
        str(output),
    ]


def test_full_result_resolves_each_relative_source_from_its_metrics_parent(
    tmp_path: Path,
) -> None:
    inputs = _full_cli_inputs(tmp_path / "bundle")
    for scene in ("apartment", "office"):
        primary = inputs[f"{scene}_metrics"]
        repeat = inputs[f"{scene}_metrics_repeat"]
        payload = json.loads(primary.read_text(encoding="utf-8"))
        for source in payload["sources"]:
            source["path"] = Path(source["path"]).relative_to(primary.parent).as_posix()
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        primary.write_bytes(encoded)
        repeat.write_bytes(encoded)

    result = finalize_tesse_t2.build_result(
        **inputs,
        method_key="OVIMAP_FROZEN",
        mode="frozen",
    )

    assert result["unavailable_evidence"]["apartment"]["change_f1"]["source"][
        "path"
    ] == "apartment_sources/static_objects.csv"
    assert result["unavailable_evidence"]["office"]["change_f1"]["source"][
        "path"
    ] == "office_sources/static_objects.csv"


def test_full_cli_requires_repeat_inputs(tmp_path: Path) -> None:
    inputs = _full_cli_inputs(tmp_path)
    command = _full_cli_command(inputs, tmp_path / "result.json")
    for flag in (
        "--apartment-metrics-repeat",
        "--office-metrics-repeat",
        "--apartment-status-repeat",
        "--office-status-repeat",
    ):
        position = command.index(flag)
        del command[position : position + 2]

    completed = subprocess.run(command, capture_output=True, text=True, check=False)

    assert completed.returncode != 0
    assert "repeat" in completed.stderr.lower()


def test_full_cli_hash_binds_primary_and_repeat_evidence(tmp_path: Path) -> None:
    inputs = _full_cli_inputs(tmp_path)
    output = tmp_path / "result.json"

    completed = subprocess.run(
        _full_cli_command(inputs, output),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "VERIFIED"
    assert result["protocol"]["deterministic_repeat"] == "byte-identical"
    for scene in ("apartment", "office"):
        for kind in ("metrics", "status"):
            pair = result["evidence_sources"][scene][kind]
            assert pair["primary"]["sha256"] == pair["repeat"]["sha256"]
            assert pair["primary"]["path"] != pair["repeat"]["path"]
            assert pair["primary"]["byte_count"] > 0
    assert result["evidence_sources"]["provenance"]["sha256"]


def test_full_cli_rejects_nonidentical_repeat_bytes(tmp_path: Path) -> None:
    inputs = _full_cli_inputs(tmp_path)
    inputs["office_status_repeat"].write_bytes(
        inputs["office_status_repeat"].read_bytes() + b" "
    )

    completed = subprocess.run(
        _full_cli_command(inputs, tmp_path / "result.json"),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "byte-identical" in completed.stderr


@pytest.mark.parametrize("link_kind", ["hardlink", "symlink"])
def test_full_cli_rejects_repeat_alias_of_primary(
    tmp_path: Path, link_kind: str
) -> None:
    inputs = _full_cli_inputs(tmp_path)
    primary = inputs["apartment_metrics"]
    repeat = inputs["apartment_metrics_repeat"]
    repeat.unlink()
    if link_kind == "hardlink":
        os.link(primary, repeat)
    else:
        repeat.symlink_to(primary)

    completed = subprocess.run(
        _full_cli_command(inputs, tmp_path / "result.json"),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert (
        "independent files" in completed.stderr
        or "regular file" in completed.stderr
    )


@pytest.mark.parametrize("method", [None, 7])
def test_full_cli_requires_exact_string_status_method(
    tmp_path: Path, method: object
) -> None:
    inputs = _full_cli_inputs(tmp_path)
    payload = json.loads(inputs["apartment_status"].read_text(encoding="utf-8"))
    if method is None:
        payload.pop("method")
    else:
        payload["method"] = method
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    inputs["apartment_status"].write_bytes(encoded)
    inputs["apartment_status_repeat"].write_bytes(encoded)

    completed = subprocess.run(
        _full_cli_command(inputs, tmp_path / "result.json"),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "run method mismatch" in completed.stderr
