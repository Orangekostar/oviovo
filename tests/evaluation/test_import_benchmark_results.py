from __future__ import annotations

import csv
import hashlib
import json
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.evaluation.baselines.ovimap_paper_audit import PAPER_PROTOCOL
from tools.import_benchmark_results import ImportFailure, import_results

FIELDS = [
    "token",
    "table",
    "method",
    "dataset",
    "split",
    "metric",
    "direction",
    "precision",
    "source_json",
    "json_pointer",
    "status",
    "note",
]
SOURCE_SHA_RE = re.compile(r"source_sha256=([0-9a-f]{64})")

T2_COMMON_METRICS = {
    "CURRENT_MIOU": 0.401,
    "GHOST_RATE": 0.052,
    "BG_F5": 0.603,
    "RECOVERY_FRAMES": 8.0,
}
T2_OFFICIAL_METRICS = ("OBJECT_F1", "DYNAMIC_F1", "CHANGE_F1")


def _write_registry(path: Path) -> None:
    rows = [
        {
            "token": "T1_DUALMAP_REPLICA8_MIOU",
            "table": "T1",
            "method": "DUALMAP",
            "dataset": "Replica",
            "split": "replica_8_compat",
            "metric": "REPLICA8_MIOU",
            "direction": "higher",
            "precision": "3",
            "source_json": "",
            "json_pointer": "",
            "status": "UNFILLED",
            "note": "Pending benchmark run.",
        },
        {
            "token": "T1_OVIV2_REPLICA8_MIOU",
            "table": "T1",
            "method": "OVIV2",
            "dataset": "Replica",
            "split": "replica_8_compat",
            "metric": "REPLICA8_MIOU",
            "direction": "higher",
            "precision": "3",
            "source_json": "",
            "json_pointer": "",
            "status": "UNFILLED",
            "note": "Pending benchmark run.",
        },
        {
            "token": "T1_OVIOVO_REPLICA8_MIOU",
            "table": "T1",
            "method": "OVIOVO",
            "dataset": "Replica",
            "split": "replica_8_compat",
            "metric": "REPLICA8_MIOU",
            "direction": "higher",
            "precision": "3",
            "source_json": "",
            "json_pointer": "",
            "status": "UNFILLED",
            "note": "Pending benchmark run.",
        },
        {
            "token": "T1_OVIMAP_REPLICA8_MIOU",
            "table": "T1",
            "method": "OVIMAP",
            "dataset": "Replica",
            "split": "replica_8_compat",
            "metric": "REPLICA8_MIOU",
            "direction": "higher",
            "precision": "3",
            "source_json": "",
            "json_pointer": "",
            "status": "UNFILLED",
            "note": "Pending benchmark run.",
        },
        {
            "token": "T1_OVIMAP_SCANNET5_MIOU",
            "table": "T1",
            "method": "OVIMAP",
            "dataset": "ScanNet200",
            "split": "scannet200_5_heldout",
            "metric": "SCANNET5_MIOU",
            "direction": "higher",
            "precision": "3",
            "source_json": "",
            "json_pointer": "",
            "status": "UNFILLED",
            "note": "Pending benchmark run.",
        },
        {
            "token": "T1_OVIMAP_REPLICA8_FMIOU",
            "table": "T1",
            "method": "OVIMAP",
            "dataset": "Replica",
            "split": "replica_8_compat",
            "metric": "REPLICA8_FMIOU",
            "direction": "higher",
            "precision": "3",
            "source_json": "",
            "json_pointer": "",
            "status": "UNFILLED",
            "note": "Pending benchmark run.",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_result(
    path: Path,
    *,
    token: str = "T1_DUALMAP_REPLICA8_MIOU",
    method: str = "DUALMAP",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "VERIFIED",
                "run_id": "dualmap-replica8-test",
                "method": {"key": method, "display_label": method, "mode": "native"},
                "dataset": {"name": "Replica", "splits": ["replica_8_compat"]},
                "metrics": {"semantic": {"miou": 0.2764}},
                "token_bindings": [
                    {"token": token, "json_pointer": "/metrics/semantic/miou", "precision": 3}
                ],
                "protocol_deviations": [],
            }
        ),
        encoding="utf-8",
    )


def _templates(tmp_path: Path) -> tuple[Path, Path]:
    markdown = tmp_path / "benchmark_tables.md"
    latex = tmp_path / "benchmark_tables.tex"
    markdown.write_text(
        "value={{T1_DUALMAP_REPLICA8_MIOU}} v2={{T1_OVIV2_REPLICA8_MIOU}} "
        "legacy={{T1_OVIOVO_REPLICA8_MIOU}}\n"
    )
    latex.write_text(
        r"value=\verb|{{T1_DUALMAP_REPLICA8_MIOU}}| "
        r"v2=\verb|{{T1_OVIV2_REPLICA8_MIOU}}| legacy=\verb|{{T1_OVIOVO_REPLICA8_MIOU}}|"
        + "\n"
    )
    return markdown, latex


def _write_oviv2_t2_registry(path: Path, *, unavailable_status: str = "UNFILLED") -> list[str]:
    rows = []
    tokens = []
    for metric in T2_COMMON_METRICS:
        token = f"T2_OVIV2_{metric}"
        tokens.append(token)
        rows.append(
            {
                "token": token,
                "table": "T2",
                "method": "OVIV2",
                "dataset": "TESSE-CD",
                "split": "macro_test",
                "metric": metric,
                "direction": "lower" if metric in {"GHOST_RATE", "RECOVERY_FRAMES"} else "higher",
                "precision": "3",
                "source_json": "",
                "json_pointer": "",
                "status": "UNFILLED",
                "note": "Pending benchmark run.",
            }
        )
    for scene in ("APARTMENT", "OFFICE"):
        for metric in T2_OFFICIAL_METRICS:
            token = f"T2_OVIV2_{scene}_{metric}"
            tokens.append(token)
            status = unavailable_status if scene == "OFFICE" else "UNFILLED"
            rows.append(
                {
                    "token": token,
                    "table": "T2",
                    "method": "OVIV2",
                    "dataset": "TESSE-CD",
                    "split": f"{scene.lower()}_test",
                    "metric": f"{scene}_{metric}",
                    "direction": "higher",
                    "precision": "3",
                    "source_json": "already.json" if status != "UNFILLED" else "",
                    "json_pointer": "/metrics/old" if status != "UNFILLED" else "",
                    "status": status,
                    "note": "Existing result." if status != "UNFILLED" else "Pending benchmark run.",
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return tokens


def _write_oviv2_t2_results(tmp_path: Path) -> tuple[Path, Path, Path]:
    evidence = tmp_path / "official_evaluator.json"
    evidence.write_text('{"status":"PASS"}\n', encoding="utf-8")
    source = {
        "path": str(evidence.resolve()),
        "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "byte_count": evidence.stat().st_size,
    }
    common = tmp_path / "common.json"
    common.write_text(
        json.dumps(
            {
                "status": "VERIFIED",
                "run_id": "oviv2-common",
                "method": {"key": "OVIV2", "display_label": "OVIV2", "mode": "online"},
                "dataset": {"name": "TESSE-CD", "splits": ["macro_test"]},
                "metrics": {"macro": T2_COMMON_METRICS},
                "token_bindings": [
                    {
                        "token": f"T2_OVIV2_{metric}",
                        "json_pointer": f"/metrics/macro/{metric}",
                        "precision": 3,
                    }
                    for metric in T2_COMMON_METRICS
                ],
                "unavailable_bindings": [],
            }
        ),
        encoding="utf-8",
    )
    reasons = {metric: f"official evaluator has no finite {metric.lower()}" for metric in T2_OFFICIAL_METRICS}
    official = tmp_path / "official.json"
    official.write_text(
        json.dumps(
            {
                "status": "VERIFIED",
                "run_id": "oviv2-official",
                "method": {"key": "OVIV2", "display_label": "OVIV2", "mode": "online"},
                "dataset": {
                    "name": "TESSE-CD",
                    "splits": ["apartment_test", "office_test"],
                },
                "metrics": {
                    "apartment": {metric: 0.5 for metric in T2_OFFICIAL_METRICS},
                },
                "unavailable": {"office": reasons},
                "unavailable_evidence": {
                    "office": {
                        metric: {"reason": reason, "source": source}
                        for metric, reason in reasons.items()
                    }
                },
                "token_bindings": [
                    {
                        "token": f"T2_OVIV2_APARTMENT_{metric}",
                        "json_pointer": f"/metrics/apartment/{metric}",
                        "precision": 3,
                    }
                    for metric in T2_OFFICIAL_METRICS
                ],
                "unavailable_bindings": [
                    {
                        "token": f"T2_OVIV2_OFFICE_{metric}",
                        "reason_pointer": f"/unavailable/office/{metric}",
                        "evidence_pointer": f"/unavailable_evidence/office/{metric}",
                    }
                    for metric in T2_OFFICIAL_METRICS
                ],
            }
        ),
        encoding="utf-8",
    )
    return common, official, evidence


def _oviv2_t2_package(tmp_path: Path, *, unavailable_status: str = "UNFILLED") -> tuple[Path, Path, Path, Path, Path]:
    registry = tmp_path / "benchmark_tokens.tsv"
    tokens = _write_oviv2_t2_registry(registry, unavailable_status=unavailable_status)
    markdown = tmp_path / "benchmark_tables.md"
    latex = tmp_path / "benchmark_tables.tex"
    markdown.write_text(" ".join(f"{{{{{token}}}}}" for token in tokens), encoding="utf-8")
    latex.write_text(
        " ".join(r"\verb|{{" + token + "}}|" for token in tokens),
        encoding="utf-8",
    )
    common, official, _ = _write_oviv2_t2_results(tmp_path)
    return registry, markdown, latex, common, official


def _write_t4_package(
    tmp_path: Path,
    *,
    method: str,
    result_method: str | None = None,
) -> tuple[Path, Path, Path, Path]:
    token = f"T4_{method}_TOTAL_SPF"
    registry = tmp_path / "benchmark_tokens.tsv"
    with registry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "token": token,
                "table": "T4",
                "method": method,
                "dataset": "local_same_hardware",
                "split": "test",
                "metric": "TOTAL_SPF",
                "direction": "lower",
                "precision": "2",
                "source_json": "",
                "json_pointer": "",
                "status": "UNFILLED",
                "note": "Pending benchmark run.",
            }
        )
    markdown = tmp_path / "benchmark_tables.md"
    latex = tmp_path / "benchmark_tables.tex"
    markdown.write_text("{{" + token + "}}", encoding="utf-8")
    latex.write_text(r"\verb|{{" + token + "}}|", encoding="utf-8")
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "status": "VERIFIED",
                "run_id": "oviv2-t4-test",
                "method": {"key": result_method or method, "mode": "online"},
                "dataset": {"name": "local_same_hardware", "splits": ["test"]},
                "metrics": {"total_spf": 1.234},
                "token_bindings": [
                    {"token": token, "json_pointer": "/metrics/total_spf", "precision": 2}
                ],
            }
        ),
        encoding="utf-8",
    )
    return registry, markdown, latex, result


def test_import_merges_oviv2_t2_common_and_official_bindings(tmp_path: Path) -> None:
    registry, markdown, latex, common, official = _oviv2_t2_package(tmp_path)

    outputs = import_results(
        registry,
        [common, official],
        markdown,
        latex,
        tmp_path / "out.md",
        tmp_path / "out.tex",
    )

    with registry.open(newline="", encoding="utf-8") as handle:
        imported = list(csv.DictReader(handle, delimiter="\t"))
    assert len(imported) == 10
    assert len({row["token"] for row in imported}) == 10
    assert sum(row["status"] == "VERIFIED" for row in imported) == 7
    assert sum(row["status"] == "N/A" for row in imported) == 3
    assert all(row["source_json"] and row["json_pointer"] for row in imported)
    assert "{{T2_OVIV2_" not in outputs["markdown"].read_text(encoding="utf-8")
    assert "{{T2_OVIV2_" not in outputs["latex"].read_text(encoding="utf-8")


def _import_source_bound_t2_na_package(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "paper"
    shutil.copytree(root / "docs" / "paper", output_dir)
    evidence = output_dir / "official-evidence.json"
    evidence.write_text('{"status":"PASS"}\n', encoding="utf-8")
    reason = "official evaluator has no finite object_f1"
    result = output_dir / "oviv2-official.json"
    result.write_text(
        json.dumps(
            {
                "status": "VERIFIED",
                "run_id": "oviv2-official-source-bound",
                "method": {"key": "OVIV2", "mode": "online"},
                "dataset": {"name": "TESSE-CD", "splits": ["office_test"]},
                "unavailable": {"office": {"OBJECT_F1": reason}},
                "unavailable_evidence": {
                    "office": {
                        "OBJECT_F1": {
                            "reason": reason,
                            "source": {
                                "path": str(evidence.resolve()),
                                "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                                "byte_count": evidence.stat().st_size,
                            },
                        }
                    }
                },
                "token_bindings": [],
                "unavailable_bindings": [
                    {
                        "token": "T2_OVIV2_OFFICE_OBJECT_F1",
                        "reason_pointer": "/unavailable/office/OBJECT_F1",
                        "evidence_pointer": "/unavailable_evidence/office/OBJECT_F1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    import_results(
        output_dir / "benchmark_tokens.tsv",
        [result],
        output_dir / "benchmark_tables.md",
        output_dir / "benchmark_tables.tex",
        output_dir / "benchmark_tables_baselines.md",
        output_dir / "benchmark_tables_baselines.tex",
    )
    return root, output_dir, result, evidence


def _run_package_check(root: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "benchmark_tables.py"),
            "check",
            "--output-dir",
            str(output_dir),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )


def _rewrite_dynamic_na_registry(
    output_dir: Path,
    mutation,
) -> None:
    registry_path = output_dir / "benchmark_tokens.tsv"
    with registry_path.open(newline="", encoding="utf-8") as handle:
        registry = list(csv.DictReader(handle, delimiter="\t"))
    row = next(item for item in registry if item["token"] == "T2_OVIV2_OFFICE_OBJECT_F1")
    mutation(row)
    with registry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(registry)


def _refresh_registry_source_sha(output_dir: Path, result: Path) -> None:
    digest = hashlib.sha256(result.read_bytes()).hexdigest()

    def refresh(row: dict[str, str]) -> None:
        if SOURCE_SHA_RE.search(row["note"]):
            row["note"] = SOURCE_SHA_RE.sub(f"source_sha256={digest}", row["note"], count=1)

    _rewrite_dynamic_na_registry(output_dir, refresh)


def test_imported_source_bound_t2_na_passes_full_package_check(tmp_path: Path) -> None:
    root, output_dir, result, _ = _import_source_bound_t2_na_package(tmp_path)
    with (output_dir / "benchmark_tokens.tsv").open(newline="", encoding="utf-8") as handle:
        row = next(
            item
            for item in csv.DictReader(handle, delimiter="\t")
            if item["token"] == "T2_OVIV2_OFFICE_OBJECT_F1"
        )
    source_hashes = SOURCE_SHA_RE.findall(row["note"])
    assert source_hashes == [hashlib.sha256(result.read_bytes()).hexdigest()]

    completed = _run_package_check(root, output_dir)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "benchmark table package: PASS" in completed.stdout


@pytest.mark.parametrize(
    "attack",
    [
        "duplicate_binding",
        "missing_reason_pointer",
        "missing_evidence_pointer",
        "empty_reason",
        "reason_mismatch",
        "evidence_missing_path",
        "evidence_missing_hash",
        "evidence_missing_byte_count",
        "evidence_hash_drift",
        "evidence_count_drift",
        "source_sha_missing",
        "source_sha_mismatch",
        "source_sha_duplicate",
        "source_sha_noncanonical",
    ],
)
def test_package_check_rejects_tampered_source_bound_na(
    tmp_path: Path, attack: str
) -> None:
    root, output_dir, result, evidence = _import_source_bound_t2_na_package(tmp_path)
    payload = json.loads(result.read_text(encoding="utf-8"))
    binding = payload["unavailable_bindings"][0]
    evidence_record = payload["unavailable_evidence"]["office"]["OBJECT_F1"]
    source = evidence_record["source"]
    result_changed = True

    if attack == "duplicate_binding":
        payload["unavailable_bindings"].append(dict(binding))
    elif attack == "missing_reason_pointer":
        binding["reason_pointer"] = "/missing/reason"
        _rewrite_dynamic_na_registry(
            output_dir,
            lambda row: row.update(json_pointer="/missing/reason"),
        )
    elif attack == "missing_evidence_pointer":
        binding["evidence_pointer"] = "/missing/evidence"
    elif attack == "empty_reason":
        payload["unavailable"]["office"]["OBJECT_F1"] = " "
        evidence_record["reason"] = " "
    elif attack == "reason_mismatch":
        evidence_record["reason"] = "different unavailable reason"
    elif attack == "evidence_missing_path":
        source.pop("path")
    elif attack == "evidence_missing_hash":
        source.pop("sha256")
    elif attack == "evidence_missing_byte_count":
        source.pop("byte_count")
    elif attack == "evidence_hash_drift":
        evidence.write_text('{"status":"FAIL"}\n', encoding="utf-8")
        result_changed = False
    elif attack == "evidence_count_drift":
        evidence.write_text(evidence.read_text(encoding="utf-8") + "x", encoding="utf-8")
        source["sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    else:
        result_changed = False

    if result_changed:
        result.write_text(json.dumps(payload), encoding="utf-8")
        _refresh_registry_source_sha(output_dir, result)

    if attack.startswith("source_sha_"):
        def mutate_source_sha(row: dict[str, str]) -> None:
            match = SOURCE_SHA_RE.search(row["note"])
            if match is None:
                return
            field = match.group(0)
            if attack == "source_sha_missing":
                row["note"] = row["note"].replace(f" [{field}]", "")
            elif attack == "source_sha_mismatch":
                row["note"] = row["note"].replace(field, f"source_sha256={'0' * 64}")
            elif attack == "source_sha_duplicate":
                row["note"] = row["note"].replace(field, f"{field} {field}")
            else:
                row["note"] = row["note"].replace(field, field.upper())

        _rewrite_dynamic_na_registry(output_dir, mutate_source_sha)

    completed = _run_package_check(root, output_dir)

    assert completed.returncode != 0
    assert "benchmark_tokens.tsv" in completed.stderr


@pytest.mark.parametrize("method", ["OVIV2_STATIC", "OVIV2"])
def test_import_accepts_t4_oviv2_methods(tmp_path: Path, method: str) -> None:
    registry, markdown, latex, result = _write_t4_package(tmp_path, method=method)

    outputs = import_results(
        registry,
        [result],
        markdown,
        latex,
        tmp_path / "out.md",
        tmp_path / "out.tex",
    )

    assert outputs["markdown"].read_text(encoding="utf-8") == "1.23"
    with registry.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert row["status"] == "VERIFIED"


@pytest.mark.parametrize(
    ("method", "wrong_method"),
    [("OVIV2_STATIC", "OVIV2"), ("OVIV2", "OVIV2_STATIC")],
)
def test_import_rejects_wrong_t4_oviv2_method(
    tmp_path: Path, method: str, wrong_method: str
) -> None:
    registry, markdown, latex, result = _write_t4_package(
        tmp_path,
        method=method,
        result_method=wrong_method,
    )

    with pytest.raises(ImportFailure, match="method mismatch"):
        import_results(registry, [result], markdown, latex, tmp_path / "out.md", tmp_path / "out.tex")


def test_import_rejects_legacy_t2_oviovo_binding(tmp_path: Path) -> None:
    registry, markdown, latex, common, _ = _oviv2_t2_package(tmp_path)
    payload = json.loads(common.read_text(encoding="utf-8"))
    payload["token_bindings"][0]["token"] = "T2_OVIOVO_CURRENT_MIOU"
    common.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ImportFailure, match="OVIOVO"):
        import_results(registry, [common], markdown, latex, tmp_path / "out.md", tmp_path / "out.tex")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("method", "DUALMAP", "method mismatch"),
        ("split", "wrong_test", "dataset or split mismatch"),
    ],
)
def test_import_rejects_oviv2_t2_method_or_split_mismatch(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    registry, markdown, latex, common, _ = _oviv2_t2_package(tmp_path)
    payload = json.loads(common.read_text(encoding="utf-8"))
    if field == "method":
        payload["method"]["key"] = value
    else:
        payload["dataset"]["splits"] = [value]
    common.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ImportFailure, match=message):
        import_results(registry, [common], markdown, latex, tmp_path / "out.md", tmp_path / "out.tex")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_pointer", "evidence pointer"),
        ("missing_hash", "source hash"),
        ("wrong_hash", "source hash"),
        ("wrong_byte_count", "source hash"),
        ("non_integer_byte_count", "byte count"),
    ],
)
def test_import_rejects_unhashed_or_unbound_unavailable_evidence(
    tmp_path: Path, mutation: str, message: str
) -> None:
    registry, markdown, latex, _, official = _oviv2_t2_package(tmp_path)
    payload = json.loads(official.read_text(encoding="utf-8"))
    binding = payload["unavailable_bindings"][0]
    evidence = payload["unavailable_evidence"]["office"]["OBJECT_F1"]["source"]
    if mutation == "missing_pointer":
        binding.pop("evidence_pointer")
    elif mutation == "missing_hash":
        evidence.pop("sha256")
    elif mutation == "wrong_hash":
        evidence["sha256"] = "0" * 64
    elif mutation == "wrong_byte_count":
        evidence["byte_count"] += 1
    else:
        evidence["byte_count"] = float(evidence["byte_count"])
    official.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ImportFailure, match=message):
        import_results(registry, [official], markdown, latex, tmp_path / "out.md", tmp_path / "out.tex")


def test_import_rejects_empty_unavailable_reason(tmp_path: Path) -> None:
    registry, markdown, latex, _, official = _oviv2_t2_package(tmp_path)
    payload = json.loads(official.read_text(encoding="utf-8"))
    payload["unavailable"]["office"]["OBJECT_F1"] = ""
    payload["unavailable_evidence"]["office"]["OBJECT_F1"]["reason"] = ""
    official.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ImportFailure, match="non-empty"):
        import_results(registry, [official], markdown, latex, tmp_path / "out.md", tmp_path / "out.tex")


def test_import_rejects_finite_and_unavailable_t2_binding_for_same_token(tmp_path: Path) -> None:
    registry, markdown, latex, _, official = _oviv2_t2_package(tmp_path)
    payload = json.loads(official.read_text(encoding="utf-8"))
    payload["metrics"]["office"] = {"OBJECT_F1": 0.5}
    payload["token_bindings"].append(
        {
            "token": "T2_OVIV2_OFFICE_OBJECT_F1",
            "json_pointer": "/metrics/office/OBJECT_F1",
            "precision": 3,
        }
    )
    official.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ImportFailure, match="duplicate token binding"):
        import_results(registry, [official], markdown, latex, tmp_path / "out.md", tmp_path / "out.tex")


def test_import_only_sets_source_bound_na_on_unfilled_registry_row(tmp_path: Path) -> None:
    registry, markdown, latex, _, official = _oviv2_t2_package(
        tmp_path, unavailable_status="VERIFIED"
    )

    with pytest.raises(ImportFailure, match="UNFILLED"):
        import_results(registry, [official], markdown, latex, tmp_path / "out.md", tmp_path / "out.tex")


def test_import_reads_json_pointer_updates_registry_and_derives_tables(tmp_path) -> None:
    registry = tmp_path / "benchmark_tokens.tsv"
    result = tmp_path / "results" / "result.json"
    markdown, latex = _templates(tmp_path)
    _write_registry(registry)
    _write_result(result)

    outputs = import_results(
        registry_path=registry,
        result_paths=[result],
        markdown_template=markdown,
        latex_template=latex,
        markdown_output=tmp_path / "benchmark_tables_baselines.md",
        latex_output=tmp_path / "benchmark_tables_baselines.tex",
    )

    with registry.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows[0]["status"] == "VERIFIED"
    assert rows[0]["json_pointer"] == "/metrics/semantic/miou"
    assert rows[1]["status"] == "UNFILLED"
    assert rows[2]["status"] == "UNFILLED"
    assert "0.276" in outputs["markdown"].read_text(encoding="utf-8")
    assert "{{T1_OVIOVO_REPLICA8_MIOU}}" in outputs["markdown"].read_text(encoding="utf-8")
    assert r"\verb|{{T1_DUALMAP_REPLICA8_MIOU}}|" not in outputs["latex"].read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update(status="BLOCKED"), "VERIFIED"),
        (lambda data: data["token_bindings"][0].update(json_pointer="/missing"), "JSON pointer"),
        (lambda data: data["token_bindings"][0].update(precision=2), "precision"),
        (lambda data: data["metrics"]["semantic"].update(miou=float("nan")), "finite"),
        (lambda data: data["method"].update(key="OVIMAP"), "method"),
        (lambda data: data["dataset"].update(name="ScanNet200"), "dataset"),
    ],
)
def test_import_rejects_invalid_result_contract(tmp_path, mutation, message) -> None:
    registry = tmp_path / "benchmark_tokens.tsv"
    result = tmp_path / "result.json"
    markdown, latex = _templates(tmp_path)
    _write_registry(registry)
    _write_result(result)
    data = json.loads(result.read_text(encoding="utf-8"))
    mutation(data)
    result.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ImportFailure, match=message):
        import_results(
            registry,
            [result],
            markdown,
            latex,
            tmp_path / "out.md",
            tmp_path / "out.tex",
        )


def test_import_rejects_oviovo_duplicate_and_missing_files(tmp_path) -> None:
    registry = tmp_path / "benchmark_tokens.tsv"
    markdown, latex = _templates(tmp_path)
    _write_registry(registry)
    result = tmp_path / "result.json"
    _write_result(result, token="T1_OVIOVO_REPLICA8_MIOU")
    with pytest.raises(ImportFailure, match="OVIOVO"):
        import_results(registry, [result], markdown, latex, tmp_path / "out.md", tmp_path / "out.tex")

    _write_result(result)
    with pytest.raises(ImportFailure, match="duplicate"):
        import_results(registry, [result, result], markdown, latex, tmp_path / "out.md", tmp_path / "out.tex")

    with pytest.raises(ImportFailure, match="does not exist"):
        import_results(
            registry,
            [tmp_path / "missing.json"],
            markdown,
            latex,
            tmp_path / "out.md",
            tmp_path / "out.tex",
        )


def test_import_accepts_verified_oviv2_table1_result(tmp_path: Path) -> None:
    registry = tmp_path / "benchmark_tokens.tsv"
    markdown, latex = _templates(tmp_path)
    _write_registry(registry)
    result = tmp_path / "oviv2.json"
    _write_result(result, token="T1_OVIV2_REPLICA8_MIOU", method="OVIV2")

    outputs = import_results(
        registry,
        [result],
        markdown,
        latex,
        tmp_path / "out.md",
        tmp_path / "out.tex",
    )

    assert "v2=0.276" in outputs["markdown"].read_text(encoding="utf-8")


def test_import_rejects_ovimap_table1_without_passing_paper_audit(tmp_path: Path) -> None:
    registry = tmp_path / "benchmark_tokens.tsv"
    markdown, latex = _templates(tmp_path)
    _write_registry(registry)
    result = tmp_path / "ovimap.json"
    _write_result(result, token="T1_OVIMAP_REPLICA8_MIOU", method="OVIMAP")

    with pytest.raises(ImportFailure, match="paper-parity audit"):
        import_results(
            registry,
            [result],
            markdown,
            latex,
            tmp_path / "out.md",
            tmp_path / "out.tex",
        )


def _attach_ovimap_audit(
    result: Path,
    audit: Path,
    *,
    result_protocol: dict | None = None,
    **audit_overrides,
) -> None:
    payload = json.loads(result.read_text(encoding="utf-8"))
    scenes = list(PAPER_PROTOCOL["scene_ids"])
    released_stdout = audit.parent / "released_semantic_evaluation.stdout.txt"
    released_stdout.write_text("mIoU\tmAcc\n0.2764\t0.322\n", encoding="utf-8")
    released_stdout_source = {
        "path": str(released_stdout.resolve()),
        "sha256": hashlib.sha256(released_stdout.read_bytes()).hexdigest(),
    }
    released_instance_json = audit.parent / "results_replica.json"
    released_instance_json.write_text(
        json.dumps({"all_ap": 0.085, "all_ap_50%": 0.212, "all_ap_25%": 0.345}),
        encoding="utf-8",
    )
    released_instance_source = {
        "path": str(released_instance_json.resolve()),
        "sha256": hashlib.sha256(released_instance_json.read_bytes()).hexdigest(),
    }
    paper_ap_metrics = {"miou": 0.363, "ap25": 0.767, "ap50": 0.508, "ap75": 0.22}
    paper_ap_output = audit.parent / "paper_ap_metrics.json"
    paper_ap_output.write_text(json.dumps(paper_ap_metrics), encoding="utf-8")
    paper_ap_output_source = {
        "path": str(paper_ap_output.resolve()),
        "sha256": hashlib.sha256(paper_ap_output.read_bytes()).hexdigest(),
    }
    evidence_payloads = {
        "native_mapping_manifest": {
            "status": "COMPLETE_NATIVE_MAPPING",
            "protocol_name": PAPER_PROTOCOL["name"],
            "scene_ids": scenes,
            "frame_ids_by_scene": {scene: list(range(0, 2000, 10)) for scene in scenes},
            "input_hashes": {
                "replica51_vocabulary": "1" * 64,
                "siglip_model": "2" * 64,
                "replica_semantic_gt": "3" * 64,
                "replica_instance_gt": "4" * 64,
            },
        },
        "released_evaluation_manifest": {
            "status": "COMPLETE_RELEASED_EVALUATION",
            "source_protocol": "released_ovimap_replica51",
            "scene_ids": scenes,
            "frame_count_per_scene": 200,
            "semantic_vocabulary": "Replica-51",
            "paper_metric_availability": {"table_3_semantic": True},
            "output_artifacts": {
                "semantic_vertex_metrics": {"miou": 0.2764, "macc": 0.322},
                "semantic_instance_metrics": {"apall": 0.085, "ap50": 0.212, "ap25": 0.345},
                "files": {
                    "semantic_stdout": released_stdout_source,
                    "results_replica_json": released_instance_source,
                },
            },
        },
        "class_agnostic_ap_manifest": {
            "status": "COMPLETE_PAPER_AP_EVALUATION",
            "protocol_name": PAPER_PROTOCOL["name"],
            "scene_ids": scenes,
            "metric_contract": PAPER_PROTOCOL["instance_metrics"],
            "metrics_present": ["miou", "ap25", "ap50", "ap75"],
            "metrics": paper_ap_metrics,
            "output_artifacts": {"metrics_json": paper_ap_output_source},
        },
    }
    protocol_evidence = {}
    for name, document in evidence_payloads.items():
        path = audit.parent / f"{name}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        protocol_evidence[name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    payload["protocol"] = {
        **PAPER_PROTOCOL,
        "scene_ids": list(PAPER_PROTOCOL["scene_ids"]),
    } if result_protocol is None else result_protocol
    payload["metrics"]["replica_8_compat"] = {
        "scene_ids": list(PAPER_PROTOCOL["scene_ids"]),
        "scene_count": len(PAPER_PROTOCOL["scene_ids"]),
    }
    payload["protocol_evidence"] = protocol_evidence
    payload["protocol_audit"] = {
        "status": "PASS",
        "protocol_name": "ovimap_cvpr2026_replica",
        "path": str(audit),
    }
    result.write_text(json.dumps(payload), encoding="utf-8")
    audit_payload = {
        "schema_version": 1,
        "status": "PASS",
        "protocol_name": "ovimap_cvpr2026_replica",
        "failures": [],
        "scene_artifacts": {},
        "feature_sources": {},
        "result_source": {
            "path": str(result.resolve()),
            "sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
        },
    }
    for scene in PAPER_PROTOCOL["scene_ids"]:
        feature_path = audit.parent / f"{scene}.pkl"
        with feature_path.open("wb") as handle:
            pickle.dump(
                {1: {"frame_id": [0, 10], "box_2d": [(1, 1, 2, 2), (1, 1, 2, 2)]}},
                handle,
            )
        audit_payload["feature_sources"][scene] = {
            "path": str(feature_path.resolve()),
            "sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        }
        audit_payload["scene_artifacts"][scene] = {
            "feature_instance_count": 1,
            "eligible_feature_instance_count": 1,
            "query_count": 2,
            "average_queries_per_feature_instance": 2.0,
            "full_frame_bbox_count": 0,
            "full_frame_bbox_ratio": 0.0,
        }
    audit_payload.update(audit_overrides)
    audit.write_text(json.dumps(audit_payload), encoding="utf-8")


def test_import_accepts_ovimap_table1_with_result_bound_paper_audit(tmp_path: Path) -> None:
    registry = tmp_path / "benchmark_tokens.tsv"
    markdown, latex = _templates(tmp_path)
    _write_registry(registry)
    audit = tmp_path / "paper_parity_audit.json"
    result = tmp_path / "ovimap.json"
    _write_result(result, token="T1_OVIMAP_REPLICA8_MIOU", method="OVIMAP")
    _attach_ovimap_audit(result, audit)

    import_results(
        registry,
        [result],
        markdown,
        latex,
        tmp_path / "out.md",
        tmp_path / "out.tex",
    )

    with registry.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    ovimap = next(row for row in rows if row["token"] == "T1_OVIMAP_REPLICA8_MIOU")
    assert ovimap["status"] == "VERIFIED"


@pytest.mark.parametrize(
    ("audit_overrides", "message"),
    [
        ({"status": "FAIL", "failures": ["protocol mismatch"]}, "audit file must PASS"),
        ({"protocol_name": "different_protocol"}, "protocol name"),
        ({"result_source": {"sha256": "0" * 64}}, "result hash"),
        ({"feature_sources": {}}, "feature sources"),
    ],
)
def test_import_rejects_ovimap_self_reported_pass_when_audit_file_is_invalid(
    tmp_path: Path,
    audit_overrides: dict,
    message: str,
) -> None:
    registry = tmp_path / "benchmark_tokens.tsv"
    markdown, latex = _templates(tmp_path)
    _write_registry(registry)
    audit = tmp_path / "paper_parity_audit.json"
    result = tmp_path / "ovimap.json"
    _write_result(result, token="T1_OVIMAP_REPLICA8_MIOU", method="OVIMAP")
    _attach_ovimap_audit(result, audit, **audit_overrides)

    with pytest.raises(ImportFailure, match=message):
        import_results(
            registry,
            [result],
            markdown,
            latex,
            tmp_path / "out.md",
            tmp_path / "out.tex",
        )


def test_import_recomputes_ovimap_protocol_instead_of_trusting_sidecar_pass(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "benchmark_tokens.tsv"
    markdown, latex = _templates(tmp_path)
    _write_registry(registry)
    audit = tmp_path / "paper_parity_audit.json"
    result = tmp_path / "ovimap.json"
    _write_result(result, token="T1_OVIMAP_REPLICA8_MIOU", method="OVIMAP")
    _attach_ovimap_audit(result, audit, result_protocol={})

    with pytest.raises(ImportFailure, match="does not satisfy"):
        import_results(
            registry,
            [result],
            markdown,
            latex,
            tmp_path / "out.md",
            tmp_path / "out.tex",
        )


def test_import_rejects_ovimap_metric_that_differs_from_evaluator_manifest(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "benchmark_tokens.tsv"
    markdown, latex = _templates(tmp_path)
    _write_registry(registry)
    audit = tmp_path / "paper_parity_audit.json"
    result = tmp_path / "ovimap.json"
    _write_result(result, token="T1_OVIMAP_REPLICA8_MIOU", method="OVIMAP")
    _attach_ovimap_audit(result, audit)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["metrics"]["semantic"]["miou"] = 0.9
    result.write_text(json.dumps(payload), encoding="utf-8")
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    audit_payload["result_source"]["sha256"] = hashlib.sha256(result.read_bytes()).hexdigest()
    audit.write_text(json.dumps(audit_payload), encoding="utf-8")

    with pytest.raises(ImportFailure, match="metric mismatch"):
        import_results(
            registry,
            [result],
            markdown,
            latex,
            tmp_path / "out.md",
            tmp_path / "out.tex",
        )


def test_import_rejects_replica8_metrics_not_defined_by_paper_evaluators(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "benchmark_tokens.tsv"
    markdown, latex = _templates(tmp_path)
    _write_registry(registry)
    audit = tmp_path / "paper_parity_audit.json"
    result = tmp_path / "ovimap.json"
    _write_result(result, token="T1_OVIMAP_REPLICA8_FMIOU", method="OVIMAP")
    _attach_ovimap_audit(result, audit)

    with pytest.raises(ImportFailure, match="not defined by the paper evaluators"):
        import_results(
            registry,
            [result],
            markdown,
            latex,
            tmp_path / "out.md",
            tmp_path / "out.tex",
        )


def test_import_does_not_apply_replica8_paper_gate_to_scannet(tmp_path: Path) -> None:
    registry = tmp_path / "benchmark_tokens.tsv"
    markdown, latex = _templates(tmp_path)
    _write_registry(registry)
    result = tmp_path / "ovimap_scannet.json"
    _write_result(result, token="T1_OVIMAP_SCANNET5_MIOU", method="OVIMAP")
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["dataset"] = {"name": "ScanNet200", "splits": ["scannet200_5_heldout"]}
    result.write_text(json.dumps(payload), encoding="utf-8")

    import_results(
        registry,
        [result],
        markdown,
        latex,
        tmp_path / "out.md",
        tmp_path / "out.tex",
    )

    with registry.open(newline="", encoding="utf-8") as handle:
        registry_rows = list(csv.DictReader(handle, delimiter="\t"))
    scannet = next(row for row in registry_rows if row["token"] == "T1_OVIMAP_SCANNET5_MIOU")
    assert scannet["status"] == "VERIFIED"


def test_import_rejects_registry_token_outside_main_table_scope(tmp_path) -> None:
    registry = tmp_path / "benchmark_tokens.tsv"
    markdown, latex = _templates(tmp_path)
    _write_registry(registry)
    with registry.open("a", encoding="utf-8") as handle:
        handle.write(
            "S1_DUALMAP_MIOU\tS1\tDUALMAP\tReplica\treplica_8_compat\tMIOU\t"
            "higher\t3\t\t\tUNFILLED\tSupplemental.\n"
        )
    result = tmp_path / "result.json"
    _write_result(result, token="S1_DUALMAP_MIOU")

    with pytest.raises(ImportFailure, match="outside"):
        import_results(registry, [result], markdown, latex, tmp_path / "out.md", tmp_path / "out.tex")


def test_importer_cli_starts_outside_repository(tmp_path: Path) -> None:
    importer = Path(__file__).resolve().parents[2] / "tools" / "import_benchmark_results.py"

    completed = subprocess.run(
        [sys.executable, str(importer), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
