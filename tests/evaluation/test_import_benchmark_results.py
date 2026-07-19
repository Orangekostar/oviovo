from __future__ import annotations

import csv
import hashlib
import json
import pickle
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
