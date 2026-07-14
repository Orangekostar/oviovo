from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

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
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_result(path: Path, *, token: str = "T1_DUALMAP_REPLICA8_MIOU") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "VERIFIED",
                "run_id": "dualmap-replica8-test",
                "method": {"key": "DUALMAP", "display_label": "DualMap", "mode": "native"},
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
    markdown.write_text("value={{T1_DUALMAP_REPLICA8_MIOU}} ours={{T1_OVIOVO_REPLICA8_MIOU}}\n")
    latex.write_text(r"value=\verb|{{T1_DUALMAP_REPLICA8_MIOU}}| ours=\verb|{{T1_OVIOVO_REPLICA8_MIOU}}|" + "\n")
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
