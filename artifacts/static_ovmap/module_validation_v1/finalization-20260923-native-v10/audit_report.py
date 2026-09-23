import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

root = Path("/mnt/shared/ww/ovimap-module-validation-v1")
attempt = root / "attempt_011"
study = root / "scannet_study_v1"


def read(path):
    return json.loads(path.read_text())


evidence = read(attempt / "scientific_evidence.json")
groups, seen = defaultdict(list), set()
for row in evidence["rows"]:
    key = row["role"], row["method_id"], row["budget"]
    identity = (*key, row["scene_id"])
    assert identity not in seen, identity
    seen.add(identity)
    groups[key].append(row)
    assert row["role"] in ("fit", "cal", "select")
    assert row["metrics"]["uap"] is not None and row["metrics"]["miou"] is not None
    source = read(Path(row["evidence_path"]))
    actual = source.get("rows", [source["row"]] if "row" in source else [])
    assert any(
        candidate["scene_id"] == row["scene_id"]
        and candidate["method_id"] == row["method_id"]
        and candidate["prediction_key"] == row["prediction_key"]
        and candidate["metrics"] == row["metrics"]
        for candidate in actual
    ), identity
for summary in evidence["summaries"]:
    key = summary["role"], summary["method_id"], summary["budget"]
    rows = groups[key]
    assert summary["scene_count"] == len(rows)
    assert summary["selection_status"] == "COMPLETE"
    for metric, value in summary["metrics"].items():
        values = [
            r["metrics"][metric] for r in rows if r["metrics"].get(metric) is not None
        ]
        assert summary["defined_scene_counts"][metric] == len(values)
        expected = sum(values) / len(values) if values else None
        assert (value is None and expected is None) or (
            value is not None
            and expected is not None
            and math.isclose(value, expected, abs_tol=1e-15)
        ), (key, metric)
    costs = {
        name: sum(r.get("logical_cost", {}).get(name, 0) for r in rows)
        for name in {name for r in rows for name in r.get("logical_cost", {})}
    }
    assert costs == summary["logical_cost_sum"]
assert len(evidence["rows"]) == 116 and len(evidence["summaries"]) == 34
matrix = read(attempt / "method_matrix.json")
matrix_rows = matrix["rows"]
assert Counter(row["disposition"] for row in matrix_rows) == {
    "MEASURED": 15,
    "BLOCKED": 1,
    "NOT_REQUIRED": 2,
}
for row in matrix_rows:
    if row["disposition"] == "MEASURED":
        summary = next(
            s
            for s in evidence["summaries"]
            if s["role"] == "select" and s["method_id"] == row["method_id"]
        )
        assert all(
            row["metrics"][name] == value for name, value in summary["metrics"].items()
        )
    else:
        assert row["reason"] == evidence["method_status"][row["method_id"]]
selection = read(attempt / "selection.json")
assert selection["status"] == "FROZEN" and selection["final_candidate"] == "N0"
assert selection["science_status"] == "NO_NET_GAIN"
assert selection["confirmation_authorized"] is False
assert selection["combinations"] == []
assert (
    selection["budget_curves"]["gate_status"] == "NOT_REQUIRED_QUERY_GAIN_NOT_ELIGIBLE"
)
confirmation = read(attempt / "confirmation.json")
assert (
    confirmation["status"] == "NOT_REQUIRED_NO_RETAINED_CANDIDATE"
    and confirmation["rows"] == []
)
frozen = Path(selection["frozen_config"]["path"])
assert (
    hashlib.sha256(frozen.read_bytes()).hexdigest()
    == selection["frozen_config"]["sha256"]
)
for identity in read(frozen)["sources"]:
    assert (
        hashlib.sha256(Path(identity["path"]).read_bytes()).hexdigest()
        == identity["sha256"]
    ), identity["path"]
assert (
    read(frozen)["prediction_code_commit"] == "b6ab45b8dadf6672d2bcc2a6cfaed9bf41d60a0c"
)
for scene in ("scene0553_00", "scene0064_00"):
    for path in (
        root / "data/scannet/exported" / scene,
        root / "scannet_runtime_v1" / scene,
        study / "annotations" / scene,
        study / "scenes" / scene,
    ):
        assert not path.exists(), path
tables = re.findall(
    r"^## Table ([A-E])\.",
    (attempt / "MODULE_VALIDATION_RESULTS.md").read_text(),
    flags=re.MULTILINE,
)
assert tables == list("ABCDE"), tables
canonical = Path("docs/paper/static_ovmap/MODULE_VALIDATION_RESULTS.md").read_text()
assert re.findall(r"^## Table ([A-E])\.", canonical, flags=re.MULTILINE) == list("ABCDE")
generated = (attempt / "MODULE_VALIDATION_RESULTS.md").read_text()
assert canonical.split("## Table A.", 1)[1].split("## Table E.", 1)[0] == generated.split(
    "## Table A.", 1
)[1].split("## Table E.", 1)[0]
for method, reason in selection["module_selection"]["combination_plan"]["not_required"].items():
    assert f"| {method} | — | NOT_REQUIRED | {reason} |" in canonical
phases = {
    path.stem: read(path)["status"] for path in (attempt / "receipts").glob("*.json")
}
assert all(
    value == ("NOT_REQUIRED_BY_FROZEN_GATE" if key == "confirm" else "COMPLETE")
    for key, value in phases.items()
), phases
assert set(phases) == {
    "bind",
    "capture",
    "semantic",
    "geometry",
    "query",
    "select",
    "confirm",
    "report",
}
headlines = {
    row["method_id"]: row["metrics"]
    for row in evidence["summaries"]
    if row["role"] == "select"
}
print(
    json.dumps(
        {
            "status": "PASS",
            "rows": len(evidence["rows"]),
            "summaries": len(evidence["summaries"]),
            "method_status": evidence["method_status"],
            "tables": tables,
            "canonical_tables_A_to_D_match": True,
            "canonical_combination_gates_match": True,
            "phases": phases,
            "native_headline": headlines["N0"],
            "query_comparator": headlines["Q_COMBINE"],
            "query_gain": headlines["Q_GAIN"],
            "holdout_unopened": True,
            "frozen_sources_checked": len(read(frozen)["sources"]),
        }
    )
)
