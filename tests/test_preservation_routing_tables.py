import copy
import csv
import json

import pytest


def test_write_table_preserves_json_types_and_serializes_displays(tmp_path):
    from static_ovmap.preservation_routing.tabulate import write_table

    columns = ["name", "score", "count", "missing", "flag"]
    rows = [{
        "name": "under_score|line\nnext",
        "score": 0.123456789,
        "count": 3,
        "missing": None,
        "flag": True,
    }]
    original_columns = copy.deepcopy(columns)
    original_rows = copy.deepcopy(rows)

    identities = write_table(tmp_path, "sample", columns, rows)

    assert columns == original_columns
    assert rows == original_rows
    assert set(identities) == {"json", "csv", "md", "tex"}
    for extension, identity in identities.items():
        path = tmp_path / f"sample.{extension}"
        assert identity == {"path": str(path.resolve()), "bytes": path.stat().st_size,
                            "sha256": identity["sha256"]}

    payload = json.loads((tmp_path / "sample.json").read_text(encoding="utf-8"))
    assert payload["columns"] == columns
    assert payload["rows"][0]["score"] == rows[0]["score"]
    assert type(payload["rows"][0]["score"]) is float
    assert payload["rows"][0]["count"] == 3
    assert type(payload["rows"][0]["count"]) is int
    assert payload["rows"][0]["missing"] is None
    assert payload["rows"][0]["flag"] is True

    with (tmp_path / "sample.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert list(csv_rows[0]) == columns
    assert csv_rows[0]["name"] == rows[0]["name"]
    assert csv_rows[0]["missing"] == ""

    markdown = (tmp_path / "sample.md").read_text(encoding="utf-8")
    assert "under_score\\|line<br>next" in markdown
    assert "0.123457" in markdown
    assert "| NA |" in markdown
    assert "| true |" in markdown

    latex = (tmp_path / "sample.tex").read_text(encoding="utf-8")
    assert "under\\_score|line next" in latex
    assert "\\toprule" in latex
    assert "\\midrule" in latex
    assert "\\bottomrule" in latex


def test_write_table_rejects_invalid_rows_before_writing(tmp_path):
    from static_ovmap.preservation_routing.tabulate import write_table

    columns = ["value"]
    with pytest.raises(ValueError):
        write_table(tmp_path, "mismatch", columns, [{"other": 1}])
    assert list(tmp_path.iterdir()) == []

    with pytest.raises(ValueError):
        write_table(tmp_path, "nonfinite", columns, [{"value": float("nan")}])
    assert list(tmp_path.iterdir()) == []
