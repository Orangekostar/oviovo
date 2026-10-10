"""Deterministic table serialization for preservation-routing reports."""

import csv
import math
from pathlib import Path

from .common import ConsumptionIndex, write


def display(value):
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("display values must contain only finite floats")
        return f"{value:.6f}"
    return str(value)


def _escape_latex(value):
    text = str(value).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    escaped = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(escaped.get(character, character) for character in text)


def write_table(folder, name, columns, rows):
    folder = Path(folder)
    expected = set(columns)
    for row in rows:
        if set(row) != expected:
            raise ValueError("row keys must exactly match columns")
        pending = list(row.values())
        while pending:
            value = pending.pop()
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("table values must contain only finite floats")
            if isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, (list, tuple)):
                pending.extend(value)

    folder.mkdir(parents=True, exist_ok=True)
    paths = {extension: folder / f"{name}.{extension}"
             for extension in ("json", "csv", "md", "tex")}
    write(paths["json"], {"columns": columns, "rows": rows})

    with paths["csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "| " + " | ".join(
            str(column).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>").replace("|", r"\|")
            for column in columns
        ) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = display(row[column])
            value = value.replace("\r\n", "\n").replace("\r", "\n")
            cells.append(value.replace("\n", "<br>").replace("|", r"\|"))
        markdown.append("| " + " | ".join(cells) + " |")
    paths["md"].write_text("\n".join(markdown) + "\n", encoding="utf-8")

    latex = [
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        " & ".join(_escape_latex(column) for column in columns) + " \\\\",
        r"\midrule",
    ]
    for row in rows:
        latex.append(" & ".join(_escape_latex(display(row[column])) for column in columns) + " \\\\")
    latex.extend((r"\bottomrule", r"\end{tabular}"))
    paths["tex"].write_text("\n".join(latex) + "\n", encoding="utf-8")

    index = ConsumptionIndex()
    return {extension: index.identity(path) for extension, path in paths.items()}
