#!/usr/bin/env python3
"""Package compact public evidence from CROVE fine current-map runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _portable_path(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        return value
    resolved = path.resolve(strict=False)
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        pass
    home = Path(os.environ["HOME"]).resolve(strict=False)
    try:
        return f"$HOME/{resolved.relative_to(home)}"
    except ValueError:
        return f"local:{resolved.name}"


def _is_path_field(key: str | None) -> bool:
    if key is None:
        return False
    return key == "path" or key.endswith(("_path", "_root", "_dir"))


def _portable_payload(value: object, *, key: str | None = None) -> object:
    if isinstance(value, dict):
        return {
            str(child_key): _portable_payload(child, key=str(child_key))
            for child_key, child in value.items()
        }
    if isinstance(value, list):
        return [_portable_payload(child, key=key) for child in value]
    if isinstance(value, str) and _is_path_field(key):
        return _portable_path(value)
    return value


def _merge_csv(
    output: Path,
    inputs: Iterable[tuple[str, Path]],
    *,
    prefix_summary_pointers: bool = False,
) -> None:
    fieldnames = ["evaluation"]
    rows: list[dict[str, str]] = []
    for evaluation, path in inputs:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")
            for field in reader.fieldnames:
                if field not in fieldnames:
                    fieldnames.append(field)
            for row in reader:
                merged = {"evaluation": evaluation, **row}
                if prefix_summary_pointers and merged.get("status") == "N/A":
                    token = merged.get("token")
                    if not isinstance(token, str) or not token:
                        raise ValueError("N/A lineage requires a table token")
                    merged["source_json"] = "table_values.json"
                    merged["json_pointer"] = f"/{token}/value"
                elif (
                    prefix_summary_pointers
                    and merged.get("source_json") == "metrics_summary.json"
                ):
                    pointer = merged.get("json_pointer")
                    if not isinstance(pointer, str) or not pointer.startswith("/"):
                        raise ValueError("metrics summary lineage requires a JSON Pointer")
                    merged["json_pointer"] = f"/{evaluation}{pointer}"
                rows.append(merged)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _run_json(run: Path, name: str) -> dict[str, Any]:
    path = run / name
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    return _json(path)


def _cost_control_payload(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    payload = _json(path)
    if payload.get("status") != "PASS":
        raise ValueError("2 cm cost control is not PASS")
    if payload.get("condition") != (
        "SOURCE_SURFACE_VOXEL_SELECTION_NOT_RECONSTRUCTION"
    ):
        raise ValueError("2 cm cost control condition is not source-surface selection")
    portable = _portable_payload(payload)
    if not isinstance(portable, dict):
        raise TypeError("2 cm cost control must remain an object")
    return portable


def _figure(run: Path) -> tuple[Path, Path]:
    images = sorted(run.glob("*_four_views.png"))
    if len(images) != 1:
        raise ValueError(f"run must contain exactly one four-view figure: {run}")
    receipt = images[0].with_suffix(".json")
    if receipt.is_symlink() or not receipt.is_file():
        raise FileNotFoundError(receipt)
    return images[0], receipt


def _legacy_figure(image: Path) -> Path:
    if image.is_symlink() or not image.is_file():
        raise FileNotFoundError(image)
    receipt_path = image.with_suffix(".json")
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = _json(receipt_path)
    if receipt.get("status") != "PASS":
        raise ValueError("legacy figure receipt is not PASS")
    if not receipt.get("display_only"):
        raise ValueError("legacy figure must be a display-only control")
    if not receipt.get("geometry_shared_across_views"):
        raise ValueError("legacy figure views must share geometry")
    return receipt_path


def package_results(
    static_run: Path,
    dynamic_run: Path,
    cost_control: Path,
    legacy_figure: Path,
    output: Path,
) -> None:
    if output.exists():
        raise FileExistsError(output)
    for run in (static_run, dynamic_run):
        receipt = _run_json(run, "run_receipt.json")
        if receipt.get("status") != "PASS":
            raise ValueError(f"source run is not PASS: {run}")
        manifest = _run_json(run / "current_map", "current_surface_manifest.json")
        if manifest.get("status") != "PASS" or not manifest.get(
            "geometry_shared_across_views"
        ):
            raise ValueError(f"source run has no shared canonical surface: {run}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        pairs = {"static_dev": static_run, "dynamic_dev": dynamic_run}
        for name in (
            "input_selection.json",
            "metrics_summary.json",
            "export_diagnostics.json",
            "effective_geometry_config.json",
            "selected_config.json",
        ):
            _write_json(
                staging / name,
                {
                    "schema_version": 1,
                    "status": "EXPERIMENT_COMPLETE_STATIC_GAIN_DYNAMIC_NO_GAIN",
                    **{
                        key: _portable_payload(_run_json(run, name))
                        for key, run in pairs.items()
                    },
                },
            )
        _write_json(staging / "cost_control_2cm.json", _cost_control_payload(cost_control))

        table_values: dict[str, object] = {}
        for run in pairs.values():
            for token, record in _run_json(run, "table_values.json").items():
                if token in table_values:
                    raise ValueError(f"duplicate table token: {token}")
                table_values[token] = record
        _write_json(staging / "table_values.json", table_values)

        for name in ("metrics_per_scene.csv", "trial_log.csv", "table_lineage.csv"):
            _merge_csv(
                staging / name,
                tuple((key, run / name) for key, run in pairs.items()),
                prefix_summary_pointers=name == "table_lineage.csv",
            )

        figures = staging / "figures"
        figures.mkdir()
        for evaluation, run in pairs.items():
            image, receipt_path = _figure(run)
            destination = figures / f"{evaluation}_four_views.png"
            shutil.copyfile(image, destination)
            receipt = _portable_payload(_json(receipt_path))
            if not isinstance(receipt, dict):
                raise TypeError("figure receipt must remain an object")
            receipt["output"] = {
                "path": f"figures/{destination.name}",
                "byte_count": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
            _write_json(figures / f"{evaluation}_four_views.json", receipt)

        legacy_receipt_path = _legacy_figure(legacy_figure)
        legacy_destination = figures / "legacy_static_display_control.png"
        shutil.copyfile(legacy_figure, legacy_destination)
        legacy_receipt = _portable_payload(_json(legacy_receipt_path))
        if not isinstance(legacy_receipt, dict):
            raise TypeError("legacy figure receipt must remain an object")
        legacy_receipt["output"] = {
            "path": f"figures/{legacy_destination.name}",
            "byte_count": legacy_destination.stat().st_size,
            "sha256": _sha256(legacy_destination),
        }
        _write_json(figures / "legacy_static_display_control.json", legacy_receipt)

        artifacts = []
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "compact_artifact_index.json":
                artifacts.append(
                    {
                        "path": str(path.relative_to(staging)),
                        "byte_count": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
        _write_json(
            staging / "compact_artifact_index.json",
            {
                "schema_version": 1,
                "status": "PASS",
                "large_map_policy": "LOCAL_ONLY_POLICY",
                "model_upload_status": "NOT_APPLICABLE_NO_NEW_TRAINING",
                "large_artifact_roots": {
                    key: _portable_path(str(run)) for key, run in pairs.items()
                },
                "artifacts": artifacts,
            },
        )
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-run", type=Path, required=True)
    parser.add_argument("--dynamic-run", type=Path, required=True)
    parser.add_argument("--cost-control", type=Path, required=True)
    parser.add_argument("--legacy-figure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    package_results(
        args.static_run,
        args.dynamic_run,
        args.cost_control,
        args.legacy_figure,
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
