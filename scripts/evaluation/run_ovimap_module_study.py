"""Single resumable entry point for the OVI-MAP module-validation study."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.static_ovmap.module_validation.assets import (
    build_scannet_inventory,
    build_scene_splits,
    exposed_scannet_families,
    resolve_assets,
)
from src.static_ovmap.module_validation.contracts import (
    PhaseReceipt,
    ReceiptStatus,
    StudySpec,
    atomic_write_json,
    canonical_digest,
)

PHASE_DEPENDENCIES = {
    "bind": (),
    "capture": ("bind",),
    "semantic": ("capture",),
    "geometry": ("capture",),
    "query": ("capture",),
    "select": ("semantic", "geometry", "query"),
    "confirm": ("select",),
    "report": (),
}
_ATTEMPT_RE = re.compile(r"^attempt_(\d{3})$")


@dataclass(frozen=True)
class PhaseResult:
    status: ReceiptStatus
    outputs: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhaseContext:
    phase: str
    attempt_dir: Path
    cache_key: str
    spec: StudySpec
    resolved_config: Mapping[str, Any]
    dependencies: Mapping[str, PhaseReceipt]


@dataclass(frozen=True)
class PhaseRun:
    receipt: PhaseReceipt
    reused: bool


PhaseHandler = Callable[[PhaseContext], PhaseResult]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _attempt_number(path: Path) -> int | None:
    match = _ATTEMPT_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _select_attempt(output_root: Path, cache_key: str) -> tuple[Path, bool]:
    output_root.mkdir(parents=True, exist_ok=True)
    numbered: list[tuple[int, Path]] = []
    compatible: list[tuple[int, Path]] = []
    for path in output_root.iterdir():
        number = _attempt_number(path)
        if number is None or not path.is_dir():
            continue
        numbered.append((number, path))
        resolved_path = path / "resolved_config.json"
        if not resolved_path.is_file():
            continue
        try:
            recorded = _read_json(resolved_path)
        except (OSError, json.JSONDecodeError):
            continue
        if recorded.get("cache_key") == cache_key:
            compatible.append((number, path))
    if compatible:
        return max(compatible)[1], True
    number = max((item[0] for item in numbered), default=0) + 1
    attempt = output_root / f"attempt_{number:03d}"
    attempt.mkdir()
    return attempt, False


class StudyRunner:
    """Select one compatible attempt and atomically execute study phases."""

    def __init__(
        self,
        spec: StudySpec,
        output_root: Path | str,
        resolved_config: Mapping[str, Any],
        *,
        handlers: Mapping[str, PhaseHandler] | None = None,
    ) -> None:
        self.spec = spec
        self.output_root = Path(output_root).resolve()
        self.resolved_config = dict(resolved_config)
        self.cache_key = canonical_digest(
            {
                "orchestrator_schema": 1,
                "spec_digest": spec.digest,
                "resolved_config": self.resolved_config,
            }
        )
        self.attempt_dir, resumed = _select_attempt(self.output_root, self.cache_key)
        self.resumed = resumed
        self.receipt_dir = self.attempt_dir / "receipts"
        self.receipt_dir.mkdir(exist_ok=True)
        self.handlers = dict(handlers or {})
        resolved_path = self.attempt_dir / "resolved_config.json"
        if not resumed:
            atomic_write_json(
                resolved_path,
                {
                    "schema_version": 1,
                    "study": spec.study,
                    "spec_digest": spec.digest,
                    "cache_key": self.cache_key,
                    "resolved_config": self.resolved_config,
                },
            )

    def _receipt_path(self, phase: str) -> Path:
        return self.receipt_dir / f"{phase}.json"

    def _load_receipt(self, phase: str) -> PhaseReceipt | None:
        path = self._receipt_path(phase)
        if not path.is_file():
            return None
        try:
            receipt = PhaseReceipt.load(path)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if receipt.cache_key != self.cache_key:
            return None
        return receipt

    def _missing_dependency(self, phase: str) -> tuple[str, ...]:
        return tuple(
            dependency
            for dependency in PHASE_DEPENDENCIES[phase]
            if self._load_receipt(dependency) is None
        )

    def _execute_phase(self, phase: str) -> PhaseRun:
        existing = self._load_receipt(phase)
        if existing is not None and existing.status in {
            ReceiptStatus.COMPLETE,
            ReceiptStatus.NOT_REQUIRED_BY_FROZEN_GATE,
        }:
            return PhaseRun(existing, reused=True)
        dependencies = {
            name: receipt
            for name in PHASE_DEPENDENCIES[phase]
            if (receipt := self._load_receipt(name)) is not None
        }
        missing = self._missing_dependency(phase)
        if missing:
            result = PhaseResult(
                status=ReceiptStatus.BLOCKED_PREREQUISITE,
                blockers=tuple(f"MISSING_PHASE_RECEIPT:{name}" for name in missing),
            )
        else:
            handler = self.handlers.get(phase)
            if handler is None:
                result = PhaseResult(
                    status=ReceiptStatus.BLOCKED_PREREQUISITE,
                    blockers=(f"UNIMPLEMENTED_PHASE:{phase}",),
                )
            else:
                result = handler(
                    PhaseContext(
                        phase=phase,
                        attempt_dir=self.attempt_dir,
                        cache_key=self.cache_key,
                        spec=self.spec,
                        resolved_config=self.resolved_config,
                        dependencies=dependencies,
                    )
                )
        receipt = PhaseReceipt(
            phase=phase,
            status=result.status,
            cache_key=self.cache_key,
            outputs=result.outputs,
            metrics=result.metrics,
            blockers=result.blockers,
        )
        receipt.write(self._receipt_path(phase))
        return PhaseRun(receipt, reused=False)

    def _write_progress(self) -> None:
        rows = [
            "# OVI-MAP module validation progress",
            "",
            f"Attempt: `{self.attempt_dir}`",
            "",
            "| Phase | Status | Blockers |",
            "|---|---|---|",
        ]
        for phase in self.spec.phases[:-1]:
            receipt = self._load_receipt(phase)
            status = receipt.status.value if receipt else "PENDING"
            blockers = ", ".join(receipt.blockers) if receipt else ""
            rows.append(f"| {phase} | {status} | {blockers} |")
        rows.extend(("", "Next action: run the first pending or blocked phase after resolving its recorded prerequisite.", ""))
        _atomic_write_text(self.attempt_dir / "progress.md", "\n".join(rows))

    def run(self, phase: str) -> dict[str, PhaseRun]:
        if phase not in self.spec.phases:
            raise ValueError(f"unsupported phase: {phase}")
        selected = self.spec.phases[:-1] if phase == "all" else (phase,)
        results: dict[str, PhaseRun] = {}
        for current in selected:
            results[current] = self._execute_phase(current)
            self._write_progress()
        return results


def bind_phase(context: PhaseContext) -> PhaseResult:
    """Materialize verified asset, inventory, exclusion, and split locks."""

    raw_resolution = context.resolved_config.get("asset_resolution")
    if not isinstance(raw_resolution, Mapping):
        return PhaseResult(
            status=ReceiptStatus.BLOCKED_PREREQUISITE,
            blockers=("MISSING_RESOLVED_ASSET_CONFIGURATION",),
        )
    atomic_write_json(context.attempt_dir / "asset_resolution.json", raw_resolution)
    raw_bindings = raw_resolution.get("bindings", {})
    bindings = raw_bindings if isinstance(raw_bindings, Mapping) else {}
    scannet_binding = bindings.get("scannet_root")
    if isinstance(scannet_binding, Mapping) and isinstance(scannet_binding.get("path"), str):
        try:
            inventory = build_scannet_inventory(Path(scannet_binding["path"]))
        except (OSError, ValueError) as error:
            inventory = {
                "schema_version": 1,
                "status": "BLOCKED_INVENTORY_VALIDATION",
                "dataset_root": scannet_binding["path"],
                "error": str(error),
                "scenes": [],
            }
    else:
        inventory = {
            "schema_version": 1,
            "status": "MISSING_SCANNET_ROOT",
            "dataset_root": None,
            "scenes": [],
        }
    repository_root = Path(
        str(context.resolved_config.get("repository_root", REPOSITORY_ROOT))
    ).resolve()
    exclusions = exposed_scannet_families(repository_root)
    splits = build_scene_splits(inventory["scenes"], exclusions, context.spec)
    atomic_write_json(context.attempt_dir / "scene_inventory.json", inventory)
    atomic_write_json(context.attempt_dir / "splits.json", splits.to_dict())

    missing = tuple(str(item) for item in raw_resolution.get("missing", ()))
    ambiguities = raw_resolution.get("ambiguities", {})
    non_scannet_missing = tuple(item for item in missing if item != "scannet_root")
    if ambiguities:
        status = ReceiptStatus.BLOCKED_ASSET_IDENTITY
        blockers = ("BLOCKED_ASSET_IDENTITY",)
    elif non_scannet_missing:
        status = ReceiptStatus.BLOCKED_PREREQUISITE
        blockers = tuple(f"MISSING_ASSET:{item}" for item in sorted(non_scannet_missing))
    elif splits.status is ReceiptStatus.BLOCKED_INDEPENDENT_SCENES:
        status = ReceiptStatus.BLOCKED_INDEPENDENT_SCENES
        blockers = ("BLOCKED_INDEPENDENT_SCENES",)
    else:
        status = ReceiptStatus.COMPLETE
        blockers = ()
    return PhaseResult(
        status=status,
        outputs={
            "asset_resolution": "asset_resolution.json",
            "scene_inventory": "scene_inventory.json",
            "splits": "splits.json",
        },
        metrics={
            "bound_asset_count": len(bindings),
            "missing_asset_count": len(missing),
            "inventory_scene_count": len(inventory["scenes"]),
            "eligible_development_families": splits.eligible_development_count,
            "eligible_confirmation_families": splits.eligible_confirmation_count,
        },
        blockers=blockers,
    )


def _code_identity(repository_root: Path) -> Mapping[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout

    try:
        head = run("rev-parse", "HEAD").strip()
        status = run("status", "--porcelain=v1", "--untracked-files=all")
        diff = run("diff", "--binary", "HEAD", "--")
    except (OSError, subprocess.SubprocessError):
        return {
            "head": None,
            "dirty": None,
            "status_digest": None,
            "tracked_diff_digest": None,
        }
    return {
        "head": head,
        "dirty": bool(status),
        "status_digest": canonical_digest(status),
        "tracked_diff_digest": canonical_digest(diff),
    }


def _base_resolved_config(spec: StudySpec, path: Path | None) -> Mapping[str, Any]:
    if path is not None:
        raw = _read_json(path)
        if not isinstance(raw, dict):
            raise ValueError("--resolved-config must contain a JSON object")
        if {
            "cache_key",
            "spec_digest",
            "resolved_config",
        }.issubset(raw):
            if raw["spec_digest"] != spec.digest:
                raise ValueError("resolved config spec digest mismatch")
            nested = raw["resolved_config"]
            if not isinstance(nested, dict):
                raise ValueError("resolved_config field must be a JSON object")
            expected_cache_key = canonical_digest(
                {
                    "orchestrator_schema": 1,
                    "spec_digest": spec.digest,
                    "resolved_config": nested,
                }
            )
            if raw["cache_key"] != expected_cache_key:
                raise ValueError("resolved config cache key mismatch")
            return nested
        return raw
    overrides = {
        name: os.environ[name]
        for name in ("OVIMAP_DATA_ROOTS", "OVIMAP_MODEL_ROOTS", "OVIMAP_REPLAY_ROOT")
        if name in os.environ
    }
    resolution = resolve_assets(
        spec,
        overrides,
        repository_root=REPOSITORY_ROOT,
    )
    return {
        "spec_path": str(spec.source_path),
        "repository_root": str(REPOSITORY_ROOT),
        "code_identity": _code_identity(REPOSITORY_ROOT),
        "environment_overrides": overrides,
        "asset_resolution": resolution.to_dict(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=tuple(PHASE_DEPENDENCIES) + ("all",))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resolved-config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = StudySpec.load(args.spec)
    output_root = args.output_root or spec.output_root
    runner = StudyRunner(
        spec,
        output_root,
        _base_resolved_config(spec, args.resolved_config),
        handlers={"bind": bind_phase},
    )
    results = runner.run(args.phase)
    print(json.dumps({
        "attempt": str(runner.attempt_dir),
        "cache_key": runner.cache_key,
        "phases": {
            phase: {"status": run.receipt.status.value, "reused": run.reused}
            for phase, run in results.items()
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
