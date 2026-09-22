"""Single resumable entry point for the OVI-MAP module-validation study."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
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
    sha256_file,
)
from src.static_ovmap.module_validation.contracts import (
    PhaseReceipt,
    ReceiptStatus,
    StudySpec,
    atomic_write_json,
    canonical_digest,
)
from src.static_ovmap.module_validation.reporting import (
    ReleaseStatuses,
    build_method_matrix,
    render_handoff,
    render_results,
    write_method_matrix,
)

PHASE_DEPENDENCIES = {
    "bind": (),
    "capture": ("bind",),
    "semantic": ("capture",),
    "geometry": ("capture",),
    "query": ("capture",),
    "select": ("semantic", "geometry", "query"),
    "confirm": ("select",),
    "report": ("bind", "capture", "semantic", "geometry", "query", "select", "confirm"),
}
ORCHESTRATOR_SCHEMA = 2
_ATTEMPT_RE = re.compile(r"^attempt_(\d{3})$")
REQUIRED_METHODS = (
    "N0",
    "S_NATIVE_AREA",
    "S_NATIVE_VOTE",
    "S_SIGLIP2_AREA",
    "S_SIGLIP2_VOTE",
    "S_WOW_VOTE",
    "S_SIMPLE",
    "S_NO_CONTEXT",
    "S_PAIRED",
    "G_ORIGINAL",
    "G_AGREEMENT",
    "G_QUALITY",
    "Q_COMBINE",
    "Q_AREA",
    "Q_UNCERTAINTY",
    "Q_GAIN",
    "COMBO_GS",
    "COMBO_Q_REFINEMENT",
)


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
                "orchestrator_schema": ORCHESTRATOR_SCHEMA,
                "spec_digest": spec.digest,
                "resolved_config": self.resolved_config,
            }
        )
        self.handlers = dict(handlers or {})
        self.output_root.mkdir(parents=True, exist_ok=True)
        with (self.output_root / ".attempt.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            self.attempt_dir, resumed = _select_attempt(self.output_root, self.cache_key)
            self.resumed = resumed
            self.receipt_dir = self.attempt_dir / "receipts"
            self.receipt_dir.mkdir(exist_ok=True)
            if not resumed:
                atomic_write_json(self.attempt_dir / "resolved_config.json", {
                    "schema_version": ORCHESTRATOR_SCHEMA,
                    "study": spec.study, "spec_digest": spec.digest,
                    "cache_key": self.cache_key, "resolved_config": self.resolved_config,
                })
        self._hashes: dict[tuple, str] = {}

    def _hash(self, path: Path) -> str:
        stat = path.stat()
        key = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in self._hashes:
            self._hashes[key] = sha256_file(path)
        return self._hashes[key]

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
        if receipt.cache_key != self.cache_key or receipt.schema_version != 2:
            return None
        for raw_path, identity in receipt.output_identities.items():
            output = Path(raw_path)
            if not output.is_file() or not isinstance(identity, Mapping) or self._hash(output) != identity.get("sha256"):
                return None
        dependencies = {name: self._load_receipt(name) for name in PHASE_DEPENDENCIES[phase]}
        if receipt.dependency_identity != self._dependency_identity(dependencies):
            return None
        return receipt

    @staticmethod
    def _dependency_identity(dependencies: Mapping[str, PhaseReceipt | None]) -> str:
        return canonical_digest({name: receipt.to_dict() if receipt else None
                                 for name, receipt in dependencies.items()})

    def _output_identities(self, outputs: Mapping[str, Any]) -> dict[str, Any]:
        identities = {}
        for value in outputs.values():
            if not isinstance(value, str):
                continue
            path = Path(value)
            if not path.is_absolute():
                path = self.attempt_dir / path
            if path.is_file():
                identities[str(path.resolve())] = {
                    "sha256": self._hash(path), "bytes": path.stat().st_size,
                }
                if path.name.endswith(("_smoke.json", "_runtime.json")):
                    for row in _read_json(path).get("input_identities", ()):
                        identities[row["path"]] = {"sha256": row["sha256"], "bytes": row["bytes"]}
        return identities

    def _missing_dependency(self, phase: str) -> tuple[str, ...]:
        return tuple(
            dependency
            for dependency in PHASE_DEPENDENCIES[phase]
            if self._load_receipt(dependency) is None
        )

    def _execute_phase(self, phase: str) -> PhaseRun:
        started = time.monotonic()
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
        if missing and phase != "report":
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
                try:
                    result = handler(PhaseContext(
                        phase=phase,
                        attempt_dir=self.attempt_dir,
                        cache_key=self.cache_key,
                        spec=self.spec,
                        resolved_config=self.resolved_config,
                        dependencies=dependencies,
                    ))
                except Exception as error:  # noqa: BLE001 - persist failure and finish independent phases
                    error_path = self.attempt_dir / "errors" / f"{phase}.json"
                    atomic_write_json(error_path, {
                        "error_type": type(error).__name__, "message": str(error),
                        "traceback": traceback.format_exc(),
                    })
                    result = PhaseResult(
                        ReceiptStatus.FAILED, outputs={"error": str(error_path)},
                        blockers=(f"PHASE_EXECUTION_FAILED:{phase}:{type(error).__name__}",),
                    )
        receipt = PhaseReceipt(
            phase=phase,
            status=result.status,
            cache_key=self.cache_key,
            outputs=result.outputs,
            metrics={**result.metrics, "elapsed_seconds": time.monotonic() - started},
            blockers=result.blockers,
            schema_version=2,
            output_identities=self._output_identities(result.outputs),
            dependency_identity=self._dependency_identity({
                name: self._load_receipt(name) for name in PHASE_DEPENDENCIES[phase]
            }),
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
        self._hashes.clear()
        with (self.attempt_dir / ".run.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"attempt is already running: {self.attempt_dir}") from error
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


def _development_blockers(context: PhaseContext) -> tuple[str, ...]:
    split_path = context.attempt_dir / "splits.json"
    if not split_path.is_file():
        return ("MISSING_SCENE_SPLIT_LOCK",)
    splits = _read_json(split_path)
    if splits.get("status") != "COMPLETE":
        return (str(splits.get("status", "BLOCKED_INDEPENDENT_SCENES")),)
    return ("MISSING_SCANNET_SCIENTIFIC_CONFIGURATION",)


def _execute_boundary(context: PhaseContext) -> PhaseResult:
    configuration = context.resolved_config.get("historical_smoke")
    if not isinstance(configuration, Mapping):
        return PhaseResult(
            ReceiptStatus.BLOCKED_PREREQUISITE,
            blockers=("MISSING_EXECUTABLE_SMOKE_CONFIGURATION", *_development_blockers(context)),
        )
    output = context.attempt_dir / f"{context.phase}_smoke.json"
    log_path = context.attempt_dir / f"{context.phase}_smoke.log"
    python = configuration.get("semantic_python", sys.executable) if context.phase == "semantic" else sys.executable
    command = [
        str(python), str(REPOSITORY_ROOT / "scripts/evaluation/run_ovimap_module_smoke.py"),
        "--phase", context.phase, "--resolved-config", str(context.attempt_dir / "resolved_config.json"),
        "--output", str(output),
    ]
    environment = dict(os.environ)
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    reused = False
    if output.is_file():
        previous = _read_json(output)
        reused = (
            previous.get("status") == "COMPLETE"
            and previous.get("phase") == context.phase
            and previous.get("config_identity") == canonical_digest(context.resolved_config)
            and bool(previous.get("input_identities"))
            and all(Path(row["path"]).is_file() and sha256_file(row["path"]) == row["sha256"]
                    for row in previous.get("input_identities", ()))
        )
    if not reused:
        with log_path.open("w") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    value = _read_json(output)
    if value.get("status") != "COMPLETE" or value.get("phase") != context.phase:
        raise ValueError("boundary execution did not produce a successful matching receipt")
    blockers = _development_blockers(context)
    return PhaseResult(
        ReceiptStatus.PARTIAL,
        outputs={"smoke": str(output), "log": str(log_path)},
        metrics={"executed_boundary_smokes": int(not reused), "reused_boundary_smokes": int(reused), "scientific_evaluation_rows": 0,
                 "physical": value.get("physical", {}),
                 "scene_count": value.get("scene_count"), "frame_count": value.get("frame_count")},
        blockers=blockers,
    )


def capture_phase(context: PhaseContext) -> PhaseResult:
    runtime = context.resolved_config.get("scannet_runtime")
    split_path = context.attempt_dir / "splits.json"
    if isinstance(runtime, Mapping) and split_path.is_file() and _read_json(split_path).get("status") == "COMPLETE":
        from src.static_ovmap.module_validation.boundary_jobs import file_identity
        from src.static_ovmap.module_validation.scannet_runtime import (
            capture_development,
        )

        locked = _read_json(Path(runtime["data_root"]) / "acquisition_lock.json")
        splits = _read_json(split_path)
        for role in ("fit", "cal", "select", "confirm"):
            if splits[role] != [row["scene_id"] for row in locked["selected"] if row["role"] == role]:
                raise ValueError("bound study splits differ from pre-inference acquisition lock")
        try:
            result = capture_development(dict(runtime))
        except RuntimeError as error:
            if not str(error).startswith("GPU_BUSY:"):
                raise
            return PhaseResult(ReceiptStatus.BLOCKED_PREREQUISITE,
                outputs={"resource_status": str(Path(runtime["output_root"]) / "resource_status.json")},
                blockers=("BLOCKED_GPU_BUSY",))
        identities = [result["acquisition_lock"]]
        for scene_receipt in result["scenes"].values():
            identities.append(scene_receipt)
            record = _read_json(Path(scene_receipt["path"]))
            identities.extend(record["outputs"])
            identities.extend(record["input_identities"])
        output = context.attempt_dir / "capture_runtime.json"
        atomic_write_json(output, {**result, "input_identities": identities,
            "runtime_config_identity": canonical_digest(runtime), "splits": file_identity(split_path)})
        return PhaseResult(ReceiptStatus.COMPLETE, outputs={"runtime": str(output)},
            metrics={"scene_count": len(result["scenes"]), "scheduled_frame_count": 2400})
    return _execute_boundary(context)


def _scientific_identities(receipt_path: Path) -> list[dict]:
    """Bind every transitive leaf input/output, including disk acquisition caches."""
    from src.static_ovmap.module_validation.boundary_jobs import file_identity

    found = {}
    pending = [file_identity(receipt_path)]
    while pending:
        row = pending.pop()
        path = Path(row["path"])
        previous = found.get(str(path))
        if previous is not None:
            if previous["sha256"] != row["sha256"]:
                raise ValueError(f"inconsistent transitive scientific input: {path}")
            continue
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"scientific input/output changed: {path}")
        found[str(path)] = row
        if path.suffix == ".json":
            value = _read_json(path)
            if isinstance(value, dict) and "input_identity" in value:
                for key in ("inputs", "sources", "outputs", "input_identities"):
                    pending.extend(item for item in value.get(key, ())
                                   if isinstance(item, dict) and {"path", "sha256", "bytes"} <= set(item))
    return [found[key] for key in sorted(found)]


def _execute_scannet_branch(context: PhaseContext) -> PhaseResult:
    if "scannet_runtime" not in context.resolved_config:
        return _execute_boundary(context)
    capture = context.dependencies.get("capture")
    if capture is None or capture.status != ReceiptStatus.COMPLETE:
        return PhaseResult(ReceiptStatus.BLOCKED_PREREQUISITE,
                           blockers=("SCANNET_CAPTURE_NOT_COMPLETE",))
    from src.static_ovmap.module_validation.study_execution import verify_receipt

    config_path = Path(context.resolved_config.get("scannet_study_config",
        REPOSITORY_ROOT / "configs/evaluation/ovimap_module_scannet_study.json"))
    if not config_path.is_absolute():
        config_path = REPOSITORY_ROOT / config_path
    config = _read_json(config_path)
    runtime_path = Path(config["runtime_config"])
    runtime = _read_json(runtime_path if runtime_path.is_absolute() else REPOSITORY_ROOT / runtime_path)
    if runtime != context.resolved_config["scannet_runtime"]:
        raise ValueError("scientific driver runtime differs from the bound capture configuration")
    command = [runtime["mapping_python"], str(REPOSITORY_ROOT / f"scripts/evaluation/run_ovimap_scannet_{context.phase}.py"),
               "--config", str(config_path), "--phase", "all"]
    log_path = context.attempt_dir / f"{context.phase}_scientific.log"
    with log_path.open("a") as log:
        log_offset = log.tell()
        completed = subprocess.run(command, cwd=REPOSITORY_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        with log_path.open(errors="replace") as handle:
            handle.seek(log_offset)
            log = handle.read()
        for marker, blocker in (("GPU_BUSY:", "BLOCKED_GPU_BUSY"), ("BLOCKED_CAUSAL_LINEAGE", "BLOCKED_CAUSAL_LINEAGE")):
            if marker in log:
                return PhaseResult(ReceiptStatus.BLOCKED_PREREQUISITE, outputs={"log": str(log_path)}, blockers=(blocker,))
        raise RuntimeError(f"scientific {context.phase} driver failed ({completed.returncode}); see {log_path}")
    source = Path(config["study_root"]) / context.phase / "select_receipt.json"
    value = verify_receipt(source)
    rows = value["rows"] if context.phase != "query" else [result["row"] for result in value["results"]]
    output = context.attempt_dir / f"{context.phase}_runtime.json"
    atomic_write_json(output, {"status": "COMPLETE", "phase": context.phase, "scope": "LOCKED_SCANNET_DEVELOPMENT",
        "study_receipt": str(source), "command": command, "rows": rows, "learned_status": value["learned_status"],
        "input_identities": _scientific_identities(source)})
    return PhaseResult(ReceiptStatus.COMPLETE, outputs={"runtime": str(output), "log": str(log_path)},
        metrics={"scientific_evaluation_rows": len(rows), "scene_count": len({row["scene_id"] for row in rows}),
                 "learned_status": value["learned_status"]})


def semantic_phase(context: PhaseContext) -> PhaseResult:
    return _execute_scannet_branch(context)


def geometry_phase(context: PhaseContext) -> PhaseResult:
    return _execute_scannet_branch(context)


def query_phase(context: PhaseContext) -> PhaseResult:
    return _execute_scannet_branch(context)


def select_phase(context: PhaseContext) -> PhaseResult:
    dependency_blockers = sorted(
        {
            blocker
            for receipt in context.dependencies.values()
            for blocker in receipt.blockers
        }
    )
    if dependency_blockers:
        selection = {
            "schema_version": 1,
            "artifact_type": "OVIMAP_MODULE_FROZEN_SELECTION",
            "status": "NOT_FROZEN_PREREQUISITES",
            "teacher_id": None,
            "semantic_method": "N0",
            "geometry_method": "G_ORIGINAL",
            "query_method": None,
            "combination_methods": [],
            "final_candidate": "N0",
            "science_status": "INCONCLUSIVE_PREREQUISITES",
            "confirmation_status": "NOT_RUN_PREREQUISITES",
            "blockers": dependency_blockers,
            "prediction_code_identity": context.resolved_config.get("code_identity"),
            "split_lock": "splits.json",
        }
        status = ReceiptStatus.BLOCKED_PREREQUISITE
        blockers = tuple(dependency_blockers)
    else:
        return PhaseResult(
            status=ReceiptStatus.BLOCKED_PREREQUISITE,
            blockers=("MISSING_SELECTION_METRICS",),
        )
    output = context.attempt_dir / "selection.json"
    atomic_write_json(output, selection)
    return PhaseResult(
        status=status,
        outputs={"selection": output.name},
        metrics={"retained_modules": 0, "combination_variants": 0},
        blockers=blockers,
    )


def confirm_phase(context: PhaseContext) -> PhaseResult:
    if context.dependencies["select"].status != ReceiptStatus.COMPLETE:
        return PhaseResult(ReceiptStatus.BLOCKED_PREREQUISITE, blockers=("SELECTION_NOT_FROZEN",))
    selection_path = context.attempt_dir / "selection.json"
    if not selection_path.is_file():
        return PhaseResult(
            status=ReceiptStatus.BLOCKED_PREREQUISITE,
            blockers=("MISSING_FROZEN_SELECTION",),
        )
    selection = _read_json(selection_path)
    if selection.get("status") == "NOT_FROZEN_PREREQUISITES":
        return PhaseResult(
            ReceiptStatus.BLOCKED_PREREQUISITE,
            blockers=("SELECTION_NOT_FROZEN",),
        )
    if selection.get("final_candidate") == "N0":
        receipt = {
            "schema_version": 1,
            "artifact_type": "OVIMAP_MODULE_CONFIRMATION",
            "status": "NOT_REQUIRED_NO_RETAINED_CANDIDATE",
            "rows": [],
            "selection_path": str(selection_path),
        }
        output = context.attempt_dir / "confirmation.json"
        atomic_write_json(output, receipt)
        return PhaseResult(
            status=ReceiptStatus.NOT_REQUIRED_BY_FROZEN_GATE,
            outputs={"confirmation": output.name},
            metrics={"confirmation_rows": 0},
        )
    return PhaseResult(
        status=ReceiptStatus.BLOCKED_PREREQUISITE,
        blockers=("MISSING_CONFIRMATION_METRICS",),
    )


def report_phase(context: PhaseContext) -> PhaseResult:
    """Render only receipt-backed evidence, inside the selected attempt."""
    selection_path = context.attempt_dir / "selection.json"
    selection = _read_json(selection_path) if selection_path.is_file() and "select" in context.dependencies else {}
    blockers = sorted({
        blocker for receipt in context.dependencies.values() for blocker in receipt.blockers
    } | {
        f"MISSING_PHASE_RECEIPT:{phase}" for phase in PHASE_DEPENDENCIES["report"]
        if phase not in context.dependencies
    } | {
        f"PHASE_NOT_COMPLETE:{phase}" for phase, receipt in context.dependencies.items()
        if receipt.status not in {ReceiptStatus.COMPLETE, ReceiptStatus.NOT_REQUIRED_BY_FROZEN_GATE}
    })
    # The current release executes historical boundaries. The development driver
    # is still unimplemented, independently of local dataset availability.
    blockers = sorted(set(blockers) | {"UNIMPLEMENTED_DEVELOPMENT_SCENE_PIPELINE"})
    blocked = {}
    for method in REQUIRED_METHODS:
        phase = ("semantic" if method.startswith("S_") else
                 "geometry" if method.startswith("G_") else
                 "query" if method.startswith("Q_") else
                 "select" if method.startswith("COMBO_") else "capture")
        receipt = context.dependencies.get(phase)
        reasons = receipt.blockers if receipt else (f"MISSING_PHASE_RECEIPT:{phase}",)
        blocked[method] = ", ".join(reasons) or "NO_MEASURED_SCIENTIFIC_ROW"
    matrix = build_method_matrix(required_methods=REQUIRED_METHODS, measured={}, blocked=blocked)
    matrix_path = context.attempt_dir / "method_matrix.json"
    write_method_matrix(matrix_path, matrix)
    supporting, costs, external = [], {}, []
    for phase, receipt in context.dependencies.items():
        smoke_path = receipt.outputs.get("smoke")
        if not isinstance(smoke_path, str):
            continue
        value = _read_json(Path(smoke_path))
        supporting.append(f"{phase}: {value['scope']}; status {value['status']}; evidence {smoke_path}.")
        costs[phase] = {key: value[key] for key in ("physical", "scene_count", "frame_count", "policy_replay") if key in value}
        external.extend(value.get("input_identities", []))
    statuses = ReleaseStatuses(
        implementation="PARTIAL",
        experiment="BLOCKED_PREREQUISITES" if blockers else "MEASURED",
        science=str(selection.get("science_status", "INCONCLUSIVE_PREREQUISITES")),
        confirmation=str(selection.get("confirmation_status", "NOT_RUN_PREREQUISITES")),
        publication="NOT_CHECKED_BY_REPORT",
    )
    code = context.resolved_config.get("code_identity", {})
    commit = code.get("head") if isinstance(code, Mapping) else None
    command = shlex.join([
        sys.executable, str(REPOSITORY_ROOT / "scripts/evaluation/run_ovimap_module_study.py"),
        "--spec", str(context.spec.source_path), "--phase", "all",
        "--output-root", str(context.attempt_dir.parent),
        "--resolved-config", str(context.attempt_dir / "resolved_config.json"),
    ])
    results_path = context.attempt_dir / "MODULE_VALIDATION_RESULTS.md"
    handoff_path = context.attempt_dir / "MODULE_VALIDATION_HANDOFF.md"
    _atomic_write_text(results_path, render_results(
        matrix, final_candidate=str(selection.get("final_candidate", "UNSELECTED")),
        science_status=statuses.science, confirmation_status=statuses.confirmation,
        supporting_evidence=supporting,
    ))
    _atomic_write_text(handoff_path, render_handoff(
        statuses, branch="research/ovimap-module-validation-v1",
        commit=commit if isinstance(commit, str) and len(commit) == 40 else "0" * 40,
        evidence_paths=(str(matrix_path), str(context.attempt_dir / "receipts")),
        next_action="Connect the independent-scene development driver and provide authorized FIT/CAL/SELECT scenes before scientific selection.",
        execution_details=supporting, reproduction_commands=(command,),
    ))
    manifest_path = context.attempt_dir / "execution_manifest.json"
    atomic_write_json(manifest_path, {
        "schema_version": 2, "artifact_type": "OVIMAP_MODULE_EXECUTION_MANIFEST",
        "statuses": asdict(statuses), "attempt": str(context.attempt_dir),
        "code_identity": code, "cost": costs, "errors": blockers,
        "timings_seconds": {phase: receipt.metrics.get("elapsed_seconds") for phase, receipt in context.dependencies.items()},
        "phase_receipts": {phase: str(context.attempt_dir / "receipts" / f"{phase}.json") for phase in PHASE_DEPENDENCIES},
        "reproduction_command": command,
    })
    external_path = context.attempt_dir / "external_artifacts.json"
    atomic_write_json(external_path, {
        "schema_version": 2, "artifact_type": "OVIMAP_EXTERNAL_ARTIFACT_MANIFEST",
        "entries": list({row["path"]: {**row, "hash_scope": "file"} for row in external}.values()),
    })
    return PhaseResult(
        ReceiptStatus.COMPLETE,
        outputs={"method_matrix": str(matrix_path), "results": str(results_path),
                 "handoff": str(handoff_path), "execution_manifest": str(manifest_path),
                 "external_artifacts": str(external_path)},
        metrics={"required_method_rows": len(matrix), "measured_scientific_rows": 0},
        blockers=tuple(blockers),
    )


def default_phase_handlers() -> Mapping[str, PhaseHandler]:
    return {
        "bind": bind_phase,
        "capture": capture_phase,
        "semantic": semantic_phase,
        "geometry": geometry_phase,
        "query": query_phase,
        "select": select_phase,
        "confirm": confirm_phase,
        "report": report_phase,
    }


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
        untracked = run("ls-files", "--others", "--exclude-standard", "-z").split("\0")
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
        "untracked_content_digest": canonical_digest({name: sha256_file(repository_root / name) for name in untracked if name and (repository_root / name).is_file()}),
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
                    "orchestrator_schema": raw.get("schema_version", 1),
                    "spec_digest": spec.digest,
                    "resolved_config": nested,
                }
            )
            if raw["cache_key"] != expected_cache_key:
                raise ValueError("resolved config cache key mismatch")
            if "code_identity" in nested:
                nested["code_identity"] = _code_identity(REPOSITORY_ROOT)
            return nested
        raw["code_identity"] = _code_identity(REPOSITORY_ROOT)
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
    runtime = _read_json(REPOSITORY_ROOT / "configs/evaluation/ovimap_module_scannet_runtime.json")
    scannet_binding = resolution.to_dict().get("bindings", {}).get("scannet_root")
    if scannet_binding:
        runtime["data_root"] = scannet_binding["path"]
    return {
        "spec_path": str(spec.source_path),
        "repository_root": str(REPOSITORY_ROOT),
        "code_identity": _code_identity(REPOSITORY_ROOT),
        "environment_overrides": overrides,
        "asset_resolution": resolution.to_dict(),
        "scannet_runtime": runtime,
        "scannet_study_config": str(REPOSITORY_ROOT / "configs/evaluation/ovimap_module_scannet_study.json"),
        **_read_json(REPOSITORY_ROOT / "configs/evaluation/ovimap_module_historical_smoke.json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=tuple(PHASE_DEPENDENCIES) + ("all",))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--export-root", type=Path, help="Export finalized small receipts/reports into a new directory after execution")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = StudySpec.load(args.spec)
    output_root = args.output_root or spec.output_root
    runner = StudyRunner(
        spec,
        output_root,
        _base_resolved_config(spec, args.resolved_config),
        handlers=default_phase_handlers(),
    )
    results = runner.run(args.phase)
    if args.export_root is not None:
        export_attempt(runner, args.export_root)
    print(json.dumps({
        "attempt": str(runner.attempt_dir),
        "cache_key": runner.cache_key,
        "phases": {
            phase: {"status": run.receipt.status.value, "reused": run.reused}
            for phase, run in results.items()
        },
    }, sort_keys=True))
    return int(any(run.receipt.status == ReceiptStatus.FAILED for run in results.values()))


def export_attempt(runner: StudyRunner, destination: Path) -> None:
    """Export after report receipt and progress are finalized; no implicit repo writes."""
    destination = destination.resolve()
    if destination.is_relative_to(runner.attempt_dir) or runner.attempt_dir.is_relative_to(destination):
        raise ValueError("export directory must be separate from the attempt")
    destination.mkdir(parents=True, exist_ok=False)
    inventory = {}
    for path in sorted(runner.attempt_dir.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".md"}:
            continue
        relative = path.relative_to(runner.attempt_dir)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        inventory[str(relative)] = sha256_file(target)
    atomic_write_json(destination / "export_manifest.json", {
        "schema_version": 1, "source_attempt": str(runner.attempt_dir),
        "cache_key": runner.cache_key, "files": inventory,
    })


if __name__ == "__main__":
    raise SystemExit(main())
