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
    "report": (),
}
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


def _path_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    raise FileNotFoundError(path)


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


def _tooling_root(context: PhaseContext) -> Path:
    configured = context.resolved_config.get("tooling_root")
    return (
        Path(str(configured)).resolve()
        if configured is not None
        else context.spec.output_root / "tooling"
    )


def _load_tooling_receipt(
    context: PhaseContext, filename: str, expected_type: str
) -> tuple[Path, Mapping[str, Any]] | None:
    path = _tooling_root(context) / filename
    if not path.is_file():
        return None
    value = _read_json(path)
    if not isinstance(value, Mapping) or value.get("artifact_type") != expected_type:
        raise ValueError(f"tooling receipt has wrong identity: {path}")
    return path, value


def capture_phase(context: PhaseContext) -> PhaseResult:
    native = _load_tooling_receipt(
        context, "native_patch_receipt.json", "OVIMAP_NATIVE_PATCH_RECEIPT"
    )
    capture_manifest = (
        _tooling_root(context)
        / "historical_two_frame_capture_v3"
        / "room0"
        / "manifest.json"
    )
    if native is None or not capture_manifest.is_file():
        missing = []
        if native is None:
            missing.append("MISSING_NATIVE_PATCH_RECEIPT")
        if not capture_manifest.is_file():
            missing.append("MISSING_CURRENT_STATE_CAPTURE")
        return PhaseResult(
            status=ReceiptStatus.BLOCKED_PREREQUISITE,
            blockers=tuple(missing),
        )
    native_path, native_receipt = native
    summary = {
        "schema_version": 1,
        "artifact_type": "OVIMAP_CAPTURE_PHASE_SUMMARY",
        "status": "HISTORICAL_SMOKE_COMPLETE",
        "scientific_result": False,
        "native_patch_receipt": str(native_path),
        "native_patch_status": native_receipt.get("status"),
        "historical_capture_manifest": str(capture_manifest),
        "captured_scene_count": 1,
        "captured_frame_count": 2,
        "blocked_status": "BLOCKED_INDEPENDENT_SCENES",
    }
    output = context.attempt_dir / "capture_summary.json"
    atomic_write_json(output, summary)
    return PhaseResult(
        status=ReceiptStatus.PARTIAL,
        outputs={"capture_summary": output.name},
        metrics={"historical_scenes": 1, "historical_frames": 2},
        blockers=("BLOCKED_INDEPENDENT_SCENES",),
    )


def _module_smoke_phase(
    context: PhaseContext,
    *,
    filename: str,
    artifact_type: str,
    output_name: str,
) -> PhaseResult:
    receipt = _load_tooling_receipt(context, filename, artifact_type)
    if receipt is None:
        return PhaseResult(
            status=ReceiptStatus.BLOCKED_PREREQUISITE,
            blockers=(f"MISSING_TOOLING_RECEIPT:{filename}",),
        )
    path, value = receipt
    summary = {
        "schema_version": 1,
        "artifact_type": f"OVIMAP_{context.phase.upper()}_PHASE_SUMMARY",
        "status": "MODULE_SMOKE_COMPLETE_SCIENTIFIC_BRANCH_BLOCKED",
        "scientific_result": False,
        "evidence_path": str(path),
        "evidence_status": value.get("status"),
        "blocked_status": "BLOCKED_INDEPENDENT_SCENES",
    }
    output = context.attempt_dir / output_name
    atomic_write_json(output, summary)
    return PhaseResult(
        status=ReceiptStatus.PARTIAL,
        outputs={f"{context.phase}_summary": output.name, "evidence": str(path)},
        metrics={"module_smokes": 1, "scientific_scene_families": 0},
        blockers=("BLOCKED_INDEPENDENT_SCENES",),
    )


def semantic_phase(context: PhaseContext) -> PhaseResult:
    return _module_smoke_phase(
        context,
        filename="semantic_adapter_receipt.json",
        artifact_type="OVIMAP_SEMANTIC_ADAPTER_RECEIPT",
        output_name="semantic_summary.json",
    )


def geometry_phase(context: PhaseContext) -> PhaseResult:
    return _module_smoke_phase(
        context,
        filename="geometry_snapshot_smoke_receipt.json",
        artifact_type="OVIMAP_GEOMETRY_SNAPSHOT_SMOKE_RECEIPT",
        output_name="geometry_summary.json",
    )


def query_phase(context: PhaseContext) -> PhaseResult:
    return _module_smoke_phase(
        context,
        filename="query_current_state_smoke_receipt.json",
        artifact_type="OVIMAP_QUERY_CURRENT_STATE_SMOKE_RECEIPT",
        output_name="query_summary.json",
    )


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
            "status": "FROZEN_NO_RETAINED_CANDIDATE",
            "teacher_id": None,
            "semantic_method": "N0",
            "geometry_method": "G_ORIGINAL",
            "query_method": None,
            "combination_methods": [],
            "final_candidate": "N0",
            "science_status": "INCONCLUSIVE_PREREQUISITES",
            "confirmation_status": "NOT_REQUIRED_NO_RETAINED_CANDIDATE",
            "blockers": dependency_blockers,
            "prediction_code_identity": context.resolved_config.get("code_identity"),
            "split_lock": "splits.json",
        }
        status = ReceiptStatus.COMPLETE
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
    selection_path = context.attempt_dir / "selection.json"
    if not selection_path.is_file():
        return PhaseResult(
            status=ReceiptStatus.BLOCKED_PREREQUISITE,
            blockers=("MISSING_FROZEN_SELECTION",),
        )
    selection = _read_json(selection_path)
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
    selection_path = context.attempt_dir / "selection.json"
    selection = (
        _read_json(selection_path)
        if selection_path.is_file()
        else {
            "final_candidate": "N0",
            "science_status": "INCONCLUSIVE_PREREQUISITES",
            "confirmation_status": "NOT_REQUIRED_NO_RETAINED_CANDIDATE",
        }
    )
    scientific_methods = tuple(
        method
        for method in REQUIRED_METHODS
        if method not in {"N0", "COMBO_GS", "COMBO_Q_REFINEMENT"}
    )
    blocked = {method: "BLOCKED_INDEPENDENT_SCENES" for method in scientific_methods}
    not_required = {
        "COMBO_GS": "NOT_REQUIRED_NO_INDEPENDENTLY_RETAINED_MODULES",
        "COMBO_Q_REFINEMENT": "NOT_REQUIRED_NO_INDEPENDENTLY_RETAINED_MODULES",
    }
    resolution = context.resolved_config.get("asset_resolution", {})
    bindings = resolution.get("bindings", {}) if isinstance(resolution, Mapping) else {}
    historical = bindings.get("historical_evaluation", {}) if isinstance(bindings, Mapping) else {}
    native_evidence = historical.get("path") if isinstance(historical, Mapping) else None
    matrix = build_method_matrix(
        required_methods=REQUIRED_METHODS,
        measured={},
        blocked=blocked,
        not_required=not_required,
        reused={"N0": str(native_evidence or "HISTORICAL_NATIVE_REFERENCE")},
        evidence_paths={
            "N0": None if native_evidence is None else str(native_evidence),
            **{
                method: str(context.attempt_dir / f"{branch}_summary.json")
                for branch, methods in {
                    "semantic": tuple(method for method in scientific_methods if method.startswith("S_")),
                    "geometry": tuple(method for method in scientific_methods if method.startswith("G_")),
                    "query": tuple(method for method in scientific_methods if method.startswith("Q_")),
                }.items()
                for method in methods
            },
        },
    )
    matrix_path = context.attempt_dir / "method_matrix.json"
    write_method_matrix(matrix_path, matrix)
    semantic_receipt = _load_tooling_receipt(
        context, "semantic_adapter_receipt.json", "OVIMAP_SEMANTIC_ADAPTER_RECEIPT"
    )
    native_receipt = _load_tooling_receipt(
        context, "native_patch_receipt.json", "OVIMAP_NATIVE_PATCH_RECEIPT"
    )
    geometry_receipt = _load_tooling_receipt(
        context,
        "geometry_snapshot_smoke_receipt.json",
        "OVIMAP_GEOMETRY_SNAPSHOT_SMOKE_RECEIPT",
    )
    query_receipt = _load_tooling_receipt(
        context,
        "query_current_state_smoke_receipt.json",
        "OVIMAP_QUERY_CURRENT_STATE_SMOKE_RECEIPT",
    )
    supporting = [
        "These are implementation-boundary smokes on historical Room0, not SELECT measurements.",
    ]
    if semantic_receipt is not None:
        semantic_value = semantic_receipt[1]
        native_siglip = semantic_value.get("native_siglip", {})
        supporting.append(
            "Semantic adapters: native SigLIP crop vectors "
            f"{native_siglip.get('crop_vector_shape')}; legacy six-crop max error "
            f"{native_siglip.get('legacy_first_six_max_abs_error')}."
        )
    if geometry_receipt is not None:
        geometry_value = geometry_receipt[1]
        supporting.append(
            f"Geometry smoke: {geometry_value.get('surface_rows')} surface rows, "
            f"{geometry_value.get('leaf_count')} native leaves, "
            f"{geometry_value.get('complete_hypothesis_count')} complete bounded hypotheses."
        )
    if query_receipt is not None:
        query_value = query_receipt[1]
        parity = query_value.get("candidate_parity", ())
        native_rows = query_value.get("native_combine_parity", {}).get(
            "frame_rows", ()
        )
        replay = query_value.get("policy_replay", {})
        supporting.append(
            f"Query smoke: {sum(int(row.get('candidate_count', 0)) for row in parity)} "
            "current-state candidates with exact request/mask/bbox parity and "
            f"{sum(int(row.get('selected_count', 0)) for row in native_rows)} native "
            "combine selections with exact request IDs; "
            f"Q_AREA logical cost {replay.get('Q_AREA')}; Q_UNCERTAINTY logical cost "
            f"{replay.get('Q_UNCERTAINTY')}; physical cost {replay.get('physical')}."
        )
    results = render_results(
        matrix,
        final_candidate=str(selection.get("final_candidate", "N0")),
        science_status=str(
            selection.get("science_status", "INCONCLUSIVE_PREREQUISITES")
        ),
        confirmation_status=str(
            selection.get(
                "confirmation_status", "NOT_REQUIRED_NO_RETAINED_CANDIDATE"
            )
        ),
        supporting_evidence=supporting,
    )
    code = context.resolved_config.get("code_identity", {})
    commit = code.get("head") if isinstance(code, Mapping) else None
    if not isinstance(commit, str) or len(commit) != 40:
        commit = "0" * 40
    statuses = ReleaseStatuses(
        implementation="COMPLETE",
        experiment="BLOCKED_INDEPENDENT_SCENES",
        science=str(selection.get("science_status", "INCONCLUSIVE_PREREQUISITES")),
        confirmation=str(
            selection.get(
                "confirmation_status", "NOT_REQUIRED_NO_RETAINED_CANDIDATE"
            )
        ),
        publication="PENDING_FINAL_COMMIT_AND_PUSH",
    )
    evidence = (
        str(context.attempt_dir / "capture_summary.json"),
        str(context.attempt_dir / "semantic_summary.json"),
        str(context.attempt_dir / "geometry_summary.json"),
        str(context.attempt_dir / "query_summary.json"),
        str(selection_path),
        str(matrix_path),
    )
    resolved_path = context.attempt_dir / "resolved_config.json"
    command_prefix = (
        "python scripts/evaluation/run_ovimap_module_study.py "
        "--spec docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json "
        f"--resolved-config {resolved_path} --output-root {context.spec.output_root}"
    )
    reproduction_commands = tuple(
        f"{command_prefix} --phase {phase}" for phase in context.spec.phases[:-1]
    )
    details = list(supporting)
    details.append(f"Worktree: {REPOSITORY_ROOT}.")
    if native_receipt is not None:
        native_value = native_receipt[1]
        upstream = native_value.get("upstream", {})
        build = native_value.get("build", {})
        details.extend(
            (
                f"Pinned OVI source: {upstream.get('root')} at {upstream.get('commit')}.",
                (
                    "Loaded native extension: "
                    f"{build.get('extension_path')} "
                    f"(sha256 {build.get('extension_sha256')})."
                ),
            )
        )
    if semantic_receipt is not None:
        semantic_value = semantic_receipt[1]
        details.append(
            "Bound model roots: "
            + ", ".join(
                str(semantic_value.get(name, {}).get("model_path"))
                for name in ("native_siglip", "siglip2", "wow", "name_mapping")
            )
            + "."
        )
    handoff = render_handoff(
        statuses,
        branch=context.spec.task_branch,
        commit=commit,
        evidence_paths=evidence,
        next_action="Provide at least 14 eligible independent scene families, including the required ScanNet captures, then resume the locked FIT/CAL/SELECT/CONFIRM phases.",
        execution_details=details,
        reproduction_commands=reproduction_commands,
    )
    docs = REPOSITORY_ROOT / "docs" / "paper" / "static_ovmap"
    results_path = docs / "MODULE_VALIDATION_RESULTS.md"
    handoff_path = docs / "MODULE_VALIDATION_HANDOFF.md"
    _atomic_write_text(results_path, results)
    _atomic_write_text(handoff_path, handoff)
    artifact_root = (
        REPOSITORY_ROOT / "artifacts" / "static_ovmap" / "module_validation_v1"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_method_matrix(artifact_root / "method_matrix.json", matrix)
    atomic_write_json(artifact_root / "selection.json", selection)
    execution_manifest = {
        "schema_version": 1,
        "artifact_type": "OVIMAP_MODULE_EXECUTION_MANIFEST",
        "experiment_code_identity": context.resolved_config.get("code_identity"),
        "commands": [
            (
                "python scripts/evaluation/run_ovimap_module_study.py "
                "--spec docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json "
                f"--phase all --output-root {context.spec.output_root}"
            ),
            *reproduction_commands,
        ],
        "costs": {
            "capture": {"scenes": 1, "frames": 2},
            "semantic_smoke": {
                "native_siglip_requests": 1,
                "siglip2_requests": 1,
                "wow_generations": 1,
                "training_runs": 0,
            },
            "geometry_smoke": (
                {}
                if geometry_receipt is None
                else {
                    key: geometry_receipt[1].get(key)
                    for key in (
                        "surface_rows",
                        "surface_faces",
                        "leaf_count",
                        "complete_hypothesis_count",
                    )
                }
            ),
            "query_smoke": (
                {} if query_receipt is None else query_receipt[1].get("policy_replay", {})
            ),
            "scientific_training_runs": 0,
            "scientific_evaluation_rows": 0,
        },
        "errors": ["BLOCKED_INDEPENDENT_SCENES"],
        "timings_seconds": None,
        "timing_status": "NOT_RECORDED_FOR_PREEXISTING_BOUNDARY_SMOKES",
    }
    atomic_write_json(artifact_root / "execution_manifest.json", execution_manifest)

    external_entries: list[dict[str, Any]] = []

    def add_external(
        artifact_id: str,
        raw_path: Any,
        digest: Any,
        *,
        hash_scope: str,
        regeneration_command: str | None,
        regeneration_blocker: str | None = None,
    ) -> None:
        if not isinstance(raw_path, str) or not isinstance(digest, str):
            return
        path = Path(raw_path)
        if not path.exists():
            return
        external_entries.append(
            {
                "artifact_id": artifact_id,
                "path": str(path),
                "bytes": _path_bytes(path),
                "sha256": digest,
                "hash_scope": hash_scope,
                "regeneration_command": regeneration_command,
                "regeneration_blocker": regeneration_blocker,
            }
        )

    if native_receipt is not None:
        native_value = native_receipt[1]
        build = native_value.get("build", {})
        smoke = native_value.get("smoke", {})
        add_external(
            "native_extension",
            build.get("extension_path"),
            build.get("extension_sha256"),
            hash_scope="file",
            regeneration_command=None,
            regeneration_blocker="The isolated ABI build command is preserved by the native receipt but is not automated by the v1 study CLI.",
        )
        capture_manifest = smoke.get("capture_manifest_path")
        if isinstance(capture_manifest, str):
            add_external(
                "native_capture",
                str(Path(capture_manifest).parent),
                smoke.get("capture_identity"),
                hash_scope="native capture identity over the manifest-bound frame/surface payload",
                regeneration_command=None,
                regeneration_blocker="Requires replaying the pinned native mapper with the receipt-bound historical trajectory.",
            )
        add_external(
            "native_instance_mesh",
            smoke.get("mesh_path"),
            smoke.get("mesh_sha256"),
            hash_scope="file",
            regeneration_command=None,
            regeneration_blocker="Requires replaying the pinned native mapper with the receipt-bound historical trajectory.",
        )
    if semantic_receipt is not None:
        semantic_value = semantic_receipt[1]
        model_specs = (
            ("native_siglip", "native_siglip", None, None),
            (
                "siglip2",
                "siglip2",
                "google/siglip2-large-patch16-384",
                semantic_value.get("siglip2", {}).get("revision"),
            ),
            (
                "wow_weights",
                "wow",
                "AAwcAA/WOW-Seg",
                semantic_value.get("wow", {}).get("weights_revision"),
            ),
            (
                "name_mapping",
                "name_mapping",
                "sentence-transformers/all-MiniLM-L6-v2",
                semantic_value.get("name_mapping", {}).get("revision"),
            ),
        )
        for artifact_id, key, repo_id, revision in model_specs:
            model = semantic_value.get(key, {})
            command = None
            blocker = "The bound native model receipt has no immutable download revision."
            if repo_id is not None and isinstance(revision, str):
                command = (
                    "python -c \"from huggingface_hub import snapshot_download; "
                    f"snapshot_download(repo_id='{repo_id}', revision='{revision}', "
                    f"local_dir='{model.get('model_path')}')\""
                )
                blocker = None
            add_external(
                artifact_id,
                model.get("model_path"),
                model.get("model_sha256"),
                hash_scope="canonical directory content identity from adapter receipt",
                regeneration_command=command,
                regeneration_blocker=blocker,
            )
    atomic_write_json(
        artifact_root / "external_artifacts.json",
        {
            "schema_version": 1,
            "artifact_type": "OVIMAP_EXTERNAL_ARTIFACT_MANIFEST",
            "entries": external_entries,
        },
    )
    for name in (
        "resolved_config.json",
        "asset_resolution.json",
        "scene_inventory.json",
        "splits.json",
        "capture_summary.json",
        "semantic_summary.json",
        "geometry_summary.json",
        "query_summary.json",
        "confirmation.json",
    ):
        source = context.attempt_dir / name
        if source.is_file():
            atomic_write_json(artifact_root / name, _read_json(source))
    receipt_artifacts = artifact_root / "receipts"
    for phase in context.spec.phases[:-1]:
        source = context.attempt_dir / "receipts" / f"{phase}.json"
        if source.is_file():
            atomic_write_json(receipt_artifacts / source.name, _read_json(source))
    tooling_artifacts = artifact_root / "tooling"
    for receipt in (native_receipt, semantic_receipt, geometry_receipt, query_receipt):
        if receipt is not None:
            atomic_write_json(tooling_artifacts / receipt[0].name, receipt[1])
    progress = context.attempt_dir / "progress.md"
    if progress.is_file():
        _atomic_write_text(
            REPOSITORY_ROOT
            / "docs/paper/static_ovmap/module_validation_v1/progress.md",
            progress.read_text(encoding="utf-8"),
        )
    return PhaseResult(
        status=ReceiptStatus.COMPLETE,
        outputs={
            "method_matrix": str(matrix_path),
            "results": str(results_path),
            "handoff": str(handoff_path),
        },
        metrics={
            "principal_tables": 5,
            "required_method_rows": len(matrix),
            "measured_scientific_rows": 0,
        },
        blockers=("BLOCKED_INDEPENDENT_SCENES",),
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
        handlers=default_phase_handlers(),
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
