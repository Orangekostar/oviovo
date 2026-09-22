"""Bounded asset discovery and leakage-safe physical-scene splits."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import ReceiptStatus, StudySpec, canonical_digest

_SCANNET_FAMILY = re.compile(r"^(scene\d{4})(?:_[0-9]+)?$")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AssetRequirement:
    key: str
    kind: str
    relative_paths: tuple[str, ...]
    configured_paths: tuple[Path, ...] = ()
    required: bool = True
    expected_git_commit: str | None = None
    manifest_names: tuple[str, ...] = (
        "manifest.json",
        "receipt.json",
        "config.json",
    )

    def __post_init__(self) -> None:
        if not self.key or self.kind not in {"file", "directory"}:
            raise ValueError("asset requirement needs a key and file/directory kind")
        if not self.relative_paths and not self.configured_paths:
            raise ValueError("asset requirement needs a discoverable or configured path")
        object.__setattr__(self, "configured_paths", tuple(map(Path, self.configured_paths)))


@dataclass(frozen=True)
class AssetIdentity:
    kind: str
    sha256: str
    size_bytes: int | None
    manifest_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "manifest_path": self.manifest_path,
        }


@dataclass(frozen=True)
class AssetBinding:
    key: str
    path: Path
    source: str
    identity: AssetIdentity
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "path": str(self.path),
            "source": self.source,
            "identity": self.identity.to_dict(),
            "aliases": list(self.aliases),
        }


@dataclass(frozen=True)
class AssetResolution:
    status: ReceiptStatus
    bindings: Mapping[str, AssetBinding]
    ambiguities: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    searched_roots: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "bindings": {
                key: binding.to_dict() for key, binding in sorted(self.bindings.items())
            },
            "ambiguities": {
                key: list(paths) for key, paths in sorted(self.ambiguities.items())
            },
            "missing": list(self.missing),
            "searched_roots": list(self.searched_roots),
        }


def _matches_kind(path: Path, kind: str) -> bool:
    return path.is_file() if kind == "file" else path.is_dir()


def _asset_identity(path: Path, requirement: AssetRequirement) -> AssetIdentity:
    if path.is_file():
        return AssetIdentity("file", sha256_file(path), path.stat().st_size)
    git_commit = _git_commit(path)
    if git_commit is not None:
        return AssetIdentity("directory", canonical_digest({"git_commit": git_commit}), None)
    for name in requirement.manifest_names:
        manifest = path / name
        if manifest.is_file():
            return AssetIdentity(
                "directory",
                sha256_file(manifest),
                None,
                str(manifest.resolve()),
            )
    digest = canonical_digest({"directory_realpath": str(path.resolve())})
    return AssetIdentity("directory", digest, None)


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if re.fullmatch(r"[0-9a-f]{40}", commit) else None


def _split_roots(value: str | None) -> tuple[Path, ...]:
    if not value:
        return ()
    normalized = value.replace("\r", "\n").replace(os.pathsep, "\n")
    return tuple(Path(item.strip()).expanduser() for item in normalized.splitlines() if item.strip())


def _bounded_index(roots: Sequence[Path], max_depth: int = 4) -> dict[str, list[Path]]:
    """Index each authorized root once without following links or leaving depth four."""

    index: dict[str, list[Path]] = {}
    for raw_root in roots:
        root = raw_root.resolve()
        if not root.exists():
            continue
        if root.is_file():
            index.setdefault(root.name, []).append(root)
            continue
        index.setdefault(root.name, []).append(root)
        for current, directories, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            depth = len(current_path.relative_to(root).parts)
            if depth >= max_depth:
                directories[:] = []
            for name in directories:
                candidate = (current_path / name).resolve()
                if candidate.is_relative_to(root):
                    index.setdefault(name, []).append(candidate)
            for name in files:
                candidate = (current_path / name).resolve()
                if candidate.is_relative_to(root):
                    index.setdefault(name, []).append(candidate)
    return index


def _json_paths(value: Any) -> Iterable[Path]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _json_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from _json_paths(item)
    elif isinstance(value, str) and Path(value).is_absolute():
        yield Path(value)


def _receipt_candidates(receipt_paths: Sequence[Path]) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for receipt in receipt_paths:
        if not receipt.is_file():
            continue
        try:
            raw = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        candidates.extend(path for path in _json_paths(raw) if path.exists())
    return tuple(candidates)


def _resolve_tier(
    requirement: AssetRequirement,
    candidates: Iterable[Path],
    source: str,
) -> tuple[AssetBinding | None, tuple[str, ...]]:
    unique_paths = sorted(
        {
            path.resolve()
            for path in candidates
            if _matches_kind(path, requirement.kind)
            and (
                requirement.expected_git_commit is None
                or _git_commit(path.resolve()) == requirement.expected_git_commit
            )
        },
        key=str,
    )
    if not unique_paths:
        return None, ()
    identities = [(path, _asset_identity(path, requirement)) for path in unique_paths]
    distinct = {identity.sha256 for _, identity in identities}
    if len(distinct) > 1:
        return None, tuple(str(path) for path, _ in identities)
    path, identity = identities[0]
    return (
        AssetBinding(
            key=requirement.key,
            path=path,
            source=source,
            identity=identity,
            aliases=tuple(str(alias) for alias, _ in identities[1:]),
        ),
        (),
    )


def resolve_assets(
    spec: StudySpec,
    env: Mapping[str, str],
    *,
    requirements: Sequence[AssetRequirement] | None = None,
    receipt_paths: Sequence[Path] = (),
    repository_root: Path | None = None,
) -> AssetResolution:
    """Resolve explicit roots, configured paths, then receipt-linked paths."""

    if requirements is None:
        root = repository_root or Path(__file__).resolve().parents[3]
        requirements, default_receipts = default_asset_requirements(spec, root)
        if not receipt_paths:
            receipt_paths = default_receipts
    roots = (
        *_split_roots(env.get("OVIMAP_DATA_ROOTS")),
        *_split_roots(env.get("OVIMAP_MODEL_ROOTS")),
        *_split_roots(env.get("OVIMAP_REPLAY_ROOT")),
    )
    index = _bounded_index(roots)
    receipt_candidates = _receipt_candidates(tuple(map(Path, receipt_paths)))
    bindings: dict[str, AssetBinding] = {}
    ambiguities: dict[str, tuple[str, ...]] = {}
    missing: list[str] = []

    for requirement in requirements:
        names = {Path(relative).name for relative in requirement.relative_paths}
        environment_candidates: list[Path] = []
        for root in roots:
            if root.is_file() and root.name in names:
                environment_candidates.append(root)
            elif root.is_dir():
                resolved_root = root.resolve()
                if requirement.key == "scannet_root" and all(
                    (root / name).is_file()
                    for name in ("scannetv2_train.txt", "scannetv2_val.txt")
                ):
                    environment_candidates.append(resolved_root)
                environment_candidates.extend(
                    candidate
                    for relative in requirement.relative_paths
                    if (candidate := (root / relative).resolve()).is_relative_to(resolved_root)
                )
        for relative in requirement.relative_paths:
            if len(Path(relative).parts) == 1:
                environment_candidates.extend(index.get(Path(relative).name, ()))
        tiers = (
            ("environment", environment_candidates),
            ("configuration", requirement.configured_paths),
            (
                "receipt",
                [path for path in receipt_candidates if path.name in names],
            ),
        )
        for source, candidates in tiers:
            binding, ambiguous = _resolve_tier(requirement, candidates, source)
            if ambiguous:
                ambiguities[requirement.key] = ambiguous
                break
            if binding is not None:
                bindings[requirement.key] = binding
                break
        else:
            if requirement.required:
                missing.append(requirement.key)

    if ambiguities:
        status = ReceiptStatus.BLOCKED_ASSET_IDENTITY
    elif missing:
        status = ReceiptStatus.BLOCKED_PREREQUISITE
    else:
        status = ReceiptStatus.COMPLETE
    return AssetResolution(
        status=status,
        bindings=bindings,
        ambiguities=ambiguities,
        missing=tuple(sorted(missing)),
        searched_roots=tuple(str(path.resolve()) for path in roots),
    )


def _load_mapping(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"configuration must be a JSON object: {path}")
    return raw


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        value = value[key]
    return value


def default_asset_requirements(
    spec: StudySpec,
    repository_root: Path,
) -> tuple[tuple[AssetRequirement, ...], tuple[Path, ...]]:
    """Build the fixed v1 requirements from the three reviewed configurations."""

    config_root = repository_root / "configs/evaluation"
    t1_path = config_root / "ovimap_t1_attribution_v1.json"
    object_path = config_root / "ovimap_object_decoupling_v1.json"
    semantic_path = config_root / "ovimap_sf_ovi_semantics_v1.json"
    t1 = _load_mapping(t1_path)
    object_config = _load_mapping(object_path)
    semantic = _load_mapping(semantic_path)
    historical_t0 = Path(_nested(t1, "runs", 0, "t0"))
    known_scannet_roots = (
        Path("/home/ww/oviovo_benchmark_assets/scannet200_release"),
        Path("/home/ww/oviovo_benchmark_assets/scannet200"),
    )
    clean_upstream = repository_root.parent / f"{repository_root.name}-upstream"
    requirements = (
        AssetRequirement(
            "historical_predictions",
            "directory",
            ("ovimap-t1-attribution-v1/predictions_verified", "predictions_verified"),
            (Path(_nested(object_config, "recorded_roots_to_verify", "frozen_prediction_root")),),
        ),
        AssetRequirement(
            "historical_evaluation",
            "directory",
            ("ovimap-t1-attribution-v1/evaluation", "evaluation"),
            (Path(_nested(object_config, "recorded_roots_to_verify", "reference_evaluation_root")),),
        ),
        AssetRequirement(
            "object_decoupling_root",
            "directory",
            ("ovimap-object-semantic-decoupling-v1",),
            (Path(_nested(semantic, "recorded_roots_to_verify", "previous_od_root")),),
        ),
        AssetRequirement(
            "semantic_study_root",
            "directory",
            ("ovimap-sf-to-ovi-semantics-v1",),
            (Path(_nested(semantic, "new_output_root")),),
        ),
        AssetRequirement(
            "historical_baseline_root",
            "directory",
            ("20260913_static_ovmap",),
            (historical_t0.parents[2],),
        ),
        AssetRequirement(
            "official_ovi_checkout",
            "directory",
            ("OVI-MAP", "ovimap-module-validation-upstream"),
            (clean_upstream, Path(t1["evaluator_root"])),
            expected_git_commit=spec.official_commit,
        ),
        AssetRequirement(
            "incumbent_model",
            "directory",
            ("google_siglip-large-patch16-384",),
            (Path(_nested(object_config, "runtime_assets_to_resolve", "native_model_root")),),
        ),
        AssetRequirement(
            "native_text_cache",
            "file",
            ("replica51_native.npz",),
            (Path(t1["text_cache"]),),
        ),
        AssetRequirement(
            "historical_native_mesh",
            "file",
            ("instance_mesh_200.ply",),
            (Path(t1["native_mesh"]),),
        ),
        AssetRequirement(
            "scannet_root",
            "directory",
            ("scannet200_release", "scannet200", "scannet", "ScanNet"),
            known_scannet_roots,
            manifest_names=("acquisition_lock.json", "acquisition_plan.json", "manifest.json", "receipt.json"),
        ),
        AssetRequirement(
            "siglip2_model",
            "directory",
            ("google_siglip2-large-patch16-384", "siglip2-large-patch16-384"),
            required=False,
        ),
        AssetRequirement(
            "wow_model",
            "directory",
            ("WOW-Seg", "models--AAwcAA--WOW-Seg"),
            required=False,
        ),
        AssetRequirement(
            "minilm_model",
            "directory",
            ("all-MiniLM-L6-v2", "models--sentence-transformers--all-MiniLM-L6-v2"),
            required=False,
        ),
    )
    receipt_paths: set[Path] = set()
    for config in (t1, object_config, semantic):
        for path in _json_paths(config):
            if "receipt" in path.name.lower():
                receipt_paths.add(path)
    return requirements, tuple(sorted(receipt_paths, key=str))


def physical_family(scene_id: str) -> str:
    match = _SCANNET_FAMILY.fullmatch(scene_id)
    return match.group(1) if match else scene_id


def exposed_scannet_families(repository_root: Path) -> tuple[str, ...]:
    """Read the frozen historical ScanNet18 protocol instead of guessing exposure."""

    protocol_path = repository_root / "artifacts/static_ovmap/protocol_manifest.json"
    try:
        protocol = _load_mapping(protocol_path)
        scene_ids = _nested(protocol, "ScanNet", "scene_ids")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return ()
    return tuple(sorted({physical_family(str(scene_id)) for scene_id in scene_ids}))


def _single_named_file(root: Path, name: str) -> Path | None:
    index = _bounded_index((root,), max_depth=4)
    matches = sorted({path.resolve() for path in index.get(name, ()) if path.is_file()}, key=str)
    if not matches:
        return None
    digests = {sha256_file(path) for path in matches}
    if len(digests) != 1:
        raise ValueError(f"ambiguous differing {name} files below {root}: {matches}")
    return matches[0]


def _split_scene_ids(path: Path) -> tuple[str, ...]:
    values = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not values or any(_SCANNET_FAMILY.fullmatch(value) is None for value in values):
        raise ValueError(f"invalid or empty ScanNet split file: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate scene ID in ScanNet split file: {path}")
    return values


def _metadata_frame_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in (
        r"(?im)^\s*num(?:Depth|Color)?Frames\s*=\s*(\d+)\s*$",
        r"(?im)^\s*frameCount\s*=\s*(\d+)\s*$",
    ):
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def native_schedule(frame_count: int) -> dict[str, Any]:
    """First 200 original native slots; never renumber or backfill frames."""
    if frame_count < 200:
        raise ValueError("native schedule requires at least 200 frames")
    step = frame_count // 200
    frames = list(range(0, frame_count, step))[:200]
    return {"start": 0, "source_end": frame_count, "source_step": -1,
            "step": step, "end": frames[-1] + 1, "frame_ids": frames}


def build_scannet_inventory(
    dataset_root: Path | str,
    *,
    train_split_path: Path | None = None,
    validation_split_path: Path | None = None,
) -> dict[str, Any]:
    """Inventory official train/validation captures without reading outcomes."""

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise ValueError(f"ScanNet root is not a directory: {root}")
    train_path = train_split_path or _single_named_file(root, "scannetv2_train.txt")
    validation_path = validation_split_path or _single_named_file(root, "scannetv2_val.txt")
    if train_path is None or validation_path is None:
        return {
            "schema_version": 1,
            "status": "BLOCKED_SPLIT_FILES",
            "dataset_root": str(root),
            "train_split_path": str(train_path) if train_path else None,
            "validation_split_path": str(validation_path) if validation_path else None,
            "scenes": [],
        }
    train_ids = _split_scene_ids(Path(train_path))
    validation_ids = _split_scene_ids(Path(validation_path))
    overlap = set(train_ids) & set(validation_ids)
    if overlap:
        raise ValueError(f"ScanNet train/validation overlap: {sorted(overlap)}")
    scans_root = root / "scans"
    if not scans_root.is_dir():
        scans_root = root
    suffixes = (
        ".sens",
        ".txt",
        "_vh_clean_2.ply",
        "_vh_clean_2.labels.ply",
        "_vh_clean_2.0.010000.segs.json",
        ".aggregation.json",
    )
    scenes: list[dict[str, Any]] = []
    for source_split, scene_ids in (("train", train_ids), ("validation", validation_ids)):
        for scene_id in scene_ids:
            scene_root = scans_root / scene_id
            paths = {suffix: scene_root / f"{scene_id}{suffix}" for suffix in suffixes}
            export_roots = sorted({
                candidate.resolve()
                for candidate in (root / "exported" / scene_id, scene_root)
                if all((candidate / name).is_dir() for name in ("color", "depth", "pose", "intrinsic"))
                and all((candidate / "intrinsic" / name).is_file()
                        for name in ("intrinsic_color.txt", "intrinsic_depth.txt"))
                and all(any((candidate / folder).glob(pattern)) for folder, pattern in
                        (("color", "[0-9]*.jpg"), ("depth", "[0-9]*.png"), ("pose", "[0-9]*.txt")))
            }, key=str)
            if len(export_roots) > 1:
                raise ValueError(f"ambiguous native ScanNet input roots: {export_roots}")
            native_input = export_roots[0] if export_roots else None
            missing = [str(path) for suffix, path in paths.items()
                       if not (suffix == ".sens" and native_input is not None)
                       and (not path.is_file() or path.stat().st_size <= 0)]
            frame_count = _metadata_frame_count(paths[".txt"])
            schedule = None
            if frame_count is None or frame_count < 200:
                missing.append("metadata:numDepthFrames>=200")
            else:
                schedule = native_schedule(frame_count)
            missing_frames = []
            if native_input is not None and schedule is not None:
                missing_frames = [index for index in schedule["frame_ids"] if not all(
                    (native_input / folder / f"{index}{suffix}").is_file()
                    for folder, suffix in (("color", ".jpg"), ("depth", ".png"), ("pose", ".txt"))
                )]
            scenes.append(
                {
                    "scene_id": scene_id,
                    "family_id": physical_family(scene_id),
                    "source_split": source_split,
                    "complete": not missing,
                    "frame_count": frame_count,
                    "scene_root": str(scene_root.resolve()),
                    "native_input_root": str(native_input) if native_input else None,
                    "input_state": "EXPORTED_NATIVE_INPUTS" if native_input else "RAW_REQUIRES_EXPORT",
                    "schedule": schedule,
                    "missing_scheduled_frame_ids": missing_frames,
                    "missing": missing,
                }
            )
    return {
        "schema_version": 1,
        "status": "COMPLETE" if any(scene["complete"] for scene in scenes) else "NO_COMPLETE_SCENES",
        "dataset_root": str(root),
        "scans_root": str(scans_root.resolve()),
        "train_split_path": str(Path(train_path).resolve()),
        "train_split_sha256": sha256_file(train_path),
        "validation_split_path": str(Path(validation_path).resolve()),
        "validation_split_sha256": sha256_file(validation_path),
        "scenes": scenes,
    }


def _family_order(family_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"ovimap-module-v1|{family_id}".encode()).hexdigest()
    return digest, family_id


def _chosen_captures(
    inventory: Sequence[Mapping[str, Any]],
    source_split: str,
    exclusions: set[str],
) -> list[tuple[str, str]]:
    families: dict[str, list[str]] = {}
    for record in inventory:
        scene_id = str(record.get("scene_id", ""))
        family_id = str(record.get("family_id") or physical_family(scene_id))
        if (
            record.get("source_split") != source_split
            or record.get("complete") is not True
            or not scene_id
            or family_id in exclusions
        ):
            continue
        families.setdefault(family_id, []).append(scene_id)
    chosen: list[tuple[str, str]] = []
    for family_id, captures in families.items():
        preferred = f"{family_id}_00"
        capture = preferred if preferred in captures else min(captures)
        chosen.append((family_id, capture))
    return sorted(chosen, key=lambda pair: _family_order(pair[0]))


@dataclass(frozen=True)
class SceneSplits:
    status: ReceiptStatus
    fit: tuple[str, ...]
    cal: tuple[str, ...]
    select: tuple[str, ...]
    confirm: tuple[str, ...]
    exclusions: tuple[str, ...]
    eligible_development_count: int
    eligible_confirmation_count: int
    digest: str

    @property
    def all_selected(self) -> tuple[str, ...]:
        return self.fit + self.cal + self.select + self.confirm

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "fit": list(self.fit),
            "cal": list(self.cal),
            "select": list(self.select),
            "confirm": list(self.confirm),
            "exclusions": list(self.exclusions),
            "eligible_development_count": self.eligible_development_count,
            "eligible_confirmation_count": self.eligible_confirmation_count,
            "digest": self.digest,
        }


def build_scene_splits(
    inventory: Sequence[Mapping[str, Any]],
    exclusions: Sequence[str],
    spec: StudySpec,
) -> SceneSplits:
    """Freeze physical-family-disjoint 8/2/2 development and 2 confirm splits."""

    excluded = {physical_family(str(item)) for item in exclusions}
    development = _chosen_captures(inventory, "train", excluded)
    development_families = {family for family, _ in development[:12]}
    confirmation = _chosen_captures(
        inventory,
        "validation",
        excluded | development_families,
    )
    data = spec.values["data"]
    required_development = int(data["development_families"])
    required_confirm = int(data["confirm"])
    complete = (
        len(development) >= required_development
        and len(confirmation) >= required_confirm
    )
    if complete:
        selected_development = tuple(capture for _, capture in development[:required_development])
        fit_count = int(data["fit"])
        cal_count = int(data["cal"])
        fit = selected_development[:fit_count]
        cal = selected_development[fit_count : fit_count + cal_count]
        select = selected_development[fit_count + cal_count :]
        confirm = tuple(capture for _, capture in confirmation[:required_confirm])
        status = ReceiptStatus.COMPLETE
    else:
        fit = cal = select = confirm = ()
        status = ReceiptStatus.BLOCKED_INDEPENDENT_SCENES
    digest_payload = {
        "fit": fit,
        "cal": cal,
        "select": select,
        "confirm": confirm,
        "exclusions": sorted(excluded),
        "eligible_development": development,
        "eligible_confirmation": confirmation,
        "spec_digest": spec.digest,
    }
    return SceneSplits(
        status=status,
        fit=fit,
        cal=cal,
        select=select,
        confirm=confirm,
        exclusions=tuple(sorted(excluded)),
        eligible_development_count=len(development),
        eligible_confirmation_count=len(confirmation),
        digest=canonical_digest(digest_payload),
    )
