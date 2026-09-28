"""Local-only physical-family exposure audit and frozen fresh-scene selection."""

import hashlib
import os
import re
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.module_validation.assets import native_schedule
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.scannet_frames import sensor_header

from .binding import ROOT, InputIndex

SCENE = re.compile(r"scene\d{4}_\d{2}")


def choose_families(inventory, exposed):
    families = {}
    for row in sorted(inventory, key=lambda row: row["scene"]):
        if row["authorized"] and row["complete"] and row["exposure_known"] and row["family"] not in exposed:
            families.setdefault(row["family"], row["scene"])
    ordered = sorted(families, key=lambda family: hashlib.sha256(("M2_REVIEW_FRESH_V1\0" + family).encode("utf-8")).hexdigest())
    enough = len(ordered) >= 4
    return {"status": "FRESH_CANDIDATES_SELECTED" if enough else "FRESH_BLOCKED_INSUFFICIENT_LOCAL_UNEXPOSED_FAMILIES",
            "eligible_unexposed_family_count": len(ordered), "required_families": 4,
            "chosen": [families[f] for f in ordered[:4]] if enough else [],
            "ordered_families": ordered, "hash_prefix_hex": b"M2_REVIEW_FRESH_V1\0".hex()}


def audit_fresh(binding):
    index = InputIndex()
    spec = read_json(binding["spec"])
    config = read_json(Path(binding["composition_root"]) / "resolved_config.json")
    raw = Path(config["runtime"]["data_root"])
    acquisition_path = raw / "acquisition_lock.json"
    index.identity(acquisition_path)
    acquisition = read_json(acquisition_path)
    authorized = {r["scene_id"]: r for r in acquisition["selected"]}
    explicit = {scene.split("_")[0] for scene in spec["datasets"]["all_historical_scannet_excluded_from_fresh"]}
    # Scan only known experiment metadata/logs, never images, models or raw arrays.
    runtime_roots = sorted(p for p in Path(binding["output_root"]).parents[1].glob("ovimap-*")
                           if p.is_dir() and p != Path(binding["output_root"]).parent)
    roots = [ROOT / "docs", ROOT / "artifacts/static_ovmap", ROOT / "configs/evaluation", *runtime_roots]
    evidence, skipped = [], []
    exposed = set(explicit)
    prune = {".git", "__pycache__", "models", "model_cache", "tooling", "scans", "data", "cache", "query_cache",
             "requests", "native_request_cache", "color", "depth", "pose", "frontend", "shared_masks", "evaluation_cache"}
    for root in roots:
        if not root.exists():
            continue
        for parent, directories, files in os.walk(root):
            parent = Path(parent)
            for name in directories:
                if SCENE.fullmatch(name):
                    family = name.split("_")[0]
                    exposed.add(family)
                    evidence.append({"path": str(parent / name), "kind": "PREPARED_SCENE_DIRECTORY", "families": [family]})
            directories[:] = [n for n in directories if n not in prune and not SCENE.fullmatch(n) and not n.isdigit()]
            for name in files:
                path = parent / name
                if path.suffix not in {".json", ".jsonl", ".yaml", ".yml", ".csv", ".md", ".log", ".txt"}:
                    continue
                if path.stat().st_size > 2 * 1024 * 1024:
                    skipped.append(str(path))
                    continue
                text = path.read_text(errors="replace")
                families = sorted({s.split("_")[0] for s in SCENE.findall(text)})
                if families:
                    exposed.update(families)
                    evidence.append({**index.identity(path), "kind": "METADATA_OR_LOG_MENTION", "families": families})
    inventory = []
    for directory in sorted((raw / "scans").glob("scene*")):
        if not directory.is_dir() or not SCENE.fullmatch(directory.name):
            continue
        scene = directory.name
        row = authorized.get(scene)
        missing = []
        if row is not None:
            for suffix, remote in row["files"].items():
                path = directory / (scene + suffix)
                receipt_path = Path(str(path) + ".download.json")
                if not path.is_file() or path.stat().st_size != remote["size_bytes"] or not receipt_path.is_file():
                    missing.append(str(path))
        schedule = None
        if row is not None and not missing:
            try:
                with (directory / (scene + ".sens")).open("rb") as handle:
                    header = sensor_header(handle)
                schedule = native_schedule(header["source_frame_count"])
                if schedule != row["schedule"]:
                    missing.append("NATIVE_200_SLOT_SCHEDULE_MISMATCH")
            except (OSError, ValueError) as error:
                missing.append("INVALID_SENSOR_METADATA: " + str(error))
        family = scene.split("_")[0]
        # Oversized uninspected metadata makes an otherwise new family's history
        # unresolved. Never promote it to fresh merely because a search missed it.
        known = family in exposed or (row is not None and not skipped)
        inventory.append({"scene": scene, "family": family, "authorized": row is not None,
                          "complete": row is not None and not missing, "missing": missing,
                          "exposure_known": known, "exposed": family in exposed,
                          "schedule": schedule, "authorization": row})
    result = {**choose_families(inventory, exposed), "binding": binding["identity"],
              "authorized_inventory_root": str(raw), "local_scene_count": len(inventory),
              "inventory": inventory, "explicit_minimum_exclusions": sorted(explicit),
              "all_observed_exposed_families": sorted(exposed), "exposure_evidence": evidence,
              "inspected_roots": list(map(str, roots)), "uninspected_large_metadata": skipped,
              "exposure_scope": "Known project repositories and runtime outputs; unknown exposure excluded",
              "inputs": index.entries(), "no_download": True}
    result["identity"] = canonical_digest(result)
    output = Path(binding["output_root"]) / "fresh"
    write_once(output / "audits" / (result["identity"] + ".json"), result)
    return result


def freeze_fresh_plan(binding, audit):
    if audit["binding"] != binding["identity"] or len(audit["chosen"]) != 4:
        raise ValueError("fresh execution requires four audited unexposed families")
    nomination = read_json(Path(binding["output_root"]) / "nomination.json")
    methods = list(dict.fromkeys(["N0", "CP_M2_EQUAL_CAL", nomination["selected"], nomination["single_alternative"]]))
    rows = [next(r for r in audit["inventory"] if r["scene"] == s) for s in audit["chosen"]]
    result = {"status": "FROZEN", "binding": binding["identity"], "audit_identity": audit["identity"],
              "scenes": audit["chosen"], "rows": rows, "methods": methods,
              "temperatures": read_json(binding["transfer"])["temperatures"],
              "no_confirmation_selection": True, "old_study_guards_unchanged": True}
    result["identity"] = canonical_digest(result)
    write_once(Path(binding["output_root"]) / "fresh/plan.json", result)
    return result
