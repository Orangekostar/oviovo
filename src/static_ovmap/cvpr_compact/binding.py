"""New-task immutable input binding and bounded, read-only path relocation."""

from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.scannet_download import DownloaderLayout, REQUIRED_SUFFIXES
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, PathResolver, read

from .protocol import experiment_matrix, final_temperatures, load_spec


def parse_path_maps(values):
    mapping = {}
    for value in values:
        items = read(value) if Path(value).is_file() else dict([value.split("=", 1)])
        for old, new in items.items():
            old, new = str(Path(old)), str(Path(new))
            if not Path(old).is_absolute() or not Path(new).is_absolute():
                raise ValueError("path-map roots must be absolute")
            if old in mapping and mapping[old] != new:
                raise ValueError("conflicting repeated path-map root")
            mapping[old] = new
    return mapping


def bounded_data_inventory(roots, scene_ids, *, max_depth=4):
    found, visited = {}, set()
    required = {scene + suffix: (scene, suffix) for scene in scene_ids for suffix in REQUIRED_SUFFIXES}
    def walk(directory, depth):
        directory = Path(directory)
        physical = str(directory.resolve())
        if physical in visited or depth > max_depth or not directory.is_dir():
            return
        visited.add(physical)
        for item in sorted(os.scandir(directory), key=lambda value: value.name):
            if item.is_file() and item.name in required:
                scene, suffix = required[item.name]
                found.setdefault(scene, {}).setdefault(suffix, str(Path(item.path).resolve()))
            elif item.is_dir(follow_symlinks=False) and not item.name.startswith("."):
                walk(item.path, depth + 1)
    for root in roots:
        walk(root, 0)
    return {"roots": list(map(str, roots)), "maximum_depth": max_depth,
            "visited_directory_count": len(visited), "files": found,
            "search_policy": "ONE_BOUNDED_SEARCH_OF_EXPLICIT_DATA_ROOTS"}


def bind_inputs(spec_path, repository_root, *, path_maps=(), gpu=None):
    spec, repo = load_spec(spec_path), Path(repository_root).resolve()
    resolver = PathResolver(parse_path_maps(path_maps))
    root = Path(resolver.resolve(spec["output_root"])).resolve()
    parents = [Path(resolver.resolve(spec[key])).resolve() for key in ("parent_recovery_root", "parent_backbone_root")]
    if any(root == parent or root.is_relative_to(parent) for parent in parents):
        raise ValueError("compact-table writes cannot target immutable parent attempts")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    if branch != spec["branch"]:
        raise ValueError("compact-table branch differs from the fixed protocol")
    subprocess.run(["git", "merge-base", "--is-ancestor", spec["base_commit"], "HEAD"], cwd=repo, check=True)
    path = root / "resolved_inputs.json"
    index = ConsumptionIndex(root / "validation/input_verifications.json")
    spec_identity = index.identity(spec_path)
    if path.is_file():
        previous = read(path)
        if (canonical_digest({k: v for k, v in previous.items() if k != "identity"}) != previous["identity"]
                or previous["spec_identity"] != spec_identity or previous["path_map"] != resolver.mapping
                or (gpu is not None and previous["gpu"] != str(gpu))):
            raise ValueError("immutable compact-table binding differs; use a new attempt")
        for item in previous["inputs"]:
            index.identity(item["path"], item)
        index.write_memo(root / "validation/input_verifications.json")
        return previous
    root.mkdir(parents=True, exist_ok=True)
    blocked = []
    def consume(value, expected=None, *, optional=False):
        resolved = resolver.resolve(value)
        try:
            return index.identity(resolved, expected)
        except FileNotFoundError:
            if not optional:
                raise
            blocked.append({"path": resolved, "reason": "MISSING_BOUND_INPUT"})
            return None
    parent_path = parents[0] / "resolved_inputs.json"
    consume(parent_path)
    original = read(parent_path)
    if canonical_digest({k: v for k, v in original.items() if k != "identity"}) != original["identity"]:
        raise ValueError("recovery parent binding identity changed")
    parent = resolver.rewrite(original)
    publication_path = parents[0] / "publication/final.json"
    consume(publication_path)
    publication = read(publication_path)
    if (publication["status"] != "PUSH_VERIFIED" or publication["local_sha"] != spec["base_commit"]
            or publication["remote_sha"] != spec["base_commit"]):
        raise ValueError("recovery parent does not match the pinned published base")
    temperatures = final_temperatures(spec, parent["scenes"])
    archive_path = repo / "artifacts/static_ovmap/recovery_wave2_v1/validation/producer_sources.json"
    consume(archive_path)
    archived = {row["source_sha256"]: row for row in read(archive_path)["sources"]}
    producer_sources = {}
    def producer(item):
        key = item["sha256"]
        if key in producer_sources:
            return
        current = Path(resolver.resolve(item["path"]))
        identity = consume(current, optional=True)
        if identity is not None and identity["sha256"] == key and identity["bytes"] == item["bytes"]:
            producer_sources[key] = {"effective_source": item, "exact_bytes": identity, "kind": "EXACT_ORIGINAL_PRODUCER_BYTES"}
            return
        if key not in archived:
            raise ValueError("effective historical producer source is unavailable: " + item["path"] + " " + key)
        row = archived[key]
        archive = repo / "artifacts/static_ovmap/recovery_wave2_v1/validation/producer_sources" / Path(row["archive"]["path"]).name
        bound = consume(archive, row["archive"])
        data = gzip.decompress(archive.read_bytes())
        if hashlib.sha256(data).hexdigest() != key or len(data) != item["bytes"]:
            raise ValueError("effective producer archive is not byte-exact")
        producer_sources[key] = {"effective_source": item, "exact_archive": bound, "kind": "BYTE_EXACT_ARCHIVED_PRODUCER"}
    scenes = {}
    for scene in [*spec["cohorts"]["replica8"], spec["smoke_scene"]]:
        data = parent["scenes"][scene]
        for key in ("parent_map_receipt", "parent_readout_receipt", "capture_manifest", "config", "deferred_metadata"):
            consume(data[key])
        mapping, readout = read(data["parent_map_receipt"]), read(data["parent_readout_receipt"])
        if (mapping["status"] != "COMPLETE" or readout["status"] != "PREDICTIONS_LOCKED"
                or mapping["scene"] != scene or mapping["map_id"] != "BB00_NATIVE"
                or mapping["recipe"]["association"] != "native" or mapping["recipe"]["depth_fusion"] != "native"
                or mapping["recipe"]["frontend"] != "cropformer"):
            raise ValueError("incompatible parent native geometry: " + scene)
        capture = read(data["capture_manifest"])
        if canonical_digest({k: v for k, v in capture.items() if k != "identity"}) != capture["identity"]:
            raise ValueError("parent capture identity changed: " + scene)
        if capture["scheduled_frame_ids"] != data["schedule"] or capture["completed_frame_ids"] != data["completed_frame_ids"]:
            raise ValueError("parent native schedule/completed frame list changed")
        consume(data["native_extension"]["path"], data["native_extension"])
        for name in ("surface", "tsdf"):
            item = capture[name]
            consume(Path(data["capture_manifest"]).parent / item["path"], item)
        config = resolver.rewrite(read(data["config"]))
        model = config["models"]["native"]
        if model["dtype"] != "float32":
            raise ValueError("inherited native precision is not FP32")
        for item in model["files"] + [model["text"], data["inherited"]["checkpoint"]]:
            consume(item["path"], item, optional=True)
        consume(model["text_receipt"])
        for source in data["sources"].values():
            consume(source["path"])
            value = read(source["path"])
            if value["identity"] != source["identity"] or canonical_digest({k: v for k, v in value.items() if k != "identity"}) != source["identity"]:
                raise ValueError("parent N/Q/F source identity changed")
        for prediction_path in data["predictions"].values():
            consume(prediction_path)
            prediction = read(prediction_path)
            consume(Path(prediction_path).parent / prediction["arrays"]["path"], prediction["arrays"])
        nq_path, fc_path = Path(data["native_query_root"]) / "native_query_receipt.json", Path(data["fc_root"]) / "receipt.json"
        consume(nq_path)
        consume(fc_path)
        nq, fc = read(nq_path), read(fc_path)
        producer(nq["worker_identity"])
        for item in fc["inputs"]:
            if item["path"].endswith(".py"):
                producer(item)
            elif "/assets/fc_frozen/" in item["path"]:
                consume(item["path"], item, optional=True)
        consume(data["FC_text"]["path"], data["FC_text"], optional=True)
        if nq["Q_budget"] != 200 or fc["physical_model_identity"] != data["FC_physical_model_identity"]:
            raise ValueError("parent frozen Q/FC identity changed")
        scenes[scene] = {**data, "models": config["models"], "checkpoint": data["inherited"]["checkpoint"],
            "runtime": data["inherited"]["runtime"], "temperatures": temperatures,
            "temperature_source": {"scene": spec["cohorts"]["replica8"][0], "parent_binding_identity": original["identity"]},
            "availability": "REUSABLE_VERIFIED_BB00_NATIVE_ANCHOR", "main_cohort": "replica8" if scene in spec["cohorts"]["replica8"] else None,
            "native_query_receipt": str(nq_path), "fc_receipt": str(fc_path),
            "original_map_options": mapping["recipe"], "actual_capture_command": data["actual_parent_command"],
            "image_convention": "captured aligned uint8 RGB", "depth_convention": "captured FP32 depth_m",
            "camera_convention": "original captured K and pose_c2w; no additional world alignment"}
        index.write_memo(root / "validation/input_verifications.json")
    template = scenes[spec["smoke_scene"]]
    data_root = Path(template["runtime"]["data_root"])
    acquisition_path = data_root / "acquisition_plan.json"
    consume(acquisition_path)
    acquisition = resolver.rewrite(read(acquisition_path))
    downloader_path = Path(acquisition["layout"]["script_path"])
    consume(downloader_path, {"sha256": acquisition["layout"]["script_sha256"]})
    layout = DownloaderLayout.from_script(downloader_path)
    search_path = root / "validation/data_search.json"
    if search_path.exists():
        inventory = read(search_path)
    else:
        inventory = bounded_data_inventory([data_root, Path("/mnt/shared/ww/datasets"), Path("/home/ww/data")], spec["cohorts"]["scannet_cf18"])
        atomic_write_json(search_path, inventory)
    for scene in spec["cohorts"]["scannet_cf18"]:
        files = {}
        for suffix in REQUIRED_SUFFIXES:
            existing = inventory["files"].get(scene, {}).get(suffix)
            location = Path(existing) if existing else root / "data/scannet/scans" / scene / (scene + suffix)
            files[suffix] = {"path": str(location), "url": layout.scene_url(scene, suffix),
                             "availability": "LOCAL_FILE_PRESENT_REQUIRES_FORMAT_VALIDATION" if location.is_file() else "MISSING_DOWNLOAD_AUTHORIZED"}
        scenes[scene] = {"dataset": "ScanNet", "main_cohort": "scannet_cf18", "physical_family": scene.split("_")[0],
            "prior_exposure": "UNVERIFIED_NOT_CLAIMED_UNTOUCHED", "availability": "ANCHOR_REQUIRES_PREPARATION",
            "raw_files": files, "models": template["models"], "checkpoint": template["checkpoint"],
            "runtime": template["runtime"], "temperatures": temperatures, "native_extension": template["native_extension"],
            "FC_physical_model_identity": template["FC_physical_model_identity"], "FC_text": template["FC_text"],
            "FC_model_reference_root": template["fc_root"], "native_template_scene": spec["smoke_scene"],
            "source_schedule": "READ_ACTUAL_SENS_HEADER_WITH_NATIVE_SCHEDULE_NO_REFILL"}
    exposure_path = data_root / "acquisition_lock.json"
    consume(exposure_path)
    roles = read(exposure_path)["selected"]
    supports = {role: [row["scene_id"] for row in roles if row["role"] == role] for role in ("fit", "cal", "select", "confirm")}
    exposure = {"known_original_query_support": supports,
        "support_source": str(exposure_path), "prior_unrecorded_exposure_excluded": False,
        "captures": {scene: {"physical_family": scene.split("_")[0],
            "known_query_training_family_overlap": scene.split("_")[0] in {s.split("_")[0] for s in supports["fit"]},
            "known_calibration_family_overlap": scene.split("_")[0] in {s.split("_")[0] for s in supports["cal"]},
            "prior_benchmark_exposure": "UNVERIFIED_NOT_CLAIMED_UNTOUCHED"} for scene in spec["cohorts"]["scannet_cf18"]},
        "replica_status": "EXPOSED_HISTORICAL_BENCHMARK", "independent_scannet_physical_families": 7}
    cache_inventory = parents[0] / "parent_cache_inventory.json"
    consume(cache_inventory)
    gpu = str(gpu) if gpu is not None else str(template["runtime"]["cuda_device"])
    result = {"schema_version": 1, "task_id": spec["task_id"], "status": "BOUND_WITH_MISSING_MAIN_INPUTS",
        "spec": str(Path(spec_path).resolve()), "spec_identity": spec_identity, "base_commit": spec["base_commit"],
        "upstream_commit": spec["upstream_commit"], "repository_root": str(repo), "output_root": str(root),
        "parent_recovery_root": str(parents[0]), "parent_backbone_root": str(parents[1]),
        "parent_binding_identity": original["identity"], "path_map": resolver.mapping, "gpu": gpu,
        "gpu_lock": f"/mnt/shared/ww/ovimap-module-validation-v1/.visual-gpu-{gpu}.lock",
        "cohorts": spec["cohorts"], "scenes": scenes, "final_temperatures": temperatures,
        "assets_root": parent["assets_root"], "fc": parent["fc"], "download_layout": asdict(layout),
        "data_search": str(search_path), "exposure_ledger": exposure, "effective_producer_sources": producer_sources,
        "parent_cache_inventory": str(cache_inventory), "parent_caches_read_only": True,
        "parent_fc_cache_roots": [str(p / "content_cache/fc") for p in parents],
        "parent_native_cache_roots": [str(p / "content_cache/native") for p in parents],
        "writable_content_cache": str(root / "content_cache"), "blocked_artifacts": blocked,
        "inputs": index.entries(), "deployment": "N0_UNCHANGED"}
    result["identity"] = canonical_digest(result)
    atomic_write_json(path, result)
    atomic_write_json(root / "matrix.json", experiment_matrix(spec))
    atomic_write_json(root / "exposure_ledger.json", exposure)
    index.write_memo(root / "validation/input_verifications.json")
    return result
