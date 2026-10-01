"""Immutable parent binding, read-only relocation and consumed-object memoization."""

import copy
import json
from pathlib import Path
import subprocess

import numpy as np

from static_ovmap.m2_reviewer_study.binding import InputIndex
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest


def read(path):
    return json.loads(Path(path).read_text())


class PathResolver:
    def __init__(self, mapping=None):
        self.mapping = {str(Path(old)): str(Path(new)) for old, new in (mapping or {}).items()}
        if any(not Path(p).is_absolute() for pair in self.mapping.items() for p in pair):
            raise ValueError("path-map roots must be absolute")
        self.order = sorted(self.mapping, key=len, reverse=True)

    def resolve(self, value):
        value = str(value)
        for old in self.order:
            if value == old or value.startswith(old.rstrip("/") + "/"):
                return self.mapping[old].rstrip("/") + value[len(old):]
        return value

    def rewrite(self, value):
        if isinstance(value, str):
            return self.resolve(value) if value.startswith("/") else value
        if isinstance(value, list):
            return [self.rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: self.rewrite(item) for key, item in value.items()}
        return copy.deepcopy(value)


class ConsumptionIndex(InputIndex):
    """Reuse a prior verification only while the complete filesystem stamp agrees."""

    def __init__(self, memo=None):
        super().__init__()
        self.seeds, self.signatures = {}, {}
        if memo is not None and Path(memo).is_file():
            self.seeds = {tuple(row["signature"]): row["identity"] for row in read(memo)["entries"]}

    def identity(self, path, expected=None):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        signature = (*key, stat.st_dev, stat.st_ino)
        if key not in self.cache and signature in self.seeds:
            self.cache[key] = self.seeds[signature]
        result = super().identity(path, expected)
        self.signatures[key] = signature
        return result

    def write_memo(self, path):
        atomic_write_json(Path(path), {"entries": [{"signature": self.signatures[key], "identity": value}
                                                 for key, value in self.cache.items()]})


def bind_inputs(spec_path, repo_root, parent_root, output_root, *, path_map=None):
    repo, root = Path(repo_root).resolve(), Path(output_root).resolve()
    resolver = PathResolver(read(path_map) if path_map else {})
    parent = Path(resolver.resolve(parent_root)).resolve()
    if root == parent or root.is_relative_to(parent):
        raise ValueError("new writes must remain outside the immutable parent attempt")
    spec = read(spec_path)
    if spec["task_id"] != "ovimap-recovery-wave2-v1":
        raise ValueError("unexpected recovery specification")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    if branch != spec["branch"]:
        raise ValueError("worktree branch differs from the task")
    subprocess.run(["git", "merge-base", "--is-ancestor", spec["base_commit"], "HEAD"], cwd=repo, check=True)
    root.mkdir(parents=True, exist_ok=True)
    index = ConsumptionIndex(root / "validation/input_verifications.json")
    index.identity(spec_path)
    if path_map:
        index.identity(path_map)
    parent_path = parent / "resolved_inputs.json"
    index.identity(parent_path)
    original = read(parent_path)
    if canonical_digest({k: v for k, v in original.items() if k != "identity"}) != original["identity"]:
        raise ValueError("parent binding content identity changed")
    parent_binding = resolver.rewrite(original)
    publication_path = parent / "publication/final.json"
    index.identity(publication_path)
    publication = read(publication_path)
    if (publication["status"] != "PUSH_VERIFIED" or publication["local_sha"] != spec["base_commit"]
            or publication["remote_sha"] != spec["base_commit"]):
        raise ValueError("parent measurement publication differs from the required base")
    if parent_binding["inherited_ratio_threshold"] != 0.:
        raise ValueError("parent ratio threshold differs from the fixed protocol")
    scene_order = spec["datasets"]["development"] + spec["datasets"]["replica"]
    scenes, blocked = {}, []
    parent_inputs = {str(Path(item["path"]).resolve()): item for item in parent_binding["inputs"]}

    def consume(path, expected=None, *, optional=False):
        path = Path(resolver.resolve(path))
        try:
            return index.identity(path, expected or parent_inputs.get(str(path.resolve())))
        except FileNotFoundError:
            if not optional:
                raise
            blocked.append({"path": str(path), "status": "BLOCKED_MISSING_ARTIFACT"})
            return None

    for scene in scene_order:
        data = copy.deepcopy(parent_binding["scenes"][scene])
        map_root = parent / "maps" / scene / "BB00_NATIVE"
        readout_root = parent / "readouts" / scene / "BB00_NATIVE"
        map_path, readout_path = map_root / "map_receipt.json", readout_root / "receipt.json"
        for path in (map_path, readout_path, map_root / "running.json"):
            consume(path)
        mapping, readout = resolver.rewrite(read(map_path)), resolver.rewrite(read(readout_path))
        if (mapping["status"] != "COMPLETE" or readout["status"] != "PREDICTIONS_LOCKED"
                or mapping["scene"] != scene or readout["scene"] != scene
                or mapping["map_id"] != "BB00_NATIVE" or readout["map_id"] != "BB00_NATIVE"):
            raise ValueError(f"incomplete or incorrect parent baseline: {scene}")
        capture_path = Path(mapping["capture_manifest"])
        consume(capture_path)
        capture = resolver.rewrite(read(capture_path))
        if (capture["scene_id"] != scene or capture["scheduled_frame_ids"] != data["schedule"]
                or capture["completed_frame_ids"] != data["parent_completed_frame_ids"]):
            raise ValueError(f"parent baseline schedule changed: {scene}")
        native_extension = consume(capture["native_extension"]["path"], capture["native_extension"])
        consume(readout["config"])
        config = resolver.rewrite(read(readout["config"]))
        if config["models"]["native"]["valid_ids"] != data["models"]["native"]["valid_ids"]:
            raise ValueError(f"parent category order differs: {scene}")
        source_paths = {}
        for name, path in readout["sources"].items():
            consume(path)
            source = read(path)
            if canonical_digest({k: v for k, v in source.items() if k != "identity"}) != source["identity"]:
                raise ValueError(f"parent source distribution identity changed: {scene}/{name}")
            ids = data["models"]["native"]["valid_ids"]
            if source["valid_ids"] != ids:
                raise ValueError(f"parent source class order differs: {scene}/{name}")
            for row in source["objects"].values():
                if row["available"]:
                    scores = np.asarray(row["scores"])
                    if scores.shape != (len(ids),) or not np.isfinite(scores).all():
                        raise ValueError(f"parent source has invalid available scores: {scene}/{name}")
            source_paths[name] = {"path": str(path), "identity": source["identity"]}
        for path in readout["predictions"].values():
            consume(path)
        for method in ("D2", "FC_EQ"):
            consume(readout_root / "decisions" / f"{method}.json")
        for path in (mapping["deferred_metadata"], readout_root / "native_query/native_query_receipt.json",
                     readout_root / "fc/receipt.json", readout_root / "evaluation_rows.json",
                     readout_root / "raw_geometry_diagnostics.json"):
            consume(path)
        for item in data["models"]["native"]["files"]:
            consume(item["path"], item, optional=True)
        consume(data["models"]["native"]["text"]["path"], data["models"]["native"]["text"], optional=True)
        consume(data["checkpoint"]["path"], data["checkpoint"], optional=True)
        fc = resolver.rewrite(read(readout_root / "fc/receipt.json"))
        consume(fc["text_identity"]["path"], fc["text_identity"], optional=True)
        sam_path = parent / "frontend" / scene / "SAM2_PAIRED/receipt.json"
        consume(sam_path, optional=True)
        scenes[scene] = {"dataset": data["dataset"], "exposure": data["exposure"],
            "inherited": data, "parent_map_receipt": str(map_path), "parent_readout_receipt": str(readout_path),
            "parent_readout_identity": readout["identity"], "capture_manifest": str(capture_path),
            "native_extension": native_extension, "surface": {
                **capture["surface"], "path": str(capture_path.parent / capture["surface"]["path"])},
            "deferred_metadata": mapping["deferred_metadata"], "predictions": readout["predictions"],
            "sources": source_paths, "config": readout["config"],
            "actual_parent_command": resolver.rewrite(read(map_root / "running.json")),
            "native_query_root": str(readout_root / "native_query"), "fc_root": str(readout_root / "fc"),
            "sam_receipt": str(sam_path), "parent_native_eligible_owners": readout["native_eligible_owners"],
            "FC_physical_model_identity": fc["physical_model_identity"], "FC_text": fc["text_identity"],
            "temperatures": data["temperatures"], "schedule": capture["scheduled_frame_ids"],
            "completed_frame_ids": capture["completed_frame_ids"], "no_new_baseline_map": True}
        index.write_memo(root / "validation/input_verifications.json")
    for leaf in ("fc_frozen", "ovrcoat"):
        path = Path(parent_binding["assets_root"]) / leaf / "download_receipt.json"
        if consume(path, optional=True):
            asset = resolver.rewrite(read(path))
            for item in asset.get("files", [asset.get("checkpoint")]):
                if item is not None:
                    consume(item["path"], item, optional=True)
    index.write_memo(root / "validation/input_verifications.json")
    inventory = {"dense": {}, "regions": {}, "text": {}, "parent_job_receipts": []}
    for path in sorted((parent / "readouts").glob("*/*/fc/receipt.json")):
        consume(path)
        fc = resolver.rewrite(read(path))
        if fc["status"] != "COMPLETE":
            continue
        inventory["parent_job_receipts"].append(str(path))
        for key, value in fc["required_dense_receipts"].items():
            inventory["dense"].setdefault(key, value)
        for key, value in fc["required_region_receipts"].items():
            inventory["regions"].setdefault(key, value)
        inventory["text"][fc["text_identity"]["sha256"]] = fc["text_identity"]
    inventory["identity"] = canonical_digest(inventory)
    atomic_write_json(root / "parent_cache_inventory.json", inventory)
    result = {"schema_version": 1, "task_id": spec["task_id"], "status": "BOUND" if not blocked else "BOUND_WITH_LEAF_BLOCKS",
        "base_commit": spec["base_commit"], "branch": branch, "spec": str(Path(spec_path).resolve()),
        "spec_sha256": index.identity(spec_path)["sha256"], "parent_root": str(parent),
        "parent_binding_identity": original["identity"], "repository_root": str(repo), "output_root": str(root),
        "path_map": resolver.mapping, "datasets": spec["datasets"], "deployment": "N0_UNCHANGED",
        "assets_root": parent_binding["assets_root"], "fc": parent_binding["fc"],
        "parent_native_cache_root": str(parent / "content_cache/native"),
        "parent_fc_cache_root": str(parent / "content_cache/fc"),
        "parent_cache_inventory": str(root / "parent_cache_inventory.json"),
        "parent_cache_inventory_identity": inventory["identity"], "scenes": scenes,
        "inherited_ratio_threshold": 0., "blocked_artifacts": blocked,
        "inputs": index.entries(), "parent_caches_read_only": True,
        "new_full_baseline_maps": 0, "map_scene_budget": spec["limits"]["new_full_map_scene_successes"]}
    result["identity"] = canonical_digest(result)
    path = root / "resolved_inputs.json"
    if path.is_file() and read(path)["identity"] != result["identity"]:
        raise ValueError("resolved recovery inputs changed; use a new attempt")
    atomic_write_json(path, result)
    index.write_memo(root / "validation/input_verifications.json")
    return result
