"""Study-local content binding, memoized within a process."""

import hashlib
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.module_validation.contracts import canonical_digest

ROOT = Path(__file__).resolve().parents[3]


class InputIndex:
    def __init__(self):
        self.cache = {}

    def identity(self, path, expected=None):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in self.cache:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            self.cache[key] = {"path": str(path), "bytes": stat.st_size, "sha256": digest.hexdigest()}
        value = self.cache[key]
        if expected and any(value[k] != expected[k] for k in ("bytes", "sha256") if k in expected):
            raise ValueError(f"bound input changed: {path}")
        return value

    def entries(self):
        return sorted(self.cache.values(), key=lambda row: row["path"])


def bind(spec_path, transfer_path, output):
    index = InputIndex()
    spec, transfer = read_json(spec_path), read_json(transfer_path)
    index.identity(spec_path)
    index.identity(transfer_path)
    selection = transfer["source_selection"]
    index.identity(selection["path"], selection)
    composition = Path(selection["path"]).parent
    scan_path = composition / "resolved_config.json"
    scan = read_json(scan_path)
    index.identity(scan_path)
    scenes = {}
    for dataset, names in (("ScanNet", spec["datasets"]["calibration"]), ("Replica", spec["datasets"]["replica"])):
        for scene in names:
            config_path = scan_path if dataset == "ScanNet" else Path(transfer["attempt_root"]) / "scenes" / scene / "resolved_config.json"
            index.identity(config_path)
            config = scan if dataset == "ScanNet" else read_json(config_path)
            local = Path(config["attempt_root"])
            data = config["scenes"][scene]
            query = read_json(local / "query" / scene / "Q_GAIN/receipt.json")
            source_paths = {name: str(local / "sources" / scene / (name + ".json")) for name in ("N0", "S_SIGLIP2_AREA")}
            source_paths["Q_GAIN"] = query["evidence_path"]
            for path in source_paths.values():
                index.identity(path)
                source = read_json(path)
                if canonical_digest({k: v for k, v in source.items() if k != "identity"}) != source["identity"]:
                    raise ValueError(f"source content identity differs: {path}")
            native_manifest = read_json(data["native_prediction"])
            index.identity(data["native_prediction"])
            index.identity(Path(data["native_prediction"]).parent / native_manifest["arrays"]["path"], native_manifest["arrays"])
            for name in ("native", "siglip2"):
                model = config["models"][name]
                index.identity(model["text"]["path"], model["text"])
            scenes[scene] = {"dataset": dataset, "role": "CAL" if dataset == "ScanNet" else "HISTORICAL_TRANSFER", "config": str(config_path), "sources": source_paths,
                             "native_prediction_key": native_manifest["prediction_key"], "native_record_key": native_manifest["record_key"], "legacy_rows": str(local / "rows" / data["role"] / scene)}
    index.identity(Path(transfer["attempt_root"]) / "report_apall/per_scene.csv")
    for filename in ("folds.json", "final_temperatures.json", "examples.json"):
        index.identity(composition / "calibration" / filename)
    result = {"schema": 1, "status": "SOURCES_BOUND", "spec": str(Path(spec_path).resolve()), "output_root": str(Path(output).resolve()),
              "transfer": str(Path(transfer_path).resolve()), "composition_root": str(composition), "scenes": scenes,
              "methods": spec["controls"] + [row["method_id"] for row in spec["ablations"]], "inputs": index.entries(),
              "deployment": "N0_UNCHANGED", "fresh_inventory_status": "PENDING_EXPOSURE_AUDIT"}
    result["identity"] = canonical_digest(result)
    write_once(Path(output) / "source_binding.json", result)
    return result
