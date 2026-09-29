"""Task-local complete-map predictions over immutable reviewer geometry."""

from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.composition_study.object_evidence import owner_labels
from src.static_ovmap.m2_reviewer_study.evaluation import SceneEvaluator, write_gzip
from src.static_ovmap.m2_reviewer_study.fusion import fuse
from src.static_ovmap.m2_reviewer_study.scores import read_scene
from src.static_ovmap.module_validation.scannet_study import (
    relabel_prediction,
    save_prediction,
)

from .calibration import base_temperatures, source_path

RANKS = ("FROZEN_N0", "OFFICIAL_CURRENT_CLASS")


def pool_methods(binding, split, methods):
    from src.static_ovmap.m2_reviewer_study.evaluation import pool
    from src.static_ovmap.released_loader import load_released_module

    if split not in ("cal", "replica"):
        raise ValueError("pool requires a single declared split")
    if not set(methods) <= set(binding["methods"] + [binding["optional_method"]]):
        raise ValueError("pool method outside frozen registry")
    root = Path(binding["output_root"])
    if split == "replica" and not (root / "nomination.json").exists():
        raise ValueError("Replica pool requires frozen nomination")
    spec = read_json(binding["spec"])
    scenes = spec["datasets"]["calibration" if split == "cal" else "replica"]
    config = read_json(binding["scenes"][scenes[0]]["config"])
    namespace = load_released_module(Path(config["runtime"]["upstream"]) / "scripts/eval_utils.py")
    namespace["init"]("Scannet200" if split == "cal" else "Replica")
    rows = [read_json(root / "rows" / scene / method / (rank + ".json"))
            for scene in scenes for method in methods for rank in RANKS]
    if any(row["status"] != "COMPLETE" for row in rows):
        raise ValueError("pool requires genuine completed scene rows")
    result = []
    for method in methods:
        for rank in RANKS:
            row = {**pool(root, rows, namespace, scenes, method, rank), "method": method, "rank_mode": rank,
                   "dataset": "ScanNet" if split == "cal" else "Replica", "split": split,
                   "aggregation": "RELEASED_DATASET_POOL"}
            write_once(root / "pooled" / split / method / (rank + ".json"), row)
            result.append(row)
    return result


def evaluate_sources(binding, scene, variants, *, controls=True):
    root = Path(binding["output_root"])
    if binding["scenes"][scene]["role"] != "CAL" and not (root / "nomination.json").is_file():
        raise ValueError("Replica evaluation requires frozen research nomination")
    parent = read_json(binding["reviewer_binding"])
    evidence = read_scene(parent, scene)
    native = owner_labels(evidence.native)
    ids = evidence.sources["N0"]["valid_ids"]
    sources = {**evidence.sources, "N0": evidence.cosine_native}
    definitions = {r["id"]: r for r in read_json(binding["spec"])["source_variants"]}
    methods = {}
    if controls:
        locked = read_json(Path(parent["output_root"]) / "predictions" / scene / "locked.json")
        for method in read_json(binding["spec"])["references"]:
            labels = {int(o): v for o, v in locked[method]["labels"].items()}
            if method.startswith("RV_A7_"):
                actual, _ = fuse(native, sources, ids, tuple(sources), base_temperatures(binding, scene, refit=method.endswith("REFIT")))
                if labels != actual:
                    raise ValueError("base cosine A7 label reconstruction differs")
            methods[method] = {**locked[method], "labels": labels}
    for variant in variants:
        source = read_json(source_path(binding, scene, variant))
        if source["native_record_key"] != evidence.native.record_key or source["valid_ids"] != ids:
            raise ValueError("variant changes owner map or vocabulary")
        slot = definitions[variant]["slot"]
        fits = read_json(root / "calibration" / (variant + ".json"))
        temperatures = base_temperatures(binding, scene)
        temperatures[slot] = (fits["folds"][scene] if binding["scenes"][scene]["role"] == "CAL" else fits["final"])["temperature"]
        for mode in ("DIRECT", "A7"):
            method = variant + "_" + mode
            if mode == "DIRECT":
                labels = {o: source["objects"][str(o)]["label"] if source["objects"][str(o)]["available"] else incumbent
                          for o, incumbent in native.items()}
                details = None
            else:
                labels, details = fuse(native, {**sources, slot: source}, ids, tuple(sources), temperatures)
            payload = relabel_prediction(evidence.native, method, "S", labels, {},
                {"binding": binding["identity"], "variant_source_identity": source["identity"],
                 "temperatures": temperatures if mode == "A7" else None,
                 "geometry": "UNCHANGED", "direct_unavailable_fallback": "N0"})
            path = save_prediction(payload, root / "predictions" / scene / method)
            methods[method] = {"labels": labels, "prediction_identity": payload.prediction_key,
                               "prediction_manifest": str(path)}
            if details is not None:
                write_gzip(root / "probabilities" / scene / (method + ".json.gz"), details)
    # Persist every whole-map label decision before any evaluator is constructed.
    for method, value in methods.items():
        write_once(root / "locked" / scene / (method + ".json"), value)
    evaluator = SceneEvaluator(evidence, root)
    rows = []
    for method, value in methods.items():
        for rank in RANKS:
            row = evaluator.evaluate(value["labels"], method, rank, value["prediction_identity"])
            if method in read_json(binding["spec"])["references"]:
                old = read_json(Path(parent["output_root"]) / "rows" / scene / method / (rank + ".json"))
                if row["metrics"] != old["metrics"]:
                    raise ValueError("same-rank reference metrics changed")
            rows.append(row)
    return rows
