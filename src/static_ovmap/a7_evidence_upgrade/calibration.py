"""Changed-slot temperatures from fixed CAL-only geometric correspondences."""

import time
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.composition_study.temperature import fit_temperature
from src.static_ovmap.m2_reviewer_study.binding import InputIndex
from src.static_ovmap.module_validation.contracts import canonical_digest


def source_path(binding, scene, variant):
    spec = read_json(binding["spec"])
    definition = next(r for r in spec["source_variants"] if r["id"] == variant)
    return Path(binding["output_root"]) / definition["family"].lower() / scene / (variant + ".json")


def fit_source(binding, variant):
    index = InputIndex()
    index.identity(__file__)
    parent = read_json(binding["reviewer_binding"])
    spec = read_json(binding["spec"])
    scenes = spec["datasets"]["calibration"]
    path = Path(parent["composition_root"]) / "calibration/examples.json"
    expected = next(r for r in parent["inputs"] if r["path"] == str(path))
    index.identity(path, expected)
    matches = read_json(path)["correspondences"]
    examples, ids = [], None
    for scene in scenes:
        path = source_path(binding, scene, variant)
        index.identity(path)
        source = read_json(path)
        if canonical_digest({k: v for k, v in source.items() if k != "identity"}) != source["identity"]:
            raise ValueError("changed-source identity differs")
        if ids is not None and ids != source["valid_ids"]:
            raise ValueError("CAL source class orders disagree")
        ids = source["valid_ids"]
        for row in matches[scene]:
            obj = source["objects"][str(row["owner_id"])]
            if row["correspondence"] == "unique" and row["geometry_iou"] > .5 and obj["available"]:
                examples.append({"scene_id": scene, "owner_id": row["owner_id"],
                                 "gt_label": row["gt_label"], "available": True, "scores": obj["scores"]})
    identity = canonical_digest({"binding": binding["identity"], "variant": variant, "inputs": index.entries()})
    path = Path(binding["output_root"]) / "calibration" / (variant + ".json")
    if path.exists():
        previous = read_json(path)
        if previous["identity"] != identity:
            raise ValueError("scalar fit dependencies changed")
        return previous
    started = time.monotonic()
    folds = {s: fit_temperature(examples, [other for other in scenes if other != s], ids) for s in scenes}
    final = fit_temperature(examples, scenes, ids)
    result = {"status": "CAL_FITS_COMPLETE", "identity": identity, "variant": variant,
              "folds": folds, "final": final, "examples": examples, "inputs": index.entries(),
              "elapsed_seconds": time.monotonic() - started, "Replica_fitting": False}
    write_once(path, result)
    return result


def base_temperatures(binding, scene, *, refit=True):
    parent = read_json(binding["reviewer_binding"])
    root = Path(parent["output_root"])
    if binding["scenes"][scene]["role"] == "CAL":
        original = read_json(Path(parent["composition_root"]) / "calibration/folds.json")["folds"][scene]
        result = {name: value["temperature"] for name, value in original.items()}
        if refit:
            result["N0"] = read_json(root / "calibration/new_folds.json")["folds"][scene]["cosine_N0"]["temperature"]
    else:
        result = dict(read_json(parent["transfer"])["temperatures"])
        if refit:
            result["N0"] = read_json(root / "calibration/new_final.json")["cosine_N0"]["temperature"]
    return result
