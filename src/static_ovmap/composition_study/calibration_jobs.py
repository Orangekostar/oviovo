"""Evaluation-only object targets, cross-fits, and one post-nomination refit."""

from pathlib import Path

from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.scannet_study import load_prediction

from .evaluation import object_correspondence
from .execution import require_access, verified_receipt
from .io import SourceIndex, read_json, write_once
from .label_fusion import SOURCES
from .temperature import CAL_SCENES, fit_temperature


def load_source(path):
    source = read_json(path)
    if (
        canonical_digest(
            {key: value for key, value in source.items() if key != "identity"}
        )
        != source["identity"]
    ):
        raise ValueError("locked source evidence content changed")
    return source


def sources_for_scene(config, scene):
    root = Path(config["attempt_root"])
    verified_receipt(root / "sources" / scene / "static_source_receipt.json")
    query = verified_receipt(root / "query" / scene / "Q_GAIN/receipt.json")
    return {
        "N0": load_source(root / "sources" / scene / "N0.json"),
        "S_SIGLIP2_AREA": load_source(root / "sources" / scene / "S_SIGLIP2_AREA.json"),
        "Q_GAIN": load_source(query["evidence_path"]),
    }


def build_calibration_examples(config):
    root = Path(config["attempt_root"])
    examples = {name: [] for name in SOURCES}
    source_ids, correspondences = {}, {}
    for scene in CAL_SCENES:
        require_access(config, scene, "calibrate")
        sources = sources_for_scene(
            config, scene
        )  # All prediction evidence is locked first.
        native = load_prediction(config["scenes"][scene]["native_prediction"])
        rows = object_correspondence(config, scene, native)
        correspondences[scene] = rows
        source_ids[scene] = {
            name: source["identity"] for name, source in sources.items()
        }
        for row in rows:
            if row["correspondence"] != "unique":
                continue
            for name in SOURCES:
                evidence = sources[name]["objects"][str(row["owner_id"])]
                if (
                    evidence["available"]
                    and row["gt_label"] in sources[name]["valid_ids"]
                ):
                    examples[name].append(
                        {
                            "scene_id": scene,
                            "owner_id": row["owner_id"],
                            "available": True,
                            "scores": evidence["scores"],
                            "gt_label": row["gt_label"],
                            "geometry_iou": row["geometry_iou"],
                            "geometry_gt_ids": row["geometry_gt_ids"],
                        }
                    )
    result = {
        "schema_version": 1,
        "source_identities": source_ids,
        "correspondences": correspondences,
        "examples": examples,
        "target_rule": "unique strict IoU>0.5; semantic class independent of prediction",
        "historical_CAL_Q_checkpoint_exposure": True,
    }
    result["identity"] = canonical_digest(result)
    write_once(root / "calibration/examples.json", result)
    return result


def fit_folds(config):
    examples = build_calibration_examples(config)
    ids = config["models"]["native"]["valid_ids"]
    result = {"examples_identity": examples["identity"], "folds": {}}
    for held_out in CAL_SCENES:
        training = [scene for scene in CAL_SCENES if scene != held_out]
        result["folds"][held_out] = {
            name: fit_temperature(examples["examples"][name], training, ids)
            for name in SOURCES
        }
    result["identity"] = canonical_digest(result)
    write_once(Path(config["attempt_root"]) / "calibration/folds.json", result)
    return result


def refit_after_nomination(config, nomination):
    root = Path(config["attempt_root"])
    if nomination["status"] not in {"NOMINATED", "NO_EFFECTIVE_INTERVENTION"}:
        raise ValueError("final temperature fit requires completed CAL nomination")
    examples = read_json(root / "calibration/examples.json")
    identity = canonical_digest(
        {
            "examples_identity": examples["identity"],
            "nomination": nomination,
            "fit_source": SourceIndex().identity(
                Path(__file__).with_name("temperature.py")
            ),
        }
    )
    path = root / "calibration/final_temperatures.json"
    if path.is_file():
        result = read_json(path)
        if result["input_identity"] != identity:
            raise ValueError("final temperature refit identity changed")
        return result
    fits = {
        name: fit_temperature(
            examples["examples"][name],
            CAL_SCENES,
            config["models"]["native"]["valid_ids"],
        )
        for name in SOURCES
    }
    result = {
        "input_identity": identity,
        "fits": fits,
        "temperatures": {name: row["temperature"] for name, row in fits.items()},
        "selection_predictions_remain_cross_fitted": True,
    }
    write_once(path, result)
    return result
