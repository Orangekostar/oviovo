"""Genuine source distributions and native cosine from frozen aggregates."""

import copy
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.calibration_jobs import load_source
from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.composition_study.object_evidence import (
    native_scores,
    owner_labels,
)
from src.static_ovmap.module_validation.scannet_study import (
    load_prediction,
    native_readout,
)

from .binding import InputIndex


@dataclass
class SourceEvidence:
    dataset: str
    scene: str
    config: dict
    native: object
    sources: dict
    cosine_native: dict
    native_aggregates: dict
    cosine_audit: list
    input_identities: list


def read_scene(binding, scene, index=None):
    import torch

    index = index or InputIndex()
    bound = binding["scenes"][scene]
    config = read_json(bound["config"])
    data = config["scenes"][scene]
    native = load_prediction(data["native_prediction"])
    if native.prediction_key != bound["native_prediction_key"] or native.record_key != bound["native_record_key"]:
        raise ValueError("native prediction differs from study binding")
    sources = {name: load_source(path) for name, path in bound["sources"].items()}
    labels = owner_labels(native)
    ids = tuple(config["models"]["native"]["valid_ids"])
    for source in sources.values():
        if set(map(int, source["objects"])) != set(labels) or tuple(source["valid_ids"]) != ids:
            raise ValueError("source registry or vocabulary differs")
        for row in source["objects"].values():
            if row["available"]:
                scores = np.asarray(row["scores"])
                if scores.shape != (len(ids),) or not np.isfinite(scores).all() or ids[int(scores.argmax())] != row["label"]:
                    raise ValueError("source score/label parity failed")
            elif row["scores"] is not None:
                raise ValueError("unavailable source contains fabricated evidence")
    feature_path = data["native_features"]
    index.identity(feature_path)
    with open(feature_path, "rb") as handle:
        saved = pickle.load(handle)
    model = config["models"]["native"]
    index.identity(model["text"]["path"], model["text"])
    with np.load(model["text"]["path"], allow_pickle=False) as arrays:
        text, canonical = arrays["text_embeddings"], arrays["canonical_embeddings"]
    readouts = native_readout(saved, text, canonical, ids)
    cosine_source = copy.deepcopy(sources["N0"])
    cosine_source.pop("identity")
    cosine_source["readout_identity"] = "native_same_FP32_saved_aggregate_cosine"
    aggregates, audit = {}, []
    for owner in labels:
        legacy = sources["N0"]["objects"][str(owner)]
        row = readouts[owner]
        if row["class_id"] != legacy["label"] or (row["status"] == "AVAILABLE") != legacy["available"]:
            raise ValueError(f"native reconstruction differs: {scene}/{owner}")
        if not legacy["available"]:
            continue
        feature = row["feature"]
        reconstructed = native_scores(feature, text, canonical)
        if not np.array_equal(reconstructed, np.asarray(legacy["scores"])):
            raise ValueError(f"native canonical scores differ: {scene}/{owner}")
        aggregates[owner] = feature
        cosine = torch.nn.functional.cosine_similarity(torch.as_tensor(feature, dtype=torch.float32), torch.as_tensor(text, dtype=torch.float32), dim=-1).numpy()
        label = ids[int(cosine.argmax())]
        cosine_source["objects"][str(owner)].update(scores=cosine.tolist(), label=label)
        if label != legacy["label"]:
            canonical_tie = reconstructed[ids.index(label)] == reconstructed[ids.index(legacy["label"])]
            audit.append({"owner_id": owner, "canonical_label": legacy["label"], "cosine_label": label,
                          "canonical_exact_tie": bool(canonical_tie), "cosine_gap": float(cosine.max() - cosine[ids.index(legacy["label"])])})
            if not canonical_tie:
                raise ValueError(f"non-tie representation top1 mismatch: {scene}/{owner}")
    return SourceEvidence(bound["dataset"], scene, config, native, sources, cosine_source, aggregates, audit, index.entries())


def write_reconstruction_audit(evidence, output):
    result = {"status": "COMPLETE", "scene": evidence.scene, "dataset": evidence.dataset,
              "owners": len(owner_labels(evidence.native)), "sources": {name: source["identity"] for name, source in evidence.sources.items()},
              "available": {name: sum(row["available"] for row in source["objects"].values()) for name, source in evidence.sources.items()},
              "native_reconstructed_aggregates": len(evidence.native_aggregates), "cosine_top1_disagreements": evidence.cosine_audit,
              "physical_image_forwards": 0, "inputs": evidence.input_identities}
    write_once(Path(output) / "source_audits" / (evidence.scene + ".json"), result)
    return result


def legacy_fusion_parity(binding, evidence):
    from .fusion import fuse

    scene = evidence.scene
    composition = Path(binding["composition_root"])
    if evidence.dataset == "ScanNet":
        fits = read_json(composition / "calibration/folds.json")["folds"][scene]
        temperatures = {name: fit["temperature"] for name, fit in fits.items()}
    else:
        temperatures = read_json(binding["transfer"])["temperatures"]
    native_labels = owner_labels(evidence.native)
    parity = {}
    for method in ("N0", "Q_GAIN", "S_SIGLIP2_AREA", "CP_M2_EQUAL_RAW", "CP_M2_EQUAL_CAL"):
        row_path = Path(binding["scenes"][scene]["legacy_rows"]) / (method + ".json")
        row = read_json(row_path)
        prediction = load_prediction(row["prediction_manifest"])
        if method in evidence.sources:
            labels = {int(k): v["label"] for k, v in evidence.sources[method]["objects"].items()}
        else:
            ts = {name: .07 for name in temperatures} if method.endswith("RAW") else temperatures
            labels, _ = fuse(native_labels, evidence.sources, evidence.config["models"]["native"]["valid_ids"], ("N0", "Q_GAIN", "S_SIGLIP2_AREA"), ts)
        expected = owner_labels(prediction)
        differences = [{"owner": owner, "actual": labels[owner], "expected": expected[owner]} for owner in labels if labels[owner] != expected[owner]]
        if differences:
            raise ValueError(f"legacy {scene}/{method} differs: {differences}")
        if not np.array_equal(prediction.owner_ids, evidence.native.owner_ids) or prediction.instance_ranks != evidence.native.instance_ranks:
            raise ValueError("legacy control geometry/ranks differ")
        parity[method] = {"status": "EXACT_LABEL_OWNER_RANK_PARITY", "prediction_key": prediction.prediction_key,
                          "historical_metrics_reference": row["metrics"], "metrics_recomputed": False}
    return {"scene": scene, "controls": parity}
