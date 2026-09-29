"""Whole-map E04 coverage control and two-slot composition over frozen A7."""

from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.composition_study.object_evidence import owner_labels
from src.static_ovmap.m2_reviewer_study.binding import InputIndex
from src.static_ovmap.m2_reviewer_study.evaluation import SceneEvaluator, write_gzip
from src.static_ovmap.m2_reviewer_study.fusion import fuse
from src.static_ovmap.m2_reviewer_study.scores import read_scene
from src.static_ovmap.module_validation.contracts import canonical_digest
from src.static_ovmap.module_validation.scannet_study import (
    relabel_prediction,
    save_prediction,
)

from .calibration import base_temperatures, source_path
from .evaluation import RANKS
from .localized import freeze_candidates


def save_and_evaluate(binding, evidence, scene, method, labels, provenance, details=None):
    root = Path(binding["output_root"])
    if binding["scenes"][scene]["role"] != "CAL" and not (root / "nomination.json").exists():
        raise ValueError("Replica decisions require frozen CAL nomination")
    if method not in binding["methods"] + [binding["optional_method"]]:
        raise ValueError("unregistered complete-map decision")
    if set(labels) != set(owner_labels(evidence.native)):
        raise ValueError("decision removes original map owners")
    payload = relabel_prediction(evidence.native, method, "S", labels, {},
        {"binding": binding["identity"], "geometry": "UNCHANGED", **provenance})
    path = save_prediction(payload, root / "predictions" / scene / method)
    write_once(root / "locked" / scene / (method + ".json"),
        {"labels": labels, "prediction_identity": payload.prediction_key, "prediction_manifest": str(path)})
    if details is not None:
        write_gzip(root / "probabilities" / scene / (method + ".json.gz"), details)
    evaluator = SceneEvaluator(evidence, root)
    return [evaluator.evaluate(labels, method, rank, payload.prediction_key) for rank in RANKS]


def shortlist(binding, scene):
    index = InputIndex()
    parent = read_json(binding["reviewer_binding"])
    evidence = read_scene(parent, scene, index)
    native = owner_labels(evidence.native)
    sources = {**evidence.sources, "N0": evidence.cosine_native}
    ids = evidence.sources["N0"]["valid_ids"]
    labels, details = fuse(native, sources, ids, tuple(sources), base_temperatures(binding, scene))
    original_path = Path(parent["output_root"]) / "predictions" / scene / "locked.json"
    index.identity(original_path)
    original = read_json(original_path)["RV_A7_COS_REFIT"]["labels"]
    if labels != {int(k): v for k, v in original.items()}:
        raise ValueError("E04 base A7 reconstruction differs")
    bound = binding["scenes"][scene]["static_manifest"]
    index.identity(bound["path"], bound)
    manifest = read_json(bound["path"])
    targets = freeze_candidates(scene, ids, evidence.sources, labels, details, manifest["views"])
    targets.update(scene=scene, binding=binding["identity"], valid_ids=ids,
                   inputs=index.entries(), base="RV_A7_COS_REFIT", new_source_outputs_used=False)
    targets["identity"] = canonical_digest(targets)
    path = Path(binding["output_root"]) / "e04" / scene / "targets.json"
    write_once(path, targets)
    return save_and_evaluate(binding, evidence, scene, "AW_E04_SHORTLIST", labels,
        {"targets_identity": targets["identity"], "expected_identical_to": "RV_A7_COS_REFIT",
         "SAM3_required": False, "localized_evidence_executed": False}, details)


def compose(binding, scene):
    root = Path(binding["output_root"])
    selection = read_json(root / "composition.json")
    if selection["status"] != "PAIR_FROZEN":
        raise ValueError("composition has no eligible frozen pair")
    parent = read_json(binding["reviewer_binding"])
    evidence = read_scene(parent, scene)
    native = owner_labels(evidence.native)
    sources = {**evidence.sources, "N0": evidence.cosine_native}
    temperatures = base_temperatures(binding, scene)
    definitions = {r["id"]: r for r in read_json(binding["spec"])["source_variants"]}
    identities = {}
    for slot, variant in (("Q_GAIN", selection["q_variant"]), ("S_SIGLIP2_AREA", selection["region_variant"])):
        if definitions[variant]["slot"] != slot:
            raise ValueError("composition attempts to replace the wrong A7 slot")
        source = read_json(source_path(binding, scene, variant))
        if canonical_digest({k: v for k, v in source.items() if k != "identity"}) != source["identity"]:
            raise ValueError("composition source identity changed")
        if source["native_record_key"] != evidence.native.record_key:
            raise ValueError("composition source owner map changed")
        fits = read_json(root / "calibration" / (variant + ".json"))
        temperatures[slot] = (fits["folds"][scene] if binding["scenes"][scene]["role"] == "CAL" else fits["final"])["temperature"]
        sources[slot], identities[slot] = source, source["identity"]
    labels, details = fuse(native, sources, sources["N0"]["valid_ids"], tuple(sources), temperatures)
    return save_and_evaluate(binding, evidence, scene, "AW_COMBO_QR", labels,
        {"composition_identity": selection["identity"], "source_identities": identities,
         "temperatures": temperatures, "new_scalar_fit": False, "new_visual_inference": False}, details)
