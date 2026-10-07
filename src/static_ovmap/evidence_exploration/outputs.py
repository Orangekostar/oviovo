"""Explicit verifier decisions, immutable D2 incumbents, and locked exports."""

from pathlib import Path
import time
from collections import Counter

import numpy as np

from static_ovmap.backbone_wave1.readouts import fuse_readout
from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.cvpr_compact.area_fallback_experiment import seal
from static_ovmap.cvpr_compact.outputs import construct_output
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.module_validation.scannet_study import load_prediction, save_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read
from static_ovmap.recovery_wave2.recovery_registry import build_registry
from static_ovmap.runtime_parity.runner import execution_config, load_common

from .verifier import decide


def prediction_content(payload):
    return canonical_digest({"geometry": payload.geometry.to_dict(), "scene": payload.scene_id,
        "owners": _array_digest(payload.owner_ids), "semantic": _array_digest(payload.semantic_labels),
        "official_ranks": payload.instance_ranks})


def decisions_for_method(method, source, prepared, frame_receipts, valid_ids):
    name = method["id"]
    decisions = {}
    for owner, baseline in source["objects"].items():
        candidate = prepared["candidates"][owner]
        triggered = bool(candidate["triggered"])
        controls, target, fallback = [], None, None
        rep = None
        if triggered and name not in ("EV00_D2", "EV01_G1_V2", "EV02_MARGIN"):
            frame = frame_receipts[str(candidate["frame_id"])]
            target_key = candidate["request_id"] + "_target"
            representation = {"EV03_CONTRAST": "coarse", "EV04_BILINEAR": "bilinear", "EV05_ANYUP": "anyup",
                "EV06_OWNER_ANYUP": "owner_depth", "EV07_COMBINATION": "owner_depth", "EV08_NO_DEPTH": "owner_only"}[name]
            if representation != "coarse":
                rep = frame["representations"][target_key][representation]
                target = rep.get("scores")
                fallback = rep.get("fallback_reason")
            if method["decision"] == "contrast_margin":
                for item in candidate["controls"]:
                    cr = frame["representations"][item["array_key"]][representation]
                    controls.append({**item, "feature_available": cr["feature_available"],
                                     "scores": cr.get("scores"), "representation": representation,
                                     "fallback_reason": cr.get("fallback_reason")})
        outcome = decide(baseline, valid_ids, triggered=triggered and name not in ("EV00_D2", "EV01_G1_V2"),
            purity=candidate["purity"] if candidate["purity"] is not None else 1., target_scores=target,
            controls=[x["scores"] for x in controls if x["feature_available"]], decision=method["decision"],
            representation_fallback=fallback)
        if name == "EV00_D2":
            outcome.update(accepted=False, defer_reason="NO_RECOVERY_BASELINE")
        outcome.update(owner=int(owner), method=name, bound_trigger=triggered, geometric_purity=candidate["purity"],
            request_id=candidate["request_id"], controls=controls, original_support=candidate.get("original_support"),
            baseline_fallback=candidate.get("baseline_fallback"), original_B1_label=baseline["label"],
            representation_vector_sha256=rep.get("aggregate_vector_sha256") if rep and rep["feature_available"] else baseline.get("aggregate_vector_sha256"),
            score_kind="RELATIVE_CONTRAST_EVIDENCE" if method["decision"] == "contrast_margin" and triggered else "ORIGINAL_FC_COSINE",
            relabelled=bool(outcome["accepted"] and outcome["proposed_class"] != baseline["label"]))
        decisions[owner] = outcome
    return decisions


def predict_scene(binding, root, scene):
    started = time.monotonic()
    root, index = Path(root), ConsumptionIndex()
    dest = root/"predictions"/scene
    prepared = read(root/"prepared"/scene/"receipt.json")
    _verified_identity(prepared)
    source = read(binding["scenes"][scene]["G1_source"]["path"])
    _verified_identity(source)
    frames = {}
    for fid, group in prepared["frames"].items():
        if group["triggered_owners"]:
            frames[fid] = read(root/"encoded"/scene/fid/"receipt.json")
            _verified_identity(frames[fid])
    producer = {name: index.identity(Path(__file__).with_name(name))["sha256"] for name in ("outputs.py", "verifier.py")}
    key = canonical_digest({"binding": binding["identity"], "prepared": prepared["identity"], "source": source["identity"],
        "frames": {fid: row["identity"] for fid, row in frames.items()}, "producer": producer})
    path = dest/"receipt.json"
    if path.exists():
        previous = read(path)
        _verified_identity(previous)
        if previous["input_identity"] != key:
            raise ValueError("locked prediction inputs changed; explicitly invalidate descendants")
        for output in previous["outputs"]:
            index.identity(output["path"], output)
        print("PREDICT_REUSE", scene, flush=True)
        return previous
    inputs = load_common(binding["reference"], scene, root/"common"/scene)
    config = execution_config(binding["reference"])
    registry = build_registry(inputs.xyz, inputs.raw, inputs.baseline.owner_ids)
    if registry["identity"] != prepared["registry_identity"]:
        raise ValueError("export registry differs from fixed predictor candidate registry")
    incumbents = owner_labels(inputs.baseline)
    d2, _ = fuse_readout(inputs.sources, config["final_temperatures"], "D2", inputs.valid_ids, incumbents)
    baselines = {name: load_prediction(row["prediction_manifest"]) for name, row in binding["scenes"][scene]["baseline_rows"].items()}
    if owner_labels(baselines["EV00_D2"]) != d2 or not np.array_equal(baselines["EV00_D2"].owner_ids, inputs.baseline.owner_ids):
        raise ValueError("imported D2 differs from exact frozen N/Q/F readout/support")
    seen = {prediction_content(value): {"method": name, "manifest": binding["scenes"][scene]["baseline_rows"][name]["prediction_manifest"],
        "payload": value} for name, value in baselines.items()}
    outcomes, outputs = {}, []
    for method in binding["specification"]["methods"]:
        name = method["id"]
        decisions = decisions_for_method(method, source, prepared, frames, inputs.valid_ids)
        accepted = {int(owner): row["proposed_class"] for owner, row in decisions.items() if row["accepted"]}
        if name in baselines:
            payload = baselines[name]
            manifest = binding["scenes"][scene]["baseline_rows"][name]["prediction_manifest"]
            alias = None
        else:
            payload = construct_output(inputs.baseline, inputs.raw, registry, accepted, inputs.valid_ids,
                inputs.nearest, inputs.matched, name, incumbent_labels=d2, metadata={"method_recipe": method,
                    "evidence_binding_identity": binding["identity"], "prepared_identity": prepared["identity"],
                    "decision_identity": canonical_digest(decisions), "deployment": "N0_UNCHANGED"})
            active = inputs.baseline.owner_ids > 0
            if not np.array_equal(payload.semantic_labels[active], baselines["EV00_D2"].semantic_labels[active]):
                raise ValueError("verifier changed immutable D2 incumbent semantics")
            content = prediction_content(payload)
            if content in seen:
                previous = seen[content]
                alias, manifest = previous["method"], previous["manifest"]
            else:
                alias, manifest = None, str(save_prediction(payload, dest/name))
                seen[content] = {"method": name, "manifest": manifest, "payload": payload}
        content = prediction_content(payload)
        decision_path = dest/name/"decisions.json"
        atomic_write_json(decision_path, seal({"method": name, "scene": scene, "objects": decisions,
            "GT_input": False, "input_identity": key, "accepted_labels": accepted}))
        summary = Counter(accepted=sum(row["accepted"] for row in decisions.values()),
            deferred=sum(row["defer_reason"] == "LOW_MARGIN" for row in decisions.values()),
            relabelled=sum(row["relabelled"] for row in decisions.values()), triggered=sum(row["bound_trigger"] for row in decisions.values()),
            representation_fallbacks=sum(row["representation_fallback"] is not None for row in decisions.values()))
        outcomes[name] = {"status": "PREDICTIONS_LOCKED", "manifest": str(manifest), "prediction_key": payload.prediction_key,
            "record_key": payload.record_key, "content_identity": content, "alias_of": alias,
            "scientific_no_op_vs_B1": content == prediction_content(baselines["EV01_G1_V2"]),
            "decisions": index.identity(decision_path), "counts": dict(summary)}
        outputs.append(index.identity(decision_path))
        prediction_manifest = read(manifest)
        outputs += [index.identity(manifest), index.identity(Path(manifest).parent/prediction_manifest["arrays"]["path"], prediction_manifest["arrays"])]
    result = seal({"status": "PREDICTIONS_LOCKED", "scene": scene, "cohort": binding["scenes"][scene]["cohort"],
        "input_identity": key, "prepared_identity": prepared["identity"], "outcomes": outcomes,
        "unique_prediction_inputs": len({x["content_identity"] for x in outcomes.values()}),
        "outputs": outputs, "incumbent_labels_verified_against_exact_D2": True,
        "elapsed_seconds": time.monotonic()-started, "GT_used_by_predictor": False})
    atomic_write_json(path, result)
    print("PREDICTIONS_LOCKED", scene, "9 outcomes", result["unique_prediction_inputs"], "unique inputs", flush=True)
    return result


def predict(binding, root, scenes=None):
    result = [predict_scene(binding, root, scene) for scene in (scenes or binding["scenes"])]
    if scenes is None:
        atomic_write_json(Path(root)/"predictions/summary.json", seal({"status": "PREDICTIONS_LOCKED",
            "scene_count": len(result), "scene_method_outcomes": sum(len(x["outcomes"]) for x in result),
            "scenes": {x["scene"]: x["identity"] for x in result}, "GT_used_by_predictor": False}))
    return result
