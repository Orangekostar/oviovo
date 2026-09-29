"""Post-prediction source interventions and damage on unchanged geometry."""

import gzip
import json
from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.attribution import attribute_scene
from src.static_ovmap.m2_reviewer_study.binding import InputIndex
from src.static_ovmap.m2_reviewer_study.evaluation import write_gzip
from src.static_ovmap.m2_reviewer_study.scores import read_scene

from .calibration import source_path


def diagnose_scene(binding, scene, methods):
    root, index = Path(binding["output_root"]), InputIndex()
    # Every complete-map prediction must already be locked, including all other
    # families, before opening evaluation-only semantic correspondences.
    labels = {}
    for method in methods:
        path = root / "locked" / scene / (method + ".json")
        index.identity(path)
        labels[method] = {int(k): v for k, v in read_json(path)["labels"].items()}
    parent = read_json(binding["reviewer_binding"])
    geometry_path = Path(parent["output_root"]) / "diagnostics" / scene / "objects.json.gz"
    index.identity(geometry_path)
    with gzip.open(geometry_path, "rt") as handle:
        geometry = json.load(handle)
    if {r["owner_id"] for r in geometry} != set(labels["N0"]):
        raise ValueError("diagnostic geometry does not cover the unchanged map")
    spec = read_json(binding["spec"])
    definitions = {v["id"]: v for v in spec["source_variants"]}
    sources = {}
    for name, path in binding["scenes"][scene]["sources"].items():
        index.identity(path)
        sources[name] = read_json(path)
    sources["N0_COSINE"] = read_scene(parent, scene, index).cosine_native
    for variant in definitions:
        path = source_path(binding, scene, variant)
        index.identity(path)
        sources[variant] = read_json(path)
    ids = sources["N0"]["valid_ids"]
    base = labels["RV_A7_COS_REFIT"]
    pair = read_json(root / "composition.json")
    objects, summaries = [], {}
    for row in geometry:
        owner = row["owner_id"]
        identifiable = row["correspondence"] == "unique" and row["geometry_iou"] > .5 and row["gt_label"] in ids
        gt = row["gt_label"] if identifiable else None
        record = {k: row[k] for k in ("owner_id", "correspondence", "geometry_iou", "geometry_gt_ids", "gt_label")}
        record.update(identifiable=identifiable, native_label=labels["N0"][owner], base_A7_label=base[owner],
                      source_evidence={}, methods={})
        for name, source in sources.items():
            value = source["objects"][str(owner)]
            scores = np.asarray(value["scores"], np.float64) if value["available"] else None
            target = ids.index(gt) if identifiable else None
            record["source_evidence"][name] = {"available": value["available"], "label": value["label"],
                "correct": value["label"] == gt if identifiable and value["available"] else None,
                "correct_class_rank": int(1 + np.sum(scores > scores[target])) if target is not None and scores is not None else None,
                "fallback_reason": value.get("fallback_reason")}
        for method in methods:
            label = labels[method][owner]
            corrected = identifiable and base[owner] != gt and label == gt
            harmed = identifiable and base[owner] == gt and label != gt
            selected_sources = ["N0_COSINE", "Q_GAIN", "S_SIGLIP2_AREA"]
            if method == "N0":
                selected_sources = ["N0"]
            elif method in {"Q_GAIN", "S_SIGLIP2_AREA"}:
                selected_sources = [method]
            elif method == "AW_COMBO_QR":
                selected_sources = ["N0_COSINE", pair["q_variant"], pair["region_variant"]]
            elif method.endswith(("_DIRECT", "_A7")):
                variant, mode = method.rsplit("_", 1)
                selected_sources = [variant] if mode == "DIRECT" else ["N0_COSINE", variant,
                    "S_SIGLIP2_AREA" if definitions[variant]["slot"] == "Q_GAIN" else "Q_GAIN"]
            correct_source = any(record["source_evidence"][s]["correct"] is True for s in selected_sources)
            record["methods"][method] = {"label": label, "changed_from_A7": label != base[owner],
                "changed_from_N0": label != labels["N0"][owner], "corrected_A7": bool(corrected), "harmed_A7": bool(harmed),
                "corrected_N0": bool(identifiable and labels["N0"][owner] != gt and label == gt),
                "harmed_N0": bool(identifiable and labels["N0"][owner] == gt and label != gt),
                "correct": label == gt if identifiable else None,
                "unused_correct_evidence": bool(identifiable and correct_source and label != gt),
                "outcome": "CORRECTED" if corrected else "HARMED" if harmed else "UNCHANGED_OR_OTHER" if identifiable else "UNIDENTIFIABLE"}
        objects.append(record)
    for method in methods:
        rows = [r["methods"][method] for r in objects]
        summaries[method] = {"scene": scene, "method": method, "owners": len(objects),
            "identifiable": sum(r["identifiable"] for r in objects),
            **{key: sum(bool(r[key]) for r in rows) for key in ("changed_from_A7", "changed_from_N0", "corrected_A7", "harmed_A7",
                                                              "corrected_N0", "harmed_N0", "unused_correct_evidence")}}
    source_summary = {}
    for variant, definition in definitions.items():
        old, new = sources[definition["slot"]]["objects"], sources[variant]["objects"]
        common = [r for r in objects if r["identifiable"] and old[str(r["owner_id"])]["available"] and new[str(r["owner_id"])]["available"]]
        source_summary[variant] = {"owners": len(objects), "available": sum(o["available"] for o in new.values()),
            "changed_suggestions_common_available": sum(o["available"] and old[k]["available"] and o["label"] != old[k]["label"] for k, o in new.items()),
            "common_identifiable": len(common),
            "old_correct_common": sum(old[str(r["owner_id"])]["label"] == r["gt_label"] for r in common),
            "new_correct_common": sum(new[str(r["owner_id"])]["label"] == r["gt_label"] for r in common),
            "requested_views": sum(len(o.get("attempted_request_ids", o.get("retained_request_ids", []))) for o in new.values()),
            "usable_views": sum(len(o.get("used_request_ids", o.get("retained_request_ids", []))) for o in new.values() if o["available"])}
    example_method = "AW_COMBO_QR" if "AW_COMBO_QR" in methods else "RV_A7_COS_REFIT"
    examples = {key: [r["owner_id"] for r in sorted(objects, key=lambda r: r["owner_id"])
                      if r["methods"][example_method][key]][:2] for key in ("corrected_A7", "harmed_A7")}
    out = root / "diagnostics" / scene
    write_gzip(out / "objects.json.gz", objects)
    # This reuses the instrumented released evaluator's actual event traces;
    # geometry-only semantic correction counts above are never called AP TPs.
    attribution = attribute_scene({**binding, "methods": methods}, scene)
    trace_summary = {}
    for key, value in attribution.items():
        trace_summary[key] = {"eligibility_added_owners": value["eligibility_added_owners"],
                              "eligibility_removed_owners": value["eligibility_removed_owners"],
                              "AP25_AP50": [{"overlap_threshold": t["overlap_threshold"],
                                  "added_gt_matches": len(t["added_gt_matches"]), "lost_gt_matches": len(t["lost_gt_matches"]),
                                  "changed_classes": len(t["class_ap_changes"])} for t in value["AP25_AP50"]]}
    result = {"status": "POST_PREDICTION_DIAGNOSTICS_COMPLETE", "scene": scene, "methods": summaries,
              "source_intervention": source_summary, "released_trace_summary": trace_summary,
              "examples": {"method": example_method, "selection": "owner_id ascending, up to two each", **examples},
              "inputs": index.entries(), "geometry_is_unchanged": True}
    write_once(out / "summary.json", result)
    return result
