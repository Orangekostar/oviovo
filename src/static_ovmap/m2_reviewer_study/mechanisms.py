"""Source-correlated errors, exact operation overlap and ablation strata."""

import gzip
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.io import read_json, write_once

from .costs import operation_key
from .diagnostics import SOURCES
from .evaluation import write_gzip


def mechanism_scene(binding, scene):
    root = Path(binding["output_root"])
    config = read_json(binding["scenes"][scene]["config"])
    data = config["scenes"][scene]
    capture = read_json(data["capture"])
    native_requests = {r["request_id"]: r for f in capture["frames"] for r in f["requests"]}
    static_requests = read_json(Path(data["source_directory"]) / "semantic_requests.json")["requests"]
    with gzip.open(root / "diagnostics" / scene / "objects.json.gz", "rt") as handle:
        objects = json.load(handle)
    operations = {}
    for row in objects:
        owner_operations = {}
        for name, source in row["source_evidence"].items():
            requests = static_requests if name == "S_SIGLIP2_AREA" else native_requests
            model = config["models"]["siglip2" if name == "S_SIGLIP2_AREA" else "native"]["identity"]
            used = source.get("used_request_ids", [])
            if not set(used) <= set(requests):
                raise ValueError("used source request lacks actual input identity")
            owner_operations[name] = {request: operation_key(model, requests[request]) for request in used}
        n, q = (set(owner_operations[name].values()) for name in ("N0", "Q_GAIN"))
        operations[row["owner_id"]] = {"sources": owner_operations, "N0_Q": {
            "intersection": sorted(n & q), "union_count": len(n | q),
            "jaccard": len(n & q) / len(n | q) if n | q else None,
            "overlap_coefficient": len(n & q) / min(len(n), len(q)) if n and q else None}}
    paired = [r for r in objects if r["paired_population"]]
    coerror = {}
    for a, b in combinations(SOURCES, 2):
        x = np.asarray([not r["source_correct"][a] for r in paired], dtype=float)
        y = np.asarray([not r["source_correct"][b] for r in paired], dtype=float)
        coerror[f"{a}/{b}"] = {"denominator": len(paired), "both_wrong": int((x * y).sum()),
            "coerror_rate": float((x * y).mean()) if len(paired) else None,
            "error_correlation": float(np.corrcoef(x, y)[0, 1]) if len(x) and x.std() > 0 and y.std() > 0 else None,
            "a_only_correct": sum(r["source_correct"][a] and not r["source_correct"][b] for r in paired),
            "b_only_correct": sum(r["source_correct"][b] and not r["source_correct"][a] for r in paired)}
    exclusive = {name: {"denominator": len(paired), "correct_only_this_source": sum(
        r["source_correct"][name] and all(not r["source_correct"][other] for other in SOURCES if other != name)
        for r in paired)} for name in SOURCES}
    strata = {}
    for available_count in range(4):
        population = [r for r in objects if r["identifiable"] and
                      sum(v["available"] for v in r["source_evidence"].values()) == available_count]
        strata[str(available_count)] = {"denominator": len(population), "methods": {method: {
            "correct": sum(r["methods"][method]["label"] == r["gt_label"] for r in population),
            "abstentions": sum(r["methods"][method]["label"] == 0 for r in population),
            "corrected": sum(r["methods"][method]["outcome"] == "CORRECTED" for r in population),
            "harmed": sum(r["methods"][method]["outcome"] == "HARMED" for r in population)} for method in binding["methods"]}}
    ablations = {}
    for method in binding["methods"]:
        if not method.startswith("RV_A"):
            continue
        changes = []
        for row in objects:
            original, ablated = row["methods"]["CP_M2_EQUAL_CAL"], row["methods"][method]
            if original["label"] != ablated["label"]:
                changes.append({"owner_id": row["owner_id"], "identifiable": row["identifiable"],
                                "gt_label": row["gt_label"], "original_label": original["label"],
                                "ablation_label": ablated["label"], "original_margin": original["top2_margin"],
                                "ablation_margin": ablated["top2_margin"]})
        ablations[method] = {"all_owner_denominator": len(objects), "changes": changes}
    report = {"status": "COMPLETE", "scene": scene, "paired_owner_ids": [r["owner_id"] for r in paired],
              "coerror": coerror, "exclusive_corrections": exclusive, "source_count_strata": strata,
              "ablation_changes": ablations,
              "oracle_object_headroom": {"denominator": len(paired), "native_wrong_any_source_right": sum(
                  not r["source_correct"]["N0"] and any(r["source_correct"].values()) for r in paired),
                  "diagnostic_only": True, "not_deployable_AP": True},
              "causal_claim": "none; shared-model observations are correlated evidence"}
    write_gzip(root / "diagnostics" / scene / "used_operations.json.gz", operations)
    write_once(root / "diagnostics" / scene / "mechanisms.json", report)
    return report
