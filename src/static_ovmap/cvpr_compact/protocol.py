"""Immutable protocol, cohort matrix and inherited numerical contracts."""

import json
from pathlib import Path

import numpy as np

from static_ovmap.module_validation.contracts import canonical_digest


PACKAGE = (Path(__file__).resolve().parents[3] / "docs/paper/static_ovmap/"
           "cvpr_compact_tables_v1/CVPR_COMPACT_TABLES_CODEX_2026-10-03")


def load_spec(path):
    path = Path(path)
    if path.read_bytes() != (PACKAGE / "PROTOCOL_SPEC.json").read_bytes():
        raise ValueError("executable specification differs from the frozen execution package")
    spec = json.loads(path.read_text())
    if spec["task_id"] != "ovimap-cvpr-compact-tables-v1":
        raise ValueError("unexpected compact-table task")
    matrix = experiment_matrix(spec)
    if tuple(map(len, (matrix["anchors"], matrix["outputs"], matrix["pools"], matrix["timings"]))) != (26, 172, 14, 24):
        raise ValueError("fixed experiment matrix changed")
    return spec


def experiment_matrix(spec):
    anchors, outputs, pools, timings = [], [], [], []
    for cohort, scenes in spec["cohorts"].items():
        anchors.extend({"scene": scene, "cohort": cohort,
                        "physical_family": scene.split("_")[0] if cohort == "scannet_cf18" else scene}
                       for scene in scenes)
        for method in spec["methods"]:
            if cohort in method["cohorts"]:
                pools.append({"cohort": cohort, "method": method["id"], "scene_order": list(scenes)})
                outputs.extend({"cohort": cohort, "scene": scene, "method": method["id"]} for scene in scenes)
    for scene in spec["cohorts"][spec["timing"]["cohort"]]:
        timings.extend({"scene": scene, "arm": arm} for arm in spec["timing"]["arms"])
    result = {"anchors": anchors, "outputs": outputs, "pools": pools, "timings": timings,
              "primary_method": spec["selection"]["primary_method"]}
    result["identity"] = canonical_digest(result)
    return result


def final_temperatures(spec, scenes):
    vectors = [scenes[scene]["temperatures"] for scene in spec["cohorts"]["replica8"]]
    first = vectors[0]
    if (set(first) != {"N", "Q", "F"} or any(row != first for row in vectors)
            or not all(np.isfinite(v) and v > 0 for v in first.values())):
        raise ValueError("all eight Replica anchors must use the exact same final N/Q/F temperatures")
    return dict(first)


def validate_overlaps(overlaps):
    values = np.asarray(overlaps)
    expected = np.r_[np.arange(.5, .95, .05), .25]
    if (values.shape != expected.shape or not np.isfinite(values).all()
            or not np.allclose(values, expected, atol=1e-12, rtol=0)):
        raise ValueError("released overlap vector differs from the fixed nine APall thresholds plus AP25")
    return overlaps
