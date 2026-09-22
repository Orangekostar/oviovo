"""Native-surface geometry pools and complete partition choices, without GT."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from .assets import sha256_file
from .boundary_jobs import file_identity
from .contracts import atomic_write_json, canonical_digest
from .entity_hypotheses import (
    ConflictGroup,
    FrameLeafEvidence,
    LeafContact,
    NativeLeaf,
    NativeLeaves,
    PartitionHypothesis,
    SurfaceGraph,
    apply_selected_partitions,
    build_conflict_groups,
    build_frame_leaf_evidence,
    build_native_leaves,
    build_surface_graph,
    generate_complete_hypotheses,
    select_geometry_frames,
)
from .native_capture import _array_digest, _write_npz
from .partition_quality import (
    PartitionFeatures,
    build_partition_features,
    partition_agreement,
)
from .scannet_runtime import reusable_job
from .static_regions import visibility_frame


def _sources():
    return [file_identity(Path(__file__).with_name(name)) for name in (
        "geometry_study.py", "entity_hypotheses.py", "partition_quality.py", "static_regions.py")]


def prepare_geometry_pool(data, output):
    output = Path(output)
    native, surface, capture = data["native"], data["surface"], data["capture"]
    if (not native.locked or _array_digest(surface["surface_xyz"]) != native.geometry.xyz_sha256
            or _array_digest(surface["surface_faces"]) != native.geometry.faces_sha256):
        raise ValueError("geometry pool requires the locked native source coordinates and faces")
    inputs = [file_identity(path) for path in (data["capture_path"], data["native_manifest_path"])]
    identity = canonical_digest({"inputs": inputs, "sources": _sources(), "native_record_key": native.record_key,
        "surface_fields": {key: _array_digest(surface[key]) for key in ("segment_labels", "surface_normals", "normal_valid")}})
    receipt_path = output / "receipt.json"
    if reusable_job(receipt_path, identity):
        return receipt_path
    if receipt_path.exists():
        raise ValueError("completed geometry pool changed; choose a new output root")
    started = time.monotonic()
    graph = build_surface_graph(surface["surface_xyz"], surface["surface_faces"],
                                surface["surface_normals"], surface["normal_valid"])
    leaves = build_native_leaves(surface["surface_xyz"], native.owner_ids, surface["segment_labels"],
                                 alias_table=capture["alias_table"], surface_graph=graph)
    print(f"{native.scene_id}: G graph {len(leaves.leaves)} leaves, {len(leaves.contacts)} contacts", flush=True)
    frames_by_id = {frame["frame_id"]: frame for frame in capture["frames"]}
    selected_frames = select_geometry_frames([SimpleNamespace(frame_id=frame["frame_id"], pose_c2w=frame["pose_c2w"])
                                              for frame in capture["frames"]])
    evidence, outputs = [], []
    for selected in selected_frames:
        frame = frames_by_id[selected.frame_id]
        source, pixels = visibility_frame(data["capture_path"], frame, surface["surface_xyz"],
                                          native.geometry.xyz_sha256, data["output"] / "surface_visibility")
        panoptic_path = Path(data["capture_path"]).parent / frame["panoptic_path"]
        if sha256_file(panoptic_path) != frame["panoptic_sha256"]:
            raise ValueError("G captured entities changed")
        local = np.asarray(Image.open(panoptic_path)).reshape(-1)[pixels]
        row = build_frame_leaf_evidence(frame["frame_id"], leaves.row_leaf_ids[source], local, leaf_count=len(leaves.leaves))
        evidence.append(row)
        path = output / "frames" / f"{row.frame_id:06d}.npz"
        _write_npz(path, {key: getattr(row, key) for key in (
            "pixel_leaf_ids", "pixel_entity_ids", "observed_pixels", "dominant_pixels", "entity_by_leaf")})
        outputs.append(file_identity(path))
    groups, untouched = build_conflict_groups(leaves, evidence)
    all_hypotheses, features, agreements = {}, {}, {}
    for index, group in enumerate(groups):
        hypotheses = generate_complete_hypotheses(native.scene_id, group, leaves, evidence)
        all_hypotheses[group.group_id] = hypotheses
        for hypothesis in hypotheses:
            features[hypothesis.hypothesis_id] = build_partition_features(hypothesis, group, leaves, graph, evidence, surface["surface_xyz"])
            agreements[hypothesis.hypothesis_id] = asdict(partition_agreement(hypothesis, evidence))
        print(f"{native.scene_id}: G pool {index + 1}/{len(groups)}, {len(group.leaf_ids)} leaves, {len(hypotheses)} hypotheses", flush=True)
    no_op = apply_selected_partitions(native.scene_id, native.owner_ids, leaves,
                                      {group: rows[0] for group, rows in all_hypotheses.items()})
    if not np.array_equal(no_op.owner_ids, native.owner_ids):
        raise ValueError("G ORIGINAL does not reproduce the native incumbent owner assignment")
    arrays_path = output / "pool.npz"
    feature_ids = list(features)
    _write_npz(arrays_path, {"row_leaf_ids": leaves.row_leaf_ids, "resolved_segment_labels": leaves.resolved_segment_labels,
        "leaf_owner_ids": np.array([leaf.owner_id for leaf in leaves.leaves], np.int64),
        "leaf_segment_labels": np.array([leaf.segment_label for leaf in leaves.leaves], np.int64),
        "leaf_centroids": np.asarray([leaf.centroid for leaf in leaves.leaves]),
        "graph_edges": graph.edges, "graph_normals": graph.normals, "normal_valid": graph.normal_valid,
        "contact_indices": np.asarray([[row.left_leaf, row.right_leaf, row.edge_count] for row in leaves.contacts], np.int64).reshape(-1, 3),
        "contact_values": np.asarray([[row.mean_distance, row.mean_abs_normal_dot] for row in leaves.contacts], float).reshape(-1, 2),
        "contact_normal_valid": np.asarray([row.normal_available for row in leaves.contacts], bool),
        "feature_ids": np.asarray(feature_ids),
        "feature_values": np.asarray([features[key].values for key in feature_ids]).reshape(-1, 20),
        "feature_available": np.asarray([features[key].available for key in feature_ids], bool).reshape(-1, 20)})
    outputs.append(file_identity(arrays_path))
    manifest_path = output / "hypotheses.json"
    atomic_write_json(manifest_path, {"groups": [asdict(group) for group in groups], "untouched_owner_ids": untouched,
        "hypotheses": {group: [asdict(row) for row in rows] for group, rows in all_hypotheses.items()},
        "agreements": agreements, "graph_source": graph.source,
        "spectral_omitted_groups": [group.group_id for group in groups if len(group.leaf_ids) > 1024],
        "GT_input": False})
    outputs.append(file_identity(manifest_path))
    atomic_write_json(receipt_path, {"status": "COMPLETE", "input_identity": identity, "inputs": inputs,
        "sources": _sources(), "outputs": outputs, "scene_id": native.scene_id, "native_record_key": native.record_key,
        "frame_ids": [row.frame_id for row in evidence], "original_owner_parity": True,
        "graph_source": graph.source, "leaf_count": len(leaves.leaves), "contact_count": len(leaves.contacts),
        "group_count": len(groups), "hypothesis_count": len(features),
        "maximum_leaf_points": max((leaf.point_count for leaf in leaves.leaves), default=0),
        "known_leaf_frame_observations": sum(int(np.count_nonzero(row.entity_by_leaf)) for row in evidence),
        "elapsed_seconds": time.monotonic() - started})
    return receipt_path


def load_geometry_pool(receipt_path):
    receipt_path = Path(receipt_path)
    output = receipt_path.parent
    receipt = json.loads(receipt_path.read_text())
    if not reusable_job(receipt_path, receipt["input_identity"]):
        raise ValueError("geometry pool output changed")
    for row in receipt["inputs"] + receipt["sources"]:
        if sha256_file(row["path"]) != row["sha256"]:
            raise ValueError("geometry pool source changed")
    manifest = json.loads((output / "hypotheses.json").read_text())
    with np.load(output / "pool.npz", allow_pickle=False) as arrays:
        row_leaf_ids = arrays["row_leaf_ids"]
        order = np.argsort(row_leaf_ids, kind="stable")
        row_groups = np.split(order, np.flatnonzero(np.diff(row_leaf_ids[order])) + 1)
        leaves = tuple(NativeLeaf(index, int(owner), int(segment), rows, len(rows), int(rows.min()), centroid)
            for index, (owner, segment, rows, centroid) in enumerate(zip(arrays["leaf_owner_ids"],
                arrays["leaf_segment_labels"], row_groups, arrays["leaf_centroids"], strict=True)))
        contacts = tuple(LeafContact(int(indices[0]), int(indices[1]), int(indices[2]), float(values[0]), float(values[1]), bool(valid))
            for indices, values, valid in zip(arrays["contact_indices"], arrays["contact_values"], arrays["contact_normal_valid"], strict=True))
        native_leaves = NativeLeaves(row_leaf_ids, arrays["resolved_segment_labels"], leaves, contacts)
        graph = SurfaceGraph(arrays["graph_edges"], arrays["graph_normals"], arrays["normal_valid"], manifest["graph_source"])
        features = {str(key): PartitionFeatures(values, valid) for key, values, valid in zip(
            arrays["feature_ids"], arrays["feature_values"], arrays["feature_available"], strict=True)}
    frames = []
    for frame_id in receipt["frame_ids"]:
        with np.load(output / "frames" / f"{frame_id:06d}.npz", allow_pickle=False) as arrays:
            frames.append(FrameLeafEvidence(frame_id=frame_id, **{key: arrays[key] for key in arrays.files}))
    groups = tuple(ConflictGroup(row["group_id"], tuple(row["owner_ids"]), tuple(row["leaf_ids"]), row["conflict_strength"])
                   for row in manifest["groups"])
    hypotheses = {group: tuple(PartitionHypothesis(**{**row, **{key: tuple(row[key]) for key in (
        "owner_ids", "leaf_ids", "components")}}) for row in rows) for group, rows in manifest["hypotheses"].items()}
    return {"receipt": receipt, "manifest": manifest, "leaves": native_leaves, "surface_graph": graph,
            "frames": tuple(frames), "groups": groups, "hypotheses": hypotheses, "features": features}


def choose_geometry(pool, method, *, scores=None, margin="KEEP_ALL"):
    """Choose from exactly the stored pool; ties prefer ORIGINAL then pool order."""
    from .partition_quality import apply_geometry_margin

    if method not in {"G_ORIGINAL", "G_AGREEMENT", "G_QUALITY"}:
        raise ValueError("unlisted geometry condition")
    result = {}
    for group, rows in pool["hypotheses"].items():
        if not rows or rows[0].kind != "ORIGINAL":
            raise ValueError("geometry hypothesis pool must retain ORIGINAL first")
        selected = rows[0]
        if method == "G_AGREEMENT":
            def score(row):
                value = pool["manifest"]["agreements"][row.hypothesis_id]["mean_frame_agreement"]
                return -float("inf") if value is None else value

            selected = max(rows, key=score)
        elif method == "G_QUALITY":
            if scores is None or any(row.hypothesis_id not in scores for row in rows):
                raise ValueError("G_QUALITY requires a frozen score for every same-pool hypothesis")
            best = max(rows, key=lambda row: scores[row.hypothesis_id])
            if apply_geometry_margin([scores[best.hypothesis_id] - scores[rows[0].hypothesis_id]], margin)[0]:
                selected = best
        result[group] = selected
    return result
