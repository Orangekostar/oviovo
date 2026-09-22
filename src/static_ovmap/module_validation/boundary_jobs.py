"""Executable historical integration checks; these never produce SELECT metrics."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .assets import sha256_file
from .contracts import canonical_digest
from .native_capture import RegionRequest, _array_digest


def file_identity(path: Path | str) -> dict[str, Any]:
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def capture_inputs(manifest_path: Path | str, manifest: dict) -> tuple[dict, ...]:
    """Resolve the complete capture payload, including the loaded native extension."""
    root = Path(manifest_path).resolve().parent
    rows = [{"path": str(Path(manifest_path).resolve())}]
    surface = manifest["surface"]
    rows.append({"path": str(root / surface["path"]), "sha256": surface["sha256"]})
    if "source_mesh_path" in surface:
        rows.append(
            {
                "path": surface["source_mesh_path"],
                "sha256": surface["source_mesh_sha256"],
            }
        )
    extension = manifest.get("native_extension", {})
    if extension.get("path"):
        rows.append(extension)
    for frame in manifest["frames"]:
        for name in ("depth", "rgb", "panoptic", "global_owner"):
            if f"{name}_path" in frame:
                rows.append(
                    {
                        "path": str(root / frame[f"{name}_path"]),
                        "sha256": frame[f"{name}_sha256"],
                    }
                )
        for name in ("preinsert", "native_state", "request_arrays"):
            if name in frame:
                rows.append({**frame[name], "path": str(root / frame[name]["path"])})
    return tuple(rows)


def verify_capture(path: Path | str, *, allow_skipped: bool = False) -> dict:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("artifact_type") != "OVIMAP_NATIVE_CAPTURE":
        raise ValueError("expected a native capture manifest")
    if manifest.get("identity") != canonical_digest(
        {k: v for k, v in manifest.items() if k != "identity"}
    ):
        raise ValueError("native capture manifest identity mismatch")
    frame_ids = [frame["frame_id"] for frame in manifest["frames"]]
    scheduled = manifest["scheduled_frame_ids"]
    expected = [value for value in scheduled if value in set(frame_ids)] if allow_skipped else scheduled
    if (
        not frame_ids
        or frame_ids != expected
        or len(frame_ids) != len(set(frame_ids))
        or frame_ids != manifest["completed_frame_ids"]
    ):
        raise ValueError("native capture has an incomplete frame schedule")
    for row in capture_inputs(path, manifest):
        if "sha256" in row and sha256_file(row["path"]) != row["sha256"]:
            raise ValueError(f"capture payload hash mismatch: {row['path']}")
    for frame in manifest["frames"]:
        for raw_request in frame.get("requests", ()):
            RegionRequest.from_dict(raw_request)
    return manifest


def _frame_arrays(root: Path, frame: dict):
    with np.load(root / frame["depth_path"], allow_pickle=False) as archive:
        depth = archive["depth_m"]
    return (
        depth,
        np.array(Image.open(root / frame["panoptic_path"])),
        np.array(Image.open(root / frame["global_owner_path"])),
    )


def geometry_smoke(manifest_path: Path | str) -> dict:
    from .entity_hypotheses import (
        apply_selected_partitions,
        build_conflict_groups,
        build_native_leaves,
        build_surface_graph,
        generate_complete_hypotheses,
        project_frame_leaf_evidence,
    )

    manifest = verify_capture(manifest_path)
    root = Path(manifest_path).resolve().parent
    with np.load(root / manifest["surface"]["path"], allow_pickle=False) as data:
        # Bound by native face order, without inventing topology for a row slice.
        source_faces = data["surface_faces"][:4096]
        rows, indices = np.unique(source_faces, return_inverse=True)
        faces = indices.reshape(source_faces.shape)
        xyz = data["surface_xyz"][rows]
        owner = data["original_owner"][rows]
        graph = build_surface_graph(
            xyz, faces, data["surface_normals"][rows], data["normal_valid"][rows]
        )
        leaves = build_native_leaves(
            xyz,
            owner,
            data["segment_labels"][rows],
            alias_table=manifest["alias_table"],
            surface_graph=graph,
        )
    frames = []
    for frame in manifest["frames"]:
        depth, entities, _ = _frame_arrays(root, frame)
        frames.append(
            project_frame_leaf_evidence(
                frame["frame_id"],
                xyz,
                leaves.row_leaf_ids,
                frame["pose_c2w"],
                frame["intrinsics"],
                depth,
                entities,
            )
        )
    groups, untouched = build_conflict_groups(leaves, frames)
    hypotheses = {
        group.group_id: generate_complete_hypotheses(
            manifest["scene_id"], group, leaves, frames
        )
        for group in groups
    }
    original = {
        key: next(row for row in values if row.kind == "ORIGINAL")
        for key, values in hypotheses.items()
    }
    result = apply_selected_partitions(manifest["scene_id"], owner, leaves, original)
    if not np.array_equal(result.owner_ids, owner):
        raise AssertionError("original geometry partition changed native ownership")
    return {
        "status": "COMPLETE",
        "scientific_result": False,
        "scope": "first_4096_native_faces",
        "scene_id": manifest["scene_id"],
        "surface_rows": len(xyz),
        "surface_faces": len(faces),
        "leaf_count": len(leaves.leaves),
        "leaf_contact_count": len(leaves.contacts),
        "conflict_group_count": len(groups),
        "untouched_owner_count": len(untouched),
        "complete_hypothesis_count": sum(map(len, hypotheses.values())),
        "noop_owner_exact": True,
        "owner0_exact": True,
    }


def query_smoke(manifest_path: Path | str, feature_root: str, text_cache: str) -> dict:
    from .query_gain_policy import run_query_policy
    from .query_state import (
        AcquisitionPayload,
        FeatureStore,
        NativeCombineState,
        build_query_candidates,
        native_combine_candidates,
    )

    manifest = verify_capture(manifest_path)
    root = Path(manifest_path).resolve().parent
    state = NativeCombineState()
    parity = []
    frame_candidates = []
    for index, frame in enumerate(manifest["frames"]):
        depth, entities, owners = _frame_arrays(root, frame)
        k = np.asarray(frame["intrinsics"])
        y, x = np.indices(depth.shape, dtype=np.float32)
        points = np.dstack(
            ((x - k[0, 2]) * depth / k[0, 0], (y - k[1, 2]) * depth / k[1, 1], depth)
        )
        requests = {
            int(row["target_id"].split(":")[-1]): row for row in frame["requests"]
        }
        candidates = build_query_candidates(
            manifest["scene_id"],
            index,
            frame["frame_id"],
            owners,
            entities,
            depth,
            np.isfinite(depth) & (depth > 0),
            frame["pose_c2w"],
            points,
            request_ids_by_owner={
                owner: row["request_id"] for owner, row in requests.items()
            },
        )
        if len(candidates) != len(requests):
            raise AssertionError("native request count differs")
        for candidate in candidates:
            request = requests[candidate.owner_id]
            if (
                list(candidate.bbox_xyxy) != request["bbox_xyxy"]
                or candidate.overlap_pixels != request["visible_target_pixels"]
                or _array_digest(candidate.global_mask) != request["target_mask_sha256"]
                or _array_digest(candidate.union_mask)
                != request["native_union_mask_sha256"]
            ):
                raise AssertionError("native request mask/bbox/overlap differs")
        selected = native_combine_candidates(candidates, state)
        if [row.request_id for row in selected] != frame["native_selected_request_ids"]:
            raise AssertionError("native combine request IDs differ")
        parity.append(
            {
                "frame_id": frame["frame_id"],
                "candidate_count": len(candidates),
                "selected_count": len(selected),
                "exact": True,
            }
        )
        frame_candidates.append(candidates)
    features = {}
    identities = [file_identity(text_cache)]
    for candidate in frame_candidates[0]:
        x1, y1, x2, y2 = candidate.bbox_xyxy
        path = (
            Path(feature_root)
            / f"siglip-l-16-384_F_{candidate.frame_id}_{x1}-{y1}-{x2 - x1}-{y2 - y1}.npy"
        )
        features[candidate.request_id] = AcquisitionPayload(
            np.load(path, allow_pickle=False).reshape(-1),
            attempted_crop_inputs=6,
            inference_seconds=0.0,
        )
        identities.append(file_identity(path))

    def no_forward(candidate):
        raise RuntimeError(f"unbound historical feature: {candidate.request_id}")

    store = FeatureStore(no_forward, initial_cache=features)
    with np.load(text_cache, allow_pickle=False) as archive:
        text = archive["text"]
    policies = {
        name: run_query_policy(name, frame_candidates[:1], store, text)
        for name in ("Q_AREA", "Q_UNCERTAINTY")
    }
    return {
        "status": "COMPLETE",
        "scientific_result": False,
        "scope": "native candidate parity on all captured frames; frame-0 cached policy replay",
        "candidate_parity": parity,
        "input_identities": identities,
        "policy_replay": {
            name: asdict(result.state.logical_ledger)
            for name, result in policies.items()
        },
        "physical": asdict(store.physical_ledger),
    }


def semantic_smoke(manifest_path: Path | str, config: dict) -> dict:
    import gc

    import torch
    from transformers import AutoModel, AutoTokenizer

    from .region_evidence import (
        FrozenSiglipBackend,
        NativeRegionEncoder,
        OfficialWowRunner,
        WowRegionAdapter,
        clean_wow_category_response,
        map_generated_name,
        native_crops,
    )

    manifest = verify_capture(manifest_path)
    root = Path(manifest_path).resolve().parent
    frame = manifest["frames"][0]
    request = next(
        row for row in frame["requests"] if row["target_id"] == config["target_id"]
    )
    request_id = request["request_id"]
    rgb = np.array(Image.open(root / frame["rgb_path"]).convert("RGB"))
    with np.load(root / frame["request_arrays"]["path"], allow_pickle=False) as archive:
        target, union = archive[f"{request_id}_target"], archive[f"{request_id}_union"]
    crops = native_crops(rgb, target, union, request["bbox_xyxy"])
    native = FrozenSiglipBackend.from_local(config["native_model"])
    encoded = NativeRegionEncoder(native).encode(crops)
    x1, y1, x2, y2 = request["bbox_xyxy"]
    legacy_path = (
        Path(config["feature_root"])
        / f"siglip-l-16-384_F_{frame['frame_id']}_{x1}-{y1}-{x2 - x1}-{y2 - y1}.npy"
    )
    legacy = np.load(legacy_path, allow_pickle=False).reshape(-1)
    error = float(np.max(np.abs(encoded.legacy_mean - legacy)))
    if error > 1e-6:
        raise AssertionError(f"native six-crop parity failed: {error}")
    del native
    gc.collect()
    torch.cuda.empty_cache()
    alternative = FrozenSiglipBackend.from_local(config["siglip2_model"])
    image_vectors = alternative.encode_images(crops.legacy_six)
    text_vectors = alternative.encode_texts(config["class_names"])
    del alternative
    gc.collect()
    torch.cuda.empty_cache()
    runner = OfficialWowRunner.from_local(
        config["wow_model"], official_repo_path=config["wow_code"]
    )
    wow = WowRegionAdapter(runner).classify(rgb, target)
    if wow.status != "COMPLETE":
        raise AssertionError(f"WOW region not classified: {wow.status}")
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(
        config["name_model"], local_files_only=True
    )
    mapper = AutoModel.from_pretrained(
        config["name_model"], local_files_only=True
    ).eval()

    def embed(values):
        tokens = tokenizer(
            list(values),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = mapper(**tokens).last_hidden_state
            weights = tokens["attention_mask"].unsqueeze(-1)
            return ((hidden * weights).sum(1) / weights.sum(1)).numpy()

    mapped = map_generated_name(
        wow.raw_generation,
        config["class_names"],
        clean_response=clean_wow_category_response,
        embed_text=embed,
    )
    fallback = map_generated_name(
        "storage cupboard",
        config["class_names"],
        clean_response=clean_wow_category_response,
        embed_text=embed,
    )
    return {
        "status": "COMPLETE",
        "scientific_result": False,
        "scope": "one native captured request per visual adapter",
        "request_id": request_id,
        "native_crop_shape": list(encoded.vectors.shape),
        "legacy_first_six_max_abs_error": error,
        "siglip2_image_shape": list(image_vectors.shape),
        "siglip2_text_shape": list(text_vectors.shape),
        "wow": {
            "raw_generation": wow.raw_generation,
            "original_mask_support": wow.original_mask_support,
            "final_mask_support": wow.final_mask_support,
            "trace": dict(wow.trace),
        },
        "name_mapping": {"exact": asdict(mapped), "embedding": asdict(fallback)},
        "input_identities": [file_identity(legacy_path)],
        "physical": {
            "visual_model_loads": 3,
            "visual_requests": 3,
            "native_crop_inputs": 9,
            "siglip2_crop_inputs": 6,
            "wow_generations": 1,
            "name_model_loads": 1,
        },
    }
