#!/usr/bin/env python3
"""Recompute evaluator-only attribution for a published Apartment B7 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.execute_ovi_rescene_two_visit_matrix import _evaluation_context
from scripts.evaluation.run_ovi_rescene_b7 import (
    B7_RUN_ARTIFACT_ID,
    B7RunError,
    _canonical_json,
    _json_object,
    _load_local_artifact,
    _normalized_record,
    _read_regular,
    authorize_b7_scene,
    canonical_relations_bytes,
    load_and_validate_b7_config,
    load_frozen_b7_inputs,
)
from src.evaluation.baselines.tesse_semantics import load_tesse_semantic_crosswalk
from src.evaluation.two_visit_b7_attribution import (
    attribute_b7_recovery,
    write_b7_attribution,
)
from src.oviv2.two_visit_dense_recovery import read_dense_recovery
from src.oviv2.two_visit_execution import SignedVisibilityConfig


def attribute_published_b7(
    *,
    config_path: str | Path,
    run_manifest_path: str | Path,
    output_root: str | Path,
) -> Path:
    """Validate a published method package and write fresh attribution only."""

    config_file = Path(config_path).absolute()
    config_bytes = _read_regular(config_file, label="B7 configuration")
    config = load_and_validate_b7_config(
        config_file,
        verify_source_bindings=True,
        verify_frozen_inputs=False,
    )
    authorize_b7_scene(config, "apartment")
    manifest_file = Path(run_manifest_path).absolute()
    manifest = _json_object(
        _read_regular(manifest_file, label="B7 run manifest"),
        label="B7 run manifest",
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_id") != B7_RUN_ARTIFACT_ID
        or manifest.get("status") != "PASS"
        or manifest.get("scene") != "apartment"
        or manifest.get("variant_id") != "B7-G"
    ):
        raise B7RunError("B7 run manifest identity is invalid")
    expected_config = {
        "path": str(config_file),
        "sha256": hashlib.sha256(config_bytes).hexdigest(),
        "byte_count": len(config_bytes),
    }
    if manifest.get("config") != expected_config:
        raise B7RunError("B7 run and configuration bindings differ")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise B7RunError("B7 run artifact inventory is invalid")
    relation_path = _load_local_artifact(
        artifacts.get("relations"),
        root=manifest_file.parent,
        label="B7 relations",
    )
    dense_manifest_path = _load_local_artifact(
        artifacts.get("dense_recovery_manifest"),
        root=manifest_file.parent,
        label="B7 dense recovery manifest",
    )
    inputs = load_frozen_b7_inputs(config)
    if _read_regular(relation_path, label="B7 relations") != canonical_relations_bytes(
        inputs.relations
    ):
        raise B7RunError("published B7 relations differ from frozen B4 materialization")
    result = read_dense_recovery(dense_manifest_path)
    if (
        result.content_sha256() != manifest.get("dense_recovery_sha256")
        or result.baseline_current_map_sha256 != inputs.baseline.content_sha256()
        or result.source_visit_map_sha256
        != (inputs.t0.snapshot_sha256, inputs.t1.snapshot_sha256)
    ):
        raise B7RunError("published dense recovery source binding mismatch")
    records = config["frozen_inputs"]["apartment"]
    schedule_path, _schedule = _normalized_record(
        records["causal_schedule"], label="causal schedule"
    )
    target_path, _target = _normalized_record(
        records["common_v2_target_manifest"], label="common-v2 target manifest"
    )
    aliases_path, _aliases = _normalized_record(
        records["semantic_aliases"], label="semantic aliases"
    )
    label_space_path, _labels = _normalized_record(
        records["semantic_label_space"], label="semantic label space"
    )
    visibility = SignedVisibilityConfig(**config["method"]["signed_visibility"])
    context, evaluation_bindings = _evaluation_context(
        protocol=inputs.protocol,
        scene="apartment",
        final_frame=inputs.t1.observed_frame_end,
        schedule_path=schedule_path,
        target_manifest_path=target_path,
        frames=inputs.frames,
        visibility_config=visibility,
    )
    load_tesse_semantic_crosswalk(aliases_path, "apartment", label_space_path)
    evaluation_source_sha256 = hashlib.sha256(
        _canonical_json(
            {role: dict(record) for role, record in sorted(evaluation_bindings.items())}
        )
    ).hexdigest()
    attribution = attribute_b7_recovery(
        result=result,
        baseline=inputs.baseline,
        t0=inputs.t0,
        t1=inputs.t1,
        relations=inputs.relations,
        evaluation=context,
        evaluation_source_sha256=evaluation_source_sha256,
    )
    return write_b7_attribution(attribution, output_root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--b7-run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = attribute_published_b7(
            config_path=args.config,
            run_manifest_path=args.b7_run_manifest,
            output_root=args.output,
        )
    except (B7RunError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(str(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
