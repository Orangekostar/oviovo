# OVIV2 Route 2 Hybrid Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build and evaluate a deterministic YOLO plus SAM ViT-H high-recall frontend for Replica room0 while preserving the existing OVIV2 tracker, TSDF, dense semantics, and evaluator.

**Architecture:** Add a pure hybrid-proposal policy and a cache-builder CLI that converts frozen YOLO, SAM ViT-H, RADSeg, and depth inputs into the frontend cache schema already consumed by CachedFrontendAdapter. Test three frozen proposal variants through 20-frame and 200-frame room0 runs, rejecting every metric regression and retaining only a strict semantic/AP improvement with byte-identical F@5cm.

**Tech Stack:** Python 3.10, NumPy, SciPy, Pillow, gzip/pickle, immutable JSON manifests, pytest, existing OVIV2 Replica runner.

---

## Scope And File Structure

Create:

- src/oviv2/hybrid_frontend.py: pure array contracts, overlap calculations, dense mask labeling, proposal selection, and diagnostics.
- src/oviv2/hybrid_cache.py: strict frontend payload loading, atomic cache writing, hashing, and manifest construction.
- scripts/build_oviv2_hybrid_frontend_cache.py: room0 cache-builder CLI with source-frame mapping.
- scripts/evaluation/compare_oviv2_pareto.py: deterministic Route 2 and final Pareto gates.
- tests/oviv2/test_hybrid_frontend.py: pure policy tests.
- tests/oviv2/test_hybrid_cache.py: cache I/O and provenance tests.
- tests/evaluation/test_build_oviv2_hybrid_frontend_cache.py: CLI integration tests.
- tests/evaluation/test_compare_oviv2_pareto.py: metric-gate tests.
- configs/oviv2_replica_room0_route2_sam_labeled.json: SAM-only runner config.
- configs/oviv2_replica_room0_route2_yolo_novel_sam.json: recommended hybrid runner config.
- configs/oviv2_replica_room0_route2_quota_nms.json: capped/NMS hybrid runner config.
- docs/superpowers/reports/2026-07-21-oviv2-route2-room0-results.md: immutable experiment comparison.

Do not modify CachedFrontendAdapter, tracker, registry, TSDF, dense semantic integration, meshing, or the Replica evaluator in Route 2. Generated data lives outside Git under /home/ww/oviovo_experiments/20260721_route2_frontend/.

### Task 1: Add Pure Hybrid Proposal Contracts

**Files:**

- Create: src/oviv2/hybrid_frontend.py
- Test: tests/oviv2/test_hybrid_frontend.py

- [ ] **Step 1: Write failing overlap and validation tests**

Add:

    import numpy as np
    import pytest

    from src.oviv2.hybrid_frontend import (
        FrontendBatch,
        HybridFrontendConfig,
        mask_overlap,
    )


    def test_mask_overlap_reports_iou_and_directed_coverages() -> None:
        left = np.asarray([[1, 1, 0], [1, 1, 0]], dtype=bool)
        right = np.asarray([[0, 1, 1], [0, 1, 1]], dtype=bool)

        overlap = mask_overlap(left, right)

        assert overlap.intersection == 2
        assert overlap.iou == pytest.approx(2 / 6)
        assert overlap.left_coverage == pytest.approx(0.5)
        assert overlap.right_coverage == pytest.approx(0.5)


    def test_frontend_batch_rejects_misaligned_vectors() -> None:
        with pytest.raises(ValueError, match="batch dimensions"):
            FrontendBatch(
                masks=np.zeros((2, 4, 5), dtype=bool),
                boxes_xyxy=np.zeros((1, 4), dtype=np.float32),
                confidences=np.ones(2, dtype=np.float32),
                labels=("chair", "table"),
                image_features=np.ones((2, 3), dtype=np.float32),
            )


    def test_hybrid_config_rejects_unbounded_thresholds() -> None:
        with pytest.raises(ValueError, match="novel_iou"):
            HybridFrontendConfig(variant="yolo_novel_sam", novel_iou=1.1)

- [ ] **Step 2: Run the tests and verify the missing module**

Run:

    /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q       tests/oviv2/test_hybrid_frontend.py

Expected: collection fails with ModuleNotFoundError for src.oviv2.hybrid_frontend.

- [ ] **Step 3: Implement the immutable contracts**

Implement these public types and function:

    @dataclass(frozen=True)
    class HybridFrontendConfig:
        variant: str
        yolo_match_iou: float = 0.50
        yolo_match_coverage: float = 0.60
        novel_iou: float = 0.80
        duplicate_iou: float = 0.85
        minimum_area_fraction: float = 0.0001
        maximum_area_fraction: float = 0.50
        minimum_valid_depth_fraction: float = 0.50
        minimum_dense_probability: float = 0.25
        minimum_dense_margin: float = 0.05
        maximum_structure_probability: float = 0.50
        maximum_proposals: int = 64
        maximum_per_class: int = 12


    @dataclass(frozen=True)
    class FrontendBatch:
        masks: np.ndarray
        boxes_xyxy: np.ndarray
        confidences: np.ndarray
        labels: tuple[str, ...]
        image_features: np.ndarray


    @dataclass(frozen=True)
    class MaskOverlap:
        intersection: int
        iou: float
        left_coverage: float
        right_coverage: float


    def mask_overlap(left: np.ndarray, right: np.ndarray) -> MaskOverlap:
        left = np.asarray(left, dtype=bool)
        right = np.asarray(right, dtype=bool)
        if left.ndim != 2 or left.shape != right.shape:
            raise ValueError("masks must be two dimensional with equal shape")
        intersection = int(np.count_nonzero(left & right))
        left_size = int(np.count_nonzero(left))
        right_size = int(np.count_nonzero(right))
        union = left_size + right_size - intersection
        return MaskOverlap(
            intersection=intersection,
            iou=float(intersection / union) if union else 0.0,
            left_coverage=float(intersection / left_size) if left_size else 0.0,
            right_coverage=float(intersection / right_size) if right_size else 0.0,
        )

Accept only variants sam_labeled, yolo_novel_sam, and quota_nms_ensemble. Copy arrays into contiguous read-only storage. Require finite normalized nonzero feature rows and probability thresholds in [0, 1].

- [ ] **Step 4: Run tests and commit**

    /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q       tests/oviv2/test_hybrid_frontend.py
    git add src/oviv2/hybrid_frontend.py tests/oviv2/test_hybrid_frontend.py
    git commit -m "feat: add hybrid frontend contracts"

Expected: focused tests pass.

### Task 2: Add RADSeg Mask Labeling

**Files:**

- Modify: src/oviv2/hybrid_frontend.py
- Modify: tests/oviv2/test_hybrid_frontend.py

- [ ] **Step 1: Write failing dense-mask aggregation tests**

Use a stride-2 DenseSemanticFrame fixture and verify:

    def test_dense_mask_aggregation_returns_nonstructural_label() -> None:
        result = aggregate_dense_mask(
            mask=_top_left_mask(),
            valid_depth=np.ones((4, 4), dtype=bool),
            dense=_dense_fixture(),
            structure_ids={1},
        )

        assert result.semantic_id == 2
        assert result.probability == pytest.approx(0.7)
        assert result.valid_depth_fraction == pytest.approx(1.0)

Add explicit fixtures containing class_ids, probabilities, entropy, and margin arrays. Also test empty masks, inadequate valid depth, structure-dominated masks, one-based class IDs, ceiling-divided sampled shapes, and deterministic class-ID ties.

- [ ] **Step 2: Run and verify failure**

Run the Task 1 test command. Expected: aggregate_dense_mask import or assertion failure.

- [ ] **Step 3: Implement sparse top-k aggregation**

Add:

    @dataclass(frozen=True)
    class DenseMaskLabel:
        semantic_id: int
        probability: float
        margin: float
        structure_probability: float
        valid_depth_fraction: float


    def aggregate_dense_mask(
        mask: np.ndarray,
        valid_depth: np.ndarray,
        dense: DenseSemanticFrame,
        *,
        structure_ids: set[int],
    ) -> DenseMaskLabel:
        mask = np.asarray(mask, dtype=bool)
        valid_depth = np.asarray(valid_depth, dtype=bool)
        if mask.shape != dense.image_shape or valid_depth.shape != dense.image_shape:
            raise ValueError("mask and valid_depth must match dense image_shape")
        sampled_mask = mask[::dense.sample_stride, ::dense.sample_stride]
        sampled_depth = valid_depth[::dense.sample_stride, ::dense.sample_stride]
        selected_count = int(np.count_nonzero(sampled_mask))
        selected = sampled_mask & sampled_depth
        valid_selected_count = int(np.count_nonzero(selected))
        if selected_count == 0 or valid_selected_count == 0:
            return DenseMaskLabel(0, 0.0, 0.0, 0.0, 0.0)
        support = np.zeros(dense.class_count + 1, dtype=np.float64)
        for slot in range(dense.class_ids.shape[2]):
            ids = dense.class_ids[..., slot][selected]
            probabilities = dense.probabilities[..., slot][selected]
            np.add.at(support, ids, probabilities)
        support /= valid_selected_count
        mean_structure_probability = float(sum(support[value] for value in structure_ids))
        object_support = support.copy()
        object_support[0] = 0.0
        object_support[list(structure_ids)] = 0.0
        order = np.lexsort((np.arange(len(object_support)), -object_support))
        winning_nonstructural_id = int(order[0])
        runner_up_id = int(order[1]) if len(order) > 1 else 0
        winning_mean_probability = float(object_support[winning_nonstructural_id])
        runner_up_mean_probability = float(object_support[runner_up_id])
        return DenseMaskLabel(
            semantic_id=winning_nonstructural_id,
            probability=winning_mean_probability,
            margin=winning_mean_probability - runner_up_mean_probability,
            structure_probability=mean_structure_probability,
            valid_depth_fraction=valid_selected_count / selected_count,
        )

Sample with mask[::dense.sample_stride, ::dense.sample_stride]. Accumulate only cached top-k class IDs and probabilities under valid samples. Do not allocate an H x W x C tensor. Return semantic_id zero when no non-structural class has positive support.

- [ ] **Step 4: Run focused tests and commit**

    /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q       tests/oviv2/test_hybrid_frontend.py
    git add src/oviv2/hybrid_frontend.py tests/oviv2/test_hybrid_frontend.py
    git commit -m "feat: label SAM masks from dense evidence"

### Task 3: Add Hybrid Proposal Selection

**Files:**

- Modify: src/oviv2/hybrid_frontend.py
- Modify: tests/oviv2/test_hybrid_frontend.py

- [ ] **Step 1: Write failing tests for all variants**

Add complete synthetic YOLO, SAM, dense, and valid-depth fixtures. Require:

    def test_yolo_novel_sam_keeps_yolo_and_adds_only_novel_sam() -> None:
        result = select_hybrid_proposals(
            yolo=_two_yolo_proposals(),
            sam=_matched_duplicate_and_novel_sam(),
            dense=_dense_fixture(),
            valid_depth=np.ones((4, 5), dtype=bool),
            class_names=("wall", "chair", "table"),
            structure_ids={1},
            config=HybridFrontendConfig(variant="yolo_novel_sam"),
        )

        assert result.batch.labels == ("chair", "table", "chair")
        assert result.diagnostics["accepted_yolo"] == 2
        assert result.diagnostics["accepted_novel_sam"] == 1
        assert result.diagnostics["rejected_duplicate"] == 1

Also test YOLO inheritance, RADSeg fallback, area rejection, structure rejection, depth rejection, SAM NMS, stable confidence ordering, class caps, global caps, and repeated-call equality. For `quota_nms_ensemble`, retain YOLO anchors under the final per-class and global caps before NMS; preserve their confidence/index ordering without YOLO-vs-YOLO suppression, initialize occupancy only with those retained YOLO masks, then reject a SAM candidate at IoU >= `duplicate_iou` against a retained YOLO or higher-priority SAM. A YOLO removed by a quota must not suppress a SAM candidate.

- [ ] **Step 2: Run and verify failure**

Run tests/oviv2/test_hybrid_frontend.py. Expected: select_hybrid_proposals is absent.

- [ ] **Step 3: Implement deterministic selection**

Add HybridFrameResult and select_hybrid_proposals(). Use this priority:

1. Original YOLO proposals, descending confidence then original index.
2. SAM proposals with reliable YOLO inheritance.
3. Novel SAM proposals with RADSeg fallback.
4. Within one class, descending confidence, descending area, then source index.

Freeze compact rescue to `quota_nms_ensemble` only. Add `compact_rescue_minimum_area_fraction=0.00002`, `compact_rescue_minimum_confidence=0.60`, and `maximum_compact_rescues=4`; validate both thresholds as finite values in [0, 1] and the cap as a positive non-bool integer. For `quota_nms_ensemble`, require the rescue area minimum to be strictly below `minimum_area_fraction`; non-quota variants do not apply that relationship constraint. A rescue candidate must be a SAM mask with area fraction in [`compact_rescue_minimum_area_fraction`, `minimum_area_fraction`), pass the normal maximum-area, depth, and structure filters, and obtain either normal reliable-YOLO inheritance or normal RADSeg fallback. Reject a rescue below the final confidence threshold as `rejected_area`. Rank regular inherited SAM first, compact rescues second, and regular RADSeg fallback third, using confidence descending, area descending, and source index ascending within each tier. Apply the same cross-source NMS, per-class cap, and global cap; accept at most `maximum_compact_rescues`, report final `accepted_compact_rescue` and deterministic `rejected_compact_rescue_cap`, and retain the original `accepted_inherited_sam` or `accepted_novel_sam` source count.

Accept YOLO inheritance at IoU >= 0.50 or either directed coverage >= 0.60. Require maximum YOLO IoU < 0.80 for novelty. Compute inherited confidence as yolo_confidence times the square root of the maximum overlap measure, clipped to [0, 1]. Use dense probability for fallback confidence.

Emit source image features and omit text_feats. The cache builder must verify that both source caches used the same ViT-H-14 checkpoint before mixing features.

- [ ] **Step 4: Run tests and commit**

    /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q       tests/oviv2/test_hybrid_frontend.py
    git add src/oviv2/hybrid_frontend.py tests/oviv2/test_hybrid_frontend.py
    git commit -m "feat: select high recall hybrid proposals"

### Task 4: Add Strict Hybrid Cache I/O

**Files:**

- Create: src/oviv2/hybrid_cache.py
- Test: tests/oviv2/test_hybrid_cache.py

- [ ] **Step 1: Write failing round-trip and rejection tests**

Add:

    def test_hybrid_cache_round_trip_is_adapter_compatible(tmp_path: Path) -> None:
        output = tmp_path / "frame000000.pkl.gz"
        write_hybrid_frame(output, _batch_fixture(), classes=("chair", "table"))
        restored = load_frontend_batch(output)

        assert restored.labels == _batch_fixture().labels
        assert restored.masks.flags.writeable is False


    def test_atomic_writer_refuses_existing_frame(tmp_path: Path) -> None:
        output = tmp_path / "frame000000.pkl.gz"
        write_hybrid_frame(output, _batch_fixture(), classes=("chair", "table"))
        with pytest.raises(FileExistsError):
            write_hybrid_frame(output, _batch_fixture(), classes=("chair", "table"))

Test corrupt gzip, non-dictionary pickle, missing keys, object arrays, non-finite values, class bounds, shape mismatch, zero feature rows, symlinks, checksum mismatch, and temporary cleanup.

- [ ] **Step 2: Run and verify missing module**

    /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q       tests/oviv2/test_hybrid_cache.py

Expected: collection failure.

- [ ] **Step 3: Implement loading, writing, and manifests**

Implement four public functions: sha256(path), load_frontend_batch(path, expected_sha256=None), write_hybrid_frame(path, batch, classes), and publish_frontend_manifest(output_dir, scene, frame_count, source_frame_ids, classes, feature_model_id, cache_files_sha256, source_sha256, policy, diagnostics). Their return types are respectively a lowercase SHA-256 string, FrontendBatch, the published frame SHA-256 string, and the exact manifest dictionary written to disk.

Write gzip with fixed mtime zero, pickle protocol 4, sorted JSON keys, allow_nan=False, fsync, and hard-link publication. Keep method="OVIV2" so the existing runner accepts the manifest.

- [ ] **Step 4: Verify CachedFrontendAdapter compatibility**

Instantiate CachedFrontendAdapter over the written frame and assert it creates normalized FrameObservation values without modifying the adapter.

- [ ] **Step 5: Run tests and commit**

    /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q       tests/oviv2/test_hybrid_cache.py tests/oviv2/test_observations.py
    git add src/oviv2/hybrid_cache.py tests/oviv2/test_hybrid_cache.py
    git commit -m "feat: add immutable hybrid frontend cache"

### Task 5: Add The Hybrid Cache Builder CLI

**Files:**

- Create: scripts/build_oviv2_hybrid_frontend_cache.py
- Create: tests/evaluation/test_build_oviv2_hybrid_frontend_cache.py

- [ ] **Step 1: Write failing source-frame tests**

Create a two-frame fixture. Cache index one must read YOLO frame000001.pkl.gz, dense frame000001.npz, materialized depth000001.png, and canonical SAM frame000010.pkl.gz for source stride 10.

    def test_builder_maps_cache_index_to_sam_source_frame(tmp_path: Path) -> None:
        fixture = write_builder_fixture(tmp_path, frame_count=2, source_stride=10)
        manifest = run(parse_args([
            "--config", str(fixture.config),
            "--variant", "yolo_novel_sam",
            "--output", str(tmp_path / "output"),
        ]))

        assert manifest["frame_count"] == 2
        assert manifest["source_frame_ids"] == [0, 10]
        assert manifest["diagnostics"]["accepted_novel_sam"] > 0

Also test CLIP hash disagreement, vocabulary mismatch, image shape mismatch, dense source-ID mismatch, missing source hash, partial output, and an existing destination.

- [ ] **Step 2: Run and verify missing script**

    /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q       tests/evaluation/test_build_oviv2_hybrid_frontend_cache.py

Expected: import failure.

- [ ] **Step 3: Implement CLI preflight**

Require arguments --config, --variant, --output, and optional --num-frames. The runner JSON contains scene, dataset_root, dense_cache_dir, manifest, source_start, source_stride, and a hybrid_frontend object. That object contains variant, yolo_cache_dir, sam_cache_dir, sam_gate, sam_generator_script, sam_runtime_patch, clip_checkpoint, and the three named policy objects. The builder must reject a CLI variant that disagrees with hybrid_frontend.variant.

Before creating output, verify every selected file and checksum. Require the YOLO manifest CLIP hash and actual ViT-H-14 checkpoint to equal 9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4. Require the canonical generator script hash to equal 7faf64705a80f920eb269b3c842319fc9eccb1e32eda1ed1b23dbe40ce064110 and its runtime patch hash to equal 74bd99e9c0a097be802b3a30f013a8fcbbc74863bae37cb78d1e3f3cb5e9dd87. Record the SAM gate, generator script, runtime patch, CLIP checkpoint, dense manifest, YOLO manifest, and all consumed frame hashes.

- [ ] **Step 4: Implement conversion**

For each cache index:

1. Compute source_frame_id = source_start + cache_index * source_stride.
2. Load YOLO cache index, SAM source frame, dense cache index, and materialized depth index.
3. Set valid_depth from nonzero depth pixels without using metric depth values.
4. Run select_hybrid_proposals().
5. Atomically publish frameNNNNNN.pkl.gz.
6. Accumulate rejection and label-source counters.

Publish frontend_manifest.json only after all frames and hashes verify.

- [ ] **Step 5: Run tests and commit**

    /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q       tests/evaluation/test_build_oviv2_hybrid_frontend_cache.py       tests/oviv2/test_hybrid_cache.py       tests/oviv2/test_hybrid_frontend.py       tests/oviv2/test_observations.py
    git add scripts/build_oviv2_hybrid_frontend_cache.py       tests/evaluation/test_build_oviv2_hybrid_frontend_cache.py
    git commit -m "feat: build Replica hybrid frontend caches"

### Task 6: Add Pareto Metric Gates

**Files:**

- Create: scripts/evaluation/compare_oviv2_pareto.py
- Test: tests/evaluation/test_compare_oviv2_pareto.py

- [ ] **Step 1: Write failing gate tests**

Add tests requiring strict improvement of semantic/AP metrics and exact F5 equality in route2 mode, strict improvement of all six in final mode, and failure for every individual regression.

    def test_route2_gate_requires_semantic_ap_gain_and_identical_f5() -> None:
        audit = compare_metrics(
            metrics(miou=.38, macc=.43, f_miou=.66, ap25=.28, ap50=.05, f5=.916),
            metrics(miou=.39, macc=.44, f_miou=.67, ap25=.29, ap50=.06, f5=.916),
            mode="route2",
        )
        assert audit["status"] == "PASS"
        assert audit["checks"]["f5"]["relation"] == "byte_identical"


    def test_final_gate_rejects_an_exact_tie() -> None:
        assert compare_metrics(metrics(), metrics(), mode="final")["status"] == "FAIL"

Also reject NaN, Inf, missing fields, protocol mismatch, and scene mismatch.

- [ ] **Step 2: Run and verify failure**

    /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q       tests/evaluation/test_compare_oviv2_pareto.py

- [ ] **Step 3: Implement library function and CLI**

Expose compare_metrics(baseline, candidate, mode). CLI arguments are --baseline, --candidate, --mode, and --output. Read semantic.{miou,macc,f_miou}, instance.class_agnostic.{ap25,ap50}, and geometry.f5. Exit zero for PASS and one for a metric failure. Atomically write a complete audit.

- [ ] **Step 4: Run tests and commit**

    /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q       tests/evaluation/test_compare_oviv2_pareto.py
    git add scripts/evaluation/compare_oviv2_pareto.py       tests/evaluation/test_compare_oviv2_pareto.py
    git commit -m "feat: gate OVIV2 Pareto improvements"

### Task 7: Freeze Three Route 2 Configurations

**Files:**

- Create: configs/oviv2_replica_room0_route2_sam_labeled.json
- Create: configs/oviv2_replica_room0_route2_yolo_novel_sam.json
- Create: configs/oviv2_replica_room0_route2_quota_nms.json
- Modify: tests/evaluation/test_build_oviv2_hybrid_frontend_cache.py

- [ ] **Step 1: Add failing config-contract tests**

Assert the three JSON files retain every mapping, association, dense, fusion, TSDF, and evaluation value from configs/oviv2_replica_room0_precision_stage3_fused.json. Only frontend_cache_dir, algorithm_hash, and hybrid_frontend provenance may differ.

- [ ] **Step 2: Create configs with fixed output roots**

Use:

    /home/ww/oviovo_experiments/20260721_route2_frontend/sam_labeled/cache
    /home/ww/oviovo_experiments/20260721_route2_frontend/yolo_novel_sam/cache
    /home/ww/oviovo_experiments/20260721_route2_frontend/quota_nms_ensemble/cache

The builder policy values are those in HybridFrontendConfig. The quota variant enforces 64 total and 12 per class. Recompute algorithm_hash through the existing runner helper.

Use these exact shared source paths in hybrid_frontend:

    "yolo_cache_dir": "/home/ww/vv/dataset/Replica/room0_s10_200f/gsa_detections_yolo_room0_s10_200f"
    "sam_cache_dir": "/home/ww/oviovo_baseline_runs/20260719_conceptgraphs_canonical/room0/input/room0/gsa_detections_none_canonical_room0_s10_200f"
    "sam_gate": "/home/ww/oviovo_baseline_runs/20260719_conceptgraphs_canonical/room0/gate.json"
    "sam_generator_script": "/home/ww/oviovo_baseline_builds/conceptgraphs-canonical-runtime/conceptgraph/scripts/generate_gsa_results.py"
    "sam_runtime_patch": "/home/ww/oviovo_baseline_runs/20260719_conceptgraphs_canonical/room0/runtime_path_localization.patch"
    "clip_checkpoint": "/home/ww/vv/paper2/DovSG/checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"

- [ ] **Step 3: Run config and runner tests**

    /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q       tests/evaluation/test_build_oviv2_hybrid_frontend_cache.py       tests/evaluation/test_run_oviv2_replica_cli.py

- [ ] **Step 4: Commit configs**

    git add configs/oviv2_replica_room0_route2_*.json       tests/evaluation/test_build_oviv2_hybrid_frontend_cache.py
    git commit -m "config: freeze OVIV2 room0 Route 2 variants"

### Task 8: Build And Inspect The 200-Frame Caches

**Files:**

- Generated only under /home/ww/oviovo_experiments/20260721_route2_frontend/

- [ ] **Step 1: Build all variants**

Run:

    /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python       scripts/build_oviv2_hybrid_frontend_cache.py       --config configs/oviv2_replica_room0_route2_yolo_novel_sam.json       --variant yolo_novel_sam       --output /home/ww/oviovo_experiments/20260721_route2_frontend/yolo_novel_sam/cache

Repeat with matching sam_labeled and quota_nms_ensemble config/output paths.

- [ ] **Step 2: Verify distributions**

Require 200 frames, matching hashes, normalized features, no structural output labels, and quota bounds. Report min, median, p95, max proposal counts plus rejection and label-source totals.

- [ ] **Step 3: Reject pathological caches**

Reject any variant with an empty frame, median proposals below current YOLO median, more than half of accepted SAM masks unlabeled, or more than 25% adapter drops during a 20-frame lifting probe.

### Task 9: Run 20-Frame Functional Experiments

**Files:**

- Generated only under /home/ww/oviovo_experiments/20260721_route2_frontend/runs20/

- [ ] **Step 1: Run fresh baseline**

    /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python scripts/run_oviv2_replica.py       --config configs/oviv2_replica_room0_precision_stage3_fused.json       --num-frames 20       --output /home/ww/oviovo_experiments/20260721_route2_frontend/runs20/baseline

- [ ] **Step 2: Run all three variants**

Use each Route 2 config, --num-frames 20, and a unique output.

- [ ] **Step 3: Apply gates**

Require finite metrics, no empty observation stream, entity count no more than three times baseline, no exception, and F5 exactly equal to fresh baseline. Compare proposal counts, accepted entities, overmerge, fragmentation, and seconds/frame.

- [ ] **Step 4: Promote candidates**

Promote variants with non-decreasing mIoU, mAcc, f-mIoU, AP25, and AP50 and strict gain in at least one. If none passes, revise one policy variable per new experiment.

### Task 10: Run 200-Frame Room0 Pareto Experiments

**Files:**

- Generated only under /home/ww/oviovo_experiments/20260721_route2_frontend/runs200/

- [ ] **Step 1: Run promoted candidates sequentially**

    /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python scripts/run_oviv2_replica.py       --config configs/oviv2_replica_room0_route2_yolo_novel_sam.json       --output /home/ww/oviovo_experiments/20260721_route2_frontend/runs200/yolo_novel_sam

- [ ] **Step 2: Apply strict Route 2 gate**

    /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python       scripts/evaluation/compare_oviv2_pareto.py       --baseline /home/ww/oviovo_final_outputs/oviv2_replica8_stage3_s049_200f_5f272a8/room0/evaluation_fused/metrics.json       --candidate /home/ww/oviovo_experiments/20260721_route2_frontend/runs200/yolo_novel_sam/evaluation_fused/metrics.json       --mode route2       --output /home/ww/oviovo_experiments/20260721_route2_frontend/runs200/yolo_novel_sam/pareto_audit.json

Repeat for every promoted variant. Do not select a failed variant.

- [ ] **Step 3: Reproduce winner**

Rerun the winner in a fresh directory. Require identical six metrics, entity count, projected IDs, cache hashes, and config/algorithm hashes.

### Task 11: Publish Route 2 Decision

**Files:**

- Create: docs/superpowers/reports/2026-07-21-oviv2-route2-room0-results.md

- [ ] **Step 1: Write report**

Record source/cache/config/code hashes, proposal distributions, label-source counts, six headline metrics, geometry precision/recall, overmerge/fragmentation, entity count, seconds/frame, gates, and reproduction status for baseline and every variant.

State the accepted config or state that Route 2 has no Pareto winner. Preserve failed variants.

- [ ] **Step 2: Run complete verification**

    /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q       tests/oviv2/test_hybrid_frontend.py       tests/oviv2/test_hybrid_cache.py       tests/oviv2/test_observations.py       tests/evaluation/test_build_oviv2_hybrid_frontend_cache.py       tests/evaluation/test_compare_oviv2_pareto.py       tests/evaluation/test_run_oviv2_replica_cli.py
    git diff --check

Expected: tests pass and git diff --check is silent.

- [ ] **Step 3: Commit report**

    git add docs/superpowers/reports/2026-07-21-oviv2-route2-room0-results.md
    git commit -m "docs: report OVIV2 Route 2 room0 results"

After this commit, create a separate Route 1 plan using the accepted Route 2 snapshot or, if Route 2 has no Pareto winner, the unchanged Stage 3 baseline.
