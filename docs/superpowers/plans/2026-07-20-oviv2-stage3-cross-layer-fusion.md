# OVIV2 Stage 3 Cross-Layer Fusion Implementation Plan

**Goal:** Add a causal uncertainty-aware semantic head that combines Stage 2 voxel evidence with the current entity posterior, freeze it on room0, then use the unchanged head for Replica-8 and Replica-7 reporting.

**Architecture:** Keep the schema-3 snapshot and mapping state unchanged. At mesh derivation, normalize the voxel top-k support and the owning entity posterior, compute `w_entity = 0.5 * ownership_confidence * entity_top_probability * (1 - dense_top_probability)`, and linearly mix the two distributions. Unowned voxels remain dense-only. Geometry and entity IDs are never changed by semantic fusion.

**Frozen development evidence:** On the accepted room0 200-frame Stage 2 snapshot, the rule changes 275 of 65,350 mesh vertices and estimates mIoU `0.3827572414247502` versus dense-only `0.3799786419574329`. The scale `0.5` is frozen before any other Replica scene is evaluated.

---

## Task 1: Implement The Pure Fusion Policy

**Files:**
- Create: `src/oviv2/semantic_fusion.py`
- Create: `tests/oviv2/test_semantic_fusion.py`

1. Add failing tests for dense-only, entity-only, agreement, disagreement, deterministic ties, unknown output, and strict validation.
2. Add immutable `SemanticFusionConfig(entity_weight_scale=0.5)` and `FusedSemanticLabel` values.
3. Implement normalized linear fusion using only runtime evidence, posterior probability, and ownership confidence.
4. Run `tests/oviv2/test_semantic_fusion.py` and commit.

## Task 2: Add Fused Mesh Derivation

**Files:**
- Modify: `src/oviv2/meshing.py`
- Modify: `tests/oviv2/test_meshing.py`

1. Add failing fixtures showing that fused semantics can select either voxel or entity evidence while preserving entity IDs and ownership confidence.
2. Add a strictly validated entity-posterior mapping and an opt-in fusion config to `derive_labeled_mesh`.
3. Reject simultaneous owner-authoritative and fused overrides.
4. Preserve byte-identical behavior for existing calls and run meshing tests.

## Task 3: Expose A Third Evaluator Head

**Files:**
- Modify: `scripts/evaluation/evaluate_oviv2_replica.py`
- Modify: `tests/evaluation/test_evaluate_oviv2_replica_cli.py`

1. Add `fused_uncertainty` to the accepted semantic heads and add a bounded `--fusion-entity-weight-scale` option.
2. Require an embedded schema-2/3 registry for fused evaluation and use its full entity posteriors.
3. Keep class-agnostic instance AP and geometry evaluation unchanged.
4. Record the exact fusion config in `metrics.json:protocol` and run evaluator tests.

## Task 4: Wire Fusion Into Dense Runner Outputs

**Files:**
- Modify: `scripts/run_oviv2_replica.py`
- Modify: `tests/evaluation/test_run_oviv2_replica_cli.py`
- Create: `configs/oviv2_replica_room0_precision_stage3_fused.json`

1. Add failing tests requiring `final/oviv2_fused_mesh.ply`, `evaluation_fused`, artifact hashes, and manifest fusion provenance when fusion is enabled.
2. Parse only `fusion_semantic_mode="uncertainty_linear"` and `fusion_entity_weight_scale=0.5` for the frozen config.
3. Preserve Stage 1 and Stage 2 output contracts when fusion is disabled.
4. Run focused runner tests and commit.

## Task 5: Gate Existing 20-Frame And 200-Frame Snapshots

**Files:**
- Generated outputs only.

1. Evaluate the fused head on the accepted B+SAM 20-frame snapshot. Require finite metrics and no AP50/F@5cm change.
2. Evaluate the fused head on `outputs/oviv2_room0_precision_stage2_selected_200f/final/oviv2_voxel_snapshot.npz` without rebuilding mapping state.
3. Require fused mIoU `> 0.3799786419574329`, AP50 `>= 0.04977236594883654`, F@5cm `>= 0.90`, and identical projected entity IDs to Stage 2.
4. Freeze the implementation commit and configuration hash. Do not tune on any other scene.

## Task 6: Freeze Replica-8 Execution And Fill The Benchmark

**Files:**
- Create per-scene frozen configs/caches and generated result directories.
- Update the canonical OVIV2 benchmark result JSON through the existing importer.
- Regenerate paper tables through the existing table generator.

1. Preflight all eight scene datasets, ground truth files, frontend manifests, and frame selections.
2. Precompute B+SAM dense caches with the exact room0 model, vocabulary, prompt, stride, top-k, AMP, and SAM hashes. Use separate GPUs only for independent scenes.
3. Run the unchanged Stage 3 config on all eight scenes; aggregate Replica-8 and the predeclared Replica-7 held-out result.
4. Require Replica-8 mIoU `> 0.271`, F@5cm `>= 0.88`, and finite per-scene metrics. Do not substitute room0 development metrics.
5. Import the immutable aggregate, regenerate benchmark tables, verify every displayed token against the result JSON, and run the complete regression suite.
