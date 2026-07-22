# OVIV2 View-Consensus Pareto Instance Design

**Date:** 2026-07-22

**Development scene:** Replica `room0`, 200 RGB-D frames

**Promotion split:** Replica-7 held out from parameter selection

**Parent result:** `oviv2-replica8-20260722-route3-surface-observation-v5`

## Scope

This sub-project targets OVIV2's remaining instance weakness, especially AP50, without changing the accepted v5 semantic map or observation-surface geometry. It adds GT-free multi-view instance candidates from the existing 2D frontend cache, then appends only independently screened candidates after the complete v5 prediction sequence.

Semantic and geometry improvements remain separate follow-up stages. The final paper-facing system still requires strict improvement of mIoU, mAcc, f-mIoU, AP25, AP50, and F@5cm over v5. This stage may leave semantic and geometry metrics byte-identical, but it may not reduce any metric.

## Immutable Baseline

| Split | mIoU | mAcc | f-mIoU | AP25 | AP50 | F@5cm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Replica-8 | 0.3487823338 | 0.4270198790 | 0.5913774679 | 0.4032140831 | 0.1404066457 | 0.9208486392 |
| Replica-7 | 0.3417121785 | 0.4232593940 | 0.5784774831 | 0.3954220759 | 0.1366688723 | 0.9175759487 |
| room0 | 0.3982734209 | 0.4533432746 | 0.6816773612 | 0.4577581331 | 0.1665710595 | 0.9437574731 |

The room0 high-recall diagnostic reached `0.503973/0.268823` AP25/AP50, but applying its global settings directly regressed AP25 on several held-out scenes. It is therefore a proposal source, not a replacement configuration.

## Design Choice

Use a multi-view-consensus proposal pyramid with a Pareto-safe suffix. The design combines ideas, not copied code, from:

- MaskClustering: supporter/observer view consensus and rejection of undersegmented observations;
- SAI3D: hierarchical region merging from cross-view mask agreement;
- Open3DIS: union of 2D-guided and existing 3D proposals;
- OnlineAnySeg: sparse overlap candidate lookup rather than dense all-pairs matching;
- Details Matter: visibility, inclusion, and boundary-quality filtering;
- ESAM: geometry, semantic, and visual-feature evidence for conservative association.

This is preferred over importing a learned external proposal stack because all required frame masks, features, depth, poses, and persistent 3D entities already exist locally. It avoids incompatible CUDA dependencies and preserves the current evaluator and map provenance.

## Architecture

New code is isolated from the source files hashed by the v5 result:

```text
hybrid frontend cache + RGB-D poses
              |
              v
    frame-mask voxel lifting
              |
              v
 sparse overlap graph + view-consensus statistics
              |
              v
 hierarchical proposal pyramid
              |
              v
 mesh projection + GT-free scoring + visibility pruning
              |
              v
 v5 primary prefix + screened low-ranked candidate suffix
              |
              v
 ordered-prediction AP compatibility wrapper + Pareto audit
```

The implementation consists of:

1. A focused view-consensus module that consumes `FrameObservation` values and emits immutable proposal records.
2. A CLI that loads the existing 200-frame frontend cache and scene geometry, builds candidates, and invokes a compatibility wrapper around the shared formal AP kernel without a second global deduplication pass.
3. A composition helper that preserves the accepted v5 primary sequence exactly and deduplicates only the new suffix stratum.
4. Audit JSON containing input hashes, configuration, proposal counts, prefix hashes, rejection reasons, runtime, and per-scene metric comparisons.

No accepted v5 artifact is overwritten.

## Observation Graph

Each accepted object mask is lifted with the existing depth, pose, and `lift_mask_to_voxels` contract. A node records semantic ID, confidence, image feature provenance, voxel support, visible pixels, border contact, frame ID, centroid, and viewing direction.

An inverted voxel index generates cross-frame node pairs that share support. Same-frame pairs never form positive temporal edges. Pair evidence contains:

- voxel IoU and both directed coverage values;
- semantic agreement or bounded semantic compatibility;
- image-feature cosine similarity when feature-model IDs match;
- centroid and extent compatibility;
- viewpoint separation and border-contact penalties.

Edges are admitted by a small frozen threshold grid. Missing compatible features remove only the feature term; they do not silently substitute another embedding space. Deterministic ordering is by frame ID, observation ID, then proposal ID.

## Proposal Pyramid

The graph produces several independent, auditable candidate strata:

1. **Consensus components:** reciprocal multi-view components supported by a minimum number of distinct frames.
2. **Conservative cores:** voxels repeatedly supported across views, reducing boundary leakage and undersegmentation.
3. **Inclusive unions:** core plus compatible partial observations, intended to recover AP50 coverage.
4. **Hierarchical merges:** adjacent components merged at progressively weaker, frozen consensus thresholds.
5. **2D+3D unions:** graph candidates combined with existing parent/component proposals when semantic and geometric evidence agree.

Visibility filtering rejects proposals dominated by border-truncated observations, single-view artifacts, implausible extent jumps, or low supporter-to-observer ratios. Minimum support and proposal-count caps are deterministic and configured per run, never inferred from ground truth.

For each surviving proposal, mesh vertices are selected through voxel membership and bounded local dilation. The score is a deterministic combination of distinct-view support, observation confidence, consensus density, semantic agreement, feature agreement, core coverage, and boundary penalty. All score terms and normalization constants are emitted in the audit.

## Pareto-Safe Composition

The accepted v5 prediction list is first reproduced using its original `0.7` deduplication and score ordering. This complete deduplicated list is the immutable primary prefix.

New candidates are processed afterward:

1. Screen each candidate against every primary prediction at a separate suffix threshold whose room0 grid includes `0.9`.
2. Deduplicate surviving candidates only against earlier suffix candidates.
3. Scale suffix scores strictly below the minimum positive primary score while preserving suffix order.
4. Concatenate `primary + suffix`; never rerun global deduplication over the concatenated list.

The evaluator's interpolated AP cannot decrease when predictions are appended strictly after an unchanged prefix: all original precision/recall points remain available to the precision envelope, while suffix predictions can only add later points. This property is enforced both structurally and empirically:

- hash and compare every primary hypothesis ID, mask, and score before evaluation;
- assert that the combined sequence begins with the exact primary sequence;
- run exhaustive synthetic TP/FP suffix tests for AP25 and AP50;
- reject a result if any per-scene AP differs downward beyond `1e-12`.

Semantic predictions and geometry are referenced from the immutable v5 artifacts, so their hashes and metrics must remain byte-identical in this stage.

## room0 Experiment Matrix

Only `room0` may select thresholds. Run one added mechanism at a time:

| Variant | Added mechanism | Purpose |
| --- | --- | --- |
| V0 | exact v5 replay | verify primary identity |
| V1 | consensus components | test independent new-instance recall |
| V2 | V1 + conservative cores/unions | improve mask IoU and AP50 |
| V3 | V2 + visibility/inclusion pruning | suppress partial or leaked masks |
| V4 | V3 + hierarchical merging | recover split instances |
| V5 | V4 + 2D+3D unions | combine complementary proposal families |

The bounded grid covers graph overlap, supporter ratio, minimum distinct views, core vote fraction, merge threshold, mesh dilation, suffix deduplication, and score coefficients. Selection is lexicographic: maximize AP50 subject to AP25 nondecrease, then maximize AP25, then minimize suffix count and runtime. Ground truth is used only by the evaluator for this selection.

The room0 stage passes when:

- AP25 is at least `0.4577581330960257` and AP50 is strictly above `0.16657105953750673`;
- the primary prefix identity audit passes;
- mIoU, mAcc, f-mIoU, and F@5cm are byte-identical to v5;
- proposal generation has no GT path or evaluator-output dependency.

## Freeze And Promotion

After room0 selection, freeze configuration and source hashes before running `room1`, `room2`, and `office0` through `office4`. No per-scene retuning or post-selection is allowed.

Promotion requires:

- every scene's AP25 and AP50 to be greater than or equal to its v5 value within `1e-12`;
- Replica-7 and Replica-8 AP25/AP50 to be nondecreasing, with AP50 strictly increasing;
- all semantic and geometry artifact hashes and metrics to match v5;
- all eight scene audits, the aggregate audit, and the provenance contract to pass.

A candidate failing any gate remains diagnostic-only. The v5 paper table is not changed until the full promotion passes.

## Error Handling

- Missing, corrupt, shape-incompatible, or provenance-mismatched cache data fails the scene; there is no silent fallback.
- A mask with invalid depth, pose, feature, or empty voxel support is rejected with a counted reason.
- Non-finite edge evidence, proposal scores, or metrics fail the run.
- Empty suffixes are valid diagnostics but cannot satisfy the strict AP50 promotion gate.
- Prefix mismatch, reordered primary predictions, or global post-concatenation deduplication is a hard failure.
- Resource caps fail explicitly when exceeded; candidates are not silently truncated unless the deterministic cap is part of the frozen configuration.

## Verification

Focused tests cover voxel-index pair generation, same-frame exclusion, edge evidence, graph determinism, consensus/core/union construction, visibility pruning, mesh projection, suffix-only deduplication, strict score separation, primary-prefix identity, exhaustive AP nondecrease, malformed cache handling, and CLI provenance.

Execution proceeds through:

1. unit and CLI tests;
2. a 20-frame room0 smoke run for functional and resource validation;
3. the full 200-frame room0 matrix;
4. freeze and hash capture;
5. fresh 200-frame runs on the seven held-out scenes;
6. per-scene and aggregate Pareto audits;
7. only after promotion, benchmark table regeneration.

## Deliverables

- New view-consensus proposal module, composition helper, CLI, and focused tests.
- Reproducible room0 ablation directory with configs, logs, metrics, and proposal diagnostics.
- Frozen Replica-7/Replica-8 evaluation package with source and artifact hashes.
- Per-scene Pareto report proving no metric regression.
- Updated benchmark table only for a fully accepted result.
