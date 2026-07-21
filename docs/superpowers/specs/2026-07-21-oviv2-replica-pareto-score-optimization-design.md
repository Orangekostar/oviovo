# OVIV2 Replica Pareto Score Optimization Design

**Date:** 2026-07-21  
**Development scene:** Replica `room0`, 200 frames, source stride 10  
**Execution order:** high-recall frontend, layered Pareto optimization, max-score ensemble

## Goal

Improve all six OVIV2 Replica headline metrics without trading one metric for another. The paper-facing configuration may use `room0` for development and threshold selection, but it must not use ground-truth labels, geometry, or evaluation matches during mapping or inference. Once frozen, the same configuration is applied unchanged to the other seven Replica scenes.

The room0 acceptance baseline is:

| Metric | Baseline |
| --- | ---: |
| mIoU | 0.38283415298675816 |
| mAcc | 0.43373079839146034 |
| f-mIoU | 0.6646659615492884 |
| AP25 | 0.2803600106840453 |
| AP50 | 0.04977236594883654 |
| F@5cm | 0.9160999937093854 |

The Replica-8 promotion baseline is:

| Metric | Baseline |
| --- | ---: |
| mIoU | 0.33730965872197777 |
| mAcc | 0.4119967648498053 |
| f-mIoU | 0.5760990806988411 |
| AP25 | 0.24999352823022047 |
| AP50 | 0.05459057647227853 |
| F@5cm | 0.8823510800998262 |

## Constraints

- A final candidate passes only when all six metrics are strictly greater than the applicable baseline.
- During staged development, a component may improve its target metrics while leaving structurally independent metrics byte-identical, but no metric may decrease. The composed `2+1` result must strictly improve all six metrics before Route 3 starts.
- Ground truth is evaluation-only. It may diagnose room0 and select a frozen configuration, but it may not create proposals, labels, scores, splits, or geometry.
- The paper-facing track uses one frozen configuration for all held-out Replica scenes. A separate max-score track may use scene-specific post-processing but is labeled non-headline.
- Every run uses immutable inputs, output directories, checksums, exact commands, and metric JSON files.

## Evidence Behind The Design

Existing room0 diagnostics show:

- 81 projected entities, with 31 significantly overmerged entities and 18 fragmented ground-truth instances.
- Reducing the prediction-size threshold from 100 to 20 does not improve recall.
- Protocol-aligned independent object projection raises AP25/AP50 to about `0.389/0.101` without rebuilding the map.
- An oracle-free raw-entity plus connected-component proposal ensemble reaches about `0.428/0.179` on room0 with view/margin scoring.
- OVIV2 geometry recall is already strong; its F@5cm deficit comes from precision and spurious TSDF surfaces.
- The canonical SAM ViT-H cache contains 8,801 room0 masks, while the current frontend yields 4,580 accepted object observations. Candidate coverage is the first runtime bottleneck.

## Architecture

The existing sparse TSDF, dense RADSeg evidence, entity registry, reversible ownership, snapshot, and evaluator contracts remain authoritative. New behavior is introduced through bounded adapters and post-processing heads:

```text
SAM ViT-H masks ----+
                    +--> hybrid proposal adapter --> existing tracker/entity backend
YOLO masks/labels --+              |
                                   +--> RADSeg fallback labels

snapshot --> protocol-aligned instance head --> raw/component hypotheses --> calibrated AP scores
         --> existing semantic head ----------> bounded fusion calibration
         --> geometry head -------------------> depth-supported surface cleanup
```

Each branch can be disabled independently, so every score change can be attributed to one experiment.

## Route 2: High-Recall Frontend First

### Inputs

- Reuse the canonical room0 SAM ViT-H segment-all cache generated from the same RGB frames.
- Reuse the frozen YOLO-World plus MobileSAM cache for semantic anchors.
- Reuse the frozen RADSeg top-k probability cache for masks without a reliable YOLO match.
- Do not rerun GPU inference until cache-based experiments establish that the hybrid adapter is beneficial.

### Hybrid Proposal Adapter

For each frame:

1. Load YOLO and SAM masks in the original 680 by 1200 image domain.
2. Keep the original YOLO proposals as semantic anchors.
3. Match each SAM mask to YOLO masks using mask IoU and directed coverage.
4. If a reliable match exists, inherit the YOLO semantic label and calibrated confidence.
5. Otherwise, aggregate RADSeg probabilities inside the valid-depth portion of the SAM mask and assign the highest supported non-structural class.
6. Reject masks dominated by wall, floor, or ceiling evidence; masks with inadequate valid depth; extreme image-area masks; and duplicate masks.
7. Apply deterministic per-frame proposal and per-class caps before the existing observation adapter.
8. Preserve the SAM image feature when its feature provenance matches the current association model; otherwise omit the feature instead of mixing feature spaces.

### Route 2 Variants

Run one variable at a time:

1. `sam_labeled`: SAM-only proposals with YOLO-overlap labels and RADSeg fallback.
2. `yolo_novel_sam`: original YOLO proposals plus only novel SAM masks. This is the recommended main candidate.
3. `quota_nms_ensemble`: first retain YOLO anchors in confidence/index order under the per-class and global caps, without YOLO-vs-YOLO suppression; initialize mask-NMS occupancy only from those final retained anchors, and then consider regular inherited SAM, compact-rescue SAM, and regular RADSeg-fallback SAM in that order. Within each tier, use confidence descending, area descending, and source index ascending; check the current class/global/compact quota before NMS, and reject a SAM mask at IoU >= `duplicate_iou` against any retained YOLO or higher-priority retained SAM. Only a SAM surviving both quota and NMS occupies the NMS set and updates quota counts. A compact rescue is SAM-only, has area fraction in [`0.00002`, `minimum_area_fraction`), still passes depth and structure filtering, receives normal reliable-YOLO inheritance or RADSeg fallback, and has final confidence >= `0.60`; the strict relationship between compact and normal area minima is quota-only. At most four compact rescues survive, and they remain subject to the same per-class and global quotas. Report final `accepted_compact_rescue` and deterministic `rejected_compact_rescue_cap` counts without changing inherited/fallback source counters.

Each variant starts with a cache-contract test, a 20-frame functional run, and then a fresh 200-frame room0 run. Route 2 passes only if all semantic and AP metrics strictly improve and F@5cm is byte-identical. A result with unchanged semantic metrics is retained only as diagnostic evidence, not as the Route 2 winner.

## Route 1: Layered Pareto Optimization

Route 1 starts from the best non-regressing Route 2 snapshot.

### Instance Evaluation Parity

- Restrict class-agnostic instance evaluation to the same non-structural ground-truth domain used by static baselines.
- Project every predicted instance independently within 5 cm instead of forcing all entities and background into one mutually exclusive nearest-neighbor label map.
- Apply the same minimum-size policy to all methods.
- Keep the old evaluator as an audit output and publish a before/after protocol comparison.

### Multi-Hypothesis Instance Head

- Retain each raw persistent entity as one proposal.
- Extract same-entity mesh connected components without ground truth.
- Add components above a frozen support threshold as child proposals rather than destructively replacing the parent.
- Score raw proposals from accepted view count and semantic margin.
- Score child proposals from the parent score and the squared child-to-parent support ratio.
- Tune only on room0, then freeze thresholds and coefficients.

The initial room0 diagnostic candidate uses a minimum child support of 200 projected vertices and a child multiplier of 2.0. These values are hypotheses, not accepted defaults, until reproduced by the implementation evaluator.

### Semantic Calibration

- Sweep only bounded entity/dense fusion parameters already available at runtime.
- Require mIoU, mAcc, and f-mIoU to all increase; no single semantic aggregate may be optimized alone.
- Keep instance IDs and geometry unchanged during semantic-only tests.

### Geometry Precision

- Measure TSDF vertex-to-input-depth residual, integration weight, component size, and multi-view depth support without ground truth at inference.
- Remove only low-support floating components and depth-discontinuity artifacts.
- Test a smaller voxel size only after post-processing is exhausted because it requires a full map rebuild.
- Require both F@5cm and geometry precision to increase, while recall and all semantic/instance metrics do not decrease.

The composed Route `2+1` candidate must strictly improve all six room0 headline metrics before promotion.

## Route 3: Max-Score Ensemble

Route 3 starts only after the paper-facing `2+1` configuration passes.

- Combine raw entities, connected components, and a small frozen set of association-threshold hypotheses.
- Calibrate hypothesis scores from runtime evidence only.
- Deduplicate near-identical hypotheses before evaluation.
- Keep a paper-facing global configuration and a separately labeled scene-specific max-score configuration.
- Never replace or relabel the paper-facing result with the scene-specific result.

## Experiment Gates

For every candidate:

1. Validate cache manifests, shapes, class vocabularies, frame counts, model hashes, and feature dimensions.
2. Run focused unit tests for mask matching, RADSeg aggregation, deterministic filtering, component extraction, scoring, and evaluator parity.
3. Run a 20-frame smoke test to catch empty proposals, entity explosion, invalid scores, or geometry changes.
4. Run all 200 room0 frames into a new output directory.
5. Compare the six headline metrics and geometry precision/recall against the immutable baseline.
6. Reject any non-finite metric, provenance mismatch, GT-dependent runtime input, or metric regression.

After room0 passes, freeze code/config hashes and run all eight Replica scenes. The final Replica-8 result passes only if all six macro metrics strictly exceed the frozen Replica-8 baseline. Per-scene regressions are reported even when macro metrics pass.

## Failure Handling

- Missing or corrupt optional SAM data fails the hybrid experiment; it does not silently fall back to a different method.
- Unmatched SAM masks without sufficient RADSeg support are dropped rather than assigned an unknown or arbitrary label.
- Feature-model mismatches disable the incompatible feature on that observation and are counted in the run manifest.
- Proposal caps, rejection reasons, label sources, and duplicate suppression counts are recorded per frame.
- A failed Pareto gate leaves the previous accepted snapshot and benchmark result untouched.

## Deliverables

- A deterministic hybrid proposal cache and manifest for room0.
- Per-variant room0 metric JSON, timing JSON, proposal diagnostics, and comparison table.
- A protocol-parity evaluator audit preserving old and new outputs.
- A frozen `2+1` room0 configuration that strictly improves all six metrics.
- Replica-8 paper-facing and separately labeled max-score result packages.
