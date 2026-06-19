# SED++ Frontend Replacement Design

## Goal

Replace the current online YOLOWorld + SAM2 + anchor-guided fusion frontend with an opt-in SED++ semantic proposal frontend. The goal is faster mapping while preserving most of the validated 200-frame stride-10 accuracy.

Current reference run:

- Run: `outputs/tmp_validation/20260612_wwai_online_yoloworld_sam_baseline_s10_200f`
- Config: `configs/replica_yoloworld_sam_online_baseline_4090.yaml`
- Frames: `200`
- Stride: `10`
- mIoU: `0.5189769907667642`
- f-mIoU: `0.605178131809614`
- Wall time: `40:37.90`
- `proposal_generation` mean: `10.4371s`
- `anchor_guided_sam_fusion` mean: `9.0352s`

The replacement should primarily remove the `anchor_guided_sam_fusion` bottleneck. YOLOWorld itself is not the bottleneck.

## Existing Frontend

The current online path creates a `FrameProposalBundle` through:

1. YOLOWorld detects class-aware anchors.
2. SAM2 creates masks using anchors.
3. `AnchorGuidedSAMModule` matches YOLO anchors to SAM masks, builds labeled proposals, fallbacks, and unknown residuals.
4. The pipeline consumes `Proposal2D` masks plus anchor assignments and then runs depth refinement, lifting, association, and object update.

This produces good semantic grounding, but the fusion step is expensive because it evaluates geometric relations between many anchors and many SAM proposals on every frame.

## Proposed Frontend

Add a `sedpp` proposal backend that directly emits class-aware `Proposal2D` instances from SED++ semantic segmentation.

Target data flow:

```text
RGB frame
  -> SED++ semantic logits/probabilities
  -> class map + confidence map
  -> connected components per class
  -> optional depth/discontinuity split
  -> Proposal2D masks with semantic metadata
  -> existing depth_refinement / patch_lifting / association / object_update
```

The backend should be selected by config and should not change existing YOLOWorld+SAM2 behavior.

Local note: the available repository is `/home/ww/vv/paper2/SED`. The OVIOVO backend name should still be `sedpp` so the experiment can point either to this local SED implementation or to a later SED++ checkout/weights without changing downstream mapping code.

## Interface Contract

SED++ proposals should use the existing `Proposal2D` contract:

- `mask`: boolean component mask.
- `bbox_xyxy`: component bounding box.
- `area`: component area.
- `confidence`: component semantic confidence.
- `backend_name`: `sedpp`.
- `metadata.anchor_class_name`: canonical Replica/OVIOVO class label.
- `metadata.anchor_confidence`: semantic confidence.
- `metadata.anchor_label_votes`: `{class_name: confidence}`.
- `metadata.anchor_label_strength`: `strong` for high-confidence direct semantic evidence, `contextual` or blocked for weak evidence.
- `metadata.semantic_commit_allowed`: true only when the SED++ evidence is strong enough.
- `metadata.mask_source`: `sedpp_semantic_component`.

The pipeline can run with:

- `proposal.backend: sedpp`
- `anchor_frontend.enabled: false`
- `anchor_guided_sam.enabled: false`
- `pipeline.yoloworld_sam_parallel_frontend_enabled: false`

No `AnchorAssignment` is required for direct SED++ proposals because `patch_lifting` copies semantic metadata from `Proposal2D` into patches, and `semantic_memory` already consumes `patch.metadata.anchor_class_name`.

## Class Vocabulary

SED++ should use a Replica-aligned class list, not generic COCO-only labels.

Sources:

- Replica class names: `Replica_original/<scene>/habitat/info_semantic.json`
- OVIOVO config classes: `anchor_frontend.classes`
- Existing SED class JSON files under `/home/ww/vv/paper2/SED/datasets`

Implementation should provide a generated class JSON for SED++ such as:

- `data/input/sedpp_replica_classes.json`

The output label must be canonicalized back to Replica/OVIOVO names before writing `anchor_class_name`. Synonym handling should be explicit, for example:

- `couch` -> `sofa`
- `potted plant` -> `indoor-plant` or `plant`
- `windowpane`, `window-blind` -> `window` / `blinds`
- `floor-wood`, `floor-tile`, `floor-other` -> `floor`
- `wall-*` -> `wall`
- `dining table`, `coffee table` -> `table`

Unknown or unmapped labels should either be dropped or emitted as unlabeled residuals with `semantic_commit_allowed: false`.

## Semantic Proposal Postprocessing

SED++ produces semantic segmentation, not instance segmentation. The backend must convert semantic regions to object-like proposals.

Minimum viable postprocessing:

1. Run SED++ and obtain `sem_seg` logits/probabilities with shape `[C, H, W]`.
2. Compute top class and top probability per pixel.
3. Drop pixels below `min_pixel_confidence`.
4. For each allowed class, run connected components.
5. Drop components below `min_component_area`.
6. Split large components by depth discontinuity when enabled.
7. Cap per-frame proposals with class-aware ranking.
8. Write semantic metadata onto each `Proposal2D`.

Precision guards:

- Do not commit low-confidence labels.
- Do not allow generic structure aliases to overwrite object labels.
- Use a stricter confidence threshold for large structural classes than for compact objects.
- Keep `semantic_commit_allowed: false` for ambiguous components, but still allow geometry to enter mapping if useful.

## Performance Strategy

The first implementation can be in-process if the SED++ dependencies import cleanly in `ww-ai`. If dependency conflicts occur, use a persistent JSON-lines worker similar to existing model-worker patterns.

Required timing keys:

- `proposal_generation`: total SED++ frontend time from `Pipeline`.
- `sedpp_inference`: model forward time.
- `sedpp_postprocess`: component extraction and metadata stamping.
- `sedpp_component_count`: count-like debug value recorded in frame metrics or backend debug payload.

`sedpp_inference` and `sedpp_postprocess` must be propagated from the proposal backend into pipeline frame metrics. The current pipeline only times the outer `proposal_generation` block, so the implementation should add a small `ProposalModule.last_generation_timings` or equivalent hook and merge those timings in `_build_proposal_bundle()`.

The expected speed win comes from removing:

- YOLOWorld/SAM dependency chain.
- SAM prompt generation per frame.
- `anchor_guided_sam_fusion` pairwise mask-anchor matching.

## Acceptance Contract

Primary 200f stride-10 contract against the current online baseline:

- `proposal_generation` mean should be at least `2x` faster than `10.4371s`; target `<= 4.0s`, stretch target `<= 2.5s`.
- Wall time should be lower than `40:37.90`; target `<= 30min`, stretch target `<= 22min`.
- mIoU should stay within an acceptable first-pass drop: target `>= 0.49`, minimum viable `>= 0.47`.
- f-mIoU should stay near current stability: target `>= 0.58`, minimum viable `>= 0.56`.
- Final object count should not explode by more than `2x` the current `554` without a compensating mIoU gain.

Smoke contract:

- 1-frame backend smoke returns non-placeholder `sedpp` proposals.
- 20-frame run completes with no fallback to placeholder.
- Frame metrics show non-empty semantic labels on a meaningful fraction of proposals.

## Fallbacks

If direct SED++ proposals are fast but accuracy drops too much, use a hybrid ladder:

1. SED++ direct semantic proposals only.
2. SED++ direct + depth split for large components.
3. SED++ direct + SAM2 rescue only for low-confidence compact object regions.
4. SED++ direct + sparse YOLOWorld semantic audit every `N` frames.

Avoid returning to full YOLOWorld+SAM2+fusion unless it is needed as the baseline comparison.

## Non-Goals

- Do not remove the existing YOLOWorld+SAM2 baseline.
- Do not rely on proposal caches for the SED++ online experiment.
- Do not change downstream map mutation semantics in the first iteration.
- Do not commit output data, checkpoints, or weights to GitHub.
