# Structure Dense Overlay Fast Design

## Goal

Build a fast OVIOVO iteration that recovers structure-class IoU without returning to the slow SAM-primary object path. The first target is room0 stride=10 200f: `window` IoU is greater than `0.0`, `wall` and `blinds` improve over the current fast/async runs, and core object classes such as `chair`, `table`, `sofa`, and `rug` do not materially regress.

## Evidence

The latest fast/async runs are fast enough to iterate but lose structure classes:

- `20260529_room0_coarse_to_fine_async_delayguard_stride10_200f`: `mIoU=0.2740`, `wall=0.347`, `blinds=0.384`, `window=0.000`.
- `20260529_room0_fast_hybrid_worker_depth_stride10_200f`: `mIoU=0.2790`, `wall=0.345`, `blinds=0.384`, `window=0.000`.

The earlier SAM-primary/weak-structure runs prove the map can represent these classes:

- `20260526_room0_observation_first_2b830e1_200f`: `mIoU=0.5226`, `wall=0.742`, `blinds=0.645`, `window=0.216`.
- `20260526_room0_semantic_vote_weak_structure_stride10_200f`: `mIoU=0.5457`, `wall=0.574`, `blinds=0.677`, `window=0.249`.

The main difference is that the fast path has `source_sam_proposal_count_total=0` and uses detector boxes as primary proposals. That protects runtime, but it makes thin, adjacent, mostly planar classes (`wall`, `window`, `blinds`, `ceiling`, `floor`, `door`) collapse into mixed object instances. In the latest audit, `window` GT vertices are mostly absorbed by door, ceiling, blinds, and wall majority objects, so no predicted object becomes `window` by evaluation majority.

## Chosen Approach

Use a conservative dense structural overlay.

The object map remains the primary representation for object-like entities. A separate structure overlay is built from existing per-frame anchors and precomputed SAM proposals, limited to structure classes:

- `wall`
- `window`
- `blinds`
- `ceiling`
- `floor`
- `door`

The overlay never runs an additional SAM model and does not feed into association or object update. It is a lightweight export/evaluation layer over the dense surface. This preserves the fast object path while restoring SAM-mask boundary information for structure classes.

## Alternatives Considered

### Full SAM-primary restoration

This matches the high-IoU 0526 behavior, but it reintroduces many proposals into runtime grouping, depth refinement, association, and object update. It is the strongest accuracy baseline and the wrong speed target for this iteration.

### Structure association rules

Adding more rules to prevent `ceiling`/`wall`/`blinds`/`window` from merging inside the object map is closer to the current architecture, but it keeps forcing stuff-like regions into an instance-centric map. The risk is rule growth and slower association.

### Aggressive overlay

An aggressive overlay would overwrite object labels whenever structure evidence is high. That may improve structure-class IoU faster, but it can eat `chair`, `sofa`, `table`, or `rug` regions. This iteration uses conservative coverage to protect object classes.

## Architecture

### Structural Evidence Collection

Add a focused module responsible for creating structural observations from a frame:

- Input: `Frame`, structure anchors from `ObjectAnchorModule.last_anchors`, and SAM-like proposals already available in the pipeline.
- Output: lightweight structural samples or voxel votes keyed by world-space voxel.
- It only considers anchors whose class is in the configured structure class set.
- It only considers SAM proposals that overlap a structure anchor enough to support a local structure boundary.
- It clips SAM masks to the anchor box so structure evidence cannot leak across the detector anchor.
- It back-projects mask pixels using existing depth and intrinsics, using the same point sampling style as the patch-lifting path where practical.

The module stores label votes and confidence per voxel rather than creating ObjectMap entries. A voxel-level layer is enough because the export target is a dense semantic surface.

### Conservative Fusion

At dense export time, project both:

- the existing pool/object instance map, and
- the structural overlay.

Object projection remains first. The structure overlay may replace an object projection only when the projected object is not a protected object label.

Protected object labels default to all non-structure classes. In practice, `chair`, `table`, `sofa`, `rug`, `lamp`, `cabinet`, `cushion`, `blanket`, `vase`, `pot`, `stool`, and other object-like classes are not overwritten by the overlay.

The overlay can overwrite:

- unlabeled dense points,
- dense points assigned to objects with no export label,
- dense points assigned to structure labels,
- dense points assigned to safe provisional/unlabeled state,
- dense points assigned to an object whose exported label is blank but whose source anchor or majority evidence is structure-like.

The overlay must not overwrite committed non-structure objects.

### Evaluation Compatibility

The current evaluator receives dense points and predicted object ids, then maps each object id to a GT-majority class. If structure overlay labels are represented as synthetic object ids, a mixed synthetic object could still fail by majority. The export path must therefore preserve structure labels cleanly enough for evaluation.

The preferred representation is to create stable synthetic ids per structure class for the dense projection, for example negative ids reserved for `wall`, `window`, `blinds`, `ceiling`, `floor`, and `door`, and teach evaluation/reporting to map those ids directly to their semantic class. This keeps the object-map evaluation behavior unchanged for real object ids while making structure overlay semantics explicit.

The exported PLY remains inspectable:

- real object ids remain positive,
- structure overlay ids are synthetic negative ids,
- colors use the existing semantic color table where available.

### Configuration

Add an explicit config block:

```yaml
structural_overlay:
  enabled: true
  classes: [wall, window, blinds, ceiling, floor, door]
  voxel_size: 0.05
  min_overlap_area: 25
  min_proposal_anchor_coverage: 0.20
  min_anchor_proposal_coverage: 0.03
  clip_to_anchor_box: true
  conservative_fusion: true
  protected_labels:
  - basket
  - blanket
  - book
  - cabinet
  - candle
  - chair
  - cushion
  - indoor-plant
  - lamp
  - picture
  - pillar
  - plant-stand
  - plate
  - pot
  - sofa
  - stool
  - switch
  - table
  - vase
  - vent
  - wall-plug
  - rug
```

The feature is disabled by default unless selected by a room0 experiment config. Existing configs and tests keep current behavior until the new config is used.

## Data Flow

1. Pipeline processes the frame through the existing fast object path.
2. The proposal backend remains `precomputed`; the primary object path still uses detector-box primary proposals.
3. The structural overlay module receives the precomputed SAM proposals and the structure anchors for the same frame.
4. The module creates voxel votes for structure labels.
5. The run exports the pool-based instance map as before.
6. The dense projection applies object labels first.
7. The dense projection applies conservative structure overlay replacements.
8. Evaluation uses direct labels for synthetic structure ids and existing majority mapping for real object ids.
9. Reports include overlay counts, overwritten point counts, protected point counts, and per-class IoU.

## Error Handling

- If no SAM proposals are available, the overlay produces no votes and reports `skip_reason: no_sam_proposals`.
- If no structure anchors are available, it reports `skip_reason: no_structure_anchors`.
- If proposal masks do not match the frame shape, those proposals are skipped and counted.
- If depth is invalid for mask pixels, those pixels are skipped.
- If the overlay is disabled, output files and metrics match the existing projection path.

## Performance Constraints

The overlay must not call YOLO, YOLOE, SAM, CLIP, or any model. It only uses data already present in the frame processing loop.

The overlay avoids per-point nearest-neighbor searches inside the frame loop. It aggregates world points into voxels and uses a hash map during dense projection, mirroring the current instance projection strategy.

The added cost target is small relative to association and async refinement. It is measured in:

- 20f smoke runtime,
- 200f stride=10 runtime,
- stage timing summary under a new `structural_overlay` or `projection_overlay` timing bucket.

## Testing Strategy

Unit tests cover:

- structure masks are clipped to anchor boxes;
- non-structure anchors do not create overlay votes;
- mismatched mask shapes are skipped;
- conservative fusion does not overwrite protected object labels;
- conservative fusion overwrites unlabeled and structure-labeled dense points;
- synthetic structure ids are evaluated as direct semantic labels;
- disabled overlay preserves existing projection behavior.

Integration tests cover:

- the room0 structural-overlay config enables the feature;
- run reports include overlay totals;
- primary object export paths still exist and keep the same names.

Experiment verification runs in this order:

1. focused unit tests;
2. existing projection/export tests;
3. 20f room0 smoke;
4. 200f stride=10 room0 comparison against `20260529_room0_fast_hybrid_worker_depth_stride10_200f` and `20260529_room0_coarse_to_fine_async_delayguard_stride10_200f`.

## Success Criteria

For room0 stride=10 200f:

- `window` IoU is greater than `0.0`;
- `wall` IoU improves over `0.347`;
- `blinds` IoU improves over `0.384`;
- `chair`, `table`, `sofa`, and `rug` do not show a large regression relative to fast-hybrid; a drop larger than `0.05` absolute IoU for any one of these classes requires inspection before accepting the iteration;
- runtime remains closer to fast-hybrid than async refinement, with no extra model inference.

This design is an iteration toward the broader goal, not the final claim that OVIOVO is fully optimized.
