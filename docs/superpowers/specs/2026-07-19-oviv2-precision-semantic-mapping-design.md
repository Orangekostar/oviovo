# OVIV2 Precision Semantic Mapping Design

**Date:** 2026-07-19
**Status:** Approved architecture; documentation review pending
**Primary objective:** Maximize causal online Replica semantic mIoU without sacrificing the OVIV2 current-state mapping mainline

## 1. Context

The verified OVIV2 Replica-8 result currently reports:

- semantic mIoU: `0.173016`;
- semantic mAcc: `0.222796`;
- frequency-weighted mIoU: `0.441345`;
- class-agnostic AP25: `0.092618`;
- class-agnostic AP50: `0.028646`;
- geometry F@5cm: `0.882351`.

The formal Table 1 baselines show that OVIV2 trails OpenFusion in semantic mIoU
(`0.271`) and trails ConceptGraphs in AP50 (`0.214`). Geometry is already competitive,
so the next optimization must target semantic representation and instance association
rather than replacing the TSDF geometry backbone.

Room0 diagnostics expose two dominant failures:

1. The frontend produces 4,580 observations across 200 frames, but the backend reduces
   them to only 19 final entities. Spatially adjacent objects and semantically different
   observations are being over-merged.
2. Each entity stores one label and one cumulative support scalar. An early label becomes
   effectively irreversible, while voxel semantic evidence keeps old positive votes even
   after ownership or entity semantics change.

The frontend cache already contains 1024-dimensional image and text features, but the
current OVIV2 path does not consume them. Only about 1.8% of cached detections are outside
the frozen Replica vocabulary, so missing detector vocabulary is not the primary cause.

## 2. Decision

Implement a hybrid, dual-layer semantic mapper while preserving the existing sparse TSDF,
signed visibility, reversible ownership, lifecycle, and immutable snapshot contracts.

The new path combines:

- dense causal open-vocabulary predictions from RADSeg;
- class-agnostic object masks and cached per-instance visual features;
- local-window evidence graph association;
- quality-weighted multi-view entity prototypes and class posteriors;
- uncertainty-aware fusion of dense and instance semantic evidence;
- owner-authoritative semantic export that can be corrected after mapping.

This design supersedes the earlier static milestone's exclusion of dense semantic fusion.
It does not supersede the voxel-first authority, runtime causality, ownership, or snapshot
invariants defined in `2026-07-15-oviv2-replica-table1-design.md`.

## 3. Evidence From Recent Work

The design reuses public implementations when licensing permits and independently
implements paper mechanisms when code is unavailable.

### Public, Reusable References

- RADSeg (CVPR 2026 Findings) provides MIT-licensed dense open-vocabulary features and a
  released Replica 3D evaluation. RADSeg+ reports `0.3305` Replica mIoU using probability
  aggregation. It is the preferred dense frontend. The RADIO checkpoint has a separate
  NVIDIA model license that must be recorded and reviewed independently from the code.
  <https://github.com/RADSeg-OVSS/RADSeg>
- RayFronts (IROS 2025) provides an MIT-licensed online dense semantic voxel mapper and
  NARADIO encoder. It is the integration fallback and a reference evaluator. Its RADIO
  model dependency is also subject to the checkpoint's separate model license.
  <https://github.com/RayFronts/RayFronts>
- OVI-MAP selects a bounded set of informative object views instead of averaging every
  observation. The pinned local checkout is MIT licensed.
  <https://arxiv.org/abs/2603.26541>
- ConceptGraphs combines spatial and visual similarity and maintains aggregated visual
  features. The pinned local checkout is MIT licensed.
- OnlineAnySeg uses hashed 3D mask associations, asymmetric overlap, visual/geometric
  filtering, and third-view support for online mask merging.
  <https://arxiv.org/abs/2503.01309>

### Design References Without Reusable Code

- RAZER reports `0.320` Replica mIoU and uses spatial candidate indexing, Hungarian
  matching, and up to three semantic embeddings per object. Its official page still marks
  code as forthcoming, so no source is copied.
  <https://razer-3d.github.io/>
- FUS3DMaps reports `0.3237` Replica mIoU for its sliding-window fused instance layer and
  `0.4117` for its fused dense layer. It motivates the dual-layer uncertainty fusion, but
  its code is also forthcoming; the implementation will be independent.
  <https://arxiv.org/abs/2605.03669>
- OnlinePG motivates local-to-global sliding-window clustering and bidirectional
  bipartite matching. Replacing OVIV2 TSDF with its 3D Gaussian map is out of scope.
  <https://arxiv.org/abs/2603.18510>

Published values are design evidence, not benchmark results. They cannot be inserted into
the OVIV2 table because model versions, class handling, projection rules, and background
domains must first be matched by the common evaluator.

## 4. Goals And Non-Goals

### Goals

1. Exceed the formal OpenFusion Replica-8 semantic mIoU of `0.271` under the existing
   common evaluator.
2. Raise class-agnostic AP50 toward or above the strongest current baseline value of
   `0.214` by preventing both over-merging and uncontrolled fragmentation.
3. Preserve Replica-8 geometry F@5cm at or above `0.88`.
4. Preserve strict frame-order causality: frame `t` may use only state committed at or
   before `t`.
5. Preserve open-vocabulary querying at entity level while allowing a frozen vocabulary
   classification head for benchmark evaluation.
6. Keep all new behavior feature-gated so every ablation produces a comparable snapshot.

### Non-Goals

- Do not use Replica semantic or instance ground truth in runtime mapping.
- Do not use future frames, backward temporal passes, or full-sequence optimization.
- Do not replace the sparse TSDF with a NeRF or 3D Gaussian reconstruction.
- Do not tune thresholds on Replica-7 after room0 optimization.
- Do not rewrite verified result directories or overwrite prior Table 1 provenance.
- Do not copy code from repositories without a compatible license.

## 5. Hard Runtime Invariants

1. RGB, depth, pose, current-frame masks, and current-frame features are the only inputs
   accepted by the runtime mapper.
2. Dense predictions are computed independently per frame. Spatial sliding windows inside
   one image are allowed; temporal windows cannot include future frames.
3. The geometry layer is independent of semantic confidence and remains authoritative.
4. Entity association emits evidence; it cannot delete TSDF geometry.
5. Ownership remains derived and reversible. Semantic reclassification cannot alter
   geometry or visibility history.
6. Every entity merge records supporting observation edges. Edges inside the active local
   window may be revoked; committed historical observations remain immutable provenance.
7. Model name, checkpoint hash, text vocabulary hash, prompt template hash, and feature
   dimension are persisted with every snapshot.
8. Evaluator-only ground truth is loaded in a separate process after snapshot finalization.

## 6. Runtime Architecture

```text
RGB-D + pose at frame t
          |
          +--------------------------+
          |                          |
  Instance observation branch   RADSeg+ dense branch
  masks + image/text features   pixel/patch class logits
          |                          |
          +------------+-------------+
                       |
             Local observation graph
        hashed candidates + reversible edges
                       |
       gated geometry/visual/semantic scoring
                       |
          one-to-one Hungarian assignment
                       |
        persistent entity posterior + view bank
                       |
      multi-hypothesis voxel ownership evidence
                       |
       uncertainty-aware dense/instance fusion
                       |
     TSDF + current owner + current semantic label
                       |
             immutable OVIV2 snapshot
```

The first implementation stage uses the existing frontend cache and leaves the RADSeg
branch disabled. This isolates backend correctness before adding a new model.

## 7. Dense Semantic Frontend

### Preferred Model

Use RADSeg+ with `C-RADIOv3-L`, the SigLIP2 language adaptor, and RADIO-SAM refinement.
The model processes only the current RGB frame. The implementation starts from the
released MIT code through a narrow adapter rather than importing its evaluator or map.

The adapter returns:

- dense normalized language-aligned features, when available;
- dense logits or probabilities for the frozen Replica-41 prompt set;
- per-pixel entropy and top-1 margin;
- model and prompt provenance;
- optional refined class masks.

The feature representation and probability representation are both supported. Benchmark
classification uses the probability head because RADSeg reports stronger Replica results
for probability-space aggregation. Open-vocabulary entity queries retain normalized
feature prototypes.

### Fallback

If RADSeg integration fails its deterministic smoke test or dependency gate, use the
RayFronts NARADIO encoder through its public MIT implementation. The fallback must produce
the same adapter contract and a separately named configuration; results from different
encoders cannot share a run ID.

### Prompt Contract

Each Replica class is encoded with a frozen ensemble of simple indoor-object templates.
Aliases map to one canonical benchmark class before posterior fusion. Prompt templates and
text embeddings are built without reading scene ground truth and are reused across scenes.

## 8. Observation And Association Model

### Observation State

Each object observation carries:

- frame-local observation ID and provenance;
- surface voxel keys and projected mask pixels;
- centroid, bounds, covariance, and depth statistics;
- detector label distribution and confidence;
- normalized visual feature and feature-model ID;
- dense semantic posterior pooled within the mask;
- visible area, image-border contact, occlusion estimate, and view direction.

Features with different model IDs or embedding dimensions are never compared directly.

### Candidate Generation

Use the sparse voxel index to retrieve entities that share supported voxels with the new
observation. A bounded spatial-neighborhood lookup adds candidates whose predicted bounds
intersect the observation uncertainty volume.

Centroid distance alone is not a valid match. A candidate must pass at least one explicit
spatial consistency condition:

- sufficient directed voxel overlap;
- compatible 3D bounds intersection after uncertainty expansion;
- projected support plus valid visibility agreement for a partially observed object.

Candidates contradicted by signed free-space evidence are rejected.

### Match Score And Gates

For every valid observation/entity pair, compute:

```text
score = wg * geometry
      + wo * directed_overlap
      + wv * visual_cosine
      + ws * semantic_compatibility
      + wt * temporal_visibility
```

Geometry and overlap establish identity feasibility. Visual similarity and semantic
compatibility resolve adjacent or overlapping candidates. High-confidence incompatible
semantic posteriors impose a conflict gate; low-confidence disagreement only lowers the
score and does not force fragmentation.

The assignment for all observations in a frame is solved jointly with Hungarian matching.
This prevents several detections from independently attaching to the same entity before
the mapper evaluates whether they are separate objects.

### Local Evidence Graph

Maintain a causal window of the latest 3-5 accepted frames. Observations are nodes and
candidate identity relations are weighted edges. Third-view support can strengthen or
reject ambiguous edges using only frames already inside the causal window.

Entity updates are produced from connected, mutually compatible observations. Weak edges
may be revoked while both endpoints remain in the active window. This corrects short-lived
mask merges without a backward pass over the full sequence.

## 9. Persistent Entity Semantics

Replace the scalar `semantic_support` and early-label replacement rule with three related
states:

1. a normalized open-vocabulary feature prototype bank with at most three hypotheses;
2. a quality-weighted Replica-41 log-posterior;
3. a bounded informative-view bank with at most ten observations.

Observation quality combines mask confidence, visible area, border truncation, valid-depth
ratio, occlusion, and view novelty. Low-quality views may support geometry association but
receive little or no semantic weight.

An incoming feature updates the closest compatible prototype. A sufficiently distinct
feature creates another hypothesis until the bank is full; replacement removes the
lowest-quality redundant hypothesis. The class posterior is updated in log space with
bounded per-view contribution so repeated near-identical frames cannot dominate one
informative view.

The exported entity label is always the current posterior maximum. Entropy, margin, and
effective observation count remain available for conflict gates and audit. No label is
permanently locked by its first observation.

## 10. Dual-Layer Voxel Semantics

### Instance Layer

Each surface voxel retains up to three entity ownership hypotheses with positive support,
negative visibility support, last-observed frame, and evidence revision. This replaces
single-winner semantic accumulation with reversible evidence.

The semantic distribution for an owned object voxel is derived from the current entity
posterior, not from historical copies of the entity's former label. Reclassifying an entity
therefore corrects all currently owned voxels without rewriting their history.

### Dense Layer

Within a configurable egocentric radius, surface voxels aggregate RADSeg class probability
vectors and observation counts. Entropy and view-angle weighting reduce the influence of
uncertain or grazing observations.

The high-dimensional dense feature grid is local and sliding. When a voxel leaves the
window, the global map keeps a bounded top-k class posterior and its sufficient statistics;
it does not retain a full dense embedding for every global voxel. Entity prototypes retain
open-vocabulary features globally.

### Cross-Layer Fusion

For voxel `v`, combine dense and instance class distributions using confidence-weighted
log probabilities:

```text
log p_fused(v) = alpha(v) * log p_dense(v)
               + beta(v)  * log p_entity(v)
```

`alpha` and `beta` depend only on runtime observation count, entropy, margin, ownership
support, and semantic view quality. They do not depend on ground truth.

Fusion policy by region:

- wall, floor, and ceiling prefer dense evidence unless a high-confidence object owner is
  present;
- owned object voxels prefer the entity posterior but use dense evidence to reject a
  semantically inconsistent merge;
- unowned surface voxels use dense evidence;
- low-confidence voxels remain unknown rather than receiving an irreversible weak label.

## 11. Dynamic And Lifecycle Compatibility

Signed visibility remains the source of presence and absence evidence. When an entity
becomes absent or loses ownership:

- its geometry history remains intact;
- its voxels may be reclaimed by background or another entity;
- its semantic prototypes remain in the dormant entity record for re-identification;
- current-map export uses only current ownership and current fused semantics.

Reactivated entities use geometry, visual prototypes, semantic posterior, and lifecycle
compatibility. Dense semantics can support re-identification but cannot reactivate an
entity without valid spatial or appearance evidence.

## 12. Persistence And Compatibility

Add a new snapshot schema version containing:

- entity class log-posteriors, entropy, margin, and effective support;
- up to three feature prototypes with model IDs and quality weights;
- informative-view metadata without raw full-resolution images;
- dense semantic top-k sufficient statistics;
- multi-hypothesis ownership support;
- association decision and conflict counters;
- dense frontend provenance and prompt hashes.

Older snapshots remain readable through the current schema reader. Missing new semantic
fields initialize as unknown and never fabricate posterior support. New snapshots are not
silently downgraded by old readers.

## 13. Implementation Stages

### Stage 0: Diagnostic Ceiling Audit

Use room0 ground truth only in an offline diagnostic process to quantify:

- semantic accuracy if current masks were associated perfectly;
- semantic accuracy if current entities retained their geometry but received an oracle
  class;
- instance AP loss from over-merging versus under-segmentation;
- coverage of the old high-recall YOLOWorld+SAM pipeline under the current evaluator.

These diagnostics cannot produce a benchmark row and cannot feed runtime parameters with
per-scene class-presence information.

### Stage 1: Correctable Entity Backend

Use the existing frozen frontend cache to implement:

- visual-feature consumption;
- class posterior semantics;
- gated Hungarian association;
- local evidence graph;
- owner-authoritative semantic export.

This stage isolates the main observed regression: 4,580 observations collapsing to 19
entities.

### Stage 2: RADSeg Dense Branch

Integrate the public RADSeg adapter, frozen text prompts, probability aggregation, and
sliding dense semantic layer. Compare RADSeg and the RayFronts fallback on identical room0
frames before choosing one.

### Stage 3: Cross-Layer Fusion

Add multi-hypothesis voxel support, entropy-aware cross-layer fusion, and structure/object
routing. Preserve separate instance-only, dense-only, and fused outputs for ablation.

### Stage 4: Frozen Evaluation

Freeze all parameters after room0. Run the unchanged configuration on the remaining seven
Replica scenes, aggregate Replica-8, and report Replica-7 as the held-out result. Only then
may the benchmark registry and generated tables be updated.

## 14. Ablation Order

Every row is cumulative and uses the same room0 200-frame stride-10 input:

1. current verified OVIV2;
2. `+` correctable entity class posterior;
3. `+` visual/semantic conflict gates and Hungarian assignment;
4. `+` local evidence graph and third-view support;
5. `+` bounded multi-prototype and informative-view banks;
6. `+` owner-authoritative semantic export;
7. `+` RADSeg dense semantic layer;
8. full method `+` uncertainty-aware cross-layer fusion.

For each row record mIoU, mAcc, f-mIoU, AP25, AP50, F@5cm, final entity count, merge
conflicts, split/revocation count, frontend time, backend time, peak GPU memory, peak RAM,
and final map size.

## 15. Validation And Acceptance

### Room0 Development Gates

- Stage 1 must reduce measured over-merge errors and improve mIoU or AP50. Final entity
  count is recorded as a diagnostic, not optimized against the GT entity count.
- No accepted stage may reduce room0 F@5cm below `0.90`.
- No accepted stage may reduce room0 AP50 below its current `0.0417`.
- A module is retained only if it improves mIoU, or improves AP50 without reducing mIoU by
  more than `0.01`.

### Frozen Replica Gates

Minimum publishable gate:

- Replica-8 mIoU `> 0.271`;
- Replica-8 F@5cm `>= 0.88`;
- Replica-7 mIoU improves over the current `0.178398`.

Primary target:

- Replica-8 mIoU `>= 0.35`;
- Replica-8 AP50 `>= 0.22`;
- Replica-8 F@5cm `>= 0.88`.

Stretch target:

- Replica-8 mIoU in the `0.42-0.50` range while preserving the primary AP50 and geometry
  gates.

The stretch range is an optimization objective, not a guaranteed claim. The historical
room0 mIoU near `0.53` came from a different high-fragmentation pipeline and is used only
as evidence that the frontend contains substantially more semantic signal than current
OVIV2 preserves.

## 16. Test Strategy

Implementation follows test-driven development.

Unit tests cover:

- quality-weighted posterior updates and reversibility;
- bounded prototype and view-bank replacement;
- feature model/dimension incompatibility;
- spatial hard gates and semantic conflict gates;
- deterministic Hungarian assignment and unmatched observations;
- local graph edge revocation without future frames;
- owner-authoritative reclassification of all owned voxels;
- dense/instance fusion under exact entropy and count fixtures;
- multi-hypothesis eviction and ownership release;
- old/new snapshot schema round trips.

Integration tests cover:

- a two-object adjacency sequence that the old distance rule over-merges;
- a partial-view sequence whose correct class changes after stronger evidence;
- a structure/object overlap sequence;
- an absent, reclaimed, and reactivated entity sequence;
- deterministic 2-frame and 20-frame Replica smoke runs;
- identical results for uninterrupted and checkpoint-resumed execution.

The formal 200-frame room0 run starts only after unit tests and 20-frame smoke gates pass.

## 17. Failure Handling

- If Stage 1 does not materially improve room0 mIoU or AP50, stop before downloading or
  integrating a heavier frontend and diagnose observation-to-entity coverage again.
- If RADSeg cannot reproduce its released Replica evaluator smoke path, record the model,
  checkpoint, environment, and error, then evaluate the RayFronts fallback.
- If dense fusion improves mIoU but harms AP, retain separate semantic and instance output
  heads and use instance evidence only for class-agnostic AP.
- If the fused map exceeds memory limits, reduce dense-window radius or store lower-rank
  features; do not drop frames or use future-frame batching.
- If a metric definition changes, create a new immutable result ID and rerun all compared
  OVIV2 ablations under the corrected evaluator.

## 18. Paper Claim Boundary

The intended claim is that a causal current-state mapper can preserve high-quality static
open-vocabulary semantics by combining correctable entity memory with bounded dense
semantic evidence. The contribution is the integration of reversible ownership,
uncertainty-aware dual-layer semantics, and causal evidence-graph association in one
current-state map.

Published numbers from RADSeg, FUS3DMaps, RAZER, or other papers remain literature context.
Only values generated from immutable OVIV2 snapshots through the common evaluator may fill
the OVIV2 benchmark row.
