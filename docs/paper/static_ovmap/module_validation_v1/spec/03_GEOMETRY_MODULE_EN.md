# 03 — G: native-leaf complete partition hypotheses

## G0. Precise scope

This is a STATIC refinement/export experiment on unchanged native TSDF surface geometry. It is not a new segmentation backbone, a LIVE multi-hypothesis mapper, or a reproduction of LEGO/GeoGuide. Native geometric leaves and frame evidence are captured through file01's active C++ integration path. Keep all surface positions, faces, projection, depth and TSDF values identical across G conditions; only instance ownership changes. No SF candidates enter G v1. Historical LOCAL/object-arbitration outputs remain negative controls, not implementation starting masks.

The experiment separates hypothesis coverage from hypothesis scoring. Never report an oracle-selected partition as a predicted method. Never prefilter hypotheses using GT or the old acceptance gates. A new learned scorer must see the SAME complete hypothesis set as its old-agreement control.

## G1. Numerical native leaves and adjacency

Export registered numerical segment labels, alias resolution and incumbent owner on the exact native surface rows. The no-op export must reproduce the native owner assignment. Do not recover IDs from arbitrary RGB colors. If this is unavailable, implement the C++ read-only API and an affected real smoke; do not synthesize labels from final SF/OVI binary masks.

A leaf is a connected component of `(resolved numerical segment label, incumbent owner)` on native surface adjacency. This ensures original ownership can always be represented by unions of leaves. Use existing mesh edges when valid faces are exported. Otherwise use mutual 8-nearest-neighbor links whose Euclidean length is<=0.03m, generated in chunks, with stable source-row tie breaking. This graph construction is a declared fixed source-domain approximation, not true mesh topology. Surface normals use native values, otherwise the same PCA16-neighbor estimate with an explicit validity flag. Do not use GT mesh topology or normals in prediction.

Keep owner0 geometry unchanged outside candidate groups. A leaf cannot be divided by a downstream pointwise argmax. If native leaves are too coarse to express a needed split, report `COARSE_LEAF_LIMIT`; such a result does not validate all possible multi-granularity methods.

Construct the leaf contact graph from the same surface adjacency. Count adjacency edges; aggregate mean absolute normal dot product, spatial distances and per-frame visible entity evidence. For each of the frozen at-most32 pose-distinct/evenly sampled frames, project the current native surface using the shared camera convention and depth tolerance0.05m. Retain depth-consistent observations with local label0 as UNKNOWN, not contrary evidence. Do not make old `positive_instance_pixels` the denominator for all visible evidence.

For a leaf, assign a frame entity ID only when at least16 of its projected pixels are observed and a positive local entity covers at least0.6 of all depth-consistent observed leaf pixels. Otherwise that frame is unknown. A pair has known same/different evidence only when both leaf labels are known in that frame. Count frame observations once; do not count every surface vertex as an independent vote. This evidence supplies features/proposals; it is not a ground-truth correctness score.

## G2. Bounded disjoint conflict groups and hypotheses

Build adjacent incumbent-owner pairs with at least one leaf contact edge. Define a prediction-side conflict strength as the fraction of known cross-owner leaf/frame observations having the same 2D entity, plus the fraction of within-owner adjacent leaf/frame observations having different entities (each fraction0 if denominator0). Sort adjacent pairs by descending strength then ascending pair IDs. Greedily accept a pair only if neither owner was assigned to an earlier group. Remaining owners form singleton groups. Take at most64 groups/scene, ordering by descending conflict strength then owner tuple; untouched owners retain N0. Pair groups allow changing TWO parents together, unlike old single-parent protection. This is bounded, not arbitrary global set partitioning.

Each group covers its original owner's entire source support, not merely its intersection with a new mask. Max1024 leaves/group for spectral candidates; oversized groups still permit original and merge candidates and have a documented spectral omission. No downsampling that silently deletes leaves.

Generate complete partitions of all group leaves:

1. ORIGINAL: exact original owner partition, always present.
2. MERGE: one group containing all leaves, only for a two-owner group.
3. FRAME_ENTITY: for each captured frame with two or three known positive entities in the group, assign known leaves by their dominant frame entity. Assign unknown leaves to the nearest centroid of known leaves of an entity, weighted by leaf surface-point count; Euclidean ties use the entity's minimum source-row ID. If no known support exists, no candidate is generated. Thus each proposal partitions all leaves; the uncertain-fill fraction is recorded as a quality feature, not hidden.
4. GRAPH_SPLIT2 / GRAPH_SPLIT3: use the leaf contact affinity `exp(-distance/0.03)*(0.1+0.9*abs_normal_dot)*(0.25+0.75*same_fraction)`. When normals are invalid use normal factor1; when there is no known pair evidence use same_fraction0.5 and a missing bit. For disconnected contact components set intercomponent affinity0. Compute the normalized graph Laplacian's smallest k eigenvectors, deterministic eigenvector sign (largest-absolute coordinate positive), row-normalize, then k-means k=2/3 with random_state17,n_init1. No alternative clustering grid. Skip k when fewer than k leaves; empty clusters invalidate that candidate. The original and all other valid proposals remain.

Canonicalize a partition by sorting output components by their minimum source-row index and encoding leaf->component integers. Deduplicate exact partitions. Keep ORIGINAL, then MERGE, GRAPH_SPLIT2/3, then FRAME_ENTITY proposals in descending known-pixel support and ascending frame ID, up to8 hypotheses/group. Do not remove a hypothesis because its old teacher agreement does not improve. Do not apply benchmark100-point filtering or the old85% support-retention rule here.

Every candidate completely partitions the same group geometry; output components cannot overlap. Empty components are invalid. Combining selected disjoint groups preserves whole-scene unique ownership, except preexisting owner0. New component IDs derive deterministically from scene/group/partition/component identity; original IDs are retained when the component support is unchanged. Keep ancestry, not just new ordinal IDs.

## G3. Old agreement and learned complete-partition score

### G_AGREEMENT

For every candidate and valid frame, rasterize its component IDs at depth-consistent visible surface samples. Match candidate components to positive 2D entities with a maximum-intersection Hungarian assignment. Frame agreement is sum assigned intersections / all depth-consistent group samples, INCLUDING unknown/unmatched samples in the denominator. Frame agreement with no denominator is missing; average over defined frames. Select the maximum mean score, tie ORIGINAL then canonical hypothesis order. There is no inherited `gain>=0.03` adoption gate. This isolates old information's ranking ability from the old conservative threshold. Name it `G_AGREEMENT`, not an exact reproduction of the earlier OD method.

### G_QUALITY features and target

For each complete partition record this 20-dimensional, permutation-invariant vector, plus20 availability bits:

1 log1p(group source size); 2 log1p(leaf count); 3 component count; 4 minimum component area fraction; 5 maximum area fraction; 6 normalized entropy of component area fractions (0 for one component); 7 count of disconnected surface components / output component count; 8 fraction of contact edges crossing output components; 9 mean absolute normal dot for within-component edges; 10 corresponding cross-component mean; 11 same-entity fraction on within-component known leaf/frame pairs; 12 same fraction across components; 13 different-entity fraction within components; 14 different fraction across components; 15 fraction of depth-consistent pixels with unknown2D labels; 16 fraction of leaves with any known view; 17 mean fraction of leaves assigned by the FRAME_ENTITY nearest-centroid completion (0 for other proposal types, but no proposal-type ID is provided); 18 visible-pixel-weighted old agreement; 19 mean component bounding-box diagonal / group diagonal; 20 mean within-component normal dispersion `1-norm(mean(unit normals))`.

Undefined continuous values have value0/availability0; standardize using FIT as in S. Feature17 is a measured proposal construction property, not GT. All sources use the same definitions. Do not pass semantic classes or GT-derived shape descriptors.

A separate FIT target builder matches predicted components to **whole** GT objects by IoU on the frozen projection. For a group include every GT object with any valid intersection with group support; use the full GT object's denominator, not its clipped group fragment. Hungarian-match pairs with strictIoU>0.5; let

`quality = sum(matched IoU) / (TP + 0.5*FP + 0.5*FN)`;

zero denominator produces0 and an explicit flag. GT objects outside the group with no intersection do not enter this local target. Label this a local PQ-like training target, not the released AP. It penalizes damage to both old objects and newly proposed components. Original is scored by exactly the same target. There are no test-oracle choices in prediction.

Train a40->32(ReLU)->1 sigmoid head, seed17, Adam0.001, weight_decay0.001, batch64 groups, max100epochs. Loss: per-group mean squared quality error plus0.1 times mean `softplus(-(score_better-score_worse))` for hypothesis pairs whose target gap>=0.05. Average groups equally within scenes, then scenes equally. Use the same CAL-every5/three-nonimprovement-check early stopping, selecting lowest CAL loss. Require at least20 FIT groups with differing quality targets across at least4 FIT scenes; otherwise code/datasets/G_AGREEMENT are delivered and learned G is `BLOCKED_PARTITION_TARGET_SUPPORT`.

Choose a CAL adoption margin from `[0,0.02,0.05,+infinity]`: accept a non-original maximum-scoring hypothesis only when its score exceeds ORIGINAL by strictly more than the margin. Pick by mean canonical class-agnostic AP50, then mean canonicalAP75, then fewer changed source points, then larger margin. Freeze before SELECT. An all-original winner is NO_INTERVENTION, not an effective geometry module.

## G4. Required experiments, same geometry and semantics conventions

| ID | Hypothesis pool | Selection |
|---|---|---|
| G_ORIGINAL | same stored pool | ORIGINAL |
| G_AGREEMENT | same stored pool | mean 2D agreement |
| G_QUALITY | same stored pool | learned complete-partition quality, CAL margin |

First evaluate class-agnostic geometric correspondence/coverage and the existing canonical AP50/AP75 diagnostic. Do not rename canonical diagnostic AP as paper-released class-agnostic AP unless an actual protocol parity test establishes this. Keep normal released semantic-instance and semantic metrics too.

For all G rows, use source component point count as the same deterministic numeric ranking rule, and re-read semantics of their ACTUAL final masks with the SAME frozen native SigLIP, max3 views and native six-crop area fusion. For G_ORIGINAL this is the matched re-read control. Do not inherit parent categories for new children. Define their ROI by final native component projection with depth visibility, bbox from its own projection, and union with the maximum-overlap original local entity, retaining full construction identity. This differs from online native capture for newly shaped masks and is explicitly a static post-map re-read. N0 is separately reported; G gains must not be attributed to semantic-input or rank bridges.

Unchanged masks can reuse exactly matching S/native requests; changed masks require new keys. At most128 final target masks/scene/condition receive re-encoding, ordered by the same prediction-only hash convention. Unchanged unavailable targets retain their original category; a new unavailable component has semantic0, not an inherited or GT label. All components remain in geometric diagnostics. Report counts of such unknown components and include their missed semantic matches; do not hide them from evaluation.

Oracle best-in-pool quality/coverage is optional only as a compulsory **diagnostic table computed from the already evaluated finite pool**, with name `ORACLE_DIAGNOSTIC_NOT_METHOD`. It does not incur new inference or select a runtime configuration. Show whether failure is inadequate leaves, no useful complete hypothesis, scorer choice, or semantic reread. The released predicted rows remain those above.

## G5. Integration output

Deliver the numerical native snapshot adapter, leaf graph, finite partition manifest, feature/target builders, old and learned selectors, final unique-map exporter and traces. No live owner override is claimed or required. Do not expand to a new TSDF mapper if this finite experiment is negative. Put implementation status and actual data limitation in the handoff separately.
