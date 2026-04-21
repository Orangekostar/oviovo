# Task: Upgrade runtime_vis into a depth-aware object consolidation module

We need to modify the current `runtime_vis` implementation.

The current version over-merges foreground objects with background planes (for example wall + cabinet), because it relies too much on 2D adjacency / planar proximity / coarse merge plausibility.

We want to fix this by introducing **depth-aware geometric reasoning**, inspired by the key idea:

> RGB masks provide candidate regions, but depth geometry should constrain boundary quality, background-vs-object gating, and merge validity.

This is not a full new system.
Patch the existing runtime_vis design with the smallest clean modification that introduces **depth-aware gating and depth-aware merge scoring**.

---

## Goal

Turn `runtime_vis` into a **depth-aware runtime mask consolidation module** that:

1. uses raw SAM2 masks as high-recall fragment proposals
2. distinguishes **background-like** vs **object-like** masks using depth cues
3. merges fragmented masks only when depth geometry supports object-level grouping
4. uses historical whole-object evidence as a strong prior
5. prevents large background planes from being merged into foreground objects

---

## Important design principle

Do not treat 2D adjacency as sufficient evidence for object merge.

Two masks should not merge just because they:

* are nearby in image space
* are roughly coplanar
* produce a plausible merged 2D bbox

Instead, merge should require:

* object-like proposal status
* depth consistency
* local structural compatibility
* optional support from historical whole-object evidence

---

## Core implementation changes

Modify runtime_vis to include these stages:

### Stage A. Depth-aware proposal gating

For each raw mask, compute:

* `objectness_score`
* `backgroundness_score`

using depth-aware cues.

Then classify each mask into one of:

* `object_like`
* `background_like`
* `uncertain`

Only `object_like` and some `uncertain` masks may enter the object consolidation graph.

Strong `background_like` masks must be excluded from object grouping.

---

### Stage B. Depth-aware pairwise merge scoring

For candidate mask pairs, compute a merge score using:

* 2D adjacency / proximity
* depth continuity at boundary
* median depth difference
* plane compatibility / coplanarity
* merged bbox compactness
* historical whole-object prior
* background conflict penalty

Final score should conceptually follow:

`S_merge = S_base + lambda * S_whole_prior - mu * S_bg_conflict`

where:

* `S_base` is built from current-frame geometric evidence
* `S_whole_prior` is a strong positive term
* `S_bg_conflict` is a strong negative term

Historical whole-object evidence should help merge fragmented parts of a known object, but must **not** override strong background conflict.

---

### Stage C. Graph grouping

Use the updated pairwise scores to build a graph and obtain groups with:

* connected components
  or
* a similar simple graph clustering method

Do not overbuild.
A simple explainable graph grouping is enough for v1.

---

## Depth-aware cues to implement

Implement these depth-aware per-mask features:

### 1. depth_valid_ratio

Fraction of valid depth pixels inside the mask.

### 2. depth_variance

Variance or robust spread of depth values inside the mask.

### 3. border_touch_ratio

How much the mask touches image borders.

### 4. planar_fit_residual

Fit a simple plane to the mask depth points or 3D points.
Large background regions often have low residual and large support.

### 5. mask_extent_ratio

Relative 2D area / bbox extent.

### 6. depth_edge_density

How strongly the mask boundary aligns with depth discontinuities.

### 7. local_closedness cue

Approximate whether the region looks like a bounded local object instead of a large layout surface.

You do not need a perfect implementation.
Use simple robust heuristics.

---

## Suggested interpretation rules

### Strong background-like tendencies

A mask is more background-like if it has several of these:

* large image area
* high border touch ratio
* low plane fit residual
* low depth variance
* large elongated extent
* spans major layout region
* weak closedness as local object

### Strong object-like tendencies

A mask is more object-like if it has several of these:

* compact extent
* limited local support
* reasonable depth variation for a bounded object
* stronger local closedness
* lower border dominance
* historical whole-object support exists

---

## Depth-aware pairwise cues

For each candidate pair of masks `(i, j)`, compute:

### 1. adjacency_score

How adjacent / near they are in image space.

### 2. median_depth_gap

Difference of median depth or local depth statistics.

### 3. boundary_depth_continuity

Whether the touching boundaries are depth-consistent.

### 4. plane_compatibility

Whether both masks fit a compatible local plane or local surface orientation.

### 5. merged_bbox_compactness

Whether merging creates a more object-like compact region.

### 6. whole_object_prior_score

Whether both masks are compatible with a historical object that was once seen as a coherent whole object.

### 7. background_conflict_penalty

High when:

* one mask is strongly background-like
* merged result resembles a large scene plane
* a foreground object would be absorbed into a layout region

---

## Required hard rules

Implement the following hard rules in v1:

### Rule 1

Strong `background_like` masks must not participate in ordinary object grouping.

### Rule 2

A merge cannot be accepted if `background_conflict_penalty` is above threshold, even if adjacency or whole prior is high.

### Rule 3

Historical whole-object prior only boosts merges among masks that are not strongly background-like.

### Rule 4

A small standalone object must not be swallowed just because it is adjacent to a large planar surface.

---

## Keep whole-object prior concept

Preserve the previous important idea:

If an object was previously observed as a coherent whole object, and the current frame splits it into fragments, those fragments should be encouraged to merge back into the same object hypothesis.

But implement this safely:

* whole prior is a strong positive term
* not a hard forced merge
* cannot override severe depth/background conflict

---

## Required data structure updates

If needed, extend current runtime_vis data structures with fields such as:

### Per proposal

* `objectness_score`
* `backgroundness_score`
* `depth_valid_ratio`
* `depth_variance`
* `planar_fit_residual`
* `border_touch_ratio`
* `depth_edge_density`
* `is_object_like`
* `is_background_like`

### Per pairwise decision

* `adjacency_score`
* `median_depth_gap`
* `boundary_depth_continuity`
* `plane_compatibility`
* `merged_bbox_compactness`
* `whole_prior_score`
* `background_conflict_penalty`
* `final_score`
* `accepted`
* `rejected_due_to_background_conflict`

### Per group

* member mask ids
* merged bbox
* linked object id if any
* merge reason summary

---

## Visualization updates

Update runtime_vis debugging outputs so that we can inspect the new logic.

At minimum produce:

1. raw masks overlay
2. mask overlay colored by:

   * object-like
   * background-like
   * uncertain
3. merged object groups overlay
4. side-by-side comparison
5. optional text annotations showing:

   * group id
   * objectness / backgroundness
   * whether whole prior was used
   * whether background conflict blocked a merge

Save these into the runtimevis output directory.

---

## JSON / manifest updates

Update JSON outputs so that they include:

### Proposal-level fields

* `objectness_score`
* `backgroundness_score`
* `is_object_like`
* `is_background_like`
* depth-aware feature summary

### Pairwise decision fields

* all pairwise score terms
* background conflict penalty
* final accept/reject reason

### Group-level fields

* grouped masks
* linked object prior if any
* whether group was formed with whole-object prior

---

## Testing requirements

Add or update tests for at least these cases:

### 1. wall + cabinet should not merge

A large planar background-like mask and a nearby object-like mask must remain separate.

### 2. cabinet door fragments can merge

Nearby object-like fragments with compatible depth structure should still merge.

### 3. small standalone object remains separate

A small local object near a big surface should not be swallowed.

### 4. historical whole prior helps merge object fragments

If an existing object prior is present, compatible fragments should more easily merge.

### 5. whole prior cannot override background conflict

Even with strong prior, a clearly background-like plane must not be merged into the object.

### 6. debug outputs expose depth-aware decisions

Ensure output structure contains the new depth-aware fields.

---

## Engineering requirements

* keep the code typed and modular
* do not rewrite the entire pipeline
* patch the current runtime_vis design cleanly
* use simple robust heuristics for v1
* add TODOs where later learned models or stronger geometry reasoning could be inserted
* preserve compatibility with no-history mode

---

## Implementation strategy

Before coding:

1. inspect current runtime_vis implementation
2. inspect where pairwise scores are computed
3. inspect where graph edges are added
4. add depth-aware proposal gating before graph construction
5. add background conflict penalty into pairwise score
6. update visualization and JSON outputs
7. update tests

---

## Final objective

The new depth-aware runtime_vis should satisfy:

* fragmented parts of one object can still merge
* historical whole-object evidence is respected
* background planes are kept out of object consolidation
* wall + cabinet over-merge is prevented
* the logic remains explainable and debuggable

Do not over-engineer.
Implement a reliable, clear v1 that is easy to inspect and iterate on.