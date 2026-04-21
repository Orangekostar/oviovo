# Task: Add a runtime_vis post-processing module for object-level mask consolidation

You are modifying an existing online open-vocabulary dynamic mapping backbone.

The current system already has:

* real Replica dataset loading
* real SAM2 proposal backend
* proposal-stage visualization (`precompute_vis`)
* a modular backbone with proposal / refinement / lifting / split / association / update modules

Now add a new **runtime_vis** stage.

## Goal

Create a **runtime post-processing module** that takes raw per-frame SAM2 masks and merges masks that likely belong to the **same physical object**, especially when a single object is fragmented into multiple masks.

This is **not** semantic labeling.
This is **object-level consolidation of fragmented masks**.

The output should be a cleaner set of per-frame object candidates for downstream 3D lifting / association.

---

## Core design principle

Do **not** treat every raw SAM2 mask as a final object.

Instead:

* raw masks are over-segmented visible surface fragments
* runtime_vis should group/merge fragments into **object-level candidates**
* historical evidence that a physical object was once observed as a **whole object** should have **higher weight** than current-frame fragmented observations

### Important rule

If an object was previously observed as a coherent whole object, and in the current frame it appears fragmented, then the current fragments should preferentially be interpreted as **partial observations of the same existing object**, not as multiple new objects.

This rule should be explicitly reflected in the scoring / merging logic.

---

## What to build

Implement a new module, e.g.:

* `src/modules/runtime_vis.py`

and integrate it into the runtime pipeline after raw proposal generation, before later downstream usage.

You may also add supporting files if needed.

---

## Required behavior

### Input

The module should take:

* current frame raw SAM2 proposals
* current RGB / depth
* optional refined masks if already available
* optional historical object hypotheses from system state
* optional historical whole-object observations / object priors

### Output

The module should produce:

* merged per-frame mask groups
* consolidated object candidate masks
* mapping from raw mask ids -> merged group ids
* metadata explaining why masks were merged
* visualization outputs for debugging

---

## Required concepts

The module must explicitly distinguish:

### 1. Raw fragment masks

These are original SAM2 masks.

### 2. Whole-object evidence

A historical signal that a certain object has previously been observed as a single coherent object.

Examples:

* a prior frame had one large coherent mask for this object
* a prior object hypothesis has a stable bbox / shape / geometry support

### 3. Split evidence

Current frame only sees multiple fragmented masks that may correspond to one object.

The merge logic must use:

* whole-object evidence as strong prior
* current-frame geometric consistency as main evidence
* semantics only as optional weak auxiliary signal

---

## High-level strategy to implement

Build the module in two layers.

### Layer A: intra-frame grouping

Within the current frame, compute candidate grouping between masks using mostly geometric / structural cues.

Possible cues:

* 2D adjacency / overlap / proximity
* depth continuity
* coplanarity or similar depth plane behavior
* merged bbox plausibility
* contour compatibility
* optional weak semantic similarity

This produces candidate mask groups for the current frame.

### Layer B: temporal object prior fusion

Use historical object evidence to bias grouping decisions.

If a historical object `O` has strong whole-object evidence, and several current fragments all fit `O`, then give a strong merge boost to grouping them together as partial observations of `O`.

This temporal prior should have higher importance than current split evidence.

---

## Scoring design

Implement an explicit scoring structure.

For two current masks `i, j`, define a base grouping score like:

* adjacency score
* depth continuity score
* coplanarity / geometric consistency score
* merged bbox plausibility score
* optional weak semantic score

Then define a temporal prior score:

* historical whole-object support
* compatibility with an existing object hypothesis
* whether both fragments fall inside an expanded canonical bbox of the same object
* whether past stable observations suggest they are parts of one object

Final runtime merge score should look conceptually like:

`S_merge = S_base + lambda * S_whole_prior`

where `lambda` is significant.

### Important requirement

Historical whole-object evidence must have **higher weight** than fragmented current-frame evidence.

Make that explicit in code comments and score design.

---

## Required constraints

### Do not do these

* do not rely on semantic labels as primary merge criterion
* do not hardcode dataset-specific object categories
* do not merge everything that shares the same text label
* do not permanently overwrite raw masks; keep raw-to-merged mapping

### Do these

* keep the module modular and typed
* expose merge scores and reasons for debugging
* make thresholds configurable
* design the module so it can run even if historical object priors are absent
* support both:

  * no-history mode
  * history-aware mode

---

## Required data structures

If needed, add clear typed structures such as:

* `WholeObjectEvidence`
* `RuntimeMaskGroup`
* `RuntimeMergeDecision`
* `RuntimeVisOutput`

Possible fields:

### WholeObjectEvidence

* object_id
* canonical_bbox_2d or canonical_bbox_3d
* whole_observation_count
* whole_observation_score
* last_seen_frame
* geometric signature summary

### RuntimeMaskGroup

* group_id
* member_mask_ids
* merged_bbox
* merged_mask
* confidence
* linked_object_id (optional)
* merge_reason

### RuntimeMergeDecision

* mask_id_a
* mask_id_b
* base_score
* whole_prior_score
* final_score
* accepted
* reason_breakdown

---

## Required implementation scope

This is still a first implementation.
Do not overbuild.

Implement a simple, explainable version:

### v1 should include

* current-frame pairwise grouping scores
* simple graph-based grouping or connected-components clustering
* historical whole-object prior boost
* support for “partial views of existing object”
* debug visualization

### v1 does not need

* learned merge model
* perfect shape reasoning
* complex graph optimization
* semantic-driven grouping

---

## Pipeline integration

Integrate runtime_vis into the pipeline at a clean point.

Suggested order:

1. raw proposal generation
2. runtime_vis mask consolidation
3. downstream refinement / lifting / split / association

If another insertion point is better given the current codebase, explain the choice clearly.

---

## Visualization requirements

Add debugging visualization outputs for runtime_vis.

At minimum:

* original raw masks overlay
* merged groups overlay
* side-by-side comparison
* optional text annotations showing:

  * group id
  * linked object id
  * whether merge was boosted by whole-object prior

Save outputs into an organized directory.

---

## README / interface documentation update

Update the README or interface doc to explain:

* what runtime_vis does
* why it exists
* how it differs from raw proposal stage
* what “whole-object evidence” means
* how it affects merging

Keep this concise.

---

## Testing requirements

Add tests for at least:

1. no-history mode:

* nearby compatible fragments can be grouped

2. history-aware mode:

* if historical whole-object evidence exists, fragmented masks that fit the same object should be more likely to merge

3. preservation of small standalone object:

* a small independent object should not be wrongly swallowed just because it is nearby

4. debug output structure:

* runtime_vis output should contain raw-to-merged mapping and merge reasons

---

## Important design message

This module is intended to implement the following research idea:

* the proposal frontend is high-recall but over-segmented
* final maintained entities should be object-level, not raw-mask-level
* historical whole-object observations should serve as strong priors
* fragmented observations should defer to stable object hypotheses when supported

Make sure the implementation reflects this idea clearly.

---

## Final instruction

Before coding:

1. inspect the current proposal module and pipeline integration points
2. identify the cleanest insertion point for runtime_vis
3. implement a minimal but rigorous first version
4. keep the module independent, typed, and debuggable
5. print or log intermediate grouping statistics in the demo

Do not rewrite the whole system.
Only add and integrate the runtime_vis consolidation stage.
