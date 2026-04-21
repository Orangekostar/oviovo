# Task: Refactor current project to match the final mapping roadmap

Refactor the current RGB-D open-vocabulary mapping project so that it clearly follows this layered design:

## Final roadmap

1. **SAM2 proposal frontend**

* keep SAM2 as high-recall raw proposal source
* raw SAM2 masks are not final objects

2. **Depth-aware refinement**

* refine proposals using depth discontinuity and depth connectivity
* reduce object/background sticking
* split obvious depth-disconnected regions
* output `RefinedProposal2D`

3. **3D patch lifting**

* lift refined proposals into world-frame `Patch3D`
* each patch is only a partial observation

4. **Global TSDF instance substrate**

* maintain a global TSDF-based instance map
* each voxel stores TSDF + owner instance id + owner support
* patch-to-instance association must be spatial/geometric first
* use spatial voting + support accumulation for instance stabilization

5. **Per-instance local geometry memory**

* each instance maintains `local_pcd`
* `local_pcd` is only an object-local maintenance memory
* it must NOT replace the global TSDF instance substrate

6. **Active set / local status checks**

* derive a local active set from the global instance map
* only use it for local reasoning, stability check, split/merge diagnostics
* active set must NOT replace global memory

7. **Semantic layer**

* semantics starts only after instance stabilization
* use object-centric view selection
* update object-level semantic memory
* semantics must not drive low-level patch association

---

## Critical correction rules

### Rule A

`runtime_vis` must not replace instance association.

It may do:

* proposal grouping
* visualization
* light proposal cleanup

It must NOT do:

* final instance id assignment
* long-term lifecycle
* object truth resolution

### Rule B

Do not use hard whole-object labels.

Use only soft evidence:

* `whole_evidence_score`
* `part_evidence_score`
* `assignment_score`

### Rule C

Do not let semantics leak into low-level association.

Patch association must remain geometry-first / spatial-first.

### Rule D

Keep the distinction explicit:

* raw proposal
* refined proposal
* 3D patch
* instance
* local geometry memory
* semantic memory

---

## Mathematical requirements

The implementation must expose explicit scores, not just hidden heuristics.

### 1. Proposal geometric features

For each refined proposal, compute and store:

* depth validity ratio
* depth variance
* border touch ratio
* planar fit residual

### 2. Proposal soft scores

Compute:

* `objectness_score`
* `backgroundness_score`
* `attachedness_score`

These should be explicit functions of geometric features, with configurable weights.

### 3. Patch lifting

Use explicit RGB-D lifting:

* depth + intrinsics + pose
* output world-frame `Patch3D`

### 4. Patch-to-instance association

Expose an explicit voting score:

* touched voxels
* owner votes
* normalized vote score
* optional local geometry consistency
* final association score
* whether a new instance is created

### 5. Voxel label stabilization

Do not hard overwrite voxel owners.
Implement owner support accumulation and decay:

* support update for selected owner
* decay for competing owners
* owner = argmax support

### 6. Active set

Define active set explicitly as a restricted local candidate set:

* visible instances
* nearby instances
* whole-prior instances
* new-object candidate

### 7. Whole-object

Do not implement `is_whole_object=True/False`.
Represent whole-ness as soft evidence only.

### 8. Local consolidation

If pairwise grouping exists, expose pairwise affinity terms such as:

* adjacency
* depth continuity
* shape plausibility
* whole prior
* background conflict penalty

### 9. Semantic update

Semantic update must happen only after stabilized instance selection.
Use instance-level feature aggregation and text matching explicitly.

---

## What to inspect and refactor

Inspect current code and fix any mixed responsibilities, especially:

* runtime_vis doing too much
* local_pcd replacing the global instance substrate
* active set becoming the only memory
* semantics entering too early
* whole-object treated as hard label

Refactor module boundaries so the final design is clear.

---

## Deliverables

Produce:

1. brief architecture summary of what was wrong and what was fixed
2. refactored code aligned with the roadmap
3. updated module boundaries
4. short README/interface note
5. mathematically explicit scores exposed in code/debug metadata

---

## Final instruction

Do not rewrite everything unnecessarily.

Preserve current runnable parts when possible, but make the final code clearly show:

* global TSDF instance map = true low-level stable backbone
* local per-instance point clouds = maintenance memory only
* active set = local reasoning tool only
* semantics begins only after stable instances exist

Also make sure the code reflects an explicit approximation of:

* geometry-based proposal refinement
* spatial-voting instance assignment
* support-based instance stabilization
* restricted active-set local reasoning
* soft whole-evidence
* post-stabilization object-level semantic aggregation

## Minimal mathematical constraints

Implement the following quantities explicitly in code and expose them in metadata/debug outputs.

### 1. Proposal geometric features

For each refined proposal, compute:

* `depth_valid_ratio`
* `depth_variance`
* `border_touch_ratio`
* `planar_fit_residual`

### 2. Proposal soft scores

Compute explicit soft scores:

* `objectness_score`
* `backgroundness_score`
* `attachedness_score`

They should be explicit weighted functions of geometric features, with configurable weights.

### 3. Patch-to-instance association

Patch association must expose:

* voxel owner vote counts
* normalized vote score
* optional local geometry consistency score
* final association score
* whether a new instance is created

### 4. Voxel label stabilization

Do not hard overwrite voxel owners.
Maintain per-voxel owner support and update it by accumulation/decay.
Final owner must come from max support.

### 5. Whole-object handling

Do not use hard `is_whole_object` labels.
Maintain only soft quantities such as:

* `whole_evidence_score`
* `part_evidence_score`
* `assignment_score`

### 6. Local consolidation

If pairwise grouping exists, expose explicit affinity terms such as:

* adjacency
* depth continuity
* merged shape plausibility
* whole prior
* background conflict penalty

### 7. Semantic update

Semantic update must happen only after instance stabilization.
Store instance-level semantic features and update them with weighted aggregation across selected views.
