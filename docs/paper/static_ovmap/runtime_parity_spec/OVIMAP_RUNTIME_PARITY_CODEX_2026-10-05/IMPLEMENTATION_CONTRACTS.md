# Implementation contracts: faster execution, unchanged scientific outputs

These are new task contracts. Paths in SOURCE_EVIDENCE identify inspected code;
interfaces below are to implement. The reference is the server's actual measured
v2, not a newly imagined area-fallback algorithm.

## C1. Reference and fast path interfaces

Expose a callable conceptually equivalent to:

```python
recover(inputs, *, arm, implementation, model_session, run_context) -> RecoveryRecord
```

`inputs` includes immutable common raw/painted geometry, capture metadata/paths,
existing N/Q/F and vocabulary, and evaluation projection. It must NOT include
preselected G1/G3 views, recovery masks/features, an already-built raycast scene,
or reference outputs. The same common inputs are given to both implementations.
`U2_CONTROL` may read the genuine archived Native-selected request definitions
inherited by the U2 method, but must recompute its required FC features in cold
runs. This input asymmetry is algorithmic and must be disclosed: U2 has an archived
observation policy, G1/G3 perform a new full-sequence visibility search.

Separate `validate_bound_reference()` from old task orchestration. The new adapter
must verify the old binding as a consumed immutable object and its own task binding
as a different object. It must not make the new worktree pretend to be the old
freeze's repository. No blanket removal of `require_frozen_execution` or mutation
of the source freeze is allowed. Thin wrapper changes may be shared by R0 and fast;
scientific producers and reference numerical operations must remain identifiable.

Verify loaded producer module paths against the bound reference/fast namespace once
at setup. Do not swap sys.path mid-call and accidentally execute a hybrid of the two.
Return compact source records and full prediction files before the timed callable
returns. Ground-truth semantic/instance labels may be read only by the post-call evaluator
or diagnostic routine, never for view selection, classification or optimization
selection. The inherited target-coordinate projection for final export/ranking is
an evaluation adapter, not a permitted source of semantic or visibility evidence. Publishing a path pointing only to an old
prediction is not a live cold run.

## C2. Scientific invariants versus provenance

A legacy request ID includes the projector source hash. Faster code therefore often
produces a different ID even when the selected evidence is byte-identical. Do not
forge the old source hash or require literal ID equality to certify same evidence.
Use two separate notions:

1. **Execution identity**: actual implementation code, flags, hardware configuration,
   adapter version and current output paths.
2. **Scientific evidence key**: scene/raw owner, final geometry, frame ID, K/pose,
   RGB/depth content, full-resolution binary mask digest/shape, both bbox conventions,
   visible pixel count, selection rank, effective preprocessing/encoding/pooling
   semantics, and ordered vocabulary. It excludes producer path/hash and run root.

Build a bijective map for selected per-view keys; compare attempted, successful,
failed and fallback sets after mapping. Compare ordered G1 and G3 requests per owner,
not only unordered sets. Preserve which arm has no views. Never infer a missing
vector from classification scores or attach features by equal integer owner alone.

Exact fields: fixed geometry arrays, raw/painted ownership, residual candidate
supports, selected mask bytes, frame/order/counts/bboxes, normal/fallback path
choice, source availability, semantic argmax, output owner and semantic arrays,
current-class area ranks including serialized rank strings, and matcher context.
Floating evidence: actual FP32 unit vectors and class-score arrays, atol=rtol=1e-5;
record max absolute error and norm. Exact final labels remain mandatory. A new source
hash is not an error; a changed mask or argmax is, even if the aggregate AP is equal.

Old global ray counts, raw positive hits and noncandidate-owner images are NOT
prediction outputs in this entrypoint. Sparse-query diagnostics must explicitly use
`diagnostic_scope='queried_pixels'`. Global hits outside queried pixels are null, not
zero. Keep all candidate depth-consistent visible area/bbox/admissibility statistics
exact. Original all-image owner arrays must not be silently returned under their
old field name with uncaptured pixels mislabeled as measured background.

## C3. Candidate geometry and numerical ray contract

Use the same full FP32 predicted mesh and nondegenerate face order as v2. A face is
assigned a positive owner only if all three raw vertex owners agree; all other
faces remain occluders. No ground-truth mesh. Do not alter ray first-hit semantics,
positive-z threshold or depth tolerance. Keep `cast_rays` CPU thread count four and
maximum batch size 65,536. Count BVH construction inside every G1/G3 call.

Construct rays with the original integer pixel convention:

    q = [u, v, 1] @ inv(K).T
    origin = pose[:3,3] (converted exactly as reference)
    direction = q @ pose[:3,:3].T (then the same FP32 conversion)

Do not normalize directions. `t_hit` is interpreted through the inherited camera-z
construction. Prepare K inverse/camera checks once per frame, but preserve operation
order. Validate ray bytes on pilot samples including image/batch edges. If a
vectorized rearrangement changes relevant ray values or parity, keep the original
ray constructor. ROI sampling must not quietly change intrinsics to a cropped image.

### Conservative candidate ROI

For each raw candidate use a bounding volume of its entire reference geometry,
not only the eventual residual output vertices. Build bounds within the measured
call. Project all eight corners when the box is safely in front of the camera.
Use outward rounding and at least a two-pixel guard, based on the actual FP32 mesh
and origins. Bound/handle ray rounding and the parent pose's non-exact orthogonality:
transposing a nearly orthogonal matrix is not automatically its exact inverse.
If conservativeness cannot be justified for a pose/bound or the box intersects the
near plane, the camera is inside it, or any numerical quantity is nonfinite or
ill-conditioned, use the entire frame. Do not infer a safe finite box by discarding
corners behind the camera. The guard is a numerical precaution, not a proof against
all floating-point error; acceptance still requires real scientific parity.

Rasterize a union of candidate rectangles into sorted original linear pixel IDs;
query each required pixel once, in batches no larger than the original cap. Invalid
measured-depth pixels can be skipped only if they could never pass the inherited
depth test; their diagnostic status is “not queried,” not “no mesh hit.” Keep full
scene geometry in the BVH: a noncandidate object must be able to occlude a candidate.
A bounds-proven empty candidate union can skip ray work. A scene with zero candidates
can return an empty recovery without building a BVH; it still has real positive
registry/output overhead and belongs in timing means.

For every candidate/view compare exact valid mask area and bbox with the baseline.
Selected mask bytes and Top-3 ranks must match across all 26 scenes. Check near-plane,
image-edge, partial occlusion, mixed-owner faces and depth-threshold cases with small
production fixtures. A general ambiguity may invoke the documented geometry-based
full-frame fallback; never add a list of troublesome scene/owner IDs. If exact parity
cannot be achieved, reject ROI globally and retain R1; do not lower tolerance until
reference mismatches disappear. This is empirical same-output acceleration over the
verified suite, not a general mathematical equivalence theorem.

## C4. Grouped owner statistics and lazy exact Top-3

Instead of `for owner: owners == owner` over the whole image for every raw owner,
compute candidate pixel counts and bbox bounds in a grouped pass (compact IDs,
`unique`/segmented operations, not a huge dense array indexed by arbitrary owner ID).
Count hit/valid owner support separately if required. Retain all actual candidate
counts/eligibility for parity. Bbox extent and minimum-area tests are unchanged.

The old code creates and hashes/packs every eligible full-resolution mask and then
keeps only three. Maintain at most three retained entries per owner. Before full mask
materialization compare the cheap leading rank terms (-pixel_count, frame_id) with
the current worst item. Only a strictly worse leading pair may be discarded without
computing its mask digest. Equal leading terms require the actual canonical mask
hash and the full legacy tie rule. Do not choose a smaller hash proxy. Unique frame
IDs usually make the tie rare; uniqueness must be validated, not assumed.

Selected masks still have their original full image shape, bool dtype and packing
order. Do not replace them with cropped masks at the FC boundary. Keep G1 as the
exact first member of the fixed Top-3 list; G3 aggregates only those preselected
successful views with the original areas/order. Do not use class scores to select
views or replace a selected failed view.

## C5. Input reuse and neural execution

Call-local immutable reuse is permitted: parsed capture/manifest; geometry already
provided to the callable; one RGB decode/conversion per physical content identity;
per-frame camera grid; per-frame dense feature used for all selected masks; and
verification memo for unchanged common inputs. It is not permission to preload
these recovery-specific products before timing or carry them across arms/repeats.

Check that an image key actually names identical bytes, preprocessing and dtype.
Do not remove effective input checks; avoid doing the same successful check many
times. The inherited ConsumptionIndex already memoizes file checks; first profile
actual duplicate work. Keep any large v2 fallback intermediates released per frame.

The original `image_tensor()` is not 320x320 despite the model name. Do not change
its 800/1333 resize, RGB layout, normalization, padding or arithmetic precision.
Do not change `signed_mask`/area-fallback selection, normal-mask pooling, text or
visual projection. Preserve the exact fallback module as a separate pinned callable.
The 175 historically normal vectors must not be rerouted through the fallback merely
for batching convenience. Count these sets using actual input identities.

R3 may keep the CUDA allocator's freed storage, but must release references to
image/dense/region tensors at call end. No `torch.cuda.empty_cache()` per image in
fast execution unless a documented real OOM recovery needs it; an OOM is not an
accuracy fallback and cannot silently drop a request. Peak reserved as well as
allocated memory must be reported. No extra background worker or GPU is allowed.

Keep encoder image batch size one and the original per-region numerical operations.
Collect already-computed region vectors into one transfer per frame when safe; do
not silently substitute a matrix reduction with different summation order. Input
hashes can be obtained before a round trip only when the exact effective tensor
identity is preserved; otherwise retain the old round trip. This task is not a
precision/kernel benchmark.

## C6. Evidence reuse and versioning

Parent 172 rows/14 pools remain immutable. Accuracy results are reusable only when
output arrays, label order, ranks and evaluation identity have been checked. Reuse
is recorded as `PREDICTION_IDENTICAL_PARENT_METRIC`, not as newly measured accuracy.
CF18 verification using old FC features is labeled `CPU_VIEW_AND_CACHED_EXPORT_PARITY`.
It does not imply that a new CF18 live GPU benchmark ran. Do not fabricate new speed
for CF18 or the full mapper from Replica incremental recovery times.

Stable scientific keys allow reuse in verification, but the cold loader must have
no access to parent recovery features/results. Use separate adapter/context types
for verification and timing; a run-mode flag that still supplies parent feature
handles is insufficient. After time stops a supervisor can compare outputs.

## C7. Minimal production checks

Use one focused suite covering actual touched call sites: (1) no-change reference
binding, (2) exact camera/rays, (3) occluders and uncertain-bound fallback, (4) lazy
Top-3/ties, (5) repeated-frame load reuse, (6) normal/fallback region identity,
(7) scientific versus producer IDs, (8) fixed-support/A5 export, (9) cold isolation
and measured repeats, (10) table units/source linkage. These are coverage themes,
not a requirement to manufacture ten or eighty named tests. No broad security test,
whole-history byte audit or unrelated dynamic/robotics suite is needed.
