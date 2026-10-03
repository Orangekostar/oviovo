# Implementation contracts — compact tables

These definitions are proposed frozen experimental choices except where explicitly
identified as inherited behavior. They are not claims that G1 has already improved
accuracy. Function names under cvpr_compact below are interfaces to IMPLEMENT.

## C0. One native anchor and a narrow new interface

Suggested modules and required responsibilities:

| New module | Reuse / reference | Required responsibility |
|---|---|---|
| binding.py | recovery_wave2.binding ConsumptionIndex/PathResolver | New 26-sequence binding; do not impersonate old task IDs |
| benchmark_inputs.py | assets.native_schedule; scannet_frames.export_sensor | Fixed CF18 acquisition/export, no random family selection |
| anchor.py | BB00_NATIVE capture; semantic_readout; recovery native-worker exhaustion adapter | Reuse/build one map and N/Q/F per sequence |
| projected_views.py | Open3D RaycastingScene; captured camera conventions | Complete-scene visibility and new independent requests |
| region_worker.py | region_adapter; recovery ContentCache | FC readout of new binary masks without old-request dependency |
| native_region_worker.py | VLModel.encode_image_with_bbox; single_native_classifier | Same G1 view under original native six-crop operator |
| outputs.py | PredictionPayload/save_prediction/native_ranks | Fixed-support A0–A5/U2/G3, A5 unknown semantics supported |
| evaluation.py | SceneEvaluator/expanded_native_registry/pool | Exact mask universe, official scores and 14 pools |
| diagnostics.py | released_trace; recovery mechanisms | n/N, actual new TP/FP entries, old losses/ranking changes |
| timing.py | same production standalone recovery callable | 24 feature-cache-cold replays with resident model |
| tables.py | TABLE_CONTRACTS.md | One source of truth -> three tables, CSV, JSON, provenance |
| workflow.py | existing runtime/resource locks | Bounded phase orchestration and restart/reuse |
| publication.py | normal Git operations | Actual release and full-SHA external receipt |

Names may be consolidated into fewer files, but responsibilities and entrypoints
must remain clear in the handoff. Reuse the frozen category order in every array.
Never connect an old owner's feature to a new map solely because the integer ID
coincides. Geometry, registry and original request lineage must match.

## C1. Candidate support and fairness

Let b(v) be the native triangle-painted owner and r(v) the raw numerical owner on
the SAME fixed source mesh rows. Let I be all positive b values. Candidate i must
not be in I; its export support is K_i={v:r(v)=i and b(v)=0}. Require |K_i|>=100
source rows, then order by (-|K_i|, spatial_support_digest), cap at 128 per scene.
Reuse recovery_registry.build_registry unchanged when its input contracts match.
A1's D2 and A0's Native have exactly the same I and b.

The cap applies before GT or visual scores, identically to every recovery condition.
Table 3 N=sum per-scene candidate counts before observation availability. No active
object boundary completion: if i is already in I, its missing boundary pixels are
not a recovery candidate in this experiment. Do not split K_i into independent
extra instances. Keep all original raw owners and all mesh faces for occlusion,
even when they are ineligible for recovery.

A0/A1 never change b. A2/A3/A4/U2/G3 preserve b on b>0 and append only K_i for a
successful classification. A2/A3/A5 share exactly the same successful G1_FC set.
Technical availability differences against A4/U2 are reported, not concealed by
only evaluating the intersection. The small common-success diagnostics reuse
existing scores and do not count as an extra method grid.

## C2. Full-scene geometric visibility (not target-only splatting)

Inputs: the final PREDICTED source mesh (X, triangles, r), each aligned captured
RGB image I_t, measured `depth_m` D_t, K_t and camera-to-world pose [R_t,t_t].
These are the 200-slot native capture inputs, not target evaluation meshes or
preprocessed GT vertex labels. Use completed frames only. Final geometry projected
into historical cameras is OFFLINE processing; do not label it causal online.

Build a CPU Open3D tensor raycasting acceleration structure once per scene in
scientific execution. Preserve original triangle ordering in a mapping to primitive
IDs. Every geometrically nondegenerate triangle is an occluder. A face receives
positive owner i only if its three raw vertex owners are the same positive i;
a mixed/zero face has owner0 but still blocks surfaces behind it. Do not erase
unknown/other-owner triangles, regenerate a mesh or change geometry normals.
Invalid indices/nonfinite geometry are input defects. Zero-area triangles can be
excluded from the raycaster with their exact original indices recorded; the saved
prediction geometry is never modified.

At integer image coordinates (u,v), construct:
q=inv(K_t)@[u,v,1], o=t_t, d=R_t@q; submit [o,d] in float32.
Do NOT normalize d. With this convention X_hit=o+t_hit*d and t_hit equals camera-z
(the third camera coordinate of q is1). This avoids confusing Euclidean ray range
with measured optical-axis depth. If using another ray helper, prove equivalent
rays rather than assuming its half-pixel/extrinsic convention matches.

At each pixel accept:
- a finite hit with t_hit>1e-6 and positive homogeneous face owner;
- finite D_t>1e-6;
- abs(t_hit-D_t)<=max(0.02,0.02*D_t).
The constants are fixed DESIGN choices for this wave, not estimated best values.
A front occluder prevents a farther candidate from receiving that pixel, even if
the occluder has owner0. Invalid/missing measured depth gives no evidence, not a
hallucinated successful projection. No dilation, connected-component completion,
SAM calls or hidden 2D ground-truth masks.

For candidate i, M_it is all accepted pixels with hit owner i. Use the whole
visible raw owner, not only the exported residual K_i. Require >=100 pixels and
half-open bbox width,height >=2. Record per-stage counts (frustum/hit owner/depth-
consistent/support eligible), so a zero-view result is distinguishable from a
camera convention bug. Process at most 65,536 rays per CPU batch, four threads.
Chunking must not change visibility or owner decisions.

## C3. G1/G3 view choice and new request identity

For every candidate build candidate views over all completed scheduled frames,
including frames that contain NO native semantic request for that owner. Do not
call reconcile_requests to decide G visibility. Sort admissible views by
(-visible_mask_pixels, original_frame_id, mask_digest). G1=first view, G3=first
up-to-three distinct frames. No semantic entropy, confidence or GT is used for
choice. Do not replace a failed selected view. G3 may aggregate the remaining
successful members of its preselected list; report exactly which members failed.

Store a new `ProjectedRegionRequest` with:
request_id, schema, scene, final_geometry_digest, raw_owner, frame_id, K/pose
identities, aligned_RGB/depth identities, mask digest/path, canonical bbox,
legacy-native bbox, visible pixels, selection rank, projector-version/config.
Its source is the final map snapshot, not a fictitious historical insertion state.
Store pixel masks compactly and share masks across A2/A3/A4/A5/G3; different models
must have different encoded-evidence keys.

The new loader must read these new masks and the actual captured frame bytes.
The legacy `_load_request` can only read real legacy request arrays. Do not create
an old RegionRequest with false `native_selected_request_ids` or paid-Q receipts.
Raw geometry IDs used for a projected request need no artificial segment lineage.

## C4. Bounding boxes and the native comparison

Internal new bbox is half-open [xmin,ymin,xmax+1,ymax+1]. The old native capture
actually stores pixel maxima and calls a crop function that slices with exclusive
upper indices and clips to W-1/H-1. Preserve this historical operator for A4:
legacy_native_box=[xmin,ymin,xmax,ymax], union_mask=M_it, and the original image.
Record both boxes. Do not silently fix the old last-pixel behavior, and do not
make cache identities collide between different crop conventions.

This yields the same selected scene/frame/target-mask evidence for A3/A4, but
not an artificial assertion that SigLIP crops and FC dense pooling consume
identical pixels/tensors. That representation difference is the intended control.
The FC mask is resized by nearest-neighbor, represented as +/-1, padded with -1,
then downsampled inside the inherited pooling operator. A 0/1 substitute changes
the `>0` support and is not permitted.

## C5. Region encoding and failures

FC: same effective `convnext_large_d_320` frozen checkpoint, operators, FP32,
800-short/1333-max resize, 32 padding and original fourteen text templates.
Preserve unit per-view vectors and visible-pixel weighting followed by final L2
normalization. A new source vector is a cosine vector in its own text space,
not a probability mistakenly softmaxed a second time. Recovery labels are argmax
of this full frozen vocabulary. No temperature fit is needed for a single recovery
source. Existing E remains exactly the frozen D2. Bind the final N/Q/F scalar vector used
by archived Replica D2 and use it for both main cohorts; do not substitute CAL
leave-one-out scalars or fit dataset-specific replacements.

Native recovery: original same-frame six-crop image encoding, stored vector and
`single_native_classifier` FP32 canonical-relative readout. Its label is genuine
native output; do not introduce a learned cross-encoder calibration.

Region evidence key must include actual image tensor/model/operator identity,
target mask, resize/crop convention and text identity for scores. Dense tensors
can share image identity across masks; pooled vectors cannot. A selected mask
that disappears at dense resolution becomes EMPTY_DENSE_MASK_SUPPORT, not an
invitation to dilate it or secretly use a different view. A worker failing every
visual request is a technical block, not a scientific zero-recovery conclusion.

A5: old owners with unavailable F retain owner support but semantic0. Existing
F scores already computed for A1 are reused; no native fallback. This is explicitly
a matched-evidence FC-only control, not a claim to test every possible all-FC
view-selection design. Its standalone dependency cost still includes the process
that constructed the common native-painted support; do not assert an end-to-end
FC-only speedup by omitting that shared prerequisite.

## C6. Exact predicted outputs and metric interface

Use PredictionPayload(branch='COMBO'), which permits nonnegative semantics and one
label per positive owner; unowned rows must remain class0. A5 unknown positive
owners require zero rank and no positive instance line under the official export.
Do not delete their target points from semantic evaluation. Extend the output
wrapper in the new module instead of weakening legacy recovered-label validation.

For each new owner partition, construct a corresponding evaluation context:
source-row geometry/projection unchanged, full positive owner dictionary includes
recovered and unknown owners, N availability correctly false for new owners.
`SceneEvaluator` currently creates masks from that dictionary; using the unexpanded
N dictionary would remove the exact effect this experiment is intended to measure.

Recompute official ranks in target space with >=100 target points, same predicted
class maximum area denominator, .6f serialization. The source row cap/threshold
must not be confused with this target test. Keep all target labels/points used by
the released protocol, including errors and unknown output.

The released np.arange vector has ordinary floating-point representation noise.
Check its nine values against the specification within1e-12 and retain its exact
loaded representation in the receipt; never rewrite evaluator thresholds.

Store complete typed scene metrics in fractions, GT-present-class confusion,
matching traces and actual scoring-context hash. Pool the full ordered cohort.
Expected unique internal pools: six on each of the two cohorts plus U2 and G3 on
Replica =14. The same pool populates every occurrence of a table cell.

## C7. Table 3 mechanism measurements

N is candidate registry size summed over8. n is actual source-available appended
owners, including owners too small after target projection; disclose the separate
scoring-eligible count in SI. Ratios use summed counts, not average scene ratios.
TP50/FP50 are added scoring entries identified from actual released matching trace.
An ambiguous tie is stored as ambiguous and not arbitrarily attributed; show a
flag in the table if it affects a displayed number. Old lost matches and old rank
changes remain in SI so "append-only" is not misreported as guaranteed harm-free.

U2 vsG1 changes the source of the view/mask; only A3 vsA4 is the same new view with
a different encoder. U2 can have fewer available objects. No GT-purity filtering
is allowed in either output. GT-majority summaries are post-lock diagnosis only.

## C8. Exact cold-feature-cache runtime definition

Table3 uses one production standalone recovery run per scene/arm, maximum24.
Start with the base anchor and model weights/text resident. Persistent dense,
pooled-vector, new-view/projection and classification result caches are disabled.
The isolated task may share one image encoding among regions within the same arm.
Measure the whole recovery callable including candidate/view work and output,
with CUDA synchronization. Rebuild the geometric projection/BVH in the G runs;
U2 performs its real archive lookup. Keep model loading and common mapping out
of this incremental column but report them separately.

Report mean=sum(all8scene measured seconds)/8. An empty-candidate scene runs and
contributes its real eligibility overhead. Do not count blank scenes as0 unless
the measured operation really is an excluded no-op reference. The no-recovery
row is0 by definition. No run on a second GPU architecture mixed into the mean.
Timing replay uses the same fixed selections/decisions; compare parity and record
FP32 numerical tolerance, but labels and exported support must match. A technical
fix requires regenerating affected scientific outputs, not using two definitions.

## C9. Scope, exposure and inference accounting

There are 172 main fixed scene-method records, at most8 additional development
smoke outputs and24 timing replays. Timing replays are computation, not independent
scientific trials. At most26 new successful main native maps if none can be reused;
no new development map and no geometry-variant grid. CropFormer/N/Q/FC image calls
needed for missing main anchors are authorized and counted, not called zeroGPU.

Do not inspect new target labels while building/picking G views. Preserve old
calibration and Q training support lists; record physical-family overlap with
CF18 without secretly dropping benchmark captures. Replica remains exposed. Do
not call CF18 a full ScanNet200 validation split, eighteen independent rooms, or
untouched until exposure evidence supports that narrow claim.

## C10. Completion is multidimensional

Use separate fields:
implementation_status; main_scene_outputs (complete/172); internal_pools
(complete/14); cold_timing_leaves(complete/24); benchmark_coverage;
external_provenance_status; performance_outcome; publication_status.

Technical no-view is a valid complete result with no additions. Missing data,
missing evidence files, inaccessible model or scorer errors are blocked leaves.
Do not write a global COMPLETE if CF18 is only partially scored. Publish the
implemented and measured work even when a real external prerequisite remains.

Final comparison names A3 and simple baselines are fixed. Report five raw metrics
and all paired deltas. A positive point estimate is not a statistical guarantee;
parameter bands are not equivalence tests. Do not retrofit the method definition
to whichever benchmark favors it.
