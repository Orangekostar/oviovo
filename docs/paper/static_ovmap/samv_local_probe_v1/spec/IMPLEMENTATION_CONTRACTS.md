# Implementation contracts — precise proposed pilot rules

The constants below are **prespecified experimental design choices**, not previously
optimized values or properties claimed by SAM-V. Existing code facts are separately
listed in SOURCE_EVIDENCE.md. No observed AP may alter these rules mid-study.

## A. Bound data and identifier domains

Use original source vertex rows r=0..M−1, unchanged xyz, mesh triangles and TSDF.
Let O0(r) and L0(r) be actual G1 owner and semantic arrays. Let OD2 be the actual D2
owner array. An incumbent is an owner present in D2; a recovered target is a positive
G1 owner not present in D2. Only owners whose G1 label belongs to the frozen valid
vocabulary may be queried. This probe resegments existing output objects, not every
object in the scene and not all 175 CF residual candidates.

Bind all values through actual parent documents. Distinguish source-row index,
owner ID, frame ID, index within a six-frame window, and class-array index. Never
use a class index as a dataset class ID. Source-row −1 means invalid, never the last
NumPy row. Do not replace the original target nearest/matched projection.

Use a new narrow binding wrapper. `source_preserving_update.binding.load_scene`
can be reused with an in-memory parent binding whose new memo output root is
redirected to this task. Do not execute the parent's bind/all/freeze routines.
Keep original source documents sealed; store resolved paths separately. Load only
the four chosen scenes. The old `units` view is not the new target universe; derive
all positive G1/D2 owners from the actual arrays.

## B. Reference observations

Select the same maximum-32 pose representatives as `pose_banks`: original frame
order; first representative within 0.20 m AND 15 degrees; otherwise new bin;
if needed retain floor(k*(M−1)/(32−1)) representative indices. Pose bins are a
sampling device, not statistically independent observations.

For each retained frame, SourceRowProjector casts against the complete original
triangle mesh with camera-ray z=1, integer pixel coordinates, batch<=65536 and
four threads. A hit is valid iff it is finite/positive and the captured FP32 metric
depth is finite/positive and |hit_z−depth| <= max(.02 m,.02*depth). Choose the hit
triangle vertex with greatest barycentric weight; exact ties take the smallest
source-row index. This is the inherited observer convention; keep it fixed.

Exact old observation arrays can be reused after verifying mesh/camera/depth/rule
identity. The panoptic field is optional here: this pilot does not use its labels
for prompting or accepting a correction. Model masks do not enter the BVH. Save
source_rows and valid; derive an old target mask as valid & (O0[source_rows]==i),
indexing only valid positions. Do not silently swap in older owner-majority masks.

## C. Target inventory, anchor and points

For every positive valid-class G1 owner compute old masks on representative frames.
A qualified old view has >=100 foreground pixels and bbox width and height >=2.
A target requires at least two such views and a promptable anchor. Build the complete
eligible inventory before selecting any target. Do not look at diagnostics/GT or
new-model logits here.

Anchor = greatest old-mask pixel area; ties use smaller original frame ID.
On its old mask, find the largest 8-connected foreground component; tied components
use smallest flattened pixel index. Pad one false pixel around it before Euclidean
distance transform. Interior candidates have distance >=2 pixels from background.
If none, mark UNPROMPTABLE and exclude before model inference.
First point maximizes distance transform, with (y,x) lexicographic tie-break.
Subsequent points maximize minimum Euclidean pixel distance to previously selected
points within that same interior set, ties by (y,x). Pick up to three DISTINCT points.
If only one/two exist, use those; do not duplicate or borrow from another component.
All labels=1. No negatives, boxes, masks-as-prompts, text prompts or corrective clicks.
Record the source vertex row of every prompt using the bound anchor observation.

Select nominally four incumbents ordered by increasing D2 top-2 probability margin,
then integer owner ID; probabilities unavailable -> not eligible for this pool.
Select nominally four recovered G1 owners ordered by decreasing maximum visible
old-mask area, then owner ID. Fill missing quota from the other eligible pool in
its original order, without duplicates, up to eight. Query execution order uses
integer owner ID. No additional object replaces a failed SAM/SAM-V target.
A zero-target scene yields unchanged outputs and a truthful no-op study result.

## D. Six-frame window and canonical pixels

For each selected target, choose a mandatory second frame among qualified old views
other than anchor: greatest d(anchor,frame), tie smaller frame ID, where
  d(a,b)=||ta−tb||/.20 + angle(Ra^T Rb)/15 degrees.
This pair is also the fixed semantic pair. Fill up to six DISTINCT frames by greedy
maximum of minimum d to already chosen frames from all retained representatives,
not just target-visible frames. Ties use smaller frame ID. Preserve selected absent
context frames. Finally sort the window by original frame ID and compute the true
anchor/second indices. No view score from either model is used.

If fewer than six representatives exist, that scene's query planning is a dependency
shortfall; do not pad with duplicates. Only the global 4-frame OOM resource profile
can lower the requested length. Both mandatory frames are retained; the remaining
frames use the same greedy rule. All four scenes use the frozen length.

For each raw captured RGB: RGB uint8 -> float32 0..255 -> bilinear 1024x1024
(align_corners=False) -> clip and truncate to uint8 -> JPEG quality=95, default
subsampling, written once to a bound canonical window directory. Decode that exact
JPEG into RGB for SAM-V. SAM2's loader reads the same JPEG files numbered
00000.jpg..00005.jpg in chronological order. Verify a decoded-pixel identity on the
real pilot (do not accept differently recompressed inputs). Keep original camera
and measured depth untouched; canonical images are model inputs only.

Point transformation is (x',y')=(x*1024/W_original, y*1024/H_original), using original
pixel integer coordinates and the upstream 1024-square convention. This is not
ResizeLongestSide. Record both coordinate systems. SAM-V then internally prepares
its 896-square VGGT input as written by the pinned model. Do not inject measured
poses/depth into its neural feature inputs.

SAM-V outputs low_res_logits of shape [1,1,h,w*N] for this single-mask profile.
Assert width divisible by N, split into N individual h×w tiles FIRST, bilinear
interpolate each to 1024×1024, then to that view's original H×W, and threshold logits
>=0. Do not interpolate a panorama across seams; do not assume forward returns
original-resolution `masks` when visualize=False. For SAM2, retrieve each target's
logits from forward then reverse propagation, assert frame coverage, restore from
1024 square to original H/W in the same way and threshold >=0.

Keep a successful zero mask as a model outcome. Anchor adherence is checked in
1024 coordinates before inverse resizing at rint(x'),rint(y') clipped to the image.
ALL positive points must be foreground. A failed target produces no structural
proposal and no paired semantic update, but its raw masks remain in diagnostics.
This criterion is the same for SAM2/SAM-V. It is not a post-hoc mask selection search.

## E. Deterministic local update domain

All domains are frozen from O0 before segmentor inference.
For target i define Bi = axis-aligned bounds of its entire O0 support expanded by
0.30 m along each dimension. This is an intervention scope, not a claim that an
entire true object necessarily lies within the band.

Build undirected mesh edges. Boundary vertices have an incident edge to a DIFFERENT
O0 owner (including owner 0). The boundary band includes these vertices and two
within-owner edge-expansion hops. Unselected-owner vertices outside this band are
protected interior. Do not infer extra mesh topology when faces are absent: a missing
required mesh is a dependency failure, not a kNN-based replacement protocol.

A source row can be edited only if inside at least one selected target's Bi and it
is (i) owned by a selected target, (ii) currently unowned, or (iii) in an unselected
owner's boundary band. All other rows are unchanged. Prompt source rows are always
forced to their O0 target owner after simultaneous arbitration. They cannot collide
because original G1 supports are exclusive; conflicting assignments are an input bug.

Unlike the previous whole-unit protocol, current raw owner=0 is NOT a universal
freeze: an observed actual surface row in this new edit domain can receive ownership.
The immutable items are xyz/faces/TSDF and out-of-domain ownership, not every prior
label. Report all old-raw-zero edits explicitly. Do not mix these results with the
old protected-raw-zero method as if protocols were identical.

## F. Visibility-normalized model evidence

For target i and source row r, in each of its window views use valid ray pixels
assigned to r. Let a_iv(r) be the fraction of those pixels inside the raw binary
model mask. That view contributes one visible observation (regardless of pixel
multiplicity), and one foreground vote iff a_iv(r)>=.5. A view with no valid rays
assigned to r is unknown and contributes neither positive nor negative evidence.

n_i(r)=number of visible views; k_i(r)=number of foreground votes; q_i=k_i/n_i.
Do not pool raw pixel counts across views, which would overweight a close view.
Do not infer positive support merely from an old owner or from model confidence.

An addition/reassignment proposal for i requires r inside Bi, n_i>=2,
and q_i>=2/3. A deletion from original selected owner i requires r inside Bi,
n_i>=2 and q_i<=1/3. Mixed or insufficient evidence preserves old ownership.
Do not require predicted mask to remain inside the old owner; that would reproduce
the fixed-fragment ceiling. No minimum-IoU-to-old-mask gate is added.

Save raw voted support AND admitted support after domain/core constraints. Report
how many raw proposed rows were disallowed; otherwise a restrictive adapter could
incorrectly be described as SAM-V having no useful support.

## G. One simultaneous partition, not independent overlapping exports

For each editable, non-prompt row collect all qualified addition candidates.
- Unique highest q -> its owner wins (compare ratios by integer cross products).
- Exact tie for best q -> retain O0 for that row, even when O0 is not a contender.
- No positive contender -> if O0 is a selected active target with qualified negative
  evidence, set owner 0; otherwise retain O0.
- Unselected protected interiors and all out-of-domain rows always retain O0.
- Enforce prompt-owner rows last, then verify all invariants.

No sequential target application, class preference, GT choice or predicted-IoU
ranking. A successful target with no anchor support is inactive and cannot alter
or delete any row. Unobserved parts remain unchanged. No new IDs are invented;
selected target IDs are preserved, though unselected boundary-only donors may
vanish. Count disappearances and fragmentation, rather than silently pruning them.

SV01 and SV02 use exactly this builder with SAM2 or SAM-V raw masks. All retained
owners keep their G1 class; unowned rows use class0. Nonselected owners may lose only
admitted boundary rows and otherwise keep their class. Record moved source rows,
source/recipient IDs, added/deleted area, protected-core preservation and prompt
retention. A non-change is a real KEEP, not evidence that the method was never run.

## H. Paired semantic readout (one additional control beyond the original sketch)

The same two mandatory frames are used for old-mask and SAM-V-mask FC, never selected
again by predicted mask area/FC score. Compute OLD=visible O0==target; NEW=SAM-V mask
& valid measured-depth support & pixels whose hit source row lies in Bi.
The semantic NEW region is not clipped to old owner membership. Its source rows do
not decide structure; segmentation and lifting are already fixed.

Use original full-resolution RGB (not canonical JPEG) for BOTH FC sources, the
inherited FC image preprocessing/model/text, signed_mask, and
`cvpr_compact.area_fallback.region_vector`. Save exact mask/RGB/camera/model/text
and input tensor identities. AnyUp, new prompts, N/Q inference and new temperature
fits are absent. Reuse exact FC dense arrays only with validated model/pixel identity.

Each source requires >=100 original valid pixels and both bbox dimensions>=2 in
both fixed frames. The common semantic update domain additionally requires valid
FC vectors in BOTH frames for BOTH sources plus SAM-V anchor adherence. Compute and
retain each source's raw availability even when its partner fails; don't delete the
object from overall evaluation. An empty/invalid representation is a KEEP for both
semantic arms, while missing files/crashed workers are execution blocks.

For each source average its two full-vocabulary cosine vectors in float64 with equal
view weights. Let c=argmax average, c0=old G1 class. Update to c iff c!=c0 and
average[c]−average[c0]>=.01; otherwise keep c0. Ties in argmax use frozen class order.
No new temperature is fitted and SAM-V never supplies text labels. Apply each final
label uniformly to its complete owner support, including raw-zero-painted rows.
This new semantic experiment does not inherit the old class-change cancellation;
both paired arms have the same permission and its historical results are not copied.

SV03 uses OLD decisions on G1, SV04 uses NEW decisions on G1. SV05 uses EXACT SV02
owners and EXACT SV04 target class map. For a nonselected owner keep its original
class. If a donor vanishes, it has no output mask. Do not re-read enlarged/shrunken
SV05 masks, invent a class for unseen objects, or use a different semantic map for
mIoU. These are future experiments, not hidden branches of this pilot.

## I. Interface records and cache correctness

Required persistent records (names may be implemented as equivalent schema fields):
- binding: source commit, parent store, four scenes, actual asset/code/config hashes;
- query_plan per target: pool/rank, owner/class, old support hash, frame IDs/poses,
  prompt coordinates/source rows, canonical RGB hashes, mandatory semantic pair,
  domain/protection hashes, effective global window length;
- segmentation per model/target: checkpoint and ordered-window identity, masks/logits,
  anchor check, forward/reverse frame coverage, actual calls/cost and failure status;
- lifted per target: visible/positive counts or compact sparse equivalents, raw
  support, admitted proposals, blocked edits; one global partition per arm;
- readout: OLD/NEW masks, FC scores, exact joint-success status and class proposals;
- prediction: actual exclusive owners/classes/ranks, scientific content identity;
- evaluation: exact baseline/subset identities, masks/traces/confusion/per-class AP.

A request key includes all source dimensions and model preprocessing. A SAM-V joint
context depends on every ordered RGB, not just the prompted image. Changed queries
invalidate masks, lifting, FC regions, predictions and scores downstream. Do not
rehash immutable multi-GB assets per target. Do not write new files in old cache roots.
