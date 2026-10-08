# Implementation contracts — minimal instance repair

All constants are in PROTOCOL_SPEC.json. These are prespecified experimental
choices, not facts learned from GT and not claims of guaranteed improvement.
Implement the rules below without a silent tuning grid. Reuse mature primitives;
the new method is local **output partition repair** on a fixed surface, not a new
TSDF integrator or an implementation of MaskClustering/MV3DIS.

## C1. Three different supports and one unchanged reference

Let X be the fixed source vertex array and F its triangle table. Let r be the raw
owner array, b the Native-painted/D2 owner array, and g the G1-v2 output owner array.
The incumbent set is I={j>0 appearing in b}, not all owners appearing in g.
For j in I, P_j={x:b(x)=j}; let R_i={x:r(x)=i} for a raw owner. For the inherited
omitted-owner candidates C, K_i={x:r(x)=i and b(x)=0}. Keep the inherited minimum
100 source rows and candidate cap 128. Revalidate the actual registry; do not
reconstruct it by arbitrary colored-mesh decoding.

Every new arm starts with the exact G1-v2 payload (IR01), except IR00 which imports
D2. Thus an untouched G1 recovered object stays exactly as it was; an unavailable
G1 object stays absent unless it is incorporated by an explicit accepted edit.
Candidate selection uses no evaluation-space point count, GT, current TP list,
class correctness, or per-dataset switch.

Two unit types are allowed:
- incumbent unit I:j uses P_j as a whole;
- residual unit C:i uses K_i as a whole, whether or not G1 could classify it.

This first study does **not** split a raw owner, transfer an incumbent source row,
merge two incumbents, modify r==0 rows, add/delete vertices, smooth a mesh or create
unobserved surface. These are deliberate scope limits. The R_i diagnostic may show
opportunities requiring forbidden transfers; report them as outside this study,
not as successes that these arms can achieve.

For deterministic priorities and edit cost, assign each source vertex one third
of the physical area of each incident nondegenerate triangle; area(U) is the sum
of those weights over U. Isolated vertices have zero weight. If all triangle area
is zero, binding is invalid. Size thresholds remain the stated source-row or
image-pixel thresholds; do not confuse these units with physical area.

A unit identity includes the same mesh identity and hash of its sorted source
rows. Numerical owner IDs are identifiers, not sufficient feature identities.
Select at most 64 residual repair seeds by (-area(K), -row_count(K), support_hash).
All candidates outside this processing cap retain their original G1 outcome and
remain in final evaluation.

## C2. Prediction-side observation table

### Frame choice and bank assignment
1. Read all original completed frames in ascending original frame_id order.
2. Greedily assign each frame to the first prior representative satisfying both:
   Euclidean camera translation distance <=0.20 m, and rotation geodesic angle
   <=15 degrees. Angle=acos(clip((trace(R_a.T R_b)-1)/2,-1,1)). Otherwise create a
   new bin with this earliest frame as representative. The representative never
   changes; distance is to the representative, not transitive chaining.
3. Sort representatives by frame ID. If M>32, use indices
   floor(k*(M-1)/(32-1)), k=0,...,31. Otherwise retain all M.
4. Even retained-position indices are proposal views; odd are verification views.
   Each bin appears in exactly one bank. One representative per bin is used.
   Do not swap a view based on how many masks, errors, or TP it later reveals.

At most 32 observer frames are used per scene. Two views per pair per bank are
required where specified; insufficient evidence yields a valid KEEP, not a new
view search. Bins are pose-diverse but not statistically independent; the full
parent geometry itself used these frames. State this in the paper.

### Full-scene occlusion and source membership
Use the fixed complete predicted mesh, including faces with noncandidate or
unknown owners. Retain the original nondegenerate-triangle index mapping.
Use parent integer-pixel camera rays with camera-direction z=1 and do not normalize
the ray. Validate K and c2w once per frame. Accept a hit only when t_hit and measured
FP32 depth are finite and >1e-6 and
abs(t_hit-depth)<=max(0.02,0.02*depth). Invalid/missed/occluded pixels are unknown.

For an accepted triangle hit, read primitive_uvs=(u,v) and barycentric weights
(1-u-v,u,v). Associate the pixel with the source vertex having maximum barycentric
weight; an exact tie is resolved by the smallest original source-row index.
Require finite barycentric weights within [-1e-5, 1+1e-5]; truly invalid outputs
are execution errors, not negative object evidence. Retain the original finite
weights for argmax (do not clip/renormalize to introduce a new tie). Batch at most
65,536 rays.

This observer intentionally uses a new label-independent pixel-to-source-row map.
It is NOT claimed to be bit-identical to the parent's homogeneous-raw-face owner
raster. It cannot replace/reinterpret the saved baseline G1 masks. Its purpose is
to query arbitrary K_i, P_j and union supports with the same spatial convention.
Because the mesh/hit is independent of the proposed label assignment, one table
can evaluate every local hypothesis without recasting each mask.

A new mask M_v(U) consists of accepted pixels whose representative source vertex
is in support U. Read the saved pre-insertion 2D integer labels L_v on those pixels.
Use only their within-frame membership, not class names or the final global owner.
Boundary/unknown labels are not competing instances.

### Per-unit evidence and pair votes
A unit's frame is usable when:
- |M_v(U)|>=25;
- at least 25 pixels have L_v>0;
- labeled fraction>=0.60;
- the dominant positive 2D ID covers >=0.65 of its labeled pixels.
Resolve equal positive-ID counts by smallest local ID. Save area, labeled area,
dominant ID and dominance for audit. Do not equate ID numbers between frames.

For a pair (a,b) in a view:
- if either unit is not usable, observation is UNKNOWN and not in the denominator;
- usable units with equal dominant IDs vote SAME=1, SEPARATE=0;
- different dominant IDs vote SEPARATE=1 only if BOTH dominance values>=0.80;
- other usable different-ID pairs are neutral, SAME=SEPARATE=0, but are in the
  comparable-view denominator. They never become positive support by omission.
For each bank define S=mean(SAME), D=mean(SEPARATE) across its comparable views.
Store n. If n=0 return unavailable rates, not zero-valued evidence. Units with no
joint observations are neither same-object nor separate-object evidence.

## C3. Spatial candidate universe and hypothesis generation

The following neighbor list is shared by the attachment controls and proposed
attachments; group generation uses the same geometry and positive evidence.

For each selected residual seed K_i, consider:
- up to 3 incumbents, sorted by exact minimum vertex-to-vertex distance then unit
  identity, when that distance<=0.08 m OR |R_i intersect P_j|/|R_i|>=0.10;
- up to 8 residual neighbors from the inherited candidate set whose exact minimum
  vertex-to-vertex distance<=0.08 m, sorted by distance then identity.
Use AABBs for conservative culling and cached per-support KD trees for exact
minimum distances, not centroid distance pretending to be surface distance. A
raw-overlap candidate is a hypothesis only, not proof of correct attachment.
The neighbor lists define undirected edges by union of either direction. A
non-seed residual may join a hypothesis but may not expand its own neighborhood.

Generate a common group library using proposal-bank SAME support only. An edge is
positive when proposal n>=2 and S>=0.60. For each seed:
1. sort its eligible positive neighbors by (-S, distance, unit_identity);
2. start a group with the seed; consider neighbors once in that order;
3. add a neighbor only when every pair in the enlarged group has a positive edge,
   group size<=4 and the group contains at most one incumbent;
4. emit every accepted prefix of size>=2. A rejected neighbor does not terminate
   the scan; a group already at size four is done.
Also emit each positive seed-incumbent pair, even if a previous group prefix used
a residual neighbor first. Deduplicate by sorted unit identities. There is no
transitive connected-component shortcut that merges A/B/C without A–C evidence.

For a group H, S_P and D_P are equal means of its per-pair proposal-bank rates;
each pair must be comparable at least twice. The edit cost is label-invariant:
- group with incumbent j: area(all added residual units)/area(H);
- residual-only group: 1-max_i area(K_i)/area(H).
If area(H)==0, the hypothesis is ineligible. Define
Q_P(H)=S_P(H)-2*D_P(H)-0.10*EditCost(H).
Sort the complete library by (-Q_P, EditCost, hypothesis_content_digest), then keep
at most 96. Proposal generation does not reject on D or Q; they order a common
library. It uses no verification views or semantic scores.

Fresh IDs for residual-only unions: assign one stable library-wide lexicographic
index by hypothesis content digest. ID=max(all raw, b, g positive IDs)+1+index.
This assignment is shared by all arms, even if some hypotheses are not applied.
Do not recycle a historical raw ID for a different new support. Groups containing
an incumbent retain that incumbent ID. Record old-to-new membership for analysis,
without mutating the original native alias table.

## C4. Exact structural arms and arbitration

### IR02_NEAREST_ATTACH
For each seed with an eligible incumbent neighbor select the closest listed one,
without using 2D or semantic scores. Sort proposed pair attachments by (distance,
seed_identity, host_identity). Apply a single greedy pass of at most 8 pair
attachments, skipping conflicts. No fallback to second-nearest after a conflict.
A host may occur in only one applied operation in this pass.

### IR03_EVIDENCE_ATTACH
Use the SAME seed/host universe. A pair is eligible when proposal n>=2, S>=0.60,
D<=0.20 and Q_P>=0.05. Each seed proposes its highest Q_P host, then ties by distance
and identity. Sort all pairs by (-Q_P, EditCost, digest) and greedily apply at most
8 disjoint-unit operations. Verification-bank data do not affect this arm.

### IR04_DIRECT_GROUP
Read the common group library from C3. In proposal priority order, directly apply
at most 8 disjoint-unit groups. No new separation/held-view threshold is added to
this control. All it knows for membership is positive proposal consensus and the
shared size/neighbor limits. The Q_P ordering is common to IR05.

### IR05_VERIFIED_REPAIR
Read the IDENTICAL library and priority. For every group, all its pairs need
verification n>=2. Compute S_V and D_V as equal pair means. Accept only if
S_V>=0.60, D_V<=0.20 and
Q_V=S_V-2*D_V-0.10*EditCost>=0.05.
Greedily apply at most 8 disjoint-unit accepted groups in original proposal order.
Do not use verification scores to rebuild, enlarge or reorder the library. Record
KEEP_INSUFFICIENT_VERIFICATION or KEEP_VERIFICATION_REJECTED explicitly. This tests
the combination of extra observation evidence and minimal-edit criterion against
IR04, not a proof that each individual term is necessary by itself.

### Classification availability and shared work
Lock the provisional structural decisions for both arms before reading new class
scores. For residual-only groups selected by either arm, choose exactly ONE view
among the fixed observer frames by (-visible_pixels, frame_id, mask_digest), with
>=100 pixels and bbox extent>=2 on each axis. Use the full union observation mask,
not only one member's old G1 mask. Compute original FC v2 pooling and visual head
against the unchanged text prototypes; argmax yields one class.

Attachments inherit the incumbent D2 class and require no new region classification.
For a residual-only group with no qualified observation or deterministic invalid
region representation, cancel that selected operation and preserve the parent
partition. Do not fill the vacated budget by choosing another hypothesis. Such a
cancellation is a real measured KEEP; a missing asset/execution fault is a block.
IR04 and IR05 share classification for every identical group, regardless of which
arm requested it first. Classification cannot alter structural priority/acceptance.
There are at most 16 selected union groups per scene across these two arms before
deduplication. There is no need to encode every library hypothesis.

### Application and invariants
Start from a copy of g and G1 semantic labels. An operation assigns all source rows
of its residual units to its retained host/fresh union ID. Former constituent G1
IDs disappear only where their entire K support is consumed. No affected residual
unit participates twice. All untouched source rows are bit-identical to the parent.
All original P_j rows retain their owner identity; structural-only arms also retain
their D2 semantic class. Attachment enlarges a host but never steals another host's
rows. r==0 is unchanged. No overlapping final masks, no union plus constituents.
Apply operations once; newly created groups are not fed back into this pass.

## C5. Semantic rereading of original incumbents

### Candidate selection and views
Compute exact parent D2 probabilities using fuse_readout and its recorded source/
temperature identities. Select at most 16 original incumbents with finite D2
probabilities, by smallest top1-top2 probability gap, then support digest. Do not
use the evaluator's IoU, GT category, known errors, or a recovery trigger inherited
from EV05. Other incumbents and ALL original G1 recoveries remain unchanged in
IR06/IR07. This is a new incumbent experiment, not relabeling a historical result.

For each selected P_j, take the largest-area qualified full mask among proposal
observer frames and the largest among verification observer frames (>=100 pixels,
bbox>=2); at most two views. Break ties by frame ID then mask digest. A missing bank
is recorded; IR06 may use one available full view, IR07 requires two. No additional
frame search beyond the 32-frame observer set, no class-driven view choice.

### Recognition masks on each selected frame
- FULL: M_v(P_j).
- CORE: erode FULL with an all-one square kernel of radius
  clip(ceil(0.01*sqrt(|FULL|)),1,8), constant-zero boundary. CORE needs >=100 pixels
  and >=25% of FULL. Do not substitute a dilated/empty core.
- FRONTEND_INTERSECTION: intersect FULL with its dominant positive saved 2D mask.
  Choose dominance by pixel count; require intersection>=100 and >=50% of FULL.
  If no positive mask qualifies, this observation type is unavailable.
Deduplicate identical masks within a frame. A FULL duplicate must retain FULL as
its type and cannot count again as an independent core/intersection vote.
Masks on different selected frames remain separate observations.

Only these recognition masks change. The actual 3D P_j support is fixed in both
arms. Additional masks are not alternative outputs from which GT picks a winner.

### Ordinary AnyUp only
For each selected frame, use the original FC image_tensor (800/1333 scaling,
right/bottom padding, original FP32 behavior) to obtain or bind clip_vis_dense.
Prepare the separate ImageNet-normalized AnyUp guidance exactly as in the audited
adapter; no mixing of image normalizations. Output size is padded H/4 by padded W/4.
Use pinned paper AnyUp, original averaged-head attention and raw dense V; disable
new owner/depth factors. Pool each recognition region by the inherited fine-grid
area weights, apply the original ConvNeXt visual prediction head, and normalize.
Text matrix and order are unchanged. Do not interpolate class probabilities.

Use the existing ordinary stream_selected path or an equivalent wrapper which
computes only the required variant. Do not pay for discarded owner_depth and
owner_only outputs in cold timing. Share image/QK/attention chunks within a frame.
A changed support gets a new region key. Check official AnyUp parity once on a
real new-mask fixture if production streaming changes; no full-frame parity loop
for all scenes. A deterministic unavailable mask is omitted with its reason, never
replaced by a different or better-scoring mask.

### IR06_ANYUP_REREAD: the simple control
Let s_v be the full-vector cosine score for each successful FULL view. q is their
EQUAL arithmetic mean (not probability/temperature mixing). c=argmax(q), tied by
original category-array order. Let c0 be the old D2 label. If c==c0, KEEP. Otherwise
change to c only if q(c)-q(c0)>=0.01; zero successful full views means KEEP.
This compares new FC scores for c and c0 in the SAME space. Never subtract a D2
probability from an FC cosine.

### IR07_BOUNDARY_STABLE
Use EXACTLY the same proposed c and full-view q as IR06, not a separately optimized
class. A relabel requires the IR06 condition, two successful full views from the
two banks, at least three content-distinct successful region observations including
at least one non-full region, and all of:
1. both full-view argmax labels equal c;
2. both full-view s_v(c)-s_v(c0)>=0;
3. at least two thirds of the distinct region observations have argmax c;
4. median over all distinct successful region observations of s(c)-s(c0)>=0.01.
Otherwise KEEP the exact old D2 label. Record which tests passed. These tests are
hypotheses about boundary robustness, not calibrated correctness probabilities.
Retain all successful scores, missing types, masks and source identities for audit.

## C6. Fixed combination without hidden extra inference

IR08 starts with the IR05 result. Apply accepted IR07 class changes to retained
ORIGINAL incumbent IDs only, uniformly over their actual current support, which
may now include attached residuals. The evidence support remains the original
P_j anchor and is explicitly recorded. This is a core-anchored semantic policy,
not a claim that an old vector was recomputed on the expanded mask. New residual-only
union IDs are not in the incumbent set and keep their shared union-FC class.
No new candidate set, no re-generation after relabeling, no extra AnyUp pass, and no
per-object choice between IR05 and IR07 using GT. All final ranks are recomputed.

## C7. Output, caching, and comparison identity

Keep GeometryIdentity for xyz/faces/tsdf/projection, and create a separate owner-array
partition digest. New PredictionPayload uses branch COMBO; the old branch G owned-
support equality rule is not appropriate to adding previously unowned residuals.
Do not modify that old invariant to accommodate this new protocol.

For each new payload, validate same source-row count and exact coordinates/faces,
nonnegative integer owners, one class per positive owner, class 0 on owner 0, no
collision of fresh IDs, and exact unchanged rows outside declared edits. Preserve
all original-incumbent owner identities even when semantic rereading changes their
class. Recompute native_ranks and export full owner-to-class decisions.

SceneEvaluator's shared_masks are indexed only by owner number and scene inside its
root. Use root/partitions/<owner_array_digest>/ for each actual partition, build its
mask registry from ALL positive output owners, and pass the current payload as its
native-shaped evidence object. Any administrative N0 unavailable rows exist only
to enumerate masks; they are not invented semantic evidence. Use original
projection/annotation source_manifest entries as load_targets requires. Never open
GT for operation acceptance or region selection.

Two methods can reuse a scoring result only when full source geometry, owner and
semantic arrays, official ranked view, target projection and scorer protocol agree.
A support change under the same owner ID is a DIFFERENT mask. A changed method name
with identical prediction is a result alias, not new neural inference.

## C8. Applicability versus implementation failures

A no-edit result is valid when all required inputs were read and no hypothesis
passed the predefined tests. Report candidate edges, positive edges, candidate
library size, verification eligibility/rejections, classification cancellations,
applied edits and changed rows. Do not merely label all branches COMPLETE without
showing whether the intended interventions happened.

Missing 2D evidence files are not no-edit observations. Corrupt cameras, mismatched
source rows, wrong model/text identities and interrupted inference are failures.
Keep independent finished arms and issue no full-cohort pool for a method missing
a required scene. If a branch is genuinely a no-op in all 26 scenes, publish that
result as such; do not add knobs or call synthetic changes a real experiment.


## C9. Dependency and accounting clarifications

A missing pre-insertion panoptic file blocks the evidence arms that consume it,
not the already bound IR00/IR01 or nearest-only IR02. IR06 needs the completed
RGB-D/camera observer but no frontend intersection. Never replace a missing
required file by unknown pixels; a present all-zero file is genuine unknown data.
IR07's declared observation recipe consumes the frontend file even when it later
fails its intersection-size check. Independently computable rows can finish, but
no partial cohort is substituted for an 18/8-scene pool.

The 832 ceilings refer to successful unique scientific observer/frame work at
most once for each of 26x32 frame identities. A true failed attempt/retry, narrow
pilot validation, and cold replay are separately bounded and separately counted;
do not hide their physical inference or time under the unique-input ceiling.

Prediction metadata must respect the inherited metadata validator. Evaluation
labels, correctness indicators and oracle assignments belong to diagnostic files,
not PredictionPayload metadata. Use operational terms such as edited_rows and
class_transition in prediction records, never labels derived from GT.
