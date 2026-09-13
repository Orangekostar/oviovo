# R6 independent extra-3D-pretraining track

Inputs are the same 200 Replica Room0 RGB-D frames, integer-pixel pinhole rays and
camera-to-world poses used by native OVI. Fuse all valid positive finite measured
depth samples at 1 cm, averaging XYZ and measured RGB by sample count. No GT mesh,
semantic IDs or GT-selected point subset enters inference. Preserve source hashes,
frame IDs, voxel size and support counts. Unknown OVI owners are not removed.

Use released SpaCeFormerInstSeg, not its semantic backbone. Release checkpoint
`chrischoy/SpaCeFormer` revision a8e81555ca9c668a5e2134da520eb76176d4d4d9 has
official MD5 0ec6a123b422053ed3988fbf442aeaa1 (verified locally). The selected
WarpConvNet wheel is 1.8.2+torch2.5cu124, build commit e3ab140bea35; installed
source and wheel hashes define execution identity, rather than the separately
inspected current source checkout. Read checkpoint missing/unexpected keys;
unexplained learned parameter mismatch blocks inference.

Match release center shift (XY bounding-box center, minimum Z), RGB /127.5 -1,
internal voxel size .02, Q=200, hidden=512, three decoder iterations, no TTA.
Output mask rows index `backbone_pc.coordinates`, NOT the raw input point order.
Restore the input center shift before strict <5 cm nearest-neighbor evaluation;
save output coordinates and row-aligned masks together. Fix random seed 0, and
report repeatability rather than assuming eval mode disables all random ordering.

Use google/siglip2-so400m-patch14-224, 1152-dimensional text, official prompt
ensemble and normalize_input=False. Keep this space distinct from native SigLIP
and R5 CLIP. Release NMS=.7, minimum mask points=20, objectness threshold=0,
mask logits >0; DBSCAN and stability disabled. Save raw objectness, mask and
embedding outputs plus processed instance scores, with the released score rule.

T0 is independent SpaCeFormer inference. T1 must combine OVI and SpaCeFormer
proposals on a common observed coordinate domain with explicit source provenance,
cross-space feature separation and a GT-free overlap rule decided before metrics.
Never call backbone-only output T0 or silently normalize the released head.
Select T1 fusion policy only after inspecting actual T0 interfaces, before reading
T0 GT metrics. Keep extra 3D training separate from B0/S1 and dense tracks.

Training corpus sources reported by the authors: ScanNet, ScanNet++, ARKitScenes,
Matterport3D. Exact training-scene exclusion has not been independently verified;
do not claim unseen-ScanNet zero-shot generalization. Official benchmark numbers
are source metadata, not results on this native RGB-D input.

Primary sources checked 2026-09-13:
- https://nvlabs.github.io/SpaCeFormer/
- https://huggingface.co/chrischoy/SpaCeFormer
- https://github.com/NVlabs/WarpConvNet/tree/main/warpconvnet/models/spaceformer

Execution gates: aligned RGB-D fusion tests; real complete-checkpoint load;
native-point-cloud forward with aligned returned coordinates; matching text and
postprocessing; independent T0/T1 evaluations and costs. No gate is complete merely
because source or weights were downloaded.

## Full-input runtime and T1 policy locked before T0 GT evaluation

The observed cloud contains 1,862,429 points. Complete FP32 inference exhausted
A40 memory in decoder projection (7.11 GiB additional allocation); BF16 alone
exhausted it while materializing attention softmax (11.10 GiB). Preserve both
failed attempts. Test `need_weights=False` on decoder MultiheadAttention: the
release consumes only tuple element 0, so attention weights are unused. This
allows the stock PyTorch SDPA path; retain all points, masks and learned weights.
Report BF16/SDPA as a runtime adaptation, not exact FP32 implementation parity.

T1 common domain is returned T0 world-coordinate points. Map native OVI owners
onto it by strict <.05 m 1NN, with 0 unknown. OVI proposals use S1a fixed-K=8
labels and query provenance. Retain unknown geometry in the common point cloud.
For each T0 proposal, reuse an OVI S1a label only if its best eligible OVI proposal
has IoU >=.5; otherwise retain its own released SigLIP2 label. This is geometric
label reuse, not averaging incompatible embeddings. Keep all source IDs and IoU
witnesses. No new image encoder queries are made for T1.

Build a proposal union with OVI proposals first (descending common-domain area,
owner-ID tie break), then T0 proposals in released score order. Apply class-agnostic
greedy NMS at .7 to this fixed ordering. No GT selects proposals, labels or NMS.
Evaluate raw overlapping proposals directly. Report T0 released confidence and
T0/T1 common area-confidence controls separately; never imply cross-model raw
confidence is calibrated. This conservative union is one explicit T1 experiment,
not a claim to be the authors' published fusion algorithm.

Full-input follow-up: omitting attention weights alone also exhausted memory.
Query-axis chunks of 16, with every key/value retained and each query's original
mask sliced consistently, passed masked-output equivalence tests and full inference.
Both BF16 (142 instances) and FP32 (141 instances) now run. Select **FP32 chunk16**
as T0/T1 input before any GT evaluation: it retains released forward precision;
the runtime difference is query chunking and unused-weight omission. Do not claim
BF16 and FP32 are interchangeable. Full FP32 repeatability is checked separately.
