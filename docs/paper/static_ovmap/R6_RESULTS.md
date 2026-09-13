# R6 Room0 extra-3D-pretrained track

Status: real complete T0/T1 inference and independent evaluation finished. This is
Room0 development evidence, not Replica8/ScanNet18. T0 is weaker than native S1a
in semantic AP; T1 improves semantic AP but has scoring/overlap and repeatability
limitations. Keep it separate from the no-extra-3D-training main track.

## Input and released interfaces

All 200 native RGB-D frames (0:10:1990) contribute 163,197,676 positive finite depth
samples, fused into 1,862,429 observed 1 cm points. Coordinates use integer camera
pixels, poses are camera-to-world, and measured RGB is averaged by sample count.
No GT mesh enters inference. Fusion took 183.997 s, peak RSS 498,253,824 bytes;
the compressed cloud is 38,254,427 bytes. Model voxelization remains its released
2 cm setting; returned masks align to its returned points, restored to world coordinates.

Complete SpaCeFormerInstSeg checkpoint strictly loads 85,801,090 parameters with
zero missing/unexpected keys. Checkpoint MD5 matches the official provenance.
WarpConvNet 1.8.2+torch2.5cu124, Torch 2.5.1+cu124, FlashAttention 2.7.4.post1 and
torch-scatter 2.1.2+pt25cu124 run in a separate venv. Source/checkpoint/binary hashes
and pinned Hugging Face revisions are recorded in artifacts.

SigLIP2-so400m-patch14-224 provides its own 1152-d text space, released prompt
ensemble and normalize_input=False. Do not mix this with native SigLIP-1024 or
R5 CLIP-512. NMS=.7, minimum 20 points, mask logits >0, objectness threshold=0;
DBSCAN/stability/TTA are off. Release foreground/mask-quality times class-probability
scores are preserved for T0.

## Runtime adaptation and repeatability

Unmodified full FP32, BF16, and BF16 with unused attention-weight omission each
exhausted A40 memory in the full decoder. These failed attempts are retained.
Splitting only the query axis into chunks of 16 preserves all keys/values and
query masks; masked-attention output tests pass at atol=1e-6. Learned parameters
and the released decoder body are unchanged; a local runtime wrapper omits unused
attention weights and chunks calls. This is runtime adaptation, not bitwise parity
with the unavailable full unchunked FP32 result.

Full **FP32 chunk16** is the primary run, chosen before GT evaluation. It retains
141 instances; model load 1.121 s, forward 12.134 s, postprocessing 1.218 s, peak
GPU allocation 37,457,695,232 bytes. Text load/encoding costs 4.223 s separately.
BF16 chunk16 ran in 8.493 s with peak 28,935,596,032 bytes and 142 instances; it is
not substituted for FP32. GPU 2 was used while R5 ran on GPU 0; host contention
is possible. These stage timings exclude failed attempts, export/serialization,
GT evaluation and historical OVI front-end costs; they are not end-to-end FPS.

A second independent FP32 invocation with seed 0 retained 143 instances. World
coordinates and text cache match exactly, but 633,226 of 372,485,800 raw mask sign
entries differ; maximum mask-logit difference is 21.6607. Query embedding and
objectness differences are also recorded. The package randomizes orders in eval,
and its CUDA implementation is not established as bitwise deterministic here;
the exact source of between-process variability remains unresolved. Do not present
the two observations as a statistical confidence interval. Keep the first preselected
run and report the repeat, rather than picking by GT score.

## T1 fixed GT-free fusion

The policy in R6_DESIGN.md was recorded before T0 GT metrics. Native OVI owners are
projected onto the same observed domain using strict <5 cm 1NN. 64 eligible S1a
objects have common-domain support (the other native objects are not invented).
T0 proposals reuse an S1a label only when best owner IoU >=.5; features are never
averaged across spaces. OVI-first, then released-T0-score NMS at .7 retains 160 of
205 candidates: all 64 OVI proposals and 96 T0 proposals. 64 T0 candidates reuse
labels; 19 of those survive NMS. No extra image encoder calls are made. Fusion,
including native projection, took 22.530 s (projection 12.894 s).

## Real evaluation

GT domain: the same 954,492 vertices; strict <5 cm 1NN leaves 111,427 unmatched.
The observed RGB-D geometry is **not** frozen B0 geometry. Overlapping proposals
are evaluated directly; collapsing them to a partition would change AP. Canonical
diagnostic AP includes void in prediction area, uses min-region=100, and greedily
matches one-to-one at IoU >= threshold over 84 GT instances. It is not the paper's
unverified class-agnostic AP implementation or released mP/mR.

| Condition / confidence | Semantic instance AP | AP50 | AP25 | Vertex mIoU | Vertex mAcc |
|---|---:|---:|---:|---:|---:|
| B0 native reference | .154316 | .343233 | .368233 | .332857 | .381587 |
| S1a native reference | .186723 | .384900 | .409900 | .362175 | .416451 |
| T0 released score | .140955 | .310082 | .361648 | .283307 | .337078 |
| T0 native-area control | .126564 | .313027 | .375177 | .215938 | .290405 |
| T1 common raw area | .205160 | .411659 | .503194 | .218774 | .272718 |
| T1 within-class normalized area | .205160 | .411659 | .503194 | .351151 | .412459 |

Semantic AP uses the released 48-class instance subset and original GT conversion,
which passes exact native-reference parity. Vertex metrics use all 51 text classes.
Overlapping vertex labels take the highest supplied proposal confidence. Area
normalization does not change within-class AP ordering, but does change cross-class
overlap assignment; both rows are necessary. T1 does not improve every metric.

| Condition / confidence | Canonical AP25 | AP50 | AP75 | Recall50 |
|---|---:|---:|---:|---:|
| T0 released score | .647757 | .486310 | .115099 | .583333 |
| T0 common source area | .329906 | .208637 | .050348 | .583333 |
| T1 common source area | .347198 | .234148 | .067554 | .654762 |
| T0 repeat, released score | .641661 | .478522 | .109782 | .571429 |

T0 repeat semantic AP is .130527 (native-area control .112197), versus .140955
(.126564) in the preselected run. T1 has not yet been repeated or validated across
scenes. Its Room0 semantic AP gain requires extra 3D training and does not establish
a robust accuracy win over the main native track.

Training sources reported by the authors are ScanNet, ScanNet++, ARKitScenes and
Matterport3D. Exact training-scene exclusion is unverified; no unseen-ScanNet
zero-shot claim is made. R7 remains off, and no additional training was performed.

Raw receipts, per-class metrics, proposal ledger, repeat differences and large-file
hashes are under `artifacts/static_ovmap/room0_spaceformer/`. External run directory:
`/home/ww/oviovo_baseline_runs/20260913_static_ovmap/room0/`. Commands are retained in
every receipt. 127 relevant tests pass across the recorded 112-test and 15-test
commands; attention tests also pass in the actual R6 Torch environment.
