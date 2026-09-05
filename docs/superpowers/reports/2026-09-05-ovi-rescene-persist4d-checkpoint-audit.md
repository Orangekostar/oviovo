# Persist4D ReScene Checkpoint Audit

Date: 2026-09-05
Status: `C0_C1_PASS_C2_BLOCKED`
Base commit: `2136865993033e35c44dac12444363a4788e8452`

## Checkpoint Identity

The canonical local artifact was found at
`/home/ww/paper5/checkpoints/rescene4d_concerto_t2_repro.ckpt`. It is a direct
regular non-symlink file with 754,917,862 bytes and SHA-256
`85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e`.
Seven additional paths are hard links to the same inode; the canonical path is
selected lexicographically from the non-worktree root and no copy is made.

The artifact is classified as `SOURCE_BOUND_RESCENE_REPRODUCTION`, not an
official ReScene checkpoint and not paper parity. Its evidence tree is
`Orangekostar/Persist4D@1380c4b9f37bec7933126ccc9bd70067de166f6f`, with a
clean working tree. The binding report is
`artifacts/P2_G2_REPRODUCTION_REPORT.md` (9,790 bytes, SHA-256
`d891fb7fd53306d8ab65db81b9bb85f08664a9689de850ac7836143b238816bc`).
It records 154/154 supervised 3RScan validation sequences, t-mAP 27.939,
t-REC 40.849, and overall mAP 36.314. The paper target t-mAP is 34.800.

The exact Concerto initialization exists at
`/home/ww/.cache/persist4d/concerto/concerto_base.pth`, has 433,987,358
bytes, and matches SHA-256
`845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`.
The reproduction binds revision
`c31f993a56129f2ba9c5d06a35957e3f05bff710`.

## Training And Runtime Sources

The official upstream base is
`GradientSpaces/rescene4d@fb2fe42eb8f1e926567c48eea9acb874e608ee10`.
The pristine checkout at `/home/ww/oviovo_references/evaluation/rescene4d` is
clean and every required file matches its pinned Git object.

Persist4D training is not unchanged upstream. Its bound differences include
the published 2/5/2 segmentation objective weighting, contrastive diagnostic
deduplication, Hydra contrastive override-order repair, fail-closed dataset and
sequence validation, DDP safety checks, full-state resume validation, and the
later exact memory-bounded contrastive implementation. These training changes
do not by themselves establish inference compatibility.

## C0 Checkpoint Structure

CPU deserialization reports a Lightning checkpoint at epoch 404 and global
step 26,730. The state dictionary contains 798 tensors and 135,149,676 tensor
elements: 796 `model.*` entries and two `criterion.*` entries. The saved
configuration resolves to Concerto, D=4, T=2, 100 non-parametric queries,
2 cm input voxels, segment training enabled, temporal masking disabled, RGB
enabled, raw coordinates enabled, and the Pointcept normal path enabled by the
collator. The machine-readable result is
`configs/evaluation/results/ovi_rescene_b5_identity/checkpoint_audit_summary.json`.

## C1 Strict Topology

The exact Persist4D evidence runtime consumes all 798 checkpoint keys with
`strict=True`, with no missing or unexpected entries. A pristine-upstream
model-only audit also consumes all 798 keys after replacing only the unrelated,
parameter-free metric object that requires an absent evaluator YAML. No model
key is dropped, renamed, ignored, or reinitialized. The correct classification
is `MODEL_TOPOLOGY_STRICT_COMPATIBLE`, not `FULL_LIGHTNING_RESTORE_PASS`.

The upstream `models/rescene.py` does not expose Persist4D's optional
`return_query_features` output, but B5 needs `pred_masks` and `pred_logits`,
which are present upstream. Therefore the preferred inference source remains
the pristine upstream commit; the complete checkpoint removes any need to
load Concerto initialization weights after model construction and strict
restore, although the construction-time initialization identity remains
bound and verified.

## C2 Input And Token Contract

One real native T=2 witness, `scene0003_00-scene0003_03`, produces 37,987
Pointcept tokens from 54,541 raw points. Its processed coordinate, grid, and
feature shapes are `[37987,5]`, `[37987,3]`, and `[37987,9]`. The model feature
channels are shared-centered XYZ, raw source camera RGB in `[0,1]`, and unit
source geometry normals. The dataset-normalized color tensor is constructed but
discarded by the pinned Pointcept collator; no RIO/ImageNet normalization enters
the actual model feature. Temporal values are exactly 0 and 1, the true sequence
batch is zero, and Pointcept stage batches contain 19,134 and 18,853 tokens.

The frozen Apartment OVI PLYs contain geometry normals, but their RGB columns
are the instance palette proven by the color-to-instance logs. The frozen B0/B2
snapshot/entity representation retains neither per-point source camera RGB nor
per-point source geometry normals, and the existing loader discards the PLY
normals. Palette RGB cannot substitute for camera RGB.

The source-correct adapter-first 2 cm inventory uses one shared pair center.
t0 contains 14,563,575 source points, 552,612 adapter tokens, and 446,153
would-be native model tokens; native spatial resampling would merge 106,459
tokens, including 2,973 cross-entity merges. t1 contains 12,327,444 source
points, 490,416 adapter tokens, and 408,276 would-be model tokens; it would
merge 82,140 tokens, including 2,069 cross-entity merges. Drop and duplication
counts are zero, but this is not a permutation and therefore cannot restore the
existing CSR entity provenance without a versioned reverse mapping.

C2 is blocked by `COLOR_NORMALIZATION_MISMATCH`,
`MISSING_SOURCE_GEOMETRY_NORMALS`, and
`BLOCKED_RESCENE_TOKEN_CONSERVATION`. This is a bridge compatibility result,
not a ReScene method-quality result. No GPU command was authorized or run.

## Current OVI Boundary

The existing backend policy already allows
`official_or_source_bound_training_only`; it was not weakened. The repo has a
fail-closed subprocess boundary and runner, but no real checkpoint-forward
executor. Because C2 failed, the conditional executor, B4 reconstruction, B5
projection, relation comparison, current-map work, learned B7, and Office were
not run.

Compatibility gates are: C0 exact checkpoint structure, C1 strict model tensor
consumption, C2 feature/color/normal/coordinate/temporal/batch/grid contract,
token conservation or a versioned reverse map, and only then one C3 frozen
Apartment GPU pair.

Compact C0/C1 and C2 receipts are under
`configs/evaluation/results/ovi_rescene_b5_identity/`. External evidence is
rooted at
`/home/ww/oviovo_baseline_runs/20260905_ovi_rescene_b5_identity/`; its Apartment
pair receipt has 7,249 bytes and SHA-256
`2fc579636d02acfa1fa026e9bfd0fa6483770a14e13e19372046481b86a646d2`.

The audit used no GPU, did not rerun OVI, and did not run any current map or B7.
