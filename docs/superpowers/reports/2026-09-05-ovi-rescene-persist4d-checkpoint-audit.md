# Persist4D ReScene Checkpoint Audit

Date: 2026-09-05
Status: `PHASE_A_PASS_C2_PENDING`
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
collator. A machine-readable rerun is part of the implementation plan.

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

## Preliminary C2 Findings

One real existing native T=2 witness,
`scene0003_00-scene0003_03`, produces 37,987 Pointcept tokens from 54,541 raw
points. The processed tensor has coordinates `[37987,5]`, grid coordinates
`[37987,3]`, and features `[37987,9]`. The nine channels are centered XYZ,
dataset-normalized RGB, and source normals. Temporal values are exactly 0 and
1, the true OVI-equivalent sequence batch is zero, and the two Pointcept stage
batches contain 19,134 and 18,853 tokens.

The native 3RScan files store RGB in `[0,1]`, then the pinned dataset code casts
it to `uint8` before Albumentations normalization. The bound mean is
`[0.001681035080447036, 0.0015699645182459389, 0.0014388616751178026]` and the
bound standard deviation is
`[0.0010721057171332617, 0.0010743444191366812, 0.001081851490549608]`.
This unusual path must be reproduced exactly; ImageNet normalization or a
second division by 255 is not equivalent.

The frozen Apartment OVI PLY has geometric normals, but its RGB columns are the
global instance palette used by the color-to-instance log. The frozen B0/B2
snapshot records do not carry source camera RGB or per-point normals, and
`load_ovimap_visit()` does not add them. The current adapter correctly rejects
palette RGB. A 2 cm geometry-only inventory yields 564,259 t0 entity tokens and
501,089 t1 entity tokens. Pure spatial Pointcept voxelization would merge 5,659
t0 tokens and 4,556 t1 tokens across entity boundaries (5,404 and 4,552
collision voxels respectively). These are C2 blockers unless a separately
versioned, source-verified feature and reverse-mapping contract is designed.

## Current OVI Boundary

The existing backend policy already allows
`official_or_source_bound_training_only`; it must not be weakened. The repo has
a fail-closed subprocess boundary and runner, but no real checkpoint-forward
executor. The smallest executor would be
`scripts/evaluation/rescene_pair_executor.py`, with preprocessing isolated in
`src/oviv2/rescene_input_bridge.py` only if C2 passes. Executor work is not
permitted while the camera-RGB and token-conservation gates are unresolved.

Compatibility gates are: C0 exact checkpoint structure, C1 strict model tensor
consumption, C2 feature/color/normal/coordinate/temporal/batch/grid contract,
token conservation or a versioned reverse map, and only then one C3 frozen
Apartment GPU pair.

Phase A used no GPU, did not run OVI, and did not run B7.
