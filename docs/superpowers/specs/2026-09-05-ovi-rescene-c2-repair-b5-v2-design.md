# OVI-MAP x ReScene C2 Repair and Apartment B5 Design

Date: 2026-09-05

Status: `FROZEN_BEFORE_C2_V2_IMPLEMENTATION`

Base commit: `f3cb93a9c49df3891279b67330e58401c20cc540`

## Objective and authority boundary

This change repairs the input contract for one source-bound ReScene inference
over the frozen Apartment t0/t1 OVI-MAP pair. OVI remains authoritative for
geometry and open-vocabulary semantics. ReScene supplies only cross-visit
identity hypotheses, while the existing signed-visibility path remains the
only removal authority.

The run does not retrain, remap OVI, alter B3 removal, rerun the B0--B7 map
matrix, evaluate Office, expand beyond T=2, or use TESSE labels/change masks.
The frozen snapshots and prior V1 failure receipt remain unchanged.

## Frozen inputs

The only pair is Apartment t0 global frames 766--1021 and t1 global frames
1217--1472. In each materialized visit, native frame IDs 0--255 map in order to
those global intervals. RGB, depth, pose, and camera intrinsics are read only
from the same materialized visit.

The following identities are parents of every V2 artifact:

| Input | t0 | t1 |
|---|---|---|
| VisitMap content SHA-256 | `cdd46a4a2b6ff82252dba22ae7b15bf3cc35ef46123a98649ba8c48f59cf33af` | `aca00e36aa6925e8a24f0f365123cc7922742df2c682a5a3da0d96a22cde88b1` |
| instance PLY SHA-256 | `e7f35ab5676c110ca72d45a2251dd53850b7b324e473b3377eb4ed6870ab790d` | `878f73ce7d6e1d44f668b6e9ed47eee442c4cd6c206169c620d4e0c9b8e0a5ae` |
| instance log SHA-256 | `0adacd7738afa0695beb7c28c33c1628440d4f3e3c232ed44fcfb197f9732b29` | `ca7d29f1c7bd79bebb9d6565463081d264003fd210c1f7ea97b18c45f9805dae` |
| native manifest SHA-256 | `e0141a2356ca719ae2bdad3061f2442671486d4f12e8abd05ebfecdb85602ea3` | `6cbfc318c25df085cb677a039ac1ca527b17cdba55b7d9fb70478a8871eddca2` |

The two-visit manifest SHA-256 is
`1996a0008649cdeb58daa920a05d30366a971b2fef77ae2e6a4ae1ca8a3f7b94`.
The complete source RGB-D export manifest is
`ee35d1843540fc88ae20b5dae5ad34caf192d50f0cab7c5564c47c454d4d5b81`.
Per-frame materialized manifests bind all 512 RGB and 512 depth files, plus
camera, trajectory, and timestamps. Their complete records are referenced by
the frozen configuration rather than duplicated here.

The model is the source-bound reproduction checkpoint
`85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e`
with Concerto initialization
`845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`.
ReScene is fixed at upstream commit
`fb2fe42eb8f1e926567c48eea9acb874e608ee10`; its native sampler source
`datasets/pointcept_utils.py` has SHA-256
`962686891547c0719670774d35917562c7f4b7c90aaad1b7002ccda676112a83`.
The installed Sonata checkout is fixed at
`18c09ff8d713494f78a8213792262b910977a65d`; its actual GridSample source
`sonata/transform.py` has SHA-256
`cf799a0c2d4d835ee722ef6300b5c1ffc401b12a129bed783c081a6cce23c615`.

## Data domains and immutable sidecars

Four domains remain explicit:

```text
D: source PLY vertices belonging to OVI entities
A: entity-aware 2 cm adapter tokens
M: native ReScene spatial tokens
Q: raw ReScene queries
```

`D -> A` is the existing source-membership CSR. `A -> M` is the actual native
GridSample inverse `g`, and `Q x M -> Q x A` is exact indexing by `g`. No stage
may infer a mapping from nearby coordinates after the fact.

New data is stored outside the frozen VisitMap as
`OVI_SURFACE_ATTRIBUTES_V2` and `OVI_RESCENE_MODEL_INPUT_V2` sidecars. Each
sidecar binds its parent snapshot, PLY, log, materialized RGB-D manifest,
producer source, configuration, arrays, byte counts, and SHA-256 digests. The
builder asserts the parent VisitMap content SHA before and after construction.
Large arrays stay under the external run root; Git contains compact manifests
and summaries only.

## Exact PLY alignment and normals

The PLY is read once per visit. The reader extracts XYZ, palette RGB,
`normal_x/y/z`, and original row index, then applies the same RGB code and
stable argsort used by `group_points_by_color()`. For every mapped entity, the
grouped float32 XYZ array must be exactly equal, including order, to the
frozen `entity.points_xyz`. Duplicate XYZ rows remain distinct through their
original PLY indices. Any mismatch blocks that visit; nearest-neighbor repair
is forbidden.

Normals are finite nonzero PLY normals normalized in their original coordinate
frame. Invalid normals are marked unsupported and never replaced by zeros or
invented directions. PLY RGB remains an instance palette used only for entity
binding and is rejected as a model color source.

## Native A-to-M sampling

Adapter coordinates and `D -> A` membership use the existing deterministic
entity-aware adapter grouping at 0.02 m. The pair shares one center

```text
[(xmin+xmax)/2, (ymin+ymax)/2, zmin]
```

computed across both visits. Each visit is then passed separately through the
pinned environment's actual Sonata `GridSample` transform with the upstream
settings:

```text
grid_size=0.02, hash_type=fnv, mode=train,
return_grid_coord=true, return_inverse=true
```

`mode=train` matches the checkpoint validation collation path. NumPy RNG state
is saved, seeded to 45 for the call, and restored afterward. The native output
provides selected A indices, per-visit grid coordinates, and inverse. Visit
offsets are applied once to form global `g`. The bridge records sampler module
path/hash/version and rejects a changed source identity.

The V2 gate requires integer `g` of length N, `0 <= g < M`, complete A
coverage, no visit crossing, no dropped or duplicated A row, and exact replay.
Within-entity and cross-entity many-to-one groups are legal diagnostics, not
automatic failures.

## Real surface representative policy

Every model token must be backed by one real D vertex. Candidate vertices are
drawn only from D members of the A contributors to that M token and must:

1. have a valid source normal;
2. quantize to the same visit-local native M grid coordinate after the shared
   center and native per-visit origin;
3. obtain a same-visit, depth-consistent camera RGB observation.

Candidate order is deterministic: the native selected A contributor first,
then remaining A contributors by adapter index; within an A token, squared
distance to its centroid followed by original PLY index. At most eight unique
candidates per M token are attempted. The first supported candidate becomes
the representative. Failure to find one for any M token makes the complete
input scope fail closed; an unsupported subset is diagnostic only.

The model coordinate feature is the chosen real D vertex minus the shared
center. The sparse grid coordinate is the already measured M coordinate, and
the builder asserts the representative quantizes back to that same coordinate.
Camera RGB and unit normal come from the same D vertex. Every A row receives
the representative M feature through `g`; this is recorded as shared voxel
support, not independent per-A color evidence.

## Same-visit RGB-D recovery

For world point `p`, the projection is

```text
p_camera = inverse(T_camera_to_world) * p
u = fx * x/z + cx
v = fy * y/z + cy
```

Projection uses camera z, NumPy nearest-integer `rint` pixel selection as in the
existing visibility implementation, depth in metres (`uint16 millimetres /
1000`), and the aligned RGB pixel. A support is valid only when z is positive,
the pixel is in bounds, depth is finite and positive, the absolute depth
residual is within the frozen tolerance, and the valid 3x3 depth neighbourhood
range is no greater than twice that tolerance. Missing neighbours do not
contribute; fewer than five valid neighbourhood depths rejects the support.
Supports never cross visits.

Before B5, exactly 2048 evenly spaced native M indices per visit (or all M if
fewer) are used for an input-only calibration. For the first valid-normal
candidate of each sampled M, the minimum finite residual over the same-visit
frames is collected without reading GT or model output. The formal tolerance is

```text
ceil_to_1mm(quantile(residual <= 0.08 m, 0.99)), clamped to [0.010, 0.050] m
```

Calibration requires at least 512 finite inlier residuals per visit; otherwise
C2 is blocked. The maximum of the two visit tolerances becomes the single
formal value. `0.08 m` is only a calibration outlier ceiling and is distinct
from the B3 removal tolerance. The value and residual summary are frozen in
one formal config before any model result. No threshold sweep follows.

Recovery streams local frames 0--255 once per candidate round, decodes each
RGB/depth pair once per round, and projects unresolved candidates in NumPy
chunks. For that candidate it retains the support with minimum
`(absolute residual, local frame, v, u)`; the first candidate with any valid
support wins. A support
record stores visit, local/global frame, pixel, source PLY index, D/A/M indices,
camera z, measured depth, residual, and source kind
`same_visit_rgbd_reprojected`.

## Model input and executor

The external model input contains M rows with:

```text
features = [shared-centered source XYZ, camera RGB / 255, unit PLY normal]
coord_bxyzt = [true sequence batch=0, centered floating x/y/z, visit t]
grid_coord = [native visit-local integer grid x/y/z]
Pointcept sparse batch = visit 0 or 1 during the backbone
true sequence batch = 0 for both visits
point2segment = arange(M) as one global identity mapping
```

Identity segments are required because the checkpoint was trained with
`train_on_segments=true`, while OVI entity IDs are reverse-projection metadata
and must not become learned features or segments. Identity segmentation makes
the model output domain exactly M without leaking OVI identity. One segment
cannot span visits.

The source-bound executor composes the exact Hydra configuration, instantiates
only `config.model`, strips the audited `model.` prefix from exactly the 796
model tensors, and loads them with `strict=True`. The two criterion tensors are
validated as the only excluded checkpoint entries; no loss or trainer is
constructed. It verifies checkpoint, Concerto, upstream checkout, input
sidecar, temporary pair arrays, feature schema, voxel size, and parent pair
identity. It runs `eval()` under `torch.inference_mode()` on one CUDA device
and records environment, shapes, runtime, peak GPU allocation/reserved memory,
and RSS.

For each raw query, `pred_masks` is M x Q and `pred_logits` is Q x (C+1).
The executor preserves raw query index. It defines positive mask support as
`mask_logit > 0`, token score as sigmoid(mask logit), foreground confidence as
the maximum softmax probability excluding no-object, and query score as that
confidence times mean positive token score. Empty-support queries are retained
in raw external output but pruned from the backend bridge with a counted reason.
No class top-k row duplicates a query and DBSCAN is disabled.

Model masks and token scores are transposed once to Q x M, then expanded as
`P[:, g]`. The existing backend receives Q x A arrays and
`token_indices=arange(N_adapter)`. This identity order describes only the
external adapter output, never the internal sampler.

## C2-V2 gate and execution budget

`OVI_RESCENE_INPUT_CONTRACT_V2` passes only when PLY alignment is exact,
normal/RGB provenance is legal, every M feature row is supported, all features
are finite with RGB in [0,1] and unit normals, shared centering is exact,
native inverse coverage is complete, and expansion tests pass. Coverage is
reported separately for D, A-independent, A-shared, and M domains.

Only after that gate passes, the frozen code/config commit is recorded and one
complete Apartment forward is run. It is both smoke and result. One targeted
retry is allowed only for a recorded correctness bug or OOM, without changing
model, voxel, query, seed, or thresholds. Office remains held out and no other
GPU job is stopped.

## Relation comparison

If no complete trusted B4 relation artifact exists, B4 is reconstructed once
from the frozen VisitMaps with `build_geometric_pair_sample()`,
`GeometricReasonerConfig()`, and `ProjectionConfig()`, then cached. B5 uses the
unchanged `project_queries_to_instances()` path.

Topology keys are `(state, sorted t0 IDs, sorted t1 IDs)`. The comparison
reports raw and unique relations, state totals, intersection, B4-only,
B5-only, unique one-to-one persistent sets, and B5-only classifications:
threshold-qualified, fallback-only, contradictory, and collision-dominated.
Collision fraction is computed from the M contributors supporting each query;
it is diagnostic and does not rewrite the projection.

At most 20 threshold-qualified, noncontradictory B5-only 1:1 relations receive
the frozen registration diagnostic, ordered by descending query score and IDs
as tie breakers. Registration is plausibility evidence, not identity GT.

## Artifacts, verdicts, and claims

Compact results are written under
`configs/evaluation/results/ovi_rescene_c2_repair_b5_v2/`; large CSR, model
input, raw logits, and masks remain under a bound external run root. Artifact
manifests avoid self-hash loops by recording the evaluated code commit, while
the final upload receipt is external.

Only these final statuses are legal:

```text
C2_REPAIR_BLOCKED
B5_EXECUTION_BLOCKED
B5_EXECUTED_NO_INCREMENTAL_SIGNAL
B5_EXECUTED_SIGNAL_CANDIDATE
```

Every decision records `gt_identity_evidence_available=false`,
`paper_superiority_established=false`, input/projection completeness, GPU
attempt count, B4 reconstruction count, and actual tests. Existing map metrics
are inherited references and are not recomputed or presented as B5 effects.

## Implementation boundaries

Core behavior is isolated in new modules:

```text
src/oviv2/ovi_surface_attributes.py
src/oviv2/rescene_input_bridge.py
scripts/evaluation/prepare_ovi_rescene_input_v2.py
scripts/evaluation/rescene_pair_executor.py
scripts/evaluation/compare_ovi_rescene_relations.py
configs/evaluation/ovi_rescene_c2_repair_b5_v2.json
configs/evaluation/rescene_two_visit_backend_persist4d_repro.json
```

Existing V1 adapter, backend schema, query projector, VisitMap contracts, B3,
B4, and B7 behavior remain unchanged. Tests cover exact PLY alignment,
projection rejection, legal many-to-one mapping, inverse expansion, 9-channel
input, query identity, fallback/contradiction diagnostics, and existing backend
stub integration before any result-bearing execution.
