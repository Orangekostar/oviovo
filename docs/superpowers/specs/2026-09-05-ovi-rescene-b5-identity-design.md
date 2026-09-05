# OVI-MAP x ReScene Persist4D B5 Identity Design

Date: 2026-09-05
Status: `FROZEN_BEFORE_RESULT_BEARING_B5`
Base commit: `2136865993033e35c44dac12444363a4788e8452`

## Objective And Stop Boundary

This task asks whether one already-trained, source-bound ReScene checkpoint
adds distinct cross-visit OVI entity relation evidence beyond frozen B4. It
does not test a B5 current map, B6, learned B7, dense recovery, Office, or a
3RScan ranking.

The causal order is fixed: checkpoint identity, strict topology, native and
OVI input contracts, token conservation, at most one frozen Apartment learned
pair, existing query-to-entity projection, and B4/B5 topology comparison. A
failed gate stops all later stages and yields `RESCENE_B5_BLOCKED` without a
claim about ReScene method quality.

## Frozen Identity And Authority

The checkpoint is
`/home/ww/paper5/checkpoints/rescene4d_concerto_t2_repro.ckpt`, SHA-256
`85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e`,
classified `SOURCE_BOUND_RESCENE_REPRODUCTION`. Its evidence source is
`Orangekostar/Persist4D@1380c4b9f37bec7933126ccc9bd70067de166f6f`.
The inference architecture is the pristine source-bound
`GradientSpaces/rescene4d@fb2fe42eb8f1e926567c48eea9acb874e608ee10` after
strict model-only topology compatibility passes. Concerto is revision
`c31f993a56129f2ba9c5d06a35957e3f05bff710`, weight SHA-256
`845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`.

OVI owns geometry and open-vocabulary semantics. ReScene or B4 owns only
cross-visit identity hypotheses. Signed t1 visibility owns existence/removal.
No closed-set ReScene class output may replace an OVI label or embedding.

## Input Contract

The adapter neural voxel is frozen at 0.02 m and the required adapter feature
schema is `rgb_normals`. The checkpoint's Pointcept stem receives nine channels:

```text
shared-centered XYZ (3)
+ native-normalized camera RGB (3)
+ unit source geometry normal (3)
```

For adapter RGB `c` already in `[0,1]`, the source-verified color operation is:

```text
u = astype_uint8(c)
normalized = (u / 255 - rio_mean) / rio_std
```

where `rio_mean` and `rio_std` are the exact values bound in the audit. No
ImageNet constants, palette colors, zero normals, or double normalization are
allowed.

All pair coordinates are centered together by
`[(xmin+xmax)/2, (ymin+ymax)/2, zmin]`. Each visit then receives its own
nonnegative 2 cm grid origin. The true sequence batch remains zero in the first
coordinate column, Pointcept batches split visits as 0 and 1, and temporal `t`
remains exactly 0 and 1. Raw centered XYZ, not integer grid coordinates, enters
the coordinate feature channels.

## Frozen OVI Pair

The only eligible pair is Apartment t0 frames 766--1021 and t1 frames
1217--1472, reopened through the frozen two-visit OVI manifest. OVI mapping,
CropFormer, SigLIP, and B0--B7 map outputs cannot be rerun. Pair materialization
is allowed once only after C2 passes, and its arrays remain under
`/home/ww/oviovo_baseline_runs/20260905_ovi_rescene_b5_identity/`.

The instance PLY RGB is an instance palette and is forbidden as a neural
feature. The current frozen VisitMap export has no source camera RGB. This task
does not introduce an unregistered RGB-D-to-mesh coloring algorithm or call an
entity-wide color average a point-level camera observation. Absence of an exact
source-bound camera-RGB tensor is `COLOR_NORMALIZATION_MISMATCH` and blocks B5.

## Token Conservation

The adapter groups by `(visit, spatial voxel, entity_id)`, while native
Pointcept GridSample groups by spatial voxel inside a visit. Before GPU use the
audit must report adapter token count, model-input token count, cross-entity
collision voxels, permutation, merge, drop, and duplication counts.

A pure permutation with zero merge/drop/duplication passes. A second spatial
GridSample is forbidden when it merges entity tokens. Duplicate sparse indices
cannot be passed to spconv as an undocumented shortcut. A future Option A must
define a versioned adapter-to-model inverse map, prove deterministic expansion
back to every adapter token, expose merge ambiguity instead of masquerading as
a permutation, and update tests and provenance. This task avoids that contract
expansion; any nonzero cross-entity merge yields
`BLOCKED_RESCENE_TOKEN_CONSERVATION` before GPU.

## Conditional Executor Semantics

An executor is created only if feature and token gates pass. It must run through
the existing `ReSceneBackend`, validate the exact pair/checkpoint/source/schema,
and retain the existing output manifest and NPZ schemas.

Mask support is `pred_masks > 0`, exactly as `_get_mask_and_scores`. Token
scores are `sigmoid(pred_masks)`. For each raw query, foreground confidence is
the maximum softmax probability after excluding the no-object class; query
confidence is that value multiplied by mean sigmoid support over positive mask
tokens. Duplicate flattened class selections collapse to their source query's
highest foreground class before a unique `query_id` is emitted. These values
are diagnostic identity confidence only.

## Projection And Comparison

If C3 passes, B5 must use the existing `project_queries_to_instances()` with:

```text
minimum_source_point_coverage = 0.25
minimum_entity_token_coverage = 0.25
static_centroid_tolerance_m = 0.10
```

B4 is rebuilt exactly once using the frozen geometric reasoner configuration
and its original geometry-only token builder. B4 is not retokenized with B5
features. The normalized topology key is `(state, sorted t0 entity IDs, sorted
t1 entity IDs)`; identity source and confidence remain attributes.

The comparison reports per-state totals, exact intersection, B4-only, B5-only,
all one-to-one persistent totals, B5-only static/moved one-to-one relations,
and optionally the already frozen registration sanity result for B5-only
one-to-one relations. TESSE has no legal identity GT, so precision, recall,
accuracy, and SOTA claims remain unavailable.

## Verdicts And Budgets

Only these verdicts are legal:

```text
RESCENE_B5_BLOCKED
RESCENE_IDENTITY_NOT_PROVEN_ON_APARTMENT
RESCENE_IDENTITY_SIGNAL_PROMISING
```

At most one real frozen Apartment GPU inference is allowed; the smoke is reused
as the final inference. No training, fine-tuning, seed, threshold, query,
backbone, or voxel sweep is allowed. The full suite runs once after freeze and
at most once more after a targeted repair. Office stays `HELD_OUT` with
`attempt_count=0`. Learned B7 stays `NOT_RUN_BY_SCOPE`. 3RScan is audited only;
ranking requires 44/44 selected visits.
