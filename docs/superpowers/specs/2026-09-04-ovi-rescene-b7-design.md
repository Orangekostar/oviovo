# OVI-MAP x ReScene B7 Dense Recovery Design

Date: 2026-09-04
Status: `FROZEN_BEFORE_B7_G_IMPLEMENTATION`
Base commit: `9db59cb3200e0bc7f48f904c4e9fa628c71c9586`

## Objective And Boundary

B7 tests one mechanism only: whether a conservative cross-visit identity and
rigid transform can recover dense historical OVI surface that t1 did not see,
without reviving geometry that t1 proved free.

```text
B7-G = frozen B4 geometric identity
     + OVI-to-OVI rigid registration
     + frozen signed t1 visibility
     + dense OVI historical completion
```

Authority is immutable: OVI owns geometry and open-vocabulary semantics;
B4/ReScene supplies identity only; signed t1 visibility alone supplies
existence/removal evidence; the evaluator never supplies method input.

B0--B6, the matrix, OVI reconstruction, semantic classification, visibility
thresholds, and evaluator definitions are frozen. B7 is a separate extension
over the exact B3 artifact. A failed or nonproductive extension returns B3
bit-equivalently.

## Data Flow

```text
frozen native/materialized OVI t0,t1
  -> strict read-only VisitMap materialization
  -> frozen B4 geometric PairRelation materialization
  -> relation-scoped rigid registration
  -> transform full-resolution OVI t0 entity
  -> classify transformed candidate voxels with frozen t1 RGB-D visibility
  -> t1 occupied / visible-free rejection
  -> current-compatible occluded/unobserved recovery
  -> replace only that entity's unwarped B3 fallback spans
  -> B7 snapshot + registration/recovery provenance
  -> unchanged common-v2 evaluator and B7 attribution
```

The B0/B2 snapshot receipts provide the frozen semantic label mapping. Loading
the source-bound native mesh is permitted; rerunning CropFormer, OVI mapping,
SigLIP, B0--B6, or the old matrix is not.

## New Module Boundary

`src/oviv2/two_visit_registration.py` owns typed registration configuration,
rigid-transform validation, deterministic point sampling, transform
estimation, quality measurement, fail-closed decisions, and canonical hashes.

`src/oviv2/two_visit_dense_recovery.py` owns typed recovery configuration,
candidate selection, B3 span replacement, provenance, immutable output, and
artifact writing. It does not estimate transforms or derive visibility.

`src/oviv2/two_visit_execution.py` gains a public
`derive_signed_visibility_for_points` entry point implemented by the same
aggregation helper as `derive_signed_visibility`. The existing function and
B3 output must remain equivalent.

`scripts/evaluation/run_ovi_rescene_b7.py` is the only standalone B7 runner.
It validates all source receipts, reconstructs the B4 relation artifact,
performs registration, derives candidate visibility, runs recovery/evaluation,
and writes an atomic result package. It never opens evaluator GT before the
method snapshot is finalized.

`src/evaluation/two_visit_b7_attribution.py` and its CLI report evaluator-only
post-hoc attribution. They cannot alter the map or gate inputs.

## Registration Contract

`RegistrationEvidence` is separate from `PairRelation.evidence` and contains:

- schema/method/config identity;
- relation ID, state, identity source, and exact t0/t1 entity IDs;
- source and target full/sample point counts;
- optional finite 4x4 `world_from_t0` transform and transform SHA-256;
- selected deterministic initialization and iteration count;
- directional inlier counts/ratios and overlap;
- directional median/p90 residuals and centroid residual;
- source/target principal extent ratios;
- accepted flag plus sorted rejection reasons.

An accepted matrix must have homogeneous bottom row `[0,0,0,1]`, finite
entries, orthogonal rotation within `1e-6`, determinant one within `1e-6`, and
no reflection or scale. Rejected evidence has no usable transform.

Only 1-to-1 `persistent_static` and `persistent_moved` relations are eligible.
Both OVI labels must match case-insensitively and belong to the frozen
recoverable set:

```text
Bed, Bin, Books, Chair, Couch, Drawer, Fridge, Lamp, Painting,
Screens, Table, Vase
```

Structural/background/unknown labels, appeared, removed, uncertain, split,
merge, and nonrigid evidence are fail-closed and retain B3. The method never
uses a whole-object warp for an explicitly nonrigid relation.

### Deterministic Estimator

Registration uses NumPy/SciPy only:

1. average full OVI points in 0.02 m voxels;
2. lexicographically cap each cloud at 20,000 uniformly spaced samples;
3. reject fewer than 64 samples or effective centered rank below two;
4. for static relations, evaluate identity only;
5. for moved relations, evaluate centroid alignment plus all proper PCA-axis
   permutations/signs;
6. run at most 40 trimmed point-to-point ICP iterations per initialization;
7. select lexicographically by maximum minimum directional overlap, minimum
   symmetric median, minimum symmetric p90, transform magnitude, then
   initialization ID.

Kabsch always corrects a negative determinant. ICP never estimates scale.
`cKDTree` queries use one worker for deterministic ordering.

### Frozen Registration Thresholds

| Parameter | Value |
| --- | ---: |
| registration voxel | 0.02 m |
| maximum samples/cloud | 20,000 |
| maximum ICP iterations | 40 |
| trimmed correspondence fraction | 0.80 |
| maximum ICP correspondence | 0.15 m |
| inlier/overlap distance | 0.05 m |
| minimum samples/cloud | 64 |
| minimum inliers/direction | 64 |
| minimum overlap/direction | 0.35 |
| maximum median/direction | 0.08 m |
| maximum p90/direction | 0.20 m |
| minimum second/first singular ratio | 0.02 |
| maximum transformed centroid residual | 0.10 m |
| principal extent ratio range | [0.40, 2.50] |
| maximum static centroid displacement | 0.15 m |
| maximum moved centroid displacement | 2.00 m |
| rotation/translation convergence | 1e-5 |

All acceptance predicates must pass. These thresholds are fixed before any
Apartment B7 result is observed; there is no sweep.

## Dense Recovery Contract

For eligible relation `r`, accepted transform `T_r`, source geometry `G0_r`,
current OVI geometry `G1`, signed t1 visibility `V1`, and compatibility `C_r`:

```text
G* = G1 union {
  T_r(x) : x in G0_r,
  V1(T_r(x)) in {occluded, unobserved},
  C_r(T_r(x))
}
```

`C_r(x)` requires accepted typed registration and either:

- nearest target-entity surface distance at most 0.20 m; or
- membership in the target entity axis-aligned bounding box expanded by 0.10 m.

The compatibility test uses method-visible OVI t1 geometry only. It never uses
GT, changed-region masks, confirmed-free voxels, or evaluator targets.

The t1 occupied set contains every t1 entity and t1 background point at the
frozen 0.05 m composition grid. Occupied always wins. Candidate visibility is
recomputed at transformed coordinates with exactly the frozen B3 thresholds:
0.05 m voxel, 0.10 m depth tolerance, 10 m maximum depth, six absent
observations, 0.80 absent fraction, three viewpoints, and 0.25 m viewpoint
baseline.

```text
VISIBLE_FREE -> reject unconditionally
OCCUPIED     -> reject unconditionally
OCCLUDED     -> recover only if C_r
UNOBSERVED   -> recover only if C_r
```

For an entity with at least one accepted recovered point, all of that source
entity's untransformed emitted B3 fallback spans are removed, and accepted
transformed points are appended to the paired t1 entity. t1 points, semantics,
entity ID, and metadata remain primary. Background is copied exactly from B3.
If registration is rejected or yields zero accepted points, no baseline span
is removed and the exact B3 object is returned.

## Provenance And Immutability

Each `DenseRecoveryPointGroup` binds:

- B3 current-map content SHA and t0 source snapshot SHA;
- relation ID/state/identity source and t0/t1 entity IDs;
- registration evidence/transform/config SHA;
- sorted source point indices;
- transformed-candidate visibility state;
- compatibility result and decision reason;
- output entity and exact output span, when emitted.

Every emitted recovered point has exactly one group; no orphan or overlapping
output span is legal. Rejected occupied, visible-free, and incompatible
candidates remain auditable. Content hashes cover arrays by dtype, shape, and
bytes. Inputs are checked before and after registration/recovery to prove no
mutation.

The result package atomically contains snapshot NPZ, entity JSONL,
`registration.json`, `recovery-provenance.jsonl`, metrics, attribution,
command, logs, and a manifest with source/config/output hashes. Large raw point
arrays remain outside Git.

## Relation Policy

| Relation | Registration | Historical transfer | Fallback |
| --- | --- | --- | --- |
| persistent static 1:1 | identity quality check only | compatible occluded/unobserved t0 | exact B3 |
| persistent moved 1:1 | deterministic PCA multi-start trimmed ICP | compatible occluded/unobserved transformed t0 | exact B3 |
| appeared | none | none | B3 t1-only behavior |
| removed candidate | none | none | B3 visibility behavior; identity cannot delete |
| uncertain | none | none | exact B3 |
| split/merge | none | none | exact B3 |
| nonrigid/ineligible label | none | none | exact B3 |

## Metrics And Pre-Registered Gate

The unchanged common-v2 evaluator reports Ghost, current mIoU, BG F@5 cm,
surface precision/recall/F1, unobserved recall, observed stale precision, and
point counts. Object/Dynamic/Change/identity remain N/A for TESSE because no
legal protocol-compatible GT contract exists.

Frozen B3 references are:

| Metric | B3 |
| --- | ---: |
| Ghost | 0.0000000000 |
| current mIoU | 0.1358861593 |
| BG F@5 cm | 0.3663600098 |
| surface precision | 0.7344127782 |
| surface recall | 0.3053313226 |
| surface F@5 cm | 0.4313354117 |
| unobserved recall | 0.0756464685 |
| observed stale precision | 0.9309634046 |
| final points | 15,622,601 |

B7-G is `GO` only if every safety/trade-off predicate passes:

```text
Ghost <= 0.0200000000
surface precision >= 0.7244127782       # B3 - 0.01
BG F@5cm >= 0.3563600098                # B3 - 0.01
observed stale precision >= 0.9259634046 # B3 - 0.005
```

and both material-gain predicates pass:

```text
unobserved recall >= 0.0806464685 # B3 + 0.005
surface F@5cm >= 0.4323354117     # B3 + 0.001
```

At least one registration and one historical point must be accepted. Current
mIoU, surface recall, final count, added/rejected points, relation-state and
quality-bin breakdowns, runtime, and memory are always reported but are not
used to select thresholds. No weighted score is allowed.

Any failed predicate yields `STOP_B7_MAPPING_EXTENSION`; no ReScene training
is then allowed to rescue the mapping mechanism.

## Failure Attribution

After the method output is immutable, evaluator-only attribution reports:

- added points matching previously uncovered current GT surface;
- added points matching confirmed free space;
- added object points in revealed-background cells;
- occupied, visible-free, incompatible, and registration-rejected candidates;
- registration failures and all quality measurements.

Rows group by entity, relation, identity source, registration quality bin,
visibility state, source visit, and OVI semantic label. Attribution may justify
one later component-specific ablation, but it cannot retroactively change this
B7-G run or gate.

## Test And Experiment Order

IT0 is a pure synthetic two-object pair covering known 3D rotation and
translation, invalid/reflection rejection, low support, linear degeneracy,
occupied priority, visible-free non-revival, unknown/occluded completion,
removed/appeared/split/merge/nonrigid fallback, and complete provenance.

IT1 uses the lexicographically first frozen B4 persistent recoverable 1:1
relation with at least 64 sampled points in each visit. Selection uses no GT or
metric. It verifies source hashes, deterministic registration, visibility
safety, B3 fallback invariants, and atomic artifact round-trip.

Only IT0 and IT1 success unlocks one full Apartment B7-G run. The architecture,
configuration, and commit are frozen before any Office result. Office remains
zero-attempt held-out until all bound assets exist, then permits one run only.

The full repository suite runs once after method/config freeze and at most one
additional time after targeted repair.

## Conditional Learned Path

ReScene work is forbidden unless B7-G is GO. If unlocked, the fixed path is
R0 adapter/source-binding smoke, R1 random-initialization plumbing with no
ranking claim, and R2 tiny overfit. Full training additionally requires
complete official data provenance. Learned B7 runs only if B5 identity
outperforms B4 under a legal 3RScan identity contract.

The selected configuration remains Concerto, 0.02 m neural grid, T=2, pinned
source/config, and no backbone/voxel sweep. Final 3RScan ranking requires the
frozen 44/44 visits; the current pilot remains 27/44. Office and incomplete
3RScan outcomes cannot affect selection.

## Resource Budget

B7-G is CPU-first and reuses all OVI outputs. No GPU is used before a GO. The
only full Apartment map pass occurs after IT0/IT1. Intermediate arrays must be
bounded by per-entity processing and 20,000-point registration samples; full
OVI points are transformed one entity at a time. Result receipts record wall
time, peak resident memory, and GPU memory when applicable.

No hyperparameter sweep, repeated OVI reconstruction, old matrix rerun,
Office tuning, or incomplete 3RScan final ranking is permitted.
