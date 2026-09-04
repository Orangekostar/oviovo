# OVI-MAP x ReScene B7 Read-Only Audit

Date: 2026-09-04

## Audit Scope And Identity

This audit was performed on branch
`research/ovi-rescene-b7-dense-recovery` at
`9db59cb3200e0bc7f48f904c4e9fa628c71c9586`. The targeted clean-baseline
suite passed with `83 passed in 5.88s`. No production code, configuration,
dataset, or frozen result was changed during Phase A.

The authority split remains frozen:

| Authority | Owner |
| --- | --- |
| Dense geometry | OVI-MAP |
| Open-vocabulary semantics | OVI-MAP/VLM |
| Cross-visit identity | B4 geometry or ReScene |
| Existence/removal | Signed t1 visibility |
| Discretization | Evaluator only |

The OVI map grid remains 0.01 m, the selected ReScene neural grid is 0.02 m,
the legacy CROVE object grid is 0.05 m, and the evaluator grid is 0.05 m.
These representations are not interchangeable.

## Current Composition Path

| Source | Current responsibility and I/O | B7 decision |
| --- | --- | --- |
| `src/oviv2/two_visit_contracts.py` | Defines immutable `VisitMap`, `NeuralSampleMap`, `PairRelation`, relation/action/source enums, and validates array and identity invariants. `PairRelation.evidence` is scalar `Mapping[str, float]`. | Keep unchanged. Add a separate typed, versioned registration contract; do not put transforms or correspondences in `evidence`. |
| `src/oviv2/two_visit_current_map.py` | Builds the frozen B3/B4 current snapshot. It emits all t1 points first, lets t1 occupied cells win, suppresses t0 history proved visible-free, and retains t0 only in occluded/unobserved cells. B4 relations only route retained points to a t1 entity; they do not warp geometry. | Keep B3 behavior unchanged. Implement B7 as a separate post-B3 extension with exact fallback to B3. |
| `src/oviv2/two_visit_execution.py` | Loads signed-visibility inputs, constructs B0-B4 inputs, derives the t1 signed grid with the frozen thresholds, and prepares B4 samples. | Reuse orchestration and factor only the smallest generic candidate-classification seam needed by B7, with B3 equivalence tests. Do not retune visibility. |
| `src/oviv2/visibility.py` | `VoxelVisibilityProjector` projects current RGB-D/poses into signed occupied, visible-free, occluded, and unobserved evidence. | Reuse as the sole visibility engine. Transformed B7 candidates must be projected at their transformed coordinates; a missing key in the original t0 grid is not valid unobserved evidence. |

The frozen B3 order is therefore:

```text
t1 dense OVI geometry
  -> t1 occupied wins
  -> signed visible-free suppresses stale t0
  -> occluded/unobserved t0 fallback
```

B4 is metric-identical to B3 because it adds identity routing but no geometric
registration or dense recovery.

## ReScene And Geometric Identity Path

| Source | Current responsibility and I/O | B7 decision |
| --- | --- | --- |
| `src/oviv2/ovi_rescene_adapter.py` | Converts per-visit OVI surfaces into transient 4D neural samples at a selected 0.01/0.02/0.04 m grid, preserving exact visit IDs and an exact CSR reverse index to OVI points/entities. It rejects instance-palette RGB. | Reuse unchanged for learned identity. It has no final-geometry authority. |
| `src/oviv2/rescene_backend.py` | Validates pinned source identity, clean checkout, checkpoint, executor, and output schema before invoking ReScene. It fails closed. | Reuse. Random initialization is diagnostic-only and cannot support ranking. |
| `src/oviv2/query_instance_projection.py` | Projects ReScene query evidence back to OVI entities and is the only legal learned query-to-OVI identity bridge. | Reuse if the learned path is unlocked. It cannot replace OVI geometry or semantics. |
| `src/oviv2/geometric_pair_reasoner.py` | Builds deterministic B4 relations from OVI entity geometry/semantics with connected-component reasoning. | Reuse as the B7-G identity authority. Materialize its relation artifact once from frozen OVI snapshots. |

Pinned ReScene source:
`/home/ww/oviovo_references/evaluation/rescene4d` at
`fb2fe42eb8f1e926567c48eea9acb874e608ee10`, clean at audit time. The pinned
defaults select Concerto, hidden dimension 128, 100 queries, 8 heads, 3 shared
decoder layers, non-parametric queries, temporal dimension 4, and a two-visit
window. The selected neural voxel is 0.02 m.

The upstream checkout still provides no official final checkpoint. Its default
base instance configuration sets the contrastive loss to null/disabled;
InfoNCE is enabled only by an explicit override. Therefore B5/B6 remain
`BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT`, and the audit does not claim that the
pinned default enables contrastive training.

## Registration And Legacy Dense Readout

The repository-wide search found no generic public SE(3)/ICP utility suitable
for B7. `src/oviv2/temporal_geometry.py` contains private ICP/Kabsch helpers,
but they are coupled to the legacy continuous-time `ObjectSubmap` contract.
They are reference implementations, not a valid B7 API.

`src/oviv2/dense_moved_readout.py` contains useful deterministic numerical
ideas: bidirectional nearest-neighbor coverage and agreement checks. It does
not estimate a rigid transform and is coupled to legacy compact/template
geometry. B7 may reuse the metric definitions or tested numerical pattern, but
must not reuse its object contract or promote legacy compact geometry to the
final map.

A new minimal two-visit registration module and a separate dense-recovery
module are required. Registration must fail closed on low support,
degeneracy, reflection, invalid SO(3), weak bidirectional overlap, or excessive
residual. A rejected registration must return the exact B3 snapshot.

## Evaluator Capabilities

`src/evaluation/two_visit_snapshot_metrics.py` and
`src/evaluation/two_visit_current_metrics.py` can measure, at the frozen 5 cm
threshold:

- current semantic mIoU;
- confirmed-free Ghost rate;
- revealed-background precision, recall, and F1;
- current-surface precision, recall, F1, and total coverage;
- t1-unobserved-region recall;
- observed historical stale precision.

Ghost is the fraction of predicted current points within 5 cm of confirmed
free space. Surface precision and recall are independent nearest-neighbor
matches in the prediction-to-GT and GT-to-prediction directions. Unobserved
recall is GT-unobserved-to-prediction coverage. These definitions are adequate
for the B7-G mechanism gate and its precision/background trade-off.

The frozen TESSE two-visit package does not provide legal object/change or
cross-visit identity ground truth for this pair. Those values must remain N/A,
not zero and not inferred from evaluator-only annotations.

`src/evaluation/rscan_temporal.py` and
`scripts/evaluation/evaluate_3rscan_temporal.py` provide causal exact-ID
diagnostics: identity switches, false re-identifications, reactivation,
current precision/recall, stale false positives, and change-type recall.
Community tAP/tREC remain separate and unavailable unless their official input
contract is satisfied. Dataset cross-session IDs are evaluator-only and must
not enter method inference.

## Frozen Apartment Artifacts

Large artifact root:
`/home/ww/oviovo_baseline_runs/20260904_ovi_rescene_two_visit/apartment-b0-b6-entity-lift-96ba693`
(575,211,024 bytes).

| Artifact | SHA-256 |
| --- | --- |
| `matrix-summary.json` | `7aa21465fcf6870a94448b2c34022251bbad79e5dadaed58dbd7ed1d4af26dab` |
| `two_visit_ovi_manifest.json` | `1996a0008649cdeb58daa920a05d30366a971b2fef77ae2e6a4ae1ca8a3f7b94` |
| B0 snapshot NPZ | `088df040bdcd908e92a426e736070f34c2bbc83c01f8f9021d18cd398110ef1a` |
| B2 snapshot NPZ | `e517639f743236620537d3bdbdebcf556b0d7e05ca304b248ae314df4432fa1e` |
| B3 snapshot NPZ | `b53d023c6257056d2d4057d1aaad9047581acc04aa225fa9ec2acbed4cd0060e` |
| B3 provenance manifest | `c431c710d02be40c365b6a20094c98034a72e1b740b2dc53422203946e9276d2` |
| B3 metrics | `7f1807db61a2ef6641a9564959ff67bf15e8bcef107a320ffd1669be21d4fc31` |
| B4 snapshot NPZ | `37ca9bdef186e75553f6f6c99314980702759677aa5319340f51db2f10dfbc63` |
| B4 provenance manifest | `15bfec5c03052de47d49b21679b075caf17c066019fba9576e2c94e30f1d81fe` |
| B4 metrics | `42b9056c2949ca829e4350295ce703bd4093b65254217375033b0540382ce763` |
| t0 native manifest | `e0141a2356ca719ae2bdad3061f2442671486d4f12e8abd05ebfecdb85602ea3` |
| t1 native manifest | `6cbfc318c25df085cb677a039ac1ca527b17cdba55b7d9fb70478a8871eddca2` |
| t0 materialized manifest | `eed51c1ebcff4c1a10425ab072a17949c02bafe27c0ed3e1d7e7a5a96c3af781` |
| t1 materialized manifest | `e94d1684ff8acbc96ecb3a7f045cda503003b75fff0bcb47550803724a3dae83` |

The checked-in matrix SHA is
`08dc52143b6c976d2b9f516fb312aa9e52ad42befe47e4fbb99cfaee6f07ee4d`;
the protocol SHA is
`720952cd75e214de36e318051d6aae18bd762109ba0d3a8ccb73a14f7398e650`.
The frozen pair is Apartment t0 frames 766--1021 and t1 frames 1217--1472.

The B4 run serialized 120 relation IDs in provenance, but no standalone
`PairRelation`/adapter artifact. B7-G therefore needs one deterministic B4
relation materialization from frozen B0/B2 OVI snapshots. This is not an OVI
reconstruction or a B0--B6 rerun.

The frozen source record names the original upstream OVI-MAP commit
`f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424`; the executed Ubuntu 24 native
runner/artifacts use the separately pinned patched source commit
`58a804e2d7c82ba05a489eb071aba3367301fed8`. These are distinct provenance
roles and must not be collapsed.

## Data Availability

The official metadata copies `/home/ww/3RScan.json` and
`/home/ww/vv/dataset/3RScan/3RScan.json` are byte-identical with SHA-256
`674a00f50f76b198b9de44efd86c390fea3da37ba8f12cf8ccd00045e265fa64`.
Across 478 reference scenes, visit-count statistics (reference plus rescans)
are:

| Condition | Scene count |
| --- | ---: |
| exactly T = 2 | 194 |
| T >= 3 | 284 |
| T >= 4 | 124 |
| T >= 5 | 56 |
| T >= 6 | 25 |

The frozen pilot manifest
`configs/evaluation/manifests/3rscan_causal_pilot_v1.json` has SHA-256
`b3bed41a0d25c7931aaf26ed460778186738f98259281099efcc35b346c7bf1f`.
Its selected ten environments contain 44 visits. Direct revalidation against
`/home/ww/vv/dataset/3RScan/scans` finds all mesh/semantic members but still
finds the same 17 missing `sequence.zip` files, so the usable pilot remains
27/44 and final ranking is blocked. The dataset root contains 72 sequences in
total, but unrelated sequences do not satisfy the frozen pilot contract.

Office remains held out with attempt count zero. Its bound RGB-D export,
timestamps, trajectory, and ground-truth directory are absent, so no Office
run is legal.

## Reuse And Compute Plan

Reuse without recomputation:

- OVI t0/t1 native and materialized maps;
- B0/B2 dense per-visit snapshots;
- B3 current snapshot, provenance, visibility binding, and metrics;
- frozen protocol, vocabulary, matrix, and evaluation inputs;
- pinned ReScene checkout and existing adapter/backend/projection code.

New work/run only:

- deterministic B4 relation materialization and receipt;
- synthetic rigid-registration, visibility-safety, fallback, and provenance tests;
- one cached Apartment entity-pair/slice integration test;
- one full Apartment B7-G run after IT0/IT1 pass;
- ReScene R0/R1/R2 only if B7-G passes its gate;
- final 3RScan ranking only after all 44 selected visits are complete;
- Office exactly once only after assets exist and method/config/commit are frozen.

Phase B/C and B7-G registration are CPU-first and require no GPU. The full
Apartment extension should reuse all dense maps and is budgeted as one run;
the previous B4 pairing took 143.514 seconds, while candidate visibility and
dense export are expected to dominate B7-G. The run should abort on IT0/IT1
failure rather than consume the full artifact pass.

Three NVIDIA A40 GPUs are available, but they are irrelevant until the B7-G
mapping gate passes. If unlocked, the learned path remains fixed to Concerto,
0.02 m, T=2, the pinned configuration, and at most the prescribed R0/R1/R2
gates. No backbone, voxel-size, or full hyperparameter sweep is authorized.

## Phase A Decision

Phase A is `PASS`. The architecture gap is specific: B4 identity currently
cannot improve dense completeness because it lacks both a source-bound rigid
transform and a signed-visibility-gated transfer of full-resolution OVI
history. The next legal step is to freeze a B7-G design that adds exactly those
two capabilities while preserving bit-equivalent B3 fallback.
