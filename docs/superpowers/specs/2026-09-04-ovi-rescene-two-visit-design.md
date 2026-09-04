# OVI-MAP x ReScene4D Two-Visit Current-State Mapping Design

Date: 2026-09-04

Status: approved by the user-provided master prompt before implementation

Master prompt SHA-256: `27b7a8c247d2fcb65ba851c92818a2dcb16c6b2943b426355859bbaf29ea993f`

Base branch: `research/crove-benchmark-alignment-audit`

Base SHA: `1e849acdcba46ab92f8704a77695ce7caefa1516`

Working branch: `research/ovi-rescene-two-visit`

## Research Question

Given two non-overlapping RGB-D visits to one indoor environment, build an
independent OVI-MAP instance map for each visit, infer cross-visit identity and
change evidence, and compose a dense current map whose stale surfaces are
removed only by signed revisit visibility. Success is a Pareto result: Ghost
must approach the OVI `t1`-only floor while retaining materially more correct
geometry and identity information than `t1`-only.

No scalar score combines Ghost, Object F1, Change F1, current mIoU, background
F-score, completeness, identity quality, runtime, or memory.

## Frozen Source Boundary

| Source | Required identity | Role | Mutation policy |
| --- | --- | --- | --- |
| Oviovo | `1e849acdcba46ab92f8704a77695ce7caefa1516` | integration and evaluation | new branch only |
| OVI-MAP | `f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424` | per-visit mapper and geometry authority | read-only external checkout |
| ReScene4D | `fb2fe42eb8f1e926567c48eea9acb874e608ee10` | optional temporal neural reasoner | read-only external checkout |

Source identity, license, required-file hashes, checkpoint availability, and
runtime assumptions are frozen in `configs/external/ovi_rescene_sources.json`.
An identity mismatch fails closed. A missing official ReScene checkpoint blocks
learned claims but not deterministic infrastructure or B0-B4 execution.

## Representation Contract

| Representation | Resolution | Lifetime | Contents | Final geometry authority |
| --- | ---: | --- | --- | --- |
| OVI mapping voxel | 0.01 m | persistent within one visit | TSDF state plus persistent label/instance fusion state | yes, through extracted OVI surface |
| ReScene neural token grid | 0.02 m primary; 0.01/0.04 m ablations | one paired forward pass | quantized XYZ, exact integer visit index, input and learned features | no |
| Legacy CROVE object submap | 0.05 m configured compact state | continuous historical baseline | one weighted representative per occupied object-local cell under a cap | no in this route |
| Common-v2 evaluator grid | 0.05 m | evaluation only | metric-domain discretization | no |

The adapter reads deterministic samples from OVI surfaces. It never mutates the
source maps, exports all-volume TSDF state, spatially scales the visit index, or
serializes neural tokens as a final map. Every token retains a CSR reverse index
to all contributing `(visit, OVI entity, OVI point index)` samples.

The voxel gate is `VOXEL_CONTRACT_PASS` only when all four representations are
distinguished and machine-checked. Otherwise it is
`VOXEL_CONTRACT_BLOCKED`, and learned integration cannot begin.

## Authority Model

```text
OVI mapping authority       = independent OVI-MAP map per visit
ReScene neural backbone     = Concerto through Pointcept when reproducible
identity authority          = ReScene query evidence or named geometric baseline
removal authority           = t1 signed visible-free evidence
semantic authority          = OVI/VLM evidence
final geometry authority    = OVI t1 surface, then visibility-safe OVI t0 fallback
evaluation voxel            = frozen 0.05 m common-v2 domain
```

A missing `t1` neural mask produces only `removed_candidate`. It cannot suppress
historical geometry. Closed-set ReScene class logits are optional association
evidence and cannot overwrite OVI open-vocabulary semantics.

## Core Contracts

`VisitMap` binds visit ID 0 or 1, scene/frame identity, independent OVI
`MapSnapshot`, map voxel size, and source manifest digest.

`NeuralSampleMap` contains `(N,4)` XYZ plus integer visit coordinates, `(N,F)`
features, per-token visit IDs, and CSR contributor arrays. Contributor spans are
non-empty, deterministic, sorted, and never mix visits in one token.

`TemporalQueryEvidence` contains backend identity, optional checkpoint identity,
query masks/scores, and explicit blocked status. Randomly initialized output is
plumbing-only and ranking-ineligible.

`PairRelation` represents 1:1 persistent/static or moved, 0:1 appeared, 1:0
removed candidate, 1:N split, N:1 merge, or uncertain evidence without forcing
one-to-one matches.

`CurrentCompositionDecision` binds each emitted or suppressed source entity to
its OVI visit, visibility evidence, relation, geometry source, identity source,
state source, and semantic source.

## Data Flow

1. Export source-bound RGB-D, camera metadata, poses, and timestamps.
2. Build `M0` and `M1` with two independent OVI-MAP executions.
3. Extract dense OVI instance/background surfaces without changing 1 cm map
   resolution.
4. Deterministically aggregate surface samples on a spatial-only neural grid;
   append exact visit index 0 or 1.
5. Run the deterministic geometric-semantic reasoner and, only when valid,
   ReScene Concerto shared-query inference.
6. Project query masks back through the CSR index to per-query, per-entity,
   per-visit evidence.
7. Classify revisit visibility as occupied, visible-free, occluded, or unknown.
8. Compose `M_current` using t1-first authority:
   - observed occupied: emit OVI t1;
   - visible-free: suppress OVI t0;
   - occluded or unknown: retain OVI t0 fallback.
9. Evaluate current correctness, geometry completeness, identity/change, open
   vocabulary semantics, and system cost as separate metric groups.
10. Attribute every Ghost/Object/change failure to its exact source path.

## Temporal Reasoners

The stable interface is `TemporalPairReasoner.infer(pair)`. The required
`geometric_semantic` backend is CPU deterministic and uses common-frame overlap,
centroid distance, size/shape evidence, and only source-bound OVI semantics. It
never consumes GT IDs or change labels.

The optional `rescene` backend is a thin boundary around the pinned checkout.
Concerto is selected first because it is an official ReScene configuration. The
fallback order is Sonata then Minkowski, and any fallback receives a distinct
backend identity. Inference requires either an official/author checkpoint with
a SHA-256 binding or a locally trained checkpoint whose data split, config,
command, environment, and bytes are source-bound. Otherwise the status is
`BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT`.

## Frozen TESSE Protocol

The new protocol ID is `TESSE_TWO_VISIT_CURRENT_V1`. Apartment is development;
Office stays `OFFICE_NOT_RUN_HELD_OUT` until source, windows, adapter, backend,
checkpoint, thresholds, composer, baselines, metrics, and gates are committed.

The pair-selection algorithm consumes source metadata and causal change timing,
not method outputs. It chooses two non-overlapping windows with enough frames,
trajectory overlap, common observable volume, changed-object support, old-site
visibility, and source-bound RGB-D/pose inputs. GT may define/evaluate the
protocol but cannot enter method input manifests.

Office is executed once after freeze for the selected method and required
baselines. Office outcomes cannot change selection or thresholds.

## Baseline and Evidence Matrix

| ID | Pair reasoner | Visibility | Final geometry | Claim tested |
| --- | --- | --- | --- | --- |
| B0 `OVI_T0_ONLY` | none | no | OVI t0 | old-map Ghost |
| B1 `OVI_UNION` | none | no | OVI t0 union OVI t1 | high-completeness stale accumulation |
| B2 `OVI_T1_ONLY` | none | no | OVI t1 | empirical low-Ghost floor and completeness lower bound |
| B3 `OVI_VISIBILITY_COMPOSE` | none | yes | OVI t1 plus safe OVI t0 fallback | signed-visibility contribution |
| B4 `OVI_GEOMETRIC_PAIRING_VISIBILITY` | geometric-semantic | yes | OVI t1 plus safe OVI t0 fallback | non-learned identity baseline |
| B5 `OVI_RESCENE_VISIBILITY` | ReScene | yes | OVI t1 plus safe OVI t0 fallback | proposed learned system |
| B6 `OVI_RESCENE_NO_VISIBILITY` | ReScene | no | OVI relation compose | visibility necessity |

Unknown and blocked measurements remain `N/A` with a reason. B5 is not the final
method by assumption. Its verdict is exactly one of
`RESCENE_COMPONENT_GO`, `RESCENE_NOT_PROVEN_NECESSARY`,
`RESCENE_BLOCKED_EXTERNAL_ASSET`, or `RESCENE_METHOD_NO_GO`.

## Quantitative Gates

First measure `G_t1 = Ghost(B2)` and B2 completeness. B5 should satisfy
`Ghost(B5) <= G_t1 + 0.02`, materially improve B2 unobserved-region
completeness, reach Object F1 at least 0.367952 (preferred 0.372762), and remain
within a pre-frozen absolute Change F1 tolerance of the 0.088 range. It must
also improve an identity/change/completeness dimension over B3/B4 at comparable
Ghost to establish ReScene necessity. These gates are not averaged.

## Failure Model

All validation fails closed on malformed shapes, mixed frames, mutable source
identity, future-frame input, evaluator-only GT leakage, invalid digests,
non-finite metrics, missing checkpoint identity, or ReScene token geometry in a
final output.

Attribution categories are:

```text
T0_RETAINED_DESPITE_VISIBLE_FREE
T0_RETAINED_UNOBSERVED
T0_RETAINED_OCCLUDED
T1_OVI_FALSE_POSITIVE
PAIR_REASONER_WRONG_ID
VISIBILITY_DEPTH_ERROR
BACKGROUND_COMPOSITION_ERROR
UNATTRIBUTED
```

## Artifact and Verification Policy

Large datasets, RGB-D, meshes, maps, third-party trees, checkpoints, caches, and
event sidecars stay outside Git. The handoff records absolute local path,
SHA-256, byte count, command, producer commit, run ID, and config/manifest hash.
Git contains code, tests, configs, small summaries, and reports.

Each behavior change follows RED-GREEN-REFACTOR and each phase is committed
separately. Before upload: run focused historical regressions, Ruff on changed
Python files when available, `git diff --check`, `python -m compileall -q src
scripts tests`, and full `pytest -q`. Push only the working branch and report
`UPLOAD_STATUS=VERIFIED` only after the remote SHA equals local HEAD.
