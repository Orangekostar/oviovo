# OVI-MAP x ReScene4D Two-Visit Experiment Ledger

Date: 2026-09-04

## Frozen Identities

| Item | Identity/status |
| --- | --- |
| Base branch | `research/crove-benchmark-alignment-audit` |
| Base commit | `1e849acdcba46ab92f8704a77695ce7caefa1516` |
| Working branch | `research/ovi-rescene-two-visit` |
| Master prompt SHA-256 | `27b7a8c247d2fcb65ba851c92818a2dcb16c6b2943b426355859bbaf29ea993f` |
| OVI-MAP | `f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424`, MIT |
| ReScene4D | `fb2fe42eb8f1e926567c48eea9acb874e608ee10`, MIT |
| Source audit | `EXTERNAL_SOURCE_PASS` |
| Voxel contract | `VOXEL_CONTRACT_PASS` |
| ReScene checkpoint | `BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT` |

Baseline repository tests at the frozen base: `5803 passed, 10 skipped` in
897.39 seconds. The five warnings are NumPy deprecation warnings in dense
semantic mutation tests.

## Historical Evidence

These results belong to the continuous CROVE path and are not two-visit
results.

| Variant | Object F1 | Dynamic F1 | Change F1 | Current mIoU | Ghost |
| --- | ---: | ---: | ---: | ---: | ---: |
| A6 | 0.372762 | N/A | 0.060853 | 0.142897 | 0.646883 |
| c553 static-anchor | 0.348472 | 0.069225 | 0.088458 | 0.149560 | 0.443760 |
| P5 | 0.367952 | 0.069225 | 0.088088 | 0.149563 | 0.441677 |
| P6-C | 0.367307 | 0.069225 | 0.087246 | 0.149449 | 0.441689 |

## Execution State

| Phase | Status | Evidence |
| --- | --- | --- |
| P0 source and voxel audit | PASS | Exact Git/file/license bindings and four-role contract |
| P1 immutable contracts | PASS | Read-only arrays, exact visit coordinates, and conserving CSR provenance |
| P2 OVI-to-ReScene adapter | PASS | Deterministic XYZ-only grouping at 1/2/4 cm neural resolutions |
| P3 deterministic reasoner | PASS | Geometry/OVI-semantics entity graph plus fail-closed learned boundary |
| P4 query projection | PASS | CSR-expanded token/soft/source-point evidence and explicit relation topology |
| P5 current-state composer | PASS | t1-first dense OVI composition with point-group provenance |
| P6 TESSE two-visit protocol | PASS | 32 source-only candidates; Apartment/Office windows frozen before method scores |
| P7 B0-B6 evaluator | PASS | Frozen Pareto metric contract and atomic B0-B4 / blocked B5-B6 orchestrator; no result claimed |
| P8 learned ReScene | BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT | Source-bound fail-closed runner/config verified; no public checkpoint and no learned result |
| P9 failure attribution | PASS | Exhaustive one-row-per-failure provenance, fixed Ghost precedence, mass conservation, and atomic sidecars |
| P10 Apartment matrix | NOT_STARTED | No result claimed |
| P11 Office confirmation | `OFFICE_NOT_RUN_HELD_OUT` | Held until method/config/gates are frozen |
| 3RScan validation | NOT_STARTED | No result claimed |

The primary comparison will report the Pareto vector rather than a hidden
weighted score: Ghost, background F-score at 5 cm, current mIoU, object F1,
dynamic F1, change F1, runtime, and memory. B5/B6 remain ineligible while the
checkpoint gate is blocked.

The blocked ReScene execution contract is frozen in
`configs/evaluation/rescene_two_visit_backend.json` and enforced by
`scripts/evaluation/run_rescene_pair_backend.py`. Against the pinned clean
checkout it publishes only `BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT`, with
`ranking_eligible=false` and no prediction arrays. The subprocess path is
covered hermetically, but it is not claimed as a scientific ReScene result;
the optional integration commit remains gated on a valid source-bound
checkpoint and environment.

Ruling: OVI labels and embeddings are carried as immutable entity side
evidence, not concatenated into the ReScene feature tensor. This preserves the
required geometric-semantic baseline without introducing an untrained feature
distribution into Concerto. If wrong, the adapter contract and learned backend
input schema will require a backward-compatible revision.

Ruling: `suppress_t0_occupied_by_t1` is added to the composition decision
vocabulary because the specified t1-first equation replaces same-cell t0
geometry, while the recommended literal list named only visible-free
suppression. Without this action, replaced source points would have no exact
provenance. If wrong, downstream readers must fold this action into `emit_t1`
without changing the composed geometry.

Ruling: TESSE two-visit candidates are event-local 256-frame windows. A single
first-change-to-last-change pair was rejected before method execution because
the continuous trajectory had moved to a different room, yielding zero
trajectory overlap in Apartment and zero common observable volume in Office.
The frozen source-only selector uses a declared 3 m indoor trajectory coverage
radius together with common view-frustum volume; it never reads method scores.
