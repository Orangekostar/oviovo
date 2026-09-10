# CROVE Entity-Episode Dynamics V2 Experiment Plan

## Research Question

Can a source-preserving current-map update improve retained valid history and current semantic quality over B3 by separating stable identity from location episodes, without increasing Ghost or changing OVI-native geometry?

## Fixed Protocol

- Evidence commit: `d5c0688bc662f8e65455cb9c62909de87c941b43`.
- Primary existing condition: TESSE-CD Apartment, t0 `[766,1021]`, t1 `[1217,1472]`.
- Geometry: independently reconstructed OVI-MAP native t0/t1 surfaces at the existing nominal 1 cm resolution.
- Current-map evaluation: visit-final, existing 5 cm evaluator and frozen semantic crosswalk/target manifest.
- State evidence: every method receives the same full authorized t1 RGB-D window.
- GT: evaluator-only. GT geometry, current-valid labels, removal labels, and Ghost labels are excluded from prediction.
- Configuration selection: one global DEV configuration; Office is confirmation-only.

## Method Boundary

The unit of identity is `stable_entity_id`; the unit of spatial state is `entity_location_episode_id`; the immutable geometry key is `(source_surface_id, source_vertex_index)`. Identity association may attach aliases and select a candidate episode, but only row-local depth/overlap evidence may change `current_valid`.

The transition precedence is:

1. A reliable t1 predicted-surface overlap retires the duplicated t0 row as `replaced`.
2. A reliable, sufficiently repeated visible-free observation keeps or makes the old location invalid.
3. A strictly later positive depth observation can locally correct an inherited tombstone only where t1 has no replacement surface.
4. Occluded, unknown, unmatched, and relation-only rows inherit their prior validity.

Moved identity requires accepted one-to-one association plus bidirectional local registration, residual improvement over identity, and at least three spatially separated 5 cm support patches. Otherwise the relation is unresolved and cannot propagate state.

## Evidence Sources

- G0: existing `GeometricPairReasoner`, retained for historical comparison.
- G1: deterministic one-to-one assignment with explicit null, centered shape/size, local registration, visible support, optional same-space appearance, soft semantics, and non-gating original-position IoU.
- R: source-bound frozen ReScene/Persist4D forward using the exact pair's RGB, native normals, D-to-A-to-M mapping, checkpoint identity, masks, and logits. Query support is projected back to fixed OVI candidates with coverage, purity, competition, and margin diagnostics.
- M: deterministic `CROVE_MEMORY` retrieval using `FeaturePrototypeBank` and `InformativeViewBank`, with active-before-dormant search and strict feature-space/projection-version separation.
- Oracle: evaluator correspondence or local boundary labels only, with no GT state mask; diagnostic and excluded from ranking.

## Experiment Matrix

| Variant | Relation evidence | State readout | Required interpretation |
| --- | --- | --- | --- |
| D0_T1 | none | t1-only | Cost of discarding all history |
| D1_B3 | B3 | frozen B3 | Strong deterministic baseline |
| D2_INHERIT | none beyond B3 | inherited local update | State-model contribution |
| D3_GEOM | G1 | shared entity-episode update | Strong simple association |
| D4_RESCENE_ID_ONLY | same R list as D5 | D1 mask/semantics | Identity-only control |
| D5_RESCENE_EPOCH | R | shared entity-episode update | ReScene evidence increment |
| D6_MEMORY_EPOCH | M | shared entity-episode update | Memory evidence increment |
| DX_ORACLE_REL | evaluator relation | shared update | Opportunity diagnostic only |

D4 must be exactly equal to D1 for the current mask, point semantics, current mIoU, Ghost, background geometry, and surface geometry. It may differ only in identity metrics and alias records.

## Data Selection

Keep the fixed Apartment pair. Before scoring any method, select at most two additional Apartment DEV pairs from `configs/evaluation/tesse_two_visit_current_v1.json` by asset readiness, declared event type, and visibility coverage. Preserve failed selected pairs rather than replacing them based on method scores.

Read the frozen Office pair from the same protocol. Report `RAW_MISSING` only when declared raw RGB-D inputs are absent, `DERIVED_NOT_BUILT` when raw inputs exist but native OVI products do not, and `READY` when both visits and evaluator inputs are bound. Build missing native products from raw inputs through the existing OVI runner. After DEV selection, run Office with D0, D1, D3, and the selected new variant without retuning.

## Metrics and Counts

Headline metrics remain unchanged: current mIoU; Ghost with `ghost_count / predicted_changed_object_count`; background precision/recall/F@5cm and match counts; object-surface precision/recall/F@5cm; unobserved-region recall; and retained-t0 state counts.

Additional diagnostics report both source-row and unique 5 cm physical-sample counts:

- deletion error: GT-supported old samples retired by the method;
- bad recovery: prior-invalid samples restored into confirmed free space;
- correct recovery: restored samples supported by current GT and not already covered by t1 prediction;
- free conflict including object, background, and unknown labels;
- association precision, recall, rejection, competition margin, and large-motion ambiguity;
- transition and retirement-reason distribution;
- identity-only versus epoch-changed surface size.

Every ratio records its numerator and denominator. Final-snapshot outputs remain distinct from official sequence Obj/Dyn/Chg scores.

## DEV Selection Rule

Evaluate 8-12 hypothesis-led configurations rather than a Cartesian sweep. First require Ghost `<= 0.02` on the established Apartment condition and frozen geometry/deletion safeguards. Among eligible configurations maximize mean current mIoU across frozen DEV pairs, then background F@5cm, then retained valid-history recall. Keep every ineligible and failed row in the result set.

## Evidence Gates

A claimed learned increment requires all of the following:

- at least one real TESSE pair-bound ReScene forward;
- D4 and D5 consuming the exact same relation list;
- D4 state/geometry equality to its baseline;
- D5 or D6 improving the selected metric under the same state kernel and constraints;
- relation/action attribution identifying which accepted relations changed which source rows;
- no Office tuning and no GT prediction input.

When D2 only prevents V1-style state regression but does not beat B3, report `STATE_CORRECTNESS_REPAIRED_NO_MAP_GAIN`. When G1 improves the map and learned evidence adds nothing, report `CROVE_UPDATE_GAIN_NO_LEARNED_INCREMENT`. Use `ENTITY_EVIDENCE_SUPPORTED_WITH_LIMITED_SCOPE` only when R or M adds a reproducible constrained gain. Otherwise preserve the negative result and identify the measured bottleneck.

## Deliverables

Compact results live under `configs/evaluation/results/crove_entity_epoch_v2/`; full PLY/NPZ caches remain under `$HOME/oviovo_baseline_runs/20260910_crove_entity_epoch_v2/`. The result package contains selection, per-pair metrics, aggregate metrics, action attribution, relation diagnostics, state transitions, frozen selected configuration, Office result/attempt, real figures, model provenance, and a compact artifact index.
