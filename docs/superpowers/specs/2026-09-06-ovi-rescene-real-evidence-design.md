# OVI-ReScene Real-Evidence Design

## Scope

This design implements the evidence program defined by `docs/01_CODEX_MASTER_PROMPT.md`,
`docs/02_EVIDENCE_AND_CODE_MAP.md`, and `docs/03_EXPERIMENT_PROTOCOL.md`. It starts from
`Orangekostar/oviovo@31130314f62a227a8f8d1b37f0784532e8c235fb`, preserves every
frozen OVI map and prior receipt, and produces only measured results.

The work has one dependency chain rather than six independent products:

1. make the legal supported domain executable;
2. obtain a real Apartment ReScene forward and resolve queries to OVI entities;
3. freeze an asset-backed 3RScan T=2 development protocol and evaluator GT;
4. compare G/F/R under identical inputs and D0/D1/D2 domains;
5. measure downstream ownership and completion opportunity;
6. adapt only the component identified by the measured failure.

## Chosen Approach

Use exact domain compaction. Let `S = flatnonzero(support_valid)` over the old model
domain `M`, let `h[S] = arange(|S|)` and `h[~S] = -1`, let
`valid_A = support_valid[adapter_to_model]`, and let
`g_compact = h[adapter_to_model[A_keep]]`. The network receives only supported model
rows, while the V3 sidecar retains new-to-old D/A/M mappings, old-to-new M, full
entity denominators, and unsupported entities.

Two alternatives are rejected. Filling unsupported RGB or normals would fabricate
model evidence. Rebuilding unchanged OVI maps would spend time without repairing the
known input-domain mismatch.

## W1: Supported Inference View

`src/oviv2/rescene_supported_view.py` owns immutable V3 domain types and pure
construction/validation. It consumes the frozen `StaticPreparedInput` and
`RecoveredModelSupport` and produces:

- a compact `ReSceneModelInput` with nine legal channels;
- a compact `NeuralSampleMap` for public projection;
- `new_to_old_model`, `old_to_new_model`, `new_to_old_adapter`, and
  `new_to_old_source` mappings;
- per-entity full/supported adapter and source counts;
- explicit unsupported entity keys and coverage counts by visit.

No negative index may be dereferenced. Token order remains old-order stable, both
visits remain nonempty, `point2segment` is compact identity, and the old geometry,
sampling, surface, and recovery artifacts are never mutated.

`scripts/evaluation/prepare_ovi_rescene_supported_v3.py` loads the prior partial V2
receipt, verifies its parent bindings with the existing audit, builds the compact
view, and atomically publishes `model_input/`, `adapter_pair/`, `mappings.npz`,
`entity_coverage.json`, and `input_contract_v3.json`. The V3 receipt binds every file
by SHA-256 and byte count and records full and supported D/A/M denominators.

## W2: Apartment Native Evidence

`scripts/evaluation/rescene_pair_executor.py` accepts either the existing complete
V2 contract or the new supported V3 contract. V3 loading verifies all sidecar hashes,
mappings, pair identity, feature schema, and voxel size before calling the unchanged
`_native_forward()` with strict checkpoint loading, `eval()`, and inference mode.
The pinned runtime is `/home/ww/miniconda3/envs/persist4d/bin/python`; the checkpoint,
Concerto weights, and ReScene source commit remain frozen.

Raw logits and masks stay in M space. Public evidence is expanded only to compact A
space. `src/oviv2/query_entity_resolver.py` aggregates each query directly over the
compact A-to-entity incidence and reports:

- input coverage and conditional query coverage;
- full-entity evidence and query-to-entity precision;
- fallback, collision, duplicate, and conflict counts;
- a deterministic one-to-one relation set selected by mutual dominance, configured
  minimum coverage, confidence, and conflict ordering.

Thresholds are selected only on GT-backed development data. Apartment visual cases
are report-only. B4 is rebuilt once from frozen inputs and serialized beside raw and
unique B4/B5 relation counts. Representative and low-support entity visuals use
actual entity geometry and query masks, with no GT-driven case selection.

## W3: 3RScan T=2 Development Protocol

`src/evaluation/rscan_t2_dev.py` creates `RSCAN_T2_DEV_V1` from the checked 3RScan
metadata, validation list, available complete assets, and Persist4D sequence database.
Selection is result-independent: validation environments only, exactly two ordered
sessions per pair, at least three environments when assets permit, deterministic UUID
order, and no overlap with checkpoint training environments. Existing 44-scan and
pilot manifests remain unchanged.

Global scan alignment is declared common method input for G/F/R. Per-object motion,
cross-time instance identity, symmetry, and ambiguity stay evaluator-only.

`src/evaluation/rscan_gt_instances.py` binds annotated PLY instances, semantic labels,
official change records, and transforms. Predicted geometry is matched to GT on a
common 5 cm grid with maximum-weight one-to-one assignment. IoU >= 0.50 is primary;
IoU >= 0.25 is sensitivity. The evaluator also reports unmatched, duplicate,
fragment, and merge errors. A static and a rigid synthetic witness prove matrix
direction before real evaluation.

## W4: Controlled G/F/R Matrix

The matrix uses the same selected pairs, alignment, 2 cm neural voxel size, label
space, checkpoint, and postprocessing:

- `G_full`: independent geometric correspondence on complete processed inputs;
- `G_supported`: the same method restricted to the measured support mask;
- `F`: independent single-visit Concerto features followed by the same association;
- `R_legacy`: joint ReScene with the legacy projection;
- `R_supported`: the same raw ReScene forward with the supported resolver.

Domain rows are D0 native processed, D1 native restricted by sensor/OVI support, and
D2 actual OVI-supported geometry when available. Missing D2 for a selected pair is an
explicit coverage result, not a fabricated zero. Custom association metrics remain
separate from official `stmetrics` t-AP, t-REC, and per-stage AP.

## W5: Downstream Ownership and Completion

One common-grid experiment compares A0 (B3), A1 (G), and A2 (R) with identical XYZ,
visibility, semantics, and thresholds. It reports AP/PQ plus fragment, merge,
duplicate, and identity errors. Identity never moves geometry by itself.

Completion rows are C0 B3, C1 G plus estimated registration, C2 R plus the same
registration, O1 GT matching plus estimated registration, and O2 GT matching plus GT
transform. O1/O2 are oracle ceilings and never ranking rows. Completion uses actual
OVI shapes and reports opportunity, recoverable surface, precision, and F-score.

## W6: Evidence-Gated Adaptation

The decision is deterministic:

- native R weak: report model limitation; do not train on evaluation pairs;
- native R strong but D2 weak: freeze the encoder and adapt only decoder/mask heads on
  disjoint training environments, selecting by validation t-AP;
- raw masks strong but resolver weak: tune only resolver thresholds on development GT;
- O1 weak and O2 strong: prioritize registration, not identity learning;
- completion opportunity negligible: change the predeclared pair/budget, then rerun
  all compared methods on that same selection;
- identity improves but geometry does not: retain the identity contribution and bound
  the geometry claim.

## Evidence, Failure Handling, and Delivery

Every run appends one row to `configs/evaluation/results/ovi_rescene_real_evidence/experiments.csv`
with command, commit, source hashes, config hash, dataset/pair/domain, seed, device,
runtime, peak memory, status, and artifact root. Large tensors remain outside Git;
small hash-bound summaries, configs, figures, and the handoff report are committed.

Any missing asset, non-finite tensor, identity mismatch, future-session input, output
collision, or evaluator-only GT leak fails closed. Focused unit and integration tests
cover each new contract. Completion requires real P1/P3/P5/P6 values or an explicit
asset-backed partial result that still includes the Apartment native B5 forward.

