# CROVE Visualization And Failure Attribution Design

## Scope

This design implements Phase 1 and Phase 2 of the dense current-state recovery
program. It does not change a formal `MapSnapshot`, association policy, motion
policy, overlay policy, or evaluator input.

Evidence labels used by this work are `HYPOTHESIS`, `CODE_EVIDENCE`, and
`MEASURED_EVIDENCE`.

## Protected Boundary

The T1 transitive source manifest remains authoritative. In particular, this
phase does not modify `src/oviv2/meshing.py`, `src/oviv2/temporal_association.py`,
`src/oviv2/temporal_runtime.py`, or `scripts/evaluation/run_oviv2_tesse_cd.py`.
New code lives under the evaluation layer and consumes immutable artifacts or
wraps one T2 development execution for observation only.

## P1: Explicit Visualization Contract

Add a standalone labeled-PLY recoloring exporter with four explicit modes:

- `rgb`: preserve the source RGB bytes and require an explicit
  `raw_tsdf` provenance declaration.
- `instance`: map `entity_id` to a deterministic, checkpoint-stable color.
- `semantic`: map `semantic_id` through an explicitly supplied palette.
- `dynamic`: map each `entity_id` through an explicitly supplied state sidecar
  using fixed colors for static, dynamic, unknown/uncertain, and
  dormant/removed.

The exporter preserves every non-RGB vertex property, all faces, and all other
PLY elements. It emits a manifest binding source, output, mode, palette/state
inputs, property invariants, and per-entity color uniformity. Missing evidence
is an error; instance colors are never called raw RGB.

This phase is `visualization-only`. Formal benchmark snapshots and metrics are
not read or written.

## P2: Causal Attribution Capture

Add a T2-only runtime proxy used by a diagnostic launcher. For each delegated
frame call, it temporarily observes the already imported temporal runtime
symbols, invokes every original function exactly once, and records only causal
inputs and returned values. The proxy restores every symbol even on failure.
Tests compare wrapped and unwrapped results, exports, snapshots, and state
digests.

Association records contain observation counts, independently admissible
active/dormant edges, re-ID-qualified pairs, accepted assignments, unmatched
observations/entities, and births. Pair admissibility is replayed through the
public association function on singleton pairs using the exact same frozen
configuration; it cannot affect the live result.

Motion records bind active assignments to the original estimator and router
calls, then record geometry/identity admission, displacement, confidence,
threshold results, streak transition, and final dynamic state. Dormant
reappearance motion is recorded separately from active continuation.

The proxy output is a diagnostic sidecar outside the formal run root. It is
`instrumentation-only`; it is never added to the formal artifact inventory.

## P2: Offline Overlay And Ghost Attribution

For each Apartment common-v2 checkpoint and event, load the composed snapshot,
its bound source temporal snapshot, the immutable anchor snapshot, and the
frozen target arrays. Partition object points into:

- `ovimap_anchor_bound_unchanged`
- `ovimap_anchor_unbound`
- `crove_temporal_moved`
- `crove_temporal_new`

Reconstruct background voxel selection with the same source order and
lexicographic tie rule as composition, partitioning selected points into
`anchor_background` and `temporal_background`. Within each changed region,
count predictions and points within 5 cm of confirmed free space using the
official metric rule. Category counts must exactly sum to the unpartitioned
count and ghost count.

Overlay records enumerate every anchor at every checkpoint with binding,
temporal identity, dynamic state, geometry epoch advancement, displacement,
threshold decision, and resulting moved/removed/occluded state.

## Gates

P1 is GO only when all four modes have explicit evidence requirements,
instance colors are uniform, geometry/IDs/faces remain identical, output is
deterministic, and the T1 source verifier passes.

P2 is GO only when the Apartment report ranks R1-R4 with code paths, measured
counts, representative IDs, and coverage ratios. No P3 threshold or algorithm
change is allowed before that report.

Office is not accessed in either phase.
