# CROVE Dense Current-State Recovery Design

## Objective

Recover CROVE's TESSE-CD current-state map quality without changing the frozen
T1 path or weakening causal evaluation. The target is a dense, stable OVI-MAP
static backbone whose moved objects retain dense object-local geometry while
CROVE remains the authority for identity, motion, lifecycle, and current state.

This design implements the approved recovery prompt in
`/home/ww/crove/docs/CROVE_DYNAMIC_MAP_RECOVERY_CODEX_PROMPT.md`. Claims use
only `HYPOTHESIS`, `CODE_EVIDENCE`, or `MEASURED_EVIDENCE` labels.

## Frozen Boundary

- Base branch: `codex/crove-input-witness-opt-20260815`.
- Base commit: `c55396b24d7706016a569c951914c79c9f641408`.
- Research branch: `research/crove-dense-current-state-recovery`.
- T1 protected files, metrics, configs, and runtime behavior remain unchanged.
- Runtime may consume only the current and prior frames. Ground truth,
  evaluator output, and future frames are forbidden runtime inputs.
- Apartment is the development scene. Office remains held out until one final
  configuration is frozen.

## Established Evidence

### PLY provenance

`MEASURED_EVIDENCE`: the Apartment causal anchor binds
`instance_mesh_263.ply` at SHA-256
`8459a59cba3f59ece14a9f3462dbe0796d00c712214e558c487bde9dbbcffb57`.
Its header declares 8,194,551 vertices and 2,731,517 faces with RGB vertex
properties.

`CODE_EVIDENCE`: the production chain is:

1. OVI-MAP `scripts/panoptic_mapping_.py` calls
   `GlobalSegmentMap_py::generateMesh(..., false, false, true)`.
2. `global_segment_map_py.cpp` invokes the instance mesh integrator and writes
   `instance_mesh_<frame-count>.ply`.
3. `label_tsdf_mesh_integrator.cc` selects `kInstance` and obtains every vertex
   color from `instance_color_map_` using the voxel's instance label.
4. `build_tesse_ovimap_static_anchor.py` groups vertices by those authoritative
   colors and stores each group as one anchor entity.

Therefore the native anchor PLY is an instance-palette visualization, not an
RGB texture mesh. Variation among its registered palette colors denotes
different instance IDs; it must not be described as RGB color noise.

### Geometry authority collapse

`MEASURED_EVIDENCE`: the frozen Apartment anchor contains 63 entities and
8,194,548 entity points. At the last measured selective checkpoint, 50
unchanged anchor entities retain 5,394,094 points. Five moved anchor entities
switch from 1,937,312 anchor points to 3,502 temporal points, a retained-point
ratio of about 0.181%. Eleven new temporal entities contain 8,734 points in
total, with a median of 294 points.

`CODE_EVIDENCE`: `compose_anchor_checkpoint` emits the temporal prediction for
a confirmed moved binding. The temporal prediction is produced from an
`ObjectSubmap` containing at most one representative point per 5 cm voxel. The
current implementation therefore changes both pose authority and geometry
source when an anchor is marked moved.

`HYPOTHESIS`: the dense-to-sparse source switch is the primary cause of visibly
degraded moved/new object surfaces. P0 will make the comparison reproducible;
P2 must independently attribute Dyn/Ghost failures before any metric tuning.

## Architecture

The recovery path separates four authorities:

1. **Static geometry authority:** immutable causal OVI-MAP anchor geometry.
2. **State authority:** CROVE temporal identity, motion, lifecycle, and epochs.
3. **Readout geometry authority:** unchanged objects use the anchor directly;
   moved bound objects may use a causally transformed anchor-local template;
   genuinely new objects use temporal geometry or a later causal reservoir.
4. **Visualization authority:** deterministic colors derived from IDs or states;
   recoloring never changes geometry, IDs, formal snapshots, or metrics.

Formal and candidate readouts are always emitted as separate artifacts. A
candidate can replace the formal readout only after its phase gate passes.

## Phase Gates

### P0: provenance and map-quality audit

Add a read-only audit tool that verifies every declared hash and byte count,
binds the exact PLY producer sources, classifies the PLY color semantics, and
compares anchor/current point counts by overlay and geometry authority. It
writes canonical JSON plus a concise Markdown report. It does not alter a map
or run the benchmark.

GO requires an exact producer chain, verified artifact bindings, and a
quantified authority transition. Ambiguous provenance is NO-GO.

### P1: visualization contract

Add explicit `rgb`, `instance`, `semantic`, and `dynamic` modes. ID-based modes
use stable palettes. Tests must prove byte-identical positions, entity IDs,
semantic IDs, lifecycle state, and formal metrics across recoloring.

### P2: causal failure attribution

Instrument association admission/rejection, motion-evidence qualification,
epoch transition, overlay state, and ghost authority. Aggregate event counts,
affected IDs, and ranked root causes on Apartment. No threshold tuning is
allowed until the failure funnel is measured.

### P3: motion-aware association

Use the already computed predicted centroid only inside the existing local
active gate. Add a normalized motion-consistency score to ranking; never widen
candidate admission. Compare C0, C1, and C2 on Apartment, where C2 must be
behaviorally equivalent to C0. Proceed only if the measured funnel implicates
association ranking or reappearance identity.

### P4A: dense moved-object readout

For a moved bound entity, preserve the anchor point set as an immutable
object-local template and transform it using only causal CROVE motion. The
first pilot is centroid translation. SE(3) is forbidden until coordinate-frame
and correspondence evidence supports it. Emit metadata for geometry authority,
state authority, template ID, transform source, and resolution.

The transformed candidate is readout-only first. The original formal readout
continues unchanged for comparison and rollback.

### P4B, P5, and P6

- P4B adds a causal multi-view reservoir only for genuinely new objects and
  only after P4A passes.
- P5 changes ghost handling only for the dominant contributor measured in P2.
  Suppression must be visibility grounded, reversible, and multi-view where
  applicable.
- P6 adds ReScene only as an identity backend if identity discontinuity remains
  dominant. Missing weights permit only an interface/converter/smoke artifact,
  not a result claim.

## Evaluation And Promotion

The reference Apartment interval metrics are Obj. `0.337883`, Dyn. `0.069225`,
Chg. `0.091272`, current mIoU `0.148771`, and Ghost `0.443760`. The existing
geometry-supported candidate reaches Obj. `0.348472` and mIoU `0.149560` with
the same Dyn. and Ghost and Chg. `0.088458`.

The final candidate must target Obj. at least `0.348472` (ideally above
`0.372762`), Dyn. above `0.069225`, Chg. at least `0.088458`, current mIoU at
least `0.149560`, and Ghost no greater than `0.443760`. These are promotion
targets, not fabricated results. Every reported value must come from a
source-bound measured artifact.

Once Apartment selection is complete, freeze the configuration and run Office
once. Any post-Office tuning invalidates the holdout and requires explicit
disclosure rather than silent reuse.

## Failure And Rollback

Each phase is a small commit with focused tests, artifact hashes, measured
results where applicable, and a Decision Ledger entry. A NO-GO blocks dependent
phases. Failed candidates remain labeled and cannot update paper tables.
Rollback is the phase commit boundary; the frozen base branch is never edited.
