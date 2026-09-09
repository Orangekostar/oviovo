# CROVE Fine Current Map V1: Implementation and Evaluation Plan

## Objective

Produce one prediction-grounded current surface with four color views while
preserving OVI-MAP's native 1 cm geometry. CROVE supplies current validity,
ownership, and semantics. Display colors never define evaluation labels.

## Fixed Inputs

- Evidence commit: `4870b928960783e8d66e9e5e6b2c751d27889400`.
- Development branch: `research/crove-fine-current-map-v1`.
- Static DEV: Replica `room0`, 200 frames at stride 10.
- Dynamic DEV: TESSE-CD Apartment two-visit artifacts, B2 and B3 controls.
- Static fine source: native OVI-MAP `instance_mesh_200.ply`, 9,282,303
  vertices and 3,094,101 faces.
- Dynamic fine sources: native OVI-MAP Apartment t0/t1 meshes and the frozen
  two-visit B3 provenance.
- The user-visible bad PLY was not identified. The evidence-bound room0
  65,350-vertex CROVE export is the diagnostic representative and is labelled
  `USER_PLY_NOT_IDENTIFIED/EVIDENCE_BOUND_REPRESENTATIVE`.

Ground-truth geometry and `*_map_gt` artifacts are evaluator-only. They must
not contribute vertices, faces, RGB, current validity, owners, or semantic
predictions.

## Data Model

Add a compact `CurrentSurfaceView` whose rows preserve source vertex order:

```text
vertices_xyz, normals_xyz, triangles
source_surface_index, source_vertex_index, source_visit, geometry_epoch
observed_rgb, rgb_valid
current_valid, evidence_state, last_supported_frame
owner_entity_id, owner_confidence
semantic_id, semantic_confidence, semantic_support_reliability, semantic_source
```

The canonical sidecar retains invalid historical rows for audit. Each
`current_*.ply` contains the same current-valid vertex selection and the same
all-valid source triangles. Only standard RGB differs between the RGB,
instance, semantic, and state views.

## Current-Validity Rule

The existing 5 cm signed grid is a candidate index, not a fine-surface delete
decision. A historical fine vertex is revoked only when:

1. its parent coarse cell is a visible-free candidate;
2. the vertex itself is depth-testable and visible-free in at least two
   observations; and
3. the supporting camera locations satisfy the configured baseline.

Current-visit vertices are observed-current. A historical vertex that is
depth-consistent with current geometry is replaced by the current-visit
surface. Occluded, invalid-depth, and out-of-view evidence cannot revoke it.
This keeps the 5 cm temporal state unchanged and prevents coarse-cell evidence
from being broadcast to every 1 cm surface row.

## Semantic Readouts

- `S0`: nearest legacy CROVE semantic and its legacy confidence.
- `S1`: bounded distance-weighted neighborhood vote on the fine surface.
- `S2`: S1 distribution plus an independent absolute support reliability.
  Low-reliability local predictions fall back to the owning OVI instance
  semantic when that source is supported. Normalized class probability is
  never reused as observation reliability.

All strategies preserve source geometry. Correspondence is bounded by a
declared metric radius and processed in chunks. Unknown remains semantic ID 0.

## Experiment Matrix

| ID | Scene | Surface | State | Semantics | Purpose |
| --- | --- | --- | --- | --- | --- |
| D0 | room0 | old CROVE 5 cm | static | existing | display-only control |
| D1 | room0 | OVI native 1 cm | static | S0 | geometry/correspondence control |
| D2 | room0 | OVI native 1 cm | static | S1/S2 | semantic selection |
| D3 | Apartment | OVI t0/t1 1 cm | B2/B3/fine gate | OVI owner | dynamic selection |
| D4 | frozen scenes | selected 1 cm | selected | selected | confirmation |

The 2 cm condition is obtained only by deterministic source-surface voxel
selection and is reported as a cost/quality readout, never as a new
reconstruction. It is labelled accordingly. If a true 2 cm reconstruction is
available, it remains a separate backend condition.

## TDD Gates

1. Four outputs share exact geometry and numeric labels; only RGB changes.
2. Source triangles survive only when all three selected vertices are current.
3. Coarse visible-free without fine evidence cannot revoke a vertex.
4. Occluded, unobserved, and invalid-depth evidence cannot revoke a vertex.
5. Low-support singleton semantics cannot report confidence one or suppress a
   reliable owner semantic.
6. Correspondence outside the configured radius remains unknown.
7. Chunk size does not change validity, ownership, semantics, or hashes.
8. Existing T1 paths and targeted regression tests remain unchanged.

## Required Evidence

The run writes `input_selection.json`, `trial_log.csv`,
`metrics_per_scene.csv`, `metrics_summary.json`, `export_diagnostics.json`,
`effective_geometry_config.json`, `selected_config.json`, `table_values.json`,
`table_lineage.csv`, `compact_artifact_index.json`, fixed-view figures, and
the four current PLYs. Large maps may stay local under `LOCAL_ONLY_POLICY`;
compact evidence and fixed views are committed.

Every table value is tagged `RECOMPUTED`, `HISTORICAL_RECORDED`,
`PROTOCOL_INCOMPARABLE`, `MISSING_ARTIFACT`, `NOT_RUN`, or `N/A`. No new model
is trained, so the final handoff records
`MODEL_UPLOAD_STATUS=NOT_APPLICABLE_NO_NEW_TRAINING`.
