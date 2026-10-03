# CVPR Compact Tables Design

The authoritative scientific specification is the unchanged package in
`docs/paper/static_ovmap/cvpr_compact_tables_v1/CVPR_COMPACT_TABLES_CODEX_2026-10-03/`.
The executable specification is its byte-identical copy at
`configs/static_ovmap/cvpr_compact_tables_v1.json`.

## Decisions

Use a new `static_ovmap.cvpr_compact` task with a task-local controller. Preserve
the historical task guards. Reuse numerical operators, payloads, consumed-file
memoization and the released scoring adapter, rather than invoking historical
high-level runners with substituted task identities.

Bind the eight BB00_NATIVE Replica maps from actual recovery/backbone receipts.
Read the final three Replica temperatures and check equality across all eight
maps. The development leave-one-out temperatures are not the main temperatures.
Build missing CF18 inputs, maps and N/Q/F with the same fixed operators. A0-A5
run on the full ordered 26 captures; U2/G3 run only on Replica. CF18 comprises
seven physical families. Exposure and training/calibration overlap remain explicit.

Build one CPU Open3D BVH over the full final predicted mesh per scene. Keep mixed
owner faces as owner-zero occluders. Use unnormalized camera rays at integer pixel
coordinates so hit time is camera-z, then apply the fixed 2cm/2% measured-depth
test. Keep original primitive indices when excluding exactly degenerate triangles.
All completed cameras are eligible without historical semantic requests. Rank
geometry-only masks before encoding, retain the first three distinct frames, and
share the first mask across G1 FC/native and G3. Do not replace a failed view.

Projected requests have a distinct schema, actual capture identities, both bbox
conventions and compact masks. FC dense features share only exact physical frame
identity, while pooled features bind masks and operators. Native uses the original
six crops with the legacy pixel-max upper bound and target mask as union mask.

Outputs preserve all incumbent support and append only absent-owner residuals.
A2/A3/A5 share successful G1 FC recovery. A5 reuses existing F and preserves a
positive owner's support with class zero when F is unavailable. Expand evaluation
mask dictionaries with genuinely unavailable Native rows for recovered owners.
Check the released nine IoU thresholds within 1e-12; never modify the evaluator.

The feature-cache-cold timer calls the production recovery function with resident
weights/text, no persistent visual-feature/view-result reads and synchronized CUDA
boundaries. Replay each Replica scene/arm exactly once, verify output parity and
average all eight measured costs. Model load, mapping and scoring costs are separate.

Typed cells from one result store generate all three tables with receipt-level
provenance. External rows are attributed arXiv-v1 Table 3 values in a separate
block; check the final CVF source for differences. No cross-source best bolding.
Incomplete internal cohorts produce null cells. A3 is fixed, deployment unchanged.

## Execution Gates

Use synthetic fixtures and only the existing scene0056_00 map for development.
Commit the complete implementation and experiment freeze before new main
predictions/metrics. Label-free input preparation and capture may precede the gate.
Later fixes require explicit amendments and affected-descendant invalidation.

Completion requires 172 complete output records, 14 whole-cohort pools, 24 real
cold timing records, generated readable PDF tables, four reports, a requirement
audit and a normal push with the full remote SHA verified. Until these all exist,
the goal remains active and no global COMPLETE status is permitted.
