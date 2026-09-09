# CROVE Fine Current Map V1 Visual Contract

- Artifact: fixed-camera four-view qualitative plate.
- Target venue / format: AAAI-style two-column paper, full-width PNG preview.
- Core claim: RGB, instance, semantic, and temporal-state renderings are views of one canonical current surface rather than independently altered geometry.
- Reviewer question: do the cleaner boundaries come from an auditable state/label readout on real OVI geometry, or only from recoloring?
- Evidence layer: qualitative inspection backed by the canonical sidecar and manifest hashes.
- Source data: the four `current_*.ply` files emitted by `run_crove_fine_current_map.py`.
- Statistics / uncertainty: none; quantitative metrics remain in the result tables.
- Figure prototype: a 2 x 2 image plate with one shared camera and direct panel titles.
- Panel map: (a) observed RGB, (b) owner instance, (c) semantic label, (d) current evidence state.
- Caption role: identify the scene, source surface, shared-geometry constraint, and gray/unknown limitation.
- Manuscript placement: qualitative current-map subsection after the quantitative T1/T2 comparison.
- Output formats: PNG for the checked-in evidence preview; canonical PLY/NPZ remain local artifacts.
- Traceability: each figure receipt records source path, byte count, SHA256, vertex count, sampled rows, and camera parameters.

The renderer may subsample rows for rasterization, but it must not change the PLY files, labels, current mask, or camera between panels. White is empty canvas, not predicted free space. Colors use the already exported standard RGB fields.
