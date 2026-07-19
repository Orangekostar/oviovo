# OVIV2 Semantic Visualization Export Design

**Date:** 2026-07-19

## 1. Objective

Export publication-ready semantic meshes from the verified OVIV2 Replica runs without
changing mapping, evaluation, or benchmark artifacts. The output must be directly
viewable in common PLY viewers and must use class-consistent colors suitable for a fair
comparison with OVI-MAP and Replica ground truth.

## 2. Reference Behavior

The reference behavior is the pinned OVI-MAP checkout at commit
`58a804e2d7c82ba05a489eb071aba3367301fed8`.

OVI-MAP's documented post-processing path:

1. Reads the reconstructed instance mesh and per-instance semantic features.
2. Classifies each valid instance against the Replica vocabulary.
3. Colors vertices with the class color from `REPLICA_52_PALETTE`.
4. Projects predictions to the GT mesh within 5 cm.
5. Writes `semantic_map_gt_<frame>.ply` with RGB colors and a vertex `label` field.

OVI-MAP also provides an interactive Rerun text-query heatmap. It does not define a
fixed-camera, publication-image export protocol. OVIV2 will reproduce the semantic mesh
behavior, while keeping paper rendering separate from evaluation.

Reference files:

- `OVI-MAP/README.md`
- `OVI-MAP/scripts/utils/mesh_postprocess_utils.py`
- `OVI-MAP/scripts/utils/semantic_const.py`
- `OVI-MAP/scripts/visualizations/search_heatmap.py`

## 3. Visualization Contract

The exporter uses the frozen `replica_runtime_semantic_41` vocabulary from
`configs/evaluation/manifests/replica8.json`.

Colors are keyed by normalized class name, never by scene-local entity ID. Shared class
names reuse the official OVI-MAP `REPLICA_52_PALETTE` RGB value. Classes present in the
frozen Replica-41 vocabulary but absent from OVI-MAP's Replica-51 vocabulary receive
explicit, collision-free RGB values in a repository-owned palette file. Semantic ID `0`
is rendered neutral gray `[200, 200, 200]`.

The same palette must be used for OVIV2, OVI-MAP, and GT panels. Random scene-specific
colors are forbidden for semantic comparison figures.

## 4. Inputs

For each scene, the exporter consumes:

- Native labeled mesh:
  `outputs/oviv2_replica8_frozen/<scene>/final/oviv2_instance_mesh.ply`
- GT-aligned predicted semantic mesh:
  `outputs/oviv2_replica8_frozen/<scene>/evaluation/semantic_map_gt.ply`
- Frozen vocabulary manifest:
  `configs/evaluation/manifests/replica8.json`

The native mesh stores `semantic_id` directly on each reconstructed vertex. The
GT-aligned mesh stores the same prediction after the evaluator's 5 cm projection onto GT
geometry. Despite its filename, `semantic_map_gt.ply` is a predicted semantic map on GT
geometry, not a ground-truth label visualization.

## 5. Outputs

Generated files live outside each verified scene directory:

```text
outputs/oviv2_replica8_frozen/paper_visualizations/
  semantic_palette.json
  semantic_legend.png
  <scene>/
    oviv2_semantic_native.ply
    oviv2_semantic_gt_aligned.ply
    export_manifest.json
```

Both PLY files contain only viewer-compatible geometry, faces, and vertex RGB fields.
Custom semantic fields are omitted from the presentation copy so JavaScript and VS Code
PLY viewers cannot miscompute the binary stride. The source labeled PLY remains the
auditable semantic record.

`export_manifest.json` records:

- scene ID;
- source and output paths;
- SHA256 of every source and output;
- vocabulary and palette hashes;
- vertex and face counts;
- semantic IDs present and their vertex counts;
- unmatched/background vertex count.

## 6. Native And GT-Aligned Roles

`oviv2_semantic_native.ply` preserves reconstructed OVIV2 geometry. It is the primary
qualitative paper artifact because it exposes geometry completeness and semantic quality
together.

`oviv2_semantic_gt_aligned.ply` preserves the evaluator's common GT geometry. It is a
supplementary diagnostic artifact aligned with OVI-MAP's official evaluation
visualization and isolates semantic errors from reconstruction geometry.

The two artifacts must not be presented as interchangeable.

## 7. CLI

Add a dedicated post-processing command:

```bash
python scripts/evaluation/export_oviv2_semantic_visualizations.py \
  --batch-root outputs/oviv2_replica8_frozen \
  --manifest configs/evaluation/manifests/replica8.json \
  --output outputs/oviv2_replica8_frozen/paper_visualizations
```

The command exports all manifest scenes by default and supports repeated `--scene`
arguments for focused runs. It never rewrites source meshes, snapshots, metrics, or the
VERIFIED result manifest.

## 8. Failure Handling

The exporter fails before publishing output when:

- a requested scene is outside the frozen manifest;
- a source PLY is missing or malformed;
- `semantic_id` is absent from a native source;
- a positive semantic ID is outside the frozen vocabulary;
- a vocabulary class has no palette entry;
- vertex colors, positions, or face indices are invalid.

Outputs are written through a temporary directory and atomically published only after all
requested scenes pass validation.

## 9. Verification

Automated tests must prove:

1. The palette covers background plus all 41 frozen classes.
2. Shared class names match OVI-MAP's official Replica palette exactly.
3. Unknown/background vertices are gray.
4. Native output positions and faces are byte-equivalent as arrays to the source.
5. GT-aligned output positions and faces are byte-equivalent as arrays to the source.
6. Every output RGB value is the palette value for its source semantic ID.
7. Viewer PLY files contain no custom `semantic_id`, `entity_id`, or confidence fields.
8. Source/output hashes and counts in `export_manifest.json` are correct.
9. Repeated export is deterministic.

Manual verification opens at least `room0` native and GT-aligned outputs in the selected
VS Code PLY viewer and confirms that both render without the prior DataView offset error.

## 10. Paper Use

All compared methods must use the same class-name palette, scene, crop, camera, background,
and image dimensions. Native-geometry panels belong in the main qualitative comparison.
GT-aligned panels may be used in supplementary material and must be labeled
"GT-aligned evaluation geometry" to avoid implying independent reconstruction quality.

This exporter produces semantic meshes and a legend. Deterministic fixed-camera PNG panel
rendering is a separate follow-up step because OVI-MAP does not publish such a protocol.

## 11. Non-Goals

- Rerunning OVIV2 mapping or evaluation.
- Changing Table 1 metrics or provenance.
- Reclassifying OVI-MAP instances.
- Producing query heatmaps.
- Selecting favorable camera views per method.
