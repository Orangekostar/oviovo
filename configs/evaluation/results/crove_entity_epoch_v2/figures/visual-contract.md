# Entity-Episode DEV Figure Contract

- Artifact: `entity_epoch_dev_summary.{svg,pdf,png}`
- Target venue / format: AAAI two-column paper, full-width vector figure plus 300 dpi preview.
- Core claim: the entity-episode update restores source-bound local surface support without Ghost or deletion errors, but the measured recovery does not improve current mIoU over frozen B3.
- Reviewer question: does the added state machinery create a real current-map gain, and what changes when the headline gain is absent?
- Evidence layer: main comparison plus mechanism and limitation.
- Source data: `../metrics_per_pair.csv`, `../aggregate_metrics.json`, the bound V1 canonical OVI surface, and the D2 entity-episode state sidecar identified in `entity_epoch_dev_summary.json`.
- Statistics / uncertainty: one fixed Apartment DEV pair; deterministic point estimates; no multi-seed uncertainty claim.
- Figure prototype: zero-based categorical bars, paired delta bars, and one fixed-camera canonical-map crop.
- Panel map: (a) current mIoU for all executed hypotheses; (b) surface precision/recall delta against B3 for state/identity variants; (c) D2 rows restored in the densest real 10 cm recovery voxel and its 45 cm context.
- Caption role: state the measured local recovery, lack of mIoU gain, zero Ghost, and single-pair limitation.
- Manuscript placement: experiments, immediately after the entity-episode ablation table.
- Output formats: editable SVG, PDF, and 300 dpi PNG.
- Traceability: exact rows, counts, crop center, metric numerator/denominator, and source hashes are stored in `entity_epoch_dev_summary.json` and bound by `../compact_artifact_index.json`.
