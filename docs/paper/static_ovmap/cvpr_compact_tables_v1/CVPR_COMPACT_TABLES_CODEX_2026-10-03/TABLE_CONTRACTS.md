# Three tables, one result store

## Table 1 — complete-system context (six rows)

Columns: Method; Replica APall/AP50/mIoU; ScanNet-CF18 APall/AP50/mIoU.
Top block, visibly marked `reported in OVI-MAP (not rerun)`:
1. Mask3D + OpenMask3D
2. Segment3D + OpenMask3D
3. OVO-SLAM (ordinary setting, NOT its 30-fps variant)
Bottom paired block:
4. OVI-MAP (reproduced): CT_A0_NATIVE
5. Ours — refinement only: CT_A1_E
6. Ours — refinement + recovery: CT_A3_ER

The table caption distinguishes attributed reported values from our paired runs.
External input/supervision/scorer verification belongs in setup/provenance. Unless
raw predictions/scorer identity are verified, do not claim uniform byte-exact
protocol or bold a global best across blocks. User-owned E uses a fitted Q head
and frozen calibrated scalars: do not describe all components as training-free.

The included literature_reference.json is a numeric transcription of the authored
arXivv1 Table3 for these THREE rows. It is not a reproduction result. Before final
publication check against the accessible final CVF paper/table, record differences
without silently overwriting the chosen source version. If the versions disagree,
keep the chosen version explicitly named; never average versions or swap columns.
Known source caution: the reported ScanNet Mask3D+OM3D AP25 is lower than its AP50;
keep an anomaly note, do not invent a correction. AP25 is not a main-table column.

## Table 2 — six controlled conditions

Columns: Configuration; existing-owner readout; recovery; same six metric columns.
Compact implementation may shorten names with a caption legend, never rename U2
as G1. Exact rows A0,A1,A2,A3,A4,A5.
A0–A3 are the refinement x recovery 2x2. Recovery FC evidence is identical in
A2/A3/A5; A4 has the same selected G1 frame/mask but original native representation.
A5 uses exactly A1's existing F evidence and unknown0 where unavailable. It may
win; report that. No additional table of dozens of historical modules.

## Table 3 — recovery mechanism, Replica only (four rows)

Columns: recovery entry; max views; recovered n/N; added TP50; added FP50;
APall; mIoU; incremental semantic time(s/scene).
Rows: none/CT_A1_E; archived U2/CT_H_U2; independent G1/CT_A3_ER;
independent G3/CT_G3.
All rows use D2 for existing owners. n/N counts source-available appended owners
in the shared geometry-defined candidate registry. TP/FP are released matcher score
entries, not an unqualified unique-object count. Retain target-min100 eligibility,
ambiguous ties, lost old matches and old rank changes in SI.
Time: feature-cache-cold, model resident; one run per8scene/3arm; include required
projection, encoding, pooling and export, not mapping or evaluator. Loading is
reported separately. This is NOT end-to-end throughput or real-time FPS.

## Single source of truth and cells

Internal `scene_metrics.json` and `pooled_metrics.json` store metric fractions.
`tables_main.json` cell objects require:
- table_id,row_id,column_id,metric;
- source_kind (MEASURED/EXACT_REUSE/AUTHOR_REPORTED/UNAVAILABLE);
- value_fraction (ornull), display_unit=percent (time/count columns typed separately);
- cohort, ordered_scene_ids, completed/required coverage;
- method_id or cited author method; scoring_protocol_id;
- receipt_path+identity OR source_url+version+table/row/column;
- unavailable_reason where needed.

No hand-written numeric LaTeX. The production generator populates the supplied
layout from these typed records. The same A3 Replica cell must numerically and
provenance-wise match across all three tables. External reported values retain
their source precision; paired results display2decimals (full precision inJSON).
A null is `--` with a caption/status reference, not0. Pool only complete cohorts.

Write per-scene and per-class values, AP25/mAcc and detailed cost in one SI bundle,
not new main experiments. Setup must state input geometry source, single-label
instance masks, official rank formula, actual overlap list, valid ID order,
200-slot schedule and offline/frozen status. Layout target: THREE readable full-
width booktabs tables, 6/6/4 rows, body >=8.5pt (prefer9pt). No large ability matrix,
extra benchmark, new figure-generation task or full manuscript-writing scope.

## Ready-to-run table outputs after the experiment

`tables/table1_main.tex`, `table2_ablation.tex`, `table3_recovery.tex`,
`table_layout_preview.tex`, `table_layout_preview.pdf`, plus JSON/CSV and
`cell_provenance.json`. The supplied PDF is a PLACEHOLDER layout only. Replace
internal placeholders from actual receipts; runLaTeX and visually inspect the
regenerated preview. Do not change a row to improve the story.
