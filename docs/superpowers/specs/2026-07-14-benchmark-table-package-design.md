# AAAI Benchmark Table Package Design

**Date:** 2026-07-14
**Status:** Approved structure; numerical cells intentionally unfilled

## 1. Goal

Create a paper-ready benchmark table package for OVIOVO with four main-paper tables and three supplementary tables. Every numerical cell must remain machine-fillable and traceable to a frozen benchmark artifact; no paper number may be manually copied into a locally reproduced row.

## 2. Deliverables

The implementation creates three synchronized files:

- `docs/paper/benchmark_tables.md`: readable protocol notes and all seven table previews.
- `docs/paper/benchmark_tables.tex`: AAAI-ready `booktabs` LaTeX tables.
- `docs/paper/benchmark_tokens.tsv`: one source-of-truth row for every numerical token.

Markdown and LaTeX use identical row labels, columns, ordering, and `{{TOKEN}}` names. The TSV records `token`, `table`, `method`, `dataset`, `split`, `metric`, `direction`, `precision`, `source_json`, `json_pointer`, `status`, and `note`.

Allowed token states are `UNFILLED`, `VERIFIED`, and `N/A`. `N/A` is used only when a method cannot produce the metric under the frozen protocol. Failed or not-yet-run experiments remain `UNFILLED`.

## 3. Main-Paper Tables

### Table 1: Static Mapping Quality

**Claim:** Lifecycle-aware OVIOVO starts from a competitive static open-vocabulary map rather than trading away static quality.

Rows:

1. OpenFusion
2. OVI-MAP
3. ConceptGraphs
4. DualMap
5. OVIOVO

Table 1 is one `table*` with two vertically stacked panels to preserve readable type size:

- Panel A, semantic quality: method, mode, Replica-8 mIoU/mAcc/f-mIoU, held-out Replica-7 mIoU, and ScanNet200-5 mIoU.
- Panel B, instance and geometry quality: method, Replica-8 AP25/AP50/Geometry F@5cm, held-out Replica-7 AP50, and ScanNet200-5 AP25/AP50/Geometry F@5cm.

OpenFusion receives `N/A` for native entity AP. Replica-8 is a compatibility result; held-out Replica-7 and ScanNet200-5 carry the generalization claim. Oracle-instance semantic scores are excluded from primary columns and may appear only as diagnostics in the supplement.

### Table 2: Dynamic Current-Map Quality

**Claim:** Visibility evidence and reversible maintenance reduce stale geometry while recovering current objects and revealed background.

Rows:

1. OVI-MAP (frozen)
2. ConceptGraphs (frozen)
3. DualMap
4. Panoptic Mapping + shared masks
5. Khronos (open-set)
6. Khronos (GT semantics oracle)
7. OVIOVO

Table 2 is one `table*` with two panels:

- Panel A, official TESSE-CD metrics: method/mode and Apartment/Office object F1, dynamic-object F1, and change F1.
- Panel B, common current-map metrics: method/mode, Current mIoU, Ghost Rate, Background Recovery F@5cm, and Recovery frames.

The Khronos oracle row is visually separated and never participates in best/second-best formatting. Frozen static baselines receive the pre-change stream once and cannot update after the intervention.

### Table 3: Causal Component Ablation

**Claim:** Each proposed mechanism addresses a distinct failure mode.

Rows form a cumulative ladder:

1. Positive-only geometry-first base.
2. Add signed visibility evidence.
3. Add reversible TSDF ownership.
4. Add background reclaim.
5. Add dormant re-identification.
6. Add calibrated `NOT_FOUND`.

Columns are variant name, newly added mechanism, Static mIoU, Change F1, Stale FP, Ghost Rate, Background F@5cm, ID switches, Reactivation R@1, and `NOT_FOUND` F1. Cumulative variant names replace five separate binary columns so the table remains readable.

Each row adds exactly one component. If component interaction cannot be inferred from the cumulative ladder, a focused two-factor mini-ablation is added in the supplement rather than widening this table.

### Table 4: Online Efficiency and Memory

**Claim:** Current-state maintenance has bounded online and memory cost without hiding deferred work.

Rows:

1. OVI-MAP
2. ConceptGraphs
3. DualMap
4. Khronos
5. OVIOVO without maintenance
6. OVIOVO

Table 4 uses two panels:

- Panel A, latency: mode, frontend/backend/maintenance seconds per frame, total online seconds per frame, processed Hz, finalization seconds, and query p50/p95 milliseconds.
- Panel B, resources: mode, peak GPU GB, peak RAM GB, final map MB, and any separately measured evaluation/I/O seconds.

Initialization, online processing, finalization, and evaluation/I/O remain separate. Paper-reported hardware numbers cannot be mixed with local same-hardware measurements in one ranked column.

## 4. Supplementary Tables

### Table S1: Open-Vocabulary Current-State Localization

**Claim:** OVIOVO answers from the current scene rather than retrieving a semantically similar stale object.

Rows:

1. OVI-MAP (frozen)
2. ConceptGraphs (frozen)
3. DualMap
4. Khronos + shared text head
5. OVIOVO

Columns:

- Present Current R@1, moved-object R@1, new-object R@1.
- `NOT_FOUND` F1, Stale FP, median localization error in meters, and Recovery frames.
- Validation-scene and held-out-test macro averages are reported separately.

`Bring me the new book` is normalized for every method to `action=locate`, `category=book`, `temporal_predicate=added_since_previous_visit`, and `scope=current`. Manipulation and navigation success are outside this benchmark.

### Table S2: Temporal Identity on 3RScan

**Claim:** Dormancy and re-identification preserve identity across visits without access to future scans.

Rows:

1. ESAM (per-visit)
2. Khronos (adapted stream)
3. ReScene4D (offline)
4. OVIOVO without re-ID
5. OVIOVO

Columns:

- Online/offline mode, per-stage AP50, t-AP, t-REC, ID switches, Reactivation R@1, and False ReID.

Offline ReScene4D is an upper bound and is not ranked as an online method. A row is removed rather than populated from a paper number when official code cannot produce protocol-valid predictions.

### Table S3: Absence Reliability and Calibration

**Claim:** A current-state map must reject absent targets rather than always return the nearest embedding.

Rows:

1. DualMap + calibrated rejector
2. ConceptGraphs + calibrated rejector
3. Khronos + shared text head + calibrated rejector
4. OVIOVO without calibration
5. OVIOVO

Columns:

- Presence AUROC.
- `NOT_FOUND` precision, recall, and F1.
- Binary ECE and risk-coverage AUC.

All rejectors use one validation-only calibration policy. The fitted threshold artifact is serialized before test execution and included in the result manifest.

## 5. Token and Formatting Rules

Token format is `{{<TABLE>_<METHOD>_<DATASET_OR_SPLIT>_<METRIC>}}`, uppercase ASCII with underscores. Examples:

- `{{T1_OVIOVO_REPLICA8_MIOU}}`
- `{{T2_OVIOVO_OFFICE_CHANGE_F1}}`
- `{{S1_OVIOVO_TEST_NOT_FOUND_F1}}`

Every token appears exactly once in the TSV and at least once in both Markdown and LaTeX. Method and dataset names are normalized in the TSV; display names may contain punctuation only in the rendered tables.

Formatting rules:

- Captions precede tables and state the protocol, split, and meaning of composed/oracle rows.
- LaTeX uses `booktabs`, no vertical rules, `\toprule/\midrule/\bottomrule`, and `\cmidrule` for grouped datasets. Wide tables use stacked panels rather than `\resizebox` or unreadably small text.
- Metric headers include direction arrows and units.
- Quality and probability metrics use three decimal places; seconds/frame, milliseconds, GB, and MB use two decimals.
- `--` is a rendering-only form of TSV state `N/A`; unfilled cells always retain visible `{{TOKEN}}` placeholders.
- Best and second-best formatting is not applied until every compatible row in the comparison group is `VERIFIED`.

## 6. Source and Fairness Contract

Each `VERIFIED` token must point to a committed JSON result through `source_json` and `json_pointer`. The associated run manifest records repository commit, environment, command, configuration hash, model weights, dataset split, pose source, hardware, and online/offline mode.

Native, composed, frozen, offline, and oracle rows are named explicitly. Runtime inputs cannot contain GT semantic, instance, change, or visibility annotations. Online rows cannot use future frames. Calibration and lifecycle thresholds are fitted only on declared validation scenes.

## 7. Verification

The table package verifier must enforce:

1. Exactly four main and three supplementary table labels.
2. Unique tokens and valid TSV states.
3. Markdown/LaTeX token sets equal the TSV token set.
4. Every `VERIFIED` token has a non-empty JSON path and pointer.
5. Every `N/A` token has an explanatory note.
6. No unfinished-marker strings or manually entered numeric result cells.
7. Captions identify offline, composed, frozen, and oracle rows.
8. `room0` is never the sole source of a headline static claim.

The initial implementation is template-only: all compatible numerical cells begin as `UNFILLED`. Number import and result highlighting are separate later tasks.
