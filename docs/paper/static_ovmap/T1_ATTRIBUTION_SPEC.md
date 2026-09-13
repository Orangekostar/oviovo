# Codex Execution Prompt: Audit and Attribute SpaCeFormer + OVI Fusion Gains and Losses

## 1. Mission and execution boundary

Work in `Orangekostar/oviovo`. Implement and run a small, controlled attribution study of the existing R6/T1 fusion. Determine which objects and regions improve or deteriorate, and whether the cause is candidate addition, label borrowing, fusion-stage suppression, scoring, geometric transfer, or final instance ownership.

**Do not assume that T1 already works reliably. Do not optimize a leaderboard column before explaining the existing result. Do not stop after producing another plan.** Deliver executable code, real cached-prediction experiments, four evidence tables, and a justified next-step decision. A well-explained negative result is a valid completion.

Reviewed baseline:

```text
Repository:       Orangekostar/oviovo
Source branch:    research/ovimap-static-benchmark-v1
Reviewed commit:  8cefe6b4bca464e0478f82e6ef1e6d0e3783e347
Suggested branch: research/ovimap-t1-attribution-v1
```

Inspect local HEAD, branch, remote, and worktree status. Protect existing changes. Use an isolated worktree/branch; do not reset, overwrite, or force-push existing work. If this task branch already contains related work, inspect and continue it rather than creating a conflicting implementation. Record relevant differences from the reviewed commit.

This is an execution specification, not evidence that the following experiments have run. Repository receipts identify historical assets; they do not establish current file availability.

**Initial scope: Room0, two existing independent FP32 predictions, cached postprocessing and evaluation only.** Do not retrain, load additional models for new inference, replace encoders, alter frame sampling, download ScanNet, or rerun the mapping/network pipelines. Missing historical assets block only dependent comparisons. Do not reopen unrelated CROVE dynamic audits or repeatedly search unchanged storage locations.

All new prediction policies must use prediction-side information only. GT belongs exclusively to evaluation and diagnostic processes. Extra 3D pretraining remains explicitly disclosed; exact training-scene exclusion remains unverified unless independently established.

## 2. Read the actual implementation before editing

Read these files and their directly relevant callers. They were checked at the reviewed commit. Do not substitute old prose summaries for source or machine-readable results.

| Existing file | Responsibility and relevant behavior |
|---|---|
| `src/static_ovmap/proposal_fusion.py` | `fuse_proposals`: OVI-first proposal union; geometric label borrowing; class-agnostic fusion NMS; common-domain area scores. |
| `scripts/evaluation/fuse_static_spaceformer.py` | Projects native OVI owners onto the **returned T0 coordinates** using strict distance `<0.05 m`; writes T1, projection, ledger, and receipt. |
| `scripts/evaluation/evaluate_static_proposals.py` | Projects overlapping proposals; couples supplied confidence to AP ranking and semantic overlap assignment; serializes AP confidence to six decimals. |
| `scripts/evaluation/run_static_spaceformer.py` | Produces postprocessed T0, raw logits/queries, and text embeddings. T0 already includes the model's internal filtering/NMS. |
| `scripts/evaluation/evaluate_static_ovmap_instances.py` | Loads released export/evaluation functions, checks historical native parity, and records explicit instance-class IDs. |
| `scripts/evaluation/evaluate_static_ovmap_readout.py` | `semantic_metrics`: evaluates valid GT vertices and averages over classes with GT support. |
| `src/static_ovmap/released_loader.py` | Loads the released evaluator; preserve its behavior and dependency isolation. |
| `src/evaluation/static_projected_instances.py` | Existing separately named geometric diagnostics; inspect its actual matching/threshold rules before using them. |
| `docs/paper/static_ovmap/R6_RESULTS.md` | Historical R6 results, adaptations, repeatability limits, and comparison boundaries. |
| `docs/paper/static_ovmap/S1_RESULTS.md` and `REPLICA8_RESULTS.md` | Separate historical Room0 evidence from rebuilt Replica8 results and S1a from native readout. |
| `artifacts/static_ovmap/room0_spaceformer/large_artifacts.json` and run receipts | Locate exact input/output assets and executed commands. |

Read the actual evaluator files under the recorded `--evaluator-root`: `scripts/eval_utils.py`, `scripts/eval_sem_seg.py`, `scripts/eval_inst_seg.py`, and `scripts/utils/semantic_const.py`. Compare their hashes with the receipts. The reviewed upstream revision is `OVI-MAP/OVI-MAP@f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424`; the receipt-bound local files, including any disclosed changes, define historical reproduction.

### Historical anchors, not acceptance targets

For the recorded Room0 R6 run, proportions were:

| Condition | Semantic instance AP | AP50 | AP25 | Semantic mIoU |
|---|---:|---:|---:|---:|
| Native OVI reference | 0.154316 | 0.343233 | 0.368233 | 0.332857 |
| OVI + S1a reference | 0.186723 | 0.384900 | 0.409900 | 0.362175 |
| T0, released score | 0.140955 | 0.310082 | 0.361648 | 0.283307 |
| T1, raw source-area score | 0.205160 | 0.411659 | 0.503194 | 0.218774 |
| T1, within-class area normalization | 0.205160 | 0.411659 | 0.503194 | 0.351151 |

T0 repeat AP was 0.130527. These are not Replica8 averages, a confidence interval, or guaranteed targets for a rebuilt run. Use full-precision artifacts for parity, not the rounded table.

The recorded primary run had 64 supported OVI/S1a candidates and 141 T0 candidates; 160 survived fusion NMS: 64 OVI and 96 SpaCeFormer. There were 64 label-borrowing events, of which 19 survived NMS. **Borrowing does not necessarily change a label and does not establish correctness.** Do not hard-code these counts for other runs.

## 3. Bind inputs and lock evaluation semantics

### 3.1 Minimum asset inventory

Locate and validate only the assets used by this task:

- Each independent FP32 run's `T0.npz`, identity, and receipt.
- The exact OVI/S1a readout, native mesh, geometry-support record, and `native_vertex_owners.npy` that generated historical T1.
- Historical `T1.npz`, `native_projection.npz`, `proposal_ledger.json`, and evaluation receipts/manifests.
- The native, pre-S1a OVI readout from the **same mesh/cache generation**, if available, for the readout-dependency control below.
- Identical evaluation meshes, converted GT instance IDs, text valid IDs, and evaluator source.
- Raw logits, query embeddings, and text embeddings only if a later, explicitly justified diagnostic needs them. Do not load the roughly 1.49 GB raw logits merely to run the core matrix.

Historical root hint:

```text
/home/ww/oviovo_baseline_runs/20260913_static_ovmap/room0/
```

Resolve exact paths from receipts; the hint is not a presence guarantee. Verify hashes for assets actually consumed, coordinate order, shape, units, finite values, and ownership alignment. Read large files once, memory-map or chunk where appropriate, and reuse validated projections. No all-repository hashing or broad repeated scanning.

Throughout this study, `source_area` is a count of supported prediction-domain points, not square metres or a calibrated quality probability. `projected_area` is the count of evaluation-domain sample vertices covered by the fixed projection. Do not conflate their sampling densities.

The common prediction domain is `T0.npz['coord']`, not the input cloud's original rows or GT mesh. Reuse nearest-neighbor indices across runs only after exact coordinate/order equality is established; otherwise bind a separate projection for each run. Never run ICP or silently reorder coordinates to obtain parity.

Stable identity:

```text
candidate_id = scene / prediction_run / source / native_owner_or_query_id
```

Also keep a canonical candidate index, original T0 row/query ID, native owner ID, model run, and OVI readout identity. Query index equality across independent runs is not object identity. Compare runs by GT-object outcomes or explicitly geometric diagnostic matching.

### 3.2 Evaluator protocol is executable behavior, not a familiar metric name

Export a small `metric_protocol.json` from the actual loaded evaluator. Record:

- Exact runtime overlap arrays, comparison operators, AP integration, duplicate handling, confidence tie handling, GT/prediction filters, void/group treatment, minimum projected region size, class IDs, and absent-class averaging.
- The 51-class semantic vocabulary and the released 48-class instance subset separately, using actual arrays rather than just their lengths.
- Native-to-common and common-to-evaluation projection thresholds and coordinate conventions.
- Confidence precision before export, exact six-decimal manifest values, and evaluator-parsed values.

At the reviewed upstream commit, `eval_utils.init` constructs `np.append(np.arange(0.5, 0.95, 0.05), 0.25)`. The all-AP portion contains nine thresholds near 0.50 through 0.90; 0.95 is not included. Matching uses strict `overlap > overlap_th`. **Record the actual floating-point values. Do not round thresholds, replace this with COCO AP, or silently fix it.** A separately named diagnostic may use different thresholds, but is not released-AP parity.

Semantic evaluation ignores GT-invalid rows, but valid GT vertices with prediction 0/unmatched geometry remain false negatives. Keep prediction 0 in confusion-matrix columns. Average over the same GT-supported classes as the existing implementation. Do not remove unmatched valid vertices or average only over jointly observed regions. Preserve no-GT class AP as null with a reason, not zero.

Released instance filtering uses the projected evaluation domain, including a minimum size of 100 vertices. Do not prefilter new predictions using GT-region overlap or dynamically remove difficult objects. Record, separately, source-empty, projected-empty, projected-too-small, invalid-instance-class, evaluator-ignored, and counted prediction outcomes.

The released evaluator is not interchangeable with a generic score-greedy or Hungarian matcher. Reuse/instrument its actual code. If a diagnostic uses maximum-cardinality one-to-one matching or max-IoU coverage, name it separately and record its denominator. Never substitute it for released TP/FP/AP.

## 4. Freeze candidates and separate four operations

Refactor minimally into testable operations while preserving the old public call and default behavior:

```text
build_frozen_candidates(...)  -> immutable prediction-only pool
apply_label_reuse(...)        -> labels and borrowing ledger only
apply_fusion_nms(...)         -> kept IDs and suppression edges only
resolve_native_owners(...)    -> unique owner on prediction-domain points
```

Separate:

```text
nms_priority       # visit order for fusion NMS
rank_score         # instance AP confidence
assignment_priority_or_score  # competition for final map ownership
canonical_index    # fixed export/tie identity, independent of NMS visit order
```

Current defaults must reproduce OVI-first ordering: OVI by descending common-domain area, then owner ID; SpaCeFormer by stable descending original released score. Borrow labels at IoU `>=0.5`; suppress at IoU `>=0.7`, irrespective of class; output raw common-domain area scores. Preserve the original best-owner tie rule and first accepted suppressor behavior.

Use explicit switches to disable borrowing/NMS. Do not simulate disabled behavior with invalid thresholds. Different OVI readouts must be accurately named; remove the misleading hard-coded `OVI_S1a` provenance only by replacing it with receipt-bound identity, not by changing predictions.

Two useful invariants: qualified OVI masks on the common domain are mutually disjoint; a label-only intervention cannot change these masks, their source areas, NMS priority, or class-agnostic kept set. Unknown native owners must remain in geometry provenance even if they supply no eligible semantic candidate.

## 5. Run the attribution experiments

All new condition IDs use `AT_` to avoid collisions with existing G1/G2 graph experiments and Q0/Q1 optional-query experiments.

### 5.1 Geometry and export bridge

| ID | Definition | Isolated change |
|---|---|---|
| `AT_GEO_NATIVE` | Exact historical OVI/S1a output on native mesh/evaluation projection | Historical reference. |
| `AT_GEO_TRANSFER` | Transfer the same historical candidate registry and labels to T0 coordinates and then the full evaluation domain | Geometry/correspondence/export support, with each candidate's original exported rank score and assignment priority frozen. |
| `AT_GEO_AREA` | Same transferred masks/classes, replacing only rank scores with common-domain raw areas | AP-scoring change; keep point assignment fixed. |

This bridge is an evaluation compatibility control. Historical evaluator-side scores and emitted registries may be reused to reproduce that reference, but must not become deployment eligibility rules, learned policy inputs, or substitutes for the prediction-only core pool.

Freeze the historical emitted candidate registry for this bridge. Keep records for candidates that acquire zero support instead of silently forgetting them. If the core OVI pool has a different eligibility registry, report that difference explicitly; it is not a pure geometry effect. Missing original scores or candidate identity locally block exact bridge attribution; do not invent scores.

`AT_GEO_TRANSFER - AT_GEO_NATIVE` is the effect of this geometric transfer, correspondence, and export path. It is not proof that SpaCeFormer reconstructed better geometry: nearest-neighbor transfer through an intermediate cloud is not generally equivalent to direct mesh projection.

### 5.2 Core factorial: label borrowing × fusion NMS

Let O be qualified OVI/S1a masks on the frozen common domain, and S be masks exported by the same T0 run. Freeze U = O union S, canonical order, source-area rank scores, and raw-area assignment priorities.

| ID | Candidate pool | Borrowing L | Fusion NMS N |
|---|---|---|---|
| `AT_O_AREA` | O | Not applicable | None added |
| `AT_S_RELEASED` | S | Not applicable | None added; retain original released scores |
| `AT_S_AREA` | S | Not applicable | None added; raw-area score control |
| `AT_U00` | U | Off | Off |
| `AT_U10` | U | On | Off |
| `AT_U01` | U | Off | On, original OVI-first order |
| `AT_U11` | U | On | On, original OVI-first order |

Keep the two S-only score baselines separate. Do not demonstrate a win only against a T0 baseline weakened by area scoring.

For the four U cells, change no other operation. In particular, do not renormalize rank scores by the newly assigned class, retune thresholds, add class-aware NMS, or re-encode imagery. T0's **upstream internal postprocessing/NMS stays fixed**; N=off means only the additional fusion NMS is disabled.

`AT_U11` must reproduce the historical T1 arrays, labels, retained IDs/order, raw scores, projection, and released metrics when bound to the original assets. Semantic labels matter more than incidental NPZ container-byte differences. Use full-precision reference metrics; any tolerance must be declared from numeric parity, not chosen to accept a better score.

For each metric M, report:

```text
Candidate-addition path: M(AT_U00) - M(AT_O_AREA)
Borrowing without NMS:   M(AT_U10) - M(AT_U00)
Borrowing with NMS:      M(AT_U11) - M(AT_U01)
NMS without borrowing:  M(AT_U01) - M(AT_U00)
NMS with borrowing:     M(AT_U11) - M(AT_U10)
Interaction:            M(AT_U11) - M(AT_U10) - M(AT_U01) + M(AT_U00)
```

These are conditional effects on the frozen pool, not universal additive module contributions. AP changes cannot be assigned to individual objects and summed as a unique causal budget.

### 5.3 Do not credit the earlier S1a readout gain to fusion

Where the native pre-S1a readout from the same geometry/cache generation is available, run a small companion table:

```text
OVI readout: native vs S1a
Candidate system: O-only vs U11 fusion
```

Reuse all masks, priorities, raw-area rank scores, and thresholds. Check equal OVI eligibility first. Report the fusion increment within each readout and their interaction. This answers whether fusion needs S1a's earlier relabeling or works with native OVI too.

If the native readout cannot be bound, mark only this control blocked and limit the claim to `SpaCeFormer + this specific OVI/S1a readout`. Do not replace it with the later rebuilt Replica8 B1 or current C1 cache.

### 5.4 Separate ranking from final map ownership

Freeze `AT_U11` candidates, labels, kept IDs, and canonical order:

| ID | AP rank score | Prediction-domain assignment rule |
|---|---|---|
| `AT_ASSIGN_RAW` | Frozen raw area | Descending raw area, then canonical index. |
| `AT_ASSIGN_CLASS_NORM` | Identical to RAW | Area divided by the maximum area among fixed retained candidates of the same final class; then canonical index. |
| `AT_ASSIGN_OVI_FILL` | Identical to RAW | Qualified OVI candidates first; SpaCeFormer fills points not covered by a qualified OVI candidate. Within each source use raw area, then canonical index. |
| `AT_RANK_CLASS_NORM` | Within-class normalized area | Exact assignment priorities/owners from RAW remain frozen. |

Points with a native unknown owner but no qualified OVI semantic candidate are not automatically protected by OVI_FILL. Preserve their original geometry-owner status separately; do not call them unobserved physical space.

Assignment-only cells must have exactly identical **overlapping-candidate AP inputs**, including serialized rank scores. Ranking-only cells must have identical owner and semantic arrays. A rank-only cell must also reuse the already fixed NMS keep set: it must not rerun NMS in the new rank order. Distinguish full-precision assignment scores from the six-decimal scores read by the AP evaluator.

Within-class scaling generally preserves within-class ordering in a single scene before quantization, but ties and six-decimal serialization require checking. Scaling independently per scene can change cross-scene class ranking; do not assert pooled-AP invariance. Preserve historical serialized values and report tie groups. Do not add epsilon perturbations to force a preferred result.

Add exactly one order control, `AT_NMS_SF_FIRST`: change only NMS visitation to SpaCeFormer-first, retaining original within-source order. Freeze borrowed labels, candidate raw-area scores, and tie keys. **Export kept candidates in the original canonical order afterward**, so a visitation-order intervention does not also change downstream equal-score order.

The current T0 released score multiplies objectness/mask quality by the original predicted class probability. After label borrowing, that original score is not automatically confidence in the borrowed class. Keep it as frozen historical priority in reproduction. Any new class-confidence proposal must recompute the probability for the actual class using the exact cached SigLIP2 alignment, not an assumed dot product or mixed SigLIP/SigLIP2 vectors. First verify reconstruction of original scores. No scoring model training in this initial study.

## 6. Evaluate both candidate quality and the actual unique map

Build unique owner IDs on the prediction-side common cloud **before** reading GT, then apply the fixed nearest-neighbor projection for evaluation. Each covered point gets one retained candidate ID; uncovered points get 0. Use wide integer IDs with an explicit inverse map. Convert canonical candidate index i to positive owner ID i+1, or another recorded injective positive mapping: candidate index 0 must never be confused with unassigned owner 0. Namespace source identities so an OVI owner and a SpaCeFormer query cannot collide. Same-class candidates still compete for instance ownership.

Do not add connected-component splitting, hole filling, mask erosion, size-based reassignment, or recomputed area confidence during this conversion. Each is another intervention. Preserve unsupported/empty candidate entries in the ledger. A candidate becoming projected-empty or falling below the evaluator's size threshold is a measured ownership/export consequence; do not silently redistribute its points to save AP.

Report separately:

1. Released semantic instance AP/AP50/AP25 for the overlapping proposal set.
2. Semantic mIoU/mAcc for the unique map.
3. Released instance AP/AP50 and a clearly named high-IoU diagnostic after converting the unique owner map back into disjoint masks, using the original frozen rank scores.

Ownership-only changes may change **unique-map AP**, although overlapping-candidate AP must remain fixed. Do not treat that expected difference as a failed isolation test.

For fixed global assignment priority and the same nearest-neighbor mapping, resolving source owners then projecting must reproduce legacy projected-mask-first semantic assignment. Check this equality for the historical rules. A mismatch indicates implementation, filtering, coordinate, or tie handling differences, not a new method gain.

A high AP from overlapping proposals plus a high mIoU from another selected output does not establish one superior instance map. Keep output representations and condition IDs explicit.

## 7. Attribute object gains, losses, and regional errors

### 7.1 Prediction-only provenance ledger

Store one record per original candidate, including suppressed and empty candidates:

```text
scene_id, prediction_run_id, condition_id, candidate_id, canonical_index
source, native_owner_id, spaceformer_query_id, original_t0_row
ovi_readout_id, source_query_ids, source_mask_reference
original_class_id, final_class_id, borrowed_from, label_actually_changed
source_area, rank_score_full_precision, rank_score_serialized, rank_score_parsed
nms_visit_rank, kept, suppressed_by, suppression_iou
assignment_rule_id, output_owner_id, owned_source_point_count
```

Evaluation-only artifacts add projected size, class eligibility, ignored reason, GT overlap relationships, matched identities, and exact evaluator events. Do not place GT fields in a reusable prediction cache.

### 7.2 GT-object transitions and candidate error events

At the released thresholds corresponding to nominal 0.25/0.50/0.75, track objects across O-only, S-only, all four U cells, and unique-map outputs. Use the evaluator's exact operators/runtime thresholds; label any separate diagnostic differently.

| Diagnostic | Required distinction |
|---|---|
| Gained/lost evaluable object | Record GT ID, source candidate, threshold, old/new match, overlap, and whether the gain survives unique ownership. |
| Candidate geometric coverage | Class-agnostic max-IoU coverage is potential, not one-to-one recall. Report class-correct coverage separately. |
| Borrowed labels | Same label; wrong-to-right; right-to-wrong; wrong-to-wrong; ambiguous/unmatched; entry/exit from the instance-evaluated class subset. Count all borrowed and retained borrowed separately. |
| Suppression harm | Record candidate/suppressor overlap and GT coverage; check whether any other retained candidate already covers that GT before calling it a lost object. |
| Duplicate/background/poor localization | Use actual evaluator events and explicit diagnostic definitions. A mixed-object mask may have multiple failure flags; do not invent mutually exclusive additive counts. |
| Ranking harm | Fixed masks/classes with changed PR curves, TP rank, and recall at fixed candidate budget. Poor rank alone is not physical disappearance when all candidates are evaluated. |
| Fragmentation/overmerge after ownership | Record owner extent changes, number of GT objects overlapped, and threshold crossings, including same-class owner conflicts. |

A label is not proven correct merely because it matches an arbitrarily chosen dominant GT region. Use a fixed geometry-based correspondence criterion, record ambiguous cases, and retain per-threshold correctness. A predicted label moving outside the 48-class instance subset can remove counted errors without recovering an object; report this explicitly.

Instrument the released evaluator with minimal observer/trace hooks or a parity-checked diagnostic copy. Its duplicate handling may generate score/label events that are not equivalent to a simple one-prediction/one-event matcher. Preserve candidate links to events and verify that the traced evaluation reproduces uninstrumented per-class AP, pooled AP, PR arrays, and false-negative counts. Do not fix evaluator behavior inside this task. If exact event tracing remains unavailable, report named overlap diagnostics and mark exact evaluator attribution incomplete instead of fabricating TP/FP labels.

If R@K is reported, use predeclared K values that fit the baseline candidate budget, identical ranking/quantization, and an explicitly named matching rule. Without a separately controlled common ranking, source-score calibration remains a confound; coverage and released AP must be reported alongside it.

### 7.3 Exhaustive, fixed regional partition

The original four-region sketch is insufficient: SpaCeFormer-only support can contain several overlapping SpaCeFormer candidates, and both-source support can be class-consistent.

From the **unmodified frozen pool U**, define a disjoint partition using the cross-product of:

```text
Source support:       neither / OVI-only / SpaCeFormer-only / both
Candidate multiplicity: zero / one / multiple
Original class agreement: none / unanimous / conflicting
```

Compute regions without GT or borrowed labels, save them once, and project with the same mapping. Report a separate `PROJECTION_UNMATCHED` evaluation region; it is not interchangeable with a matched point having no candidate support. Valid-GT unmatched vertices remain in the global confusion matrix.

For the fixed retained U11 pool, also compute active multiplicity. Assignment-only policies can change owner or class only where at least two retained candidates cover the point. **A single-source region is not necessarily a single-candidate region.** Same-class multi-owner changes may leave semantic mIoU unchanged while damaging instance AP.

For each region and condition pair, save correct-to-wrong, wrong-to-correct, wrong-to-wrong, correct-to-correct, owner changes, and class changes. Emit full confusion matrices including prediction 0. Region matrices must sum exactly to the global matrix; recompute global mIoU from that sum. Regional mIoUs and net correct-point counts cannot be summed into a valid global mIoU explanation.

Select qualitative examples using a deterministic reporting rule, not only attractive gains: include a recovered object, a lost object, a corrupted borrowed label, harmful suppression, and same-class ownership conflict when present. Keep the full ledger available; do not fabricate a missing example category.

## 8. Reuse the two real model runs; optimize only after attribution

Run the same frozen matrix on both saved independent FP32 predictions. First complete historical parity, the core factorial, and its object/region accounting; then finish the bounded score/order/readout controls. Do not allow optional upstream-logit or oracle investigations to delay the core deliverable. Use the same OVI source generation and compare each fusion result with its paired common-domain baselines. Re-reading one NPZ twice is not a model-repeat experiment. Record exact coordinate equality before projection reuse; do not assume query IDs align across runs.

Report both sets of effects and the objects responsible for agreement/disagreement. Two results do not support a reliable confidence interval or a significance claim. Do not select the better run using GT. Inconsistency should trigger a focused object-level explanation, not an indefinite determinism investigation.

After all required diagnostics, you may implement **at most one small, prediction-only postprocessing repair** supported by an identified failure mechanism. Write its rule, expected mechanism, and frozen parameters before evaluating the repair; run it on both cached runs and report all metrics, unique-map quality, and costs. No threshold grid, new inference, extra backbone, or training. If evidence does not identify a clear repair, stop with `NO_REPAIR_JUSTIFIED` and deliver the diagnosis.

A repair chosen using Room0 GT and tested on another Room0 model run is still development evidence, not independent scene generalization. Propose a predeclared scene-level confirmation plan in the handoff, but do not silently launch additional mapping/network jobs or tune on later benchmark scenes in this task.

GT-assisted counterfactuals are optional and must be labeled `ORACLE_DIAGNOSTIC`. They can test recoverability within a fixed candidate pool, but are not deployable predictions or guaranteed achievable upper bounds. Group-removal/restoration dAP is conditional and generally nonadditive.

## 9. Minimal implementation and verification

Prefer small changes to existing modules. Suggested new paths are **not pre-existing interfaces**:

```text
src/static_ovmap/fusion_attribution.py
scripts/evaluation/run_static_t1_attribution.py
scripts/evaluation/diagnose_static_t1_attribution.py
configs/evaluation/ovimap_t1_attribution_v1.json
tests/evaluation/test_static_t1_attribution.py
artifacts/static_ovmap/t1_attribution_v1/
docs/paper/static_ovmap/T1_ATTRIBUTION_RESULTS.md
```

Follow actual project conventions. New command examples must not be reported as executed until implemented and run. Recover real environment/command arguments from receipts; do not merge the native mapping and newer model environments. Prediction generation needs no GT arguments. Evaluator tasks may receive both prediction and GT artifacts.

Cache masks, source areas, and small candidate-by-candidate IoU matrices per run where useful; reuse them across factorial cells. Keep counts in a safe integer dtype. Never allocate a point-by-point dense matrix or copy the full mask bank for every condition. Memory-map optional logits. Report measured cache preparation, projection, fusion, assignment, and evaluation times separately; zero new inference does not mean zero cost. Do not infer end-to-end FPS from warm-cache timings.

Required tests and one real cached smoke, not a full dynamic-repository rerun:

- Legacy fusion parity: masks, classes, IDs/order, areas, suppression decisions, and bound metrics.
- Label-only isolation and class-agnostic keep-set invariance; exact threshold-boundary cases.
- Rank-only versus assignment-only isolation, including serialized-score ties and same-class ownership conflicts.
- NMS visitation changes do not change canonical export/tie order.
- Native-owner assignment followed by projection matches legacy semantics under frozen global priorities.
- Ownership stays inside original retained masks; each supported point gets exactly one owner; empty/small candidates are recorded without redistribution.
- Fixed region partition is exhaustive/disjoint; regional confusion matrices sum exactly to the global one, including unmatched valid GT points.
- Evaluator trace parity; correct distinctions among no GT, invalid class, small region, ignored region, and counted FP.

Use small deterministic fixtures and relevant existing regression tests. Never write “AP must increase” as a unit test. Where a real asset is unavailable, explicitly separate unit-test success from unexecuted benchmark parity.

## 10. Deliverables and completion decision

Produce four tables, machine-readable rows, and a short evidence-based decision:

1. **Paired performance table:** scene/run/condition; OVI readout identity; geometry/output representation; candidate/kept/eligible/ignored counts; released AP/AP50/AP25; separate high-IoU diagnostic; semantic mIoU/mAcc; unique-map AP; measured costs; inference counts.
2. **Intervention effects:** geometry/export and score bridges; all four factorial differences plus interaction; native-vs-S1a dependency; independent ranking, assignment, and NMS-order effects.
3. **Object gains/losses:** actual recovered/lost GT IDs; corrected/corrupted labels; harmful suppressions; duplicate/background/ignored outcomes; whether each gain survives unique ownership; exact evidence references.
4. **Regional losses:** fixed support/multiplicity/agreement strata; correctness and owner transitions; confusion-matrix deltas; unmatched-region contribution and consistency checks.

Keep machine-readable proportions and display percentages explicitly; express differences in percentage points, not relative percentages unless separately labeled. For future multi-scene reporting, pool the released evaluator across scenes and pool semantic confusion matrices; do not substitute mean per-scene AP for pooled AP. A scene-local confidence normalization may alter pooled ordering.

Use transparent decisions such as:

```text
DIAGNOSIS_COMPLETE_NO_NET_GAIN
DIAGNOSIS_COMPLETE_COMPLEMENTARITY_SUPPORTED
DIAGNOSIS_COMPLETE_OWNERSHIP_LIMITED
DIAGNOSIS_COMPLETE_LABEL_REUSE_HARMFUL
INCONCLUSIVE_MISSING_ASSET
INCONCLUSIVE_REPEAT_SENSITIVITY
NO_REPAIR_JUSTIFIED
```

These describe evidence, not publication readiness. Explain separately whether candidate recall, semantic labeling, or unique-map quality improved. Do not require a positive result for completion or claim a universal causal explanation from Room0.

The final handoff must include actual executed commands and exit status, environment and source identities, small-result paths, large-asset references, condition-level `COMPLETE / NOT_RUN / BLOCKED / FAILED`, negative results, remaining limitations, and the single next recommended experiment.

Commit only scoped code, configuration, tests, and small results. Do not upload restricted data, credentials, model weights, or large masks/meshes. Push only under existing explicit project authorization, only to the task branch, and verify the remote SHA before reporting `PUSH_VERIFIED`; otherwise report the local commit and `NOT_PUSHED`. Do not fabricate a completed remote write or create repeated commits just to embed a commit's own hash in itself.

## Source notes for this reviewed prompt

This specification supersedes the earlier Chinese attribution prompt for execution details. It preserves the frozen-prediction study and adds explicit evaluator semantics, native-vs-S1a attribution, NMS/export-order isolation, complete regional partitions, and unique-map filtering accounting.

Source basis: the repository files listed in Section 2 at `8cefe6b4bca464e0478f82e6ef1e6d0e3783e347`, plus the upstream evaluator at `OVI-MAP/OVI-MAP@f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424`. Runtime receipt-bound hashes take precedence over branch labels. TIDE was a conceptual reference in the earlier note; no TIDE installation or COCO conversion is needed for this task.

Selected immutable source locations:

```text
https://github.com/Orangekostar/oviovo/blob/8cefe6b4bca464e0478f82e6ef1e6d0e3783e347/src/static_ovmap/proposal_fusion.py
https://github.com/Orangekostar/oviovo/blob/8cefe6b4bca464e0478f82e6ef1e6d0e3783e347/scripts/evaluation/fuse_static_spaceformer.py
https://github.com/Orangekostar/oviovo/blob/8cefe6b4bca464e0478f82e6ef1e6d0e3783e347/scripts/evaluation/evaluate_static_proposals.py
https://github.com/Orangekostar/oviovo/blob/8cefe6b4bca464e0478f82e6ef1e6d0e3783e347/scripts/evaluation/evaluate_static_ovmap_readout.py
https://github.com/Orangekostar/oviovo/blob/8cefe6b4bca464e0478f82e6ef1e6d0e3783e347/scripts/evaluation/evaluate_static_ovmap_instances.py
https://github.com/Orangekostar/oviovo/blob/8cefe6b4bca464e0478f82e6ef1e6d0e3783e347/docs/paper/static_ovmap/R6_RESULTS.md
https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/scripts/eval_utils.py
https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/scripts/eval_sem_seg.py
```
