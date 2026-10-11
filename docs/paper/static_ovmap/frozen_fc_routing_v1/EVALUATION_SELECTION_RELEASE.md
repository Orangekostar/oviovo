# Evaluation, selection, state transitions and publication

## 1. What is selected and on which data?

The scientific predictors are four FG models, with at most two seeds. No diagnostic, FC-initialization state, old R head, ideal GT filter, matched-mass intervention or best-per-object mixture is eligible for selection.

At each fixed checkpoint, use DEV's69 base originals with the complete200-class text competitors. For each object i, with correctness b_i0..b_i3 in clean/truncate/append/both:

`a_i = .5*b_i0 + (b_i1+b_i2+b_i3)/6`.

- **A**: mean of family means of a_i.
- **M**: mean of present DEV base-class means of a_i.
- **C**: mean of family means of b_i0.
- **CE**: same family/object condition weighting,160-base-class denominator, tie-break only.
- **NLL200**: same weighting with full200 competitors, descriptive.
- **net_corrections**: unweighted count of FC-wrong/model-right minus FC-right/model-wrong across69x4 correlated records. It is not an independent instance count and is not algebraically identical to delta_A.

Qualifies if step>=250, A>=A_FC+.005-1e-10, M>=M_FC-1e-10, C>=C_FC-1e-10 and net_corrections>0. Compute from raw fractions, never displayed percentages. Rank qualified checkpoints by A/M/C descending, CE ascending, earlier step. If none qualifies, retain the same ranking's best diagnostic checkpoint with `NOT_QUALIFIED`, not an invented trained gain. Method tie order is FG1,FG2,FG3,FG4. Store all five checkpoint records; step0 is diagnostic only.

`nominate17` picks one best qualified architecture. If none, no repeat or new maps. If there is one, run all four seed29 models. Select their checkpoints independently on the same DEV rule. The second seed never replaces the architecture nominated at17; full expansion requires that particular architecture qualify again. Even if another seed29 architecture qualifies, report it and stop full transfer when the original nominee does not repeat.

No training/selection decisions depend on runtime, oldH, H2, map labels, a dataset-name switch or a desired publication conclusion. Report same-step paired DEV results in addition to independently selected-checkpoint contrasts.

## 2. Exactly three legitimate completed science paths

| path | training | new full-map evaluation | new H2 / real-proposal measurements | scientific terminal |
|---|---:|---:|---|---|
| no seed17 model qualifies |8000 |0 |not triggered |COMPLETE_NO_FG_SIGNAL |
| nominee does not repeat |16000 |0 |not triggered |COMPLETE_FG_NOT_REPEATED |
| same nominee qualifies twice |16000 |286 rows /22 pools |required, subject to honest dependency status |COMPLETE_NO_TARGET_GAIN or TARGET_MET_RESEARCH_ONLY |

All paths execute applicable D0/D1, engineering, selected-head DEV/old-H diagnostics, FG4 interventions and report/export. Negative complete paths return0. A real missing active stage returns1 with partial/block reasons, while report still packages useful completed work. Do not trigger new work merely to fill a table cell.

A valid but small confirmation sample is `LIMITED_CONFIRMATION_SUPPORT`, not a runtime failure or permission to change scene lists. Unavailable required inputs are `PARTIAL_ACTIVE_STAGE`. Model/nonfinite/cache mismatches are errors. Model disagreement or negative metrics are not errors.

## 3. Freeze before transfers

After all activated seed training/selection, write `predictor_lock.json`: method->selected weights, architecture, code/spec/data/bank identities, fixed nominee, whether repeat and expansion activate, unchanged H2 names/real frames, and enabled diagnostics. All branches are locked before evaluating new map outputs or H2 labels. Ordinary oldH remains exposed post-lock regression, not selection.

For full maps, construct all11 methods for all26 scenes before accessing evaluator labels for the new run. Early engineering is an explicit exception for literal unchanged-parent outputs on two fixed scenes; it cannot inspect a trained candidate or select parameters.

## 4. Real full-map prediction

Methods: REF_D2, REF_G1, FG0_FC8, FG1_VIEW, FG2_SURFACE, FG3_AUX, FG4_ROUTE and S29_ counterparts of the four trained models. These are7 main and4 repeat paths:182+104=286 scene records and14+8=22 pools.

REF_D2 and REF_G1 reuse exact LR baselines. FG0_FC8 reuses exact LR04_FC_8 only after reproducing its ordinary readout/source inputs and scorer identity. FG0 has no new trained weights. Do not label source-update or PR result rows as new FG predictions.

Use the fixed LR common eligible owners and its existing8-view bank. Each method produces a new F cosine vector against that scene's exact frozen text/IDs. Replace only F, keep N/Q and inherited temperatures/missing-source behavior, call `pool` then `simple_dependence(...,'PE_D2_GROUPED')`. With all sources this is .25*N+.25*Q+.50*newF. No .75 oldD2+.25 new extra mixture, learned gate, per-class thresholds or separate dataset configuration.

Noneligible owners preserve G1. G1 recovered objects never get new labels. Keep raw-zero and output protections identical, but report their counts. Partition, coordinates, mesh faces, target-nearest projection and evaluator remain unchanged. No geometry-improvement claim is possible from this experiment.

Use `disagreement_query.outputs.build_payload` with the real G1/D2/raw/nearest/matched structures; rebuild current-class official area ranks after relabeling. Keep exact old support even when a new classifier is wrong. A no-information local head returns FC8, not historical G1 labels. The system's literal no-update identity and the neural head's FC8 identity are separate tests.

## 5. Scoring and reuse

Reuse `partition_evaluator`, source-update scorer primitives, `scorer_compat.adapt`, `official_view`, `equivalent_scoring_input`, `released_pool_with_classes`, and original min100 protocol. Build a task-owned evaluator loop with11 methods/286 rows instead of the old fixed390-row loop. Do not patch old counts or run methods under false old IDs.

Validate the actual positive-owner registry, including recovered owners, and all current-class ranks. Evaluate instance and semantic metrics from the SAME real payload. Preserve confusion's full target denominator. Report APall/AP50/AP25/mIoU/mAcc as fractions in machine files and percentages in tables.

Exact reuse requires equal owner arrays, semantic arrays, current-class ranks, evaluator/GT geometry context and the same ordered cohort input sequence. Equal rounded metrics, equal owner IDs alone or a common parent are not proof. Otherwise execute the scorer. Pool the original ordered eight Replica scenes and18 CF captures; never average per-scene AP to substitute a full pool. Keep seed29 separate. No seven-family/18-capture independence claim.

## 6. Research recommendation vs mechanism conclusion

Every trained map row is checked against current G1 on all5 metrics in BOTH cohorts. Candidate CF APall must be > D2+1e-10 and AP50>=D2-1e-10. Material flag additionally needs CF APall>=D2+.001-1e-10. Runtime has no role.

Report the locked DEV nominee, its seed29 outcome, all raw method gate flags and an exposed regression recommendation separately. Among passing seed17 rows recommend highest CF APall, then Replica APall, then sum of the two mIoUs, then simpler FG method order. This is explicitly benchmark-exposed research selection, not H2-controlled tuning. A same-method pass at both seeds supports repeatability; do not create a hybrid of the best metrics/seeds. No passing row -> research retained REF_G1. Deployment always N0_UNCHANGED.

H2 does not change the recommendation or checkpoint. Report its confirmatory comparison for the frozen DEV nominee as predeclared, and all other locked heads descriptively. Do not claim independent map generalization from proposal recognition.

**Mechanism statements need their own comparisons:** FG2 vsFG1; FG4 vsFG3; FG4 vs matched-mass and matched-norm controls. A gate pass or a drop under corrupted intervention alone is insufficient. Added BCE supervision is extra supervision relative to FG2, so only FG4/FG3 is the directly matched supervision comparison.

## 7. Three report/table families

1. **Full-map performance**: one main table (7 methods x2 cohorts,5 metrics) and separate seed29 supplement. In inactive paths, show parent references as inherited and leave new map cells null with NOT_TRIGGERED; do not invent286 executed rows.
2. **Recognition and mechanisms**: DEV A/M/C and correction/harm with69-original denominator; separate all-class micro/macro tables for post-lock TRAIN/DEV/oldH/H2; exact same-step and selected-step contrast labels; D1 and mass/norm controls explicitly diagnostic; calibration, negative-mass and zero-support denominators.
3. **Coverage, transfer and compute**: actual distinct frames/views/tokens and fallback counts; known/unknown targets; real-proposal generator vs liftable vs multi-view vs matched denominators; H2 true family/class support; actual optimizer/encoding/head calls and stage time. No runtime rejection, cold grid or online FPS claim.

Every table has MD/CSV/JSON/LaTeX with consistent units. Add short Chinese findings in RESULTS, but keep exact definitions in English handoff/spec. A figure or PDF is not mandatory; no need to spend time on rendering before a performance signal exists. Qualitative claims require actually inspected examples, not invented images.

## 8. Process evidence and publication

Export every selected trained head (4 or8), model configuration, seed/step, base FC/phi/text identity and relative weight path. No R/MA checkpoint is an inference dependency. Small architecture-only weights are suitable for normal Git; optimizer/last weights remain external but their keys and hashes are documented. Save all scalar component logs, known-target counts, DEV checkpoint tables, compact embeddings/full200 cosine matrices, decisions and declared interventions. Reconstruct predictions with exact original text order.

Strictly reload each exported head. Compare outputs on a fixed small DEV sample against the selected training checkpoint; don't use `strict=False`. Verify base FC parameters unchanged and actual input autograd once. Avoid hundreds of redundant tests or signing every record.

Required docs:

- FROZEN_FC_ROUTING_RESULTS.md
- FROZEN_FC_ROUTING_HANDOFF.md
- FROZEN_FC_ROUTING_SELECTION.md
- FROZEN_FC_ROUTING_CLAIMS.md

HANDOFF states: original task/spec/base, all resolved roots/environments, one complete CLI, phase recovery and exact load interface, trained counts, table provenance, conditionally skipped/blocked stages, rerun boundaries and remaining uncertainty. A separate FINAL_REVIEW.md may provide a concise requirement matrix, not thousands of checkboxes.

Observe actual `all --resume` subprocess exit externally (shell PIPESTATUS or parent subprocess return code). Preserve original failed attempts and null unknown timings. Do not sum nested/resumed stage clocks as a single cold total. Commit code/results/docs normally and push the new branch. Verify full `git rev-parse HEAD` and remote branch hash after push, then write external publication/final.json with scientific status, observed exit, result identity, local/remote SHAs and timestamp. Publication can succeed even for a negative/partial scientific result; do not confuse those statuses. A commit never contains its own future SHA.
