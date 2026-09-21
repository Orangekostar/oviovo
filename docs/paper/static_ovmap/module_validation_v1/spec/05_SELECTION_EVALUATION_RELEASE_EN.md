# 05 — Evaluation, selection, confirmation and GitHub publication

## E1. Preserve the evaluator, separate input domains

Reuse the existing released semantic-instance evaluator and its trace instrumentation (`bind_protocol`, `evaluate_set`, `semantic_metrics`, `attribution_objects`) through a NEW adapter. Inspect their actual signatures at the pinned project ref. Do not pass new methods through historical hard-coded METHODS lists. Do not change native GT filtering, strict IoU comparisons, minimum region size, instance/semantic subsets, ignored/duplicate handling, nearest-neighbor projection tolerance or class-name order.

The old SF evaluator hard-codes new inference=0 and compares against a fixed64-row H pool. Neither is correct for this study. Replace those fields in the new adapter with actual branch/model/capture ledgers and a scene-specific registry. Preserve the old adapter and tests.

Distinguish released semantic-instance uAP and AP50/AP25; semantic mIoU/mAcc; class-agnostic canonical geometric diagnostics; overlapping-candidate AP; and actual runtime/cost. Do not substitute one for another or call canonical AP the paper metric without parity evidence. A class-only change may alter instance eligibility even with fixed shape; this is not an invariant violation. Report full precision values plus percentages, and differences in percentage POINTS, not percentages. Scene averaging is unweighted mean of scene metric values where defined, with denominators shown. Undefined values stay null, not0. A required selection metric undefined on any selection scene makes that comparison INCONCLUSIVE rather than silently dropping the scene.

Calculate full input/prediction identity, not method-name identity. Identical final payloads may reuse evaluations once checked, but all logical method rows remain. A copied row is not an independent validation. Every S/Q row must pass fixed-geometry and fixed-rank equality. G rows must share source XYZ/faces/TSDF/projection and satisfy unique ownership and partition completeness. GT-dependent diagnostics only append AFTER prediction manifests are locked.

## E2. Fixed selection hierarchy — no cherry-picking

FIT fits model weights; CAL selects teacher/head checkpoint/adoption margin; SELECT evaluates frozen options and chooses retained modules. CONFIRM is opened only after selection is locked. This study defaults to12 development families (8FIT/2CAL/2SELECT) and2CONFIRM families, not24+4; do not expand v1 after seeing outcomes. The small numbers justify exploratory conclusions, not universal guarantees.

Use tolerance1e-10 on metrics in their [0,1] scale for tie/no-difference classification. A positive result should additionally be reported with per-scene changes and a descriptive scene-bootstrap interval (seed17,2000 resamples, no training reruns). The interval with two scenes is not robust population inference. No required arbitrary AP gain is promised.

### S selection

On SELECT, identify the best available DIRECT alternative by the CAL-frozen teacher ID; do not reselect the teacher using SELECT. Also show all direct controls for interpretation. Compare S_PAIRED to N0, to its same-teacher S_SIMPLE and S_NO_CONTEXT, and to the CAL-selected direct teacher.

S_PAIRED is mechanism-eligible only if: it makes>=1 actual nontechnical replacement; mean uAP improves over N0 AND S_SIMPLE beyond tolerance; mean mIoU is not below N0 or S_SIMPLE; and the uAP gain over N0 occurs in both SELECT scenes. If its only win is over a degraded matched-input native control, it is not eligible. If S_NO_CONTEXT matches/beats it within these rules, retain the simpler NO_CONTEXT as an engineering choice, but the object/background feature contribution is UNSUPPORTED. When S_SIMPLE matches/beats the complex heads, report useful supervision/selection baseline if applicable, not validation of the proposed contextual mechanism.

Select an S deployment candidate among N0, the CAL-selected raw teacher and the3 learned heads using SELECT mean uAP, then mIoU, then lower added cost, then complexity order N0 < raw teacher < SIMPLE < NO_CONTEXT < PAIRED. It is usable only if mean uAP strictly exceeds N0, mean mIoU is noninferior, and both SELECT scenes have nonnegative uAP deltas. Otherwise retain N0; label other points TRADEOFF rather than hide them. The no-context and simple options remain scientifically distinct.

### G selection

G_QUALITY is eligible when it changes>=1 entire partition; mean canonical class-agnostic AP50 exceeds both G_ORIGINAL and G_AGREEMENT; mean canonical AP75 does not decrease versus G_ORIGINAL; its matched native semantic readout has mean released uAP and mIoU not below G_ORIGINAL; and canonical AP50 improves in both SELECT scenes. If the native N0 bridge is materially different, expose it and require a paired-domain interpretation. A gain solely from fresh semantic reread does not qualify as a geometry contribution.

Select between ORIGINAL/AGREEMENT/QUALITY by canonical AP50, thenAP75, then semantic uAP, then fewer changed points, with originality as final tie preference. Retain a changed G candidate for combination only if it passes the same geometric+semantic noninferiority rule against ORIGINAL; choose ORIGINAL otherwise. A useful G_AGREEMENT is an engineering result, not evidence for learned quality. Oracle rows never enter selection.

### Q selection

At B=200 choose the strongest nonlearned Q comparator on CAL by mean uAP, then mIoU, then lower logical attempted cost, then ID order COMBINE/AREA/UNCERTAINTY. Run those three CAL policies once; reuse compatible acquired requests and debit each policy separately. Q_GAIN training checkpoint still uses CAL MSE, not another policy-driven hyperparameter search.

Q_GAIN is eligible if its SELECT mean uAP exceeds that locked comparator, mIoU does not decrease, and both SELECT scenes have nonnegative uAP deltas with at least one strictly positive. All rows use B=200. If Q_GAIN spent more actual logical cost, call it an allowance-constrained accuracy result, not equal-realized-cost dominance; the B100/400 curve tests efficiency separately. A cheaper noninferior result without strict AP gain is EFFICIENCY_ONLY and may be retained as an engineering option, not a third precision contribution. Unknown-label output and failed costs cannot be omitted. No final-label warm start.

## E3. Combination gate — at most two new variants, no giant factorial

The three standalone branches are mandatory when their prerequisites exist. Combinations are conditional:

1. If an S candidate and a changed G candidate are independently retained, run `COMBO_GS` on SELECT. Apply G to native geometry, produce fresh semantic evidence on each actual final component, then apply the frozen S decision to the corresponding native-versus-teacher alternatives. Regenerate evidence for changed masks; do not copy parent features. The S head weights and threshold remain frozen; failure on changed masks is a real distribution-shift result. Unchanged requests may be reused. Use the128-target cap, count unknown/new components and all extra encodings.
2. If COMBO_GS (or retained S-only/G-only when only one is retained) passes its noninferiority check, and Q_GAIN is independently retained, run `COMBO_Q_REFINEMENT`. First complete Q's causal mapping/readout. Then apply the retained static G/S refinement on actually observed RGB-D, with all refinement queries counted separately. Its incumbent labels come from Q or explicitly paid fresh-mask native reread, never free final N0 labels; all model/head settings remain frozen. This is a **causal-query plus static-refinement hybrid**, not fully online mapping and not a B200-total-cost method. Provide a matched baseline with the same static-refinement allowance. Do not claim the three modules add linearly. If the required fresh-mask input or feature provenance is unsupported, block this combination specifically.

Select a final candidate only if mean released uAP exceeds its matched native baseline, mIoU is noninferior, and every SELECT scene has nonnegative uAP delta. Otherwise prefer the best independently supported simpler candidate. Never combine the best AP of one map and best mIoU of another. Save every measured negative combination too. At most two combination variants; their necessary paired controls are not new model choices.

No retained modules means final selection=N0 and scientific result=NO_NET_GAIN. This is a legitimate conclusion and still requires code, results and publication.

## E4. Confirmation without redesign

Write `selection.json`, frozen resolved configuration, teacher/heads, split and prediction-code commit identity BEFORE opening CONFIRM labels. On the2preselected confirmation scenes run N0, the selected final candidate, and its nearest required same-model/simple comparison. Maximum4semantic/pipeline rows per confirmation scene. For a selected Q candidate include the locked B200 comparator; budget curves are SELECT-only. Do not evaluate every teacher and head on confirmation and select another winner there.

If candidate=N0 or prerequisites for a real retained candidate are absent, confirmation is `NOT_REQUIRED_NO_RETAINED_CANDIDATE` or the actual asset blocker. If a candidate fails confirmation, publish NOT_CONFIRMED without changing thresholds, retraining or choosing a different model on those results. Record study holdout separately from unknown foundation-model pretraining overlap and prior benchmark exposure. Reusing existing exposed Replica8/ScanNet18 scenes is retrospective validation, never the same as this confirmation split.

## E5. Required reports and machine artifacts

Commit small artifacts to `artifacts/static_ovmap/module_validation_v1/`:

- resolved config, environment/model/source lock, splits/exclusions and asset inventory;
- native-domain bridge/parity and active C++ path evidence;
- complete method matrix with required/run/reused/blocked/not-required rows;
- target-view/input validity and teacher generation/name-mapping ledgers;
- S feature schemas, FIT/CAL sizes, checkpoint and all CAL threshold decisions;
- G hypothesis/partition summaries, provenance, scoring and completeness records;
- Q prefix/acquisition/cost summaries, training exploration and causal checks;
- per-scene metrics, all actual object-match gains/losses and eligibility transitions;
- teacher/module/final `selection.json`, confirmation results;
- compact tests, exact commands, stage timings, errors and reproducibility entrypoints.

Large raw surfaces, raster sequences, crops, all logits and backbone weights stay outside ordinary Git. Commit their actual paths, byte sizes, hashes and regeneration commands; a manifest is not a file upload. Save small learned weights/scalers as NPZ/JSON or safe framework weights under the experiment directory if each<=10MiB and total release additions<=100MiB. Do not serialize arbitrary Python model objects when simple weights suffice. Retain checkpoint choice and training provenance.

Required Markdown:

`docs/paper/static_ovmap/MODULE_VALIDATION_RESULTS.md`
`docs/paper/static_ovmap/MODULE_VALIDATION_HANDOFF.md`

RESULTS has exactly five principal tables (supporting appendices allowed):
A. all branches/per-scene metrics and cost; B. semantic suggestion->adoption->correction/damage; C. geometry hypotheses->selected complete partitions->object preservation; D. query availability->paid acquisition->readout improvement/cost; E. module selection/combination/confirmation with evidence paths and status.

HANDOFF specifies actual source and extension identities, worktree, branch, data/model paths, completed/blocked scope, how to reproduce each leaf job, remaining technical dependencies and the one next evidence-based research action. Do not repeat the full prior historical audit or manufacture an attractive positive narrative.

## E6. Audit once, then publish actual work

Before the final push, perform ONE scoped final audit:

- every required row has a measured payload or an explicit evidenced prerequisite/gate status;
- every positive claim matches the actual domain/control and a changed effective prediction;
- model/threshold selection uses CAL and module selection uses SELECT only;
- no GT label/error ledger reached prediction or query features;
- G exports complete unique partitions on the same source geometry;
- Q decisions use prefix state and pay for every acquired query;
- numeric summary reconciles to machine metrics; no hidden fallback, wrong-inference=0 field, duplicated independent run or best-column splice;
- all runnable code/configs and small result/head files are in the publication set, not only MDs.

Fix concrete faults and rerun ONLY invalidated artifacts. This is not permission for a second broad audit campaign or global regression suite.

Record a code commit BEFORE experiments (after executable configuration/code is ready). Experiment receipts refer to that exact code commit plus any dirty diff digest; final corrected predictions must reference the actual generating code. The final result/handoff commit may be later. Do not attempt to embed a final Git commit's own SHA inside a file belonging to that same commit.

Stage explicit scoped paths, inspect `git diff --cached --stat`, commit actual code/config/results/head weights/docs. Run ordinary push of the task branch; no force push and no automatic merge into main. Resolve a normal authentication/network failure only through existing permitted credentials; never print tokens or upload credentials. Fetch the remote task branch, compare `git rev-parse HEAD` with `git ls-remote origin refs/heads/<actual-branch>`; equality after successful push is required for `PUSH_VERIFIED`.

Write a local publication receipt AFTER final commit containing the compared SHAs and commands; this receipt can remain a delivery artifact outside that self-referential commit. The remote branch contains the source/results/handoff, not a promise to upload them. A GitHub Release/tag is optional and not required; branch publication satisfies this task. If push fails, deliver the commit, actual error and a local scoped archive, report PUSH_FAILED, and do not claim remote success.

The final Codex response must state implementation/experiment/science/confirmation/publication statuses separately; list final candidate and measured headline metrics, actual inference/training/capture costs, report paths, branch and full verified SHA. Complete negative results and accurately published partial results are preferable to fake completion or another plan.
