# Evaluation, interpretation, selection and reporting

## 1. Preserve the two complete cohorts

Use the exact order in `PROTOCOL_SPEC.json`: Replica's eight scans and the existing ConceptFusion-style ScanNet set of eighteen captures. Do not replace a scene after seeing results, choose “positive” captures, or combine per-dataset winners. All are already-exposed research cohorts. Eighteen CF18 captures are not eighteen independent physical scene families and are not the full ScanNet200 validation set.

The complete-map metric denominator is unchanged even though only at most 16 incumbents are inspected per scene. The new common-domain restriction is applied consistently to seven comparative decision arms, not to the ground-truth set. SU00 and SU01 remain the actual prior baselines.

## 2. Nine rows, not copied historical values

There are 234 logical scene-method rows and 18 full pools. The two inherited baseline methods contribute 52 rows and four pools. The remaining matched/new rows must be reconstructed and genuinely scored or proven identical by actual output/scoring identity. Identical outputs in multiple methods are scientific findings and computational aliases, not permission to omit a condition.

The historical IR06/IR07 values are context. Matched SU02/SU03 only act on the shared available-F, paired-A/C, nonprotected domain and can therefore have different metrics. Show an appendix panel with historical metrics, eligible counts and the matched replay's output identity. Do not put the historical 9.94/10.63 Replica AP figures directly into matched rows unless exact output identity establishes they are unchanged.

Read all five unrounded metrics from actual pool receipts. No rounding before tests. APall follows the existing .50:.05:.90 average; AP25 uses .25. Keep the released comparison, ignores, class list, area-based score rule, minimum size and target projection. Never use scene-average AP as the official dataset pool.

New semantics on fixed masks can change ranks in several scenes, so AP cannot be inferred from the count of corrected objects alone. Pool the actual changed predicted labels and re-evaluate the exact score files. Per-class deltas, corrected object labels and unique GT match changes are complementary evidence.

## 3. Predefined comparisons and what they can support

| Contrast | Isolated question | Limitation |
|---|---|---|
| SU04 − SU02 | Is retaining N/Q better than the matched hard replacement? | These also use different decision rules (fusion versus raw margin); do not claim a new representation |
| SU05 − SU04 | Does keeping old F matter after adding A? | Fixed blend .5 only; not an optimized weighting law |
| SU05 − SU06 | Does AnyUp add benefit over coarse FC on the same new FULL views under the same F blend? | Coarse uses established v2 fallback, not bilinear fine interpolation |
| SU05 − SU07 | Is preserving old source grouping useful at the same A probability mass? | Degenerate F-only cases are identical by construction |
| SU08 − SU05 | Does the paired representation change outperform adding A in absolute form? | Same-view difference is not automatically independent information or a calibrated likelihood ratio |
| SU03 − SU02 | What does the old boundary-stability check do on the matched domain? | It is a replay control, not a new mechanism |

Additionally compare **every matched/new method against SU01 and CF18 D2**. A method beating a damaged control but remaining below G1 is not an upgrade. If coarse FC and AnyUp perform similarly, report the negative representation attribution instead of attributing all gains to AnyUp.

Historical F used different observations/aggregation from C. SU06-versus-G1 measures the combined effect of the new coarse observation set and injecting it with a specified weight; it is not a perfectly isolated pure view-selection experiment. SU05-versus-SU06 is the stronger same-view representation contrast.

## 4. Diagnostic records without leakage

After prediction lock, build all changed-object records and parent-opportunity references. Record the closest eligible GT IoU and label only in evaluation artifacts. Use unchanged original owner masks to define geometrically matchable labels; report at least IoU .50 and .75. Do not call each matchable predicted owner a newly discovered GT object.

Required categories include selected but not eligible, actual applied change, old correct/new correct, old wrong/new right, old right/new wrong, wrong-to-wrong, geometry insufficient, no eligible GT, and protected raw-zero cancellation. Include both success and failure entries; do not select only a few attractive edits.

For released matching, preserve duplicate score-entry multiplicity and unresolved old/new tied attribution. Report unique GT gained/lost separately. Include updated incumbent and recovered rank changes, and the per-class AP/IoU deltas. Do not claim probability vectors identical because argmaxes or aggregate AP match.

Compute diagnostic source margins for old and new class separately for N, Q, F, A and C. Use missing flags, not zero-filled scores. Record whether the hard-control damage was prevented and whether its useful corrections survived. These diagnostic features are available to a later study but are not used to fit anything in this run.

## 5. Exact selection

Let G be SU01, D be SU00, x a complete matched/new candidate and epsilon=1e-10 in fraction units. A candidate passes iff:

```
for each metric in {apall, ap50, ap25, miou, macc}:
    x.replica[metric] >= G.replica[metric] - epsilon
    x.cf18[metric]    >= G.cf18[metric] - epsilon
x.cf18.apall > D.cf18.apall + epsilon
x.cf18.ap50 >= D.cf18.ap50 - epsilon
```

The separate material marker adds `x.cf18.apall-D.cf18.apall >= .001-epsilon`. It is 0.10 percentage points, not 0.10 fraction and not a significance test. Missing/infinite metrics or incomplete cohorts cannot pass.

Select among SU02–SU08 with the configured lexicographic metrics and simplicity order. Simple controls may win. If no method passes, select SU01 with `COMPLETE_NO_TARGET_GAIN`. Do not alter a gate because a rounded number “looks the same.” Record Pareto tradeoffs separately, without upgrading deployment or changing the target definition.

A passing configuration remains a retrospective candidate on exposed data; no untouched confirmation, universal robustness or guaranteed nondegradation claim follows. The experiment's goal is empirical compatibility of new and old evidence, not proof of a monotone classifier.

## 6. Costs without another expensive benchmark

Measure only the **score-resident CPU decision kernel** in this task, on the same controller environment/CPU/thread setting, eight Replica scenes, two reverse-order rounds, 50 repeated evaluations per timed sample and one untimed warmup. Return mean milliseconds/scene and per-scene values; never report the fastest sample. The repeat count improves measurement resolution; it is not independent scientific replication.

Explicitly exclude data loading, mesh/observer search, FC/AnyUp computation, file I/O, ranks and payload construction from this timing. If these are recorded separately, retain their own labels. Report SU01's nonempty decision cost when it reconstructs probabilities rather than assuming it is zero; if implemented as a literal pass-through, disclose that it performs no update.

Actual one-time coarse acquisition costs include successful/failed FC attempts, dense cache hits, coarse pools, worker/model load and wall time, counted separately from the CPU microbenchmark. Existing AnyUp scores/FC sources are reused, not free. There is no new standalone GPU latency or peak-memory comparison and no claim that the new final pipeline has zero GPU work in deployment.

No end-to-end cold calls are authorized here. A strong retained candidate is a prerequisite for a later same-boundary standalone cost test; do not fill this task's table with timings from earlier 16.23/18.85/22.71-second series.

## 7. Three table outputs

All cells derive automatically from one canonical store.

**Table 1:** nine primary methods, Replica and CF18 APall/AP50/mIoU. AP25/mAcc in CSV/JSON and compact supplement, plus the exact gate flags. Headings explicitly mark matched controls.

**Table 2:** six predefined contrasts and their paired metric differences, unique GT50 gained/lost, and wrong-to-right/right-to-wrong counts. If a contrast is just an output-identity tie, say so.

**Table 3:** eligible and actually changed incumbent counts per cohort, source availability/paired FULL coverage, cached CPU decision milliseconds/scene, and required evidence types. One separate cost paragraph gives the study's actual new FC/AnyUp counts; do not apportion shared acquisition cost as independent method latency without a real standalone run.

Write MD, CSV, JSON and compact booktabs LaTeX fragments. The user requested an execution package, not a new slide/document artifact. Production table formatting should be readable; a PDF preview is optional only when an existing compiler is available and must not become a reason to rerun science or install a TeX distribution. Undefined values are `null`/dash with a reason; never zero.

## 8. Research outcome and bounded next steps

`next_stage_assessment.json` must state evidence for:

- source preservation versus hard overwrite;
- old-F retention versus F replacement;
- fine versus coarse representation on the same observations;
- source grouping versus same-A-mass global ensemble;
- paired change versus absolute new evidence;
- remaining correct-to-wrong versus wrong-to-right changes;
- whether common availability or protected-boundary exclusions materially limited the experiment.

If fixed source preservation works, keep the simplest passing rule and record why. If new evidence remains useful but damaging, recommend a separately specified update-risk study with proper training/family separation. If paired ordinary FC is equally good, shift emphasis away from AnyUp. If there is no benefit, stop these five fixed transformations rather than launching an unbounded weight sweep.

No learned selector, calibration fit, new source model, new test subset, adaptive query scheduler or structural operation is executed by this package. This is deliberate scope control, not a claim that those research directions have been completed.
