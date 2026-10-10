# Selection, causal comparisons, outputs and release

## 1. Common proposal-level metrics

DEV selection uses only the parent's base-category objects and labels, but predictions compete against the full200 frozen text table. For original object i define `a_i=.5*I(clean correct)+(I(truncate)+I(append)+I(both))/6`.

- A: average a_i within family, then equal average over the four families.
- M: average a_i within true base class, then equal average over present DEV base classes.
- C: clean Top1 averaged within family then equally across families.
- CE: parent weighted base160 CE, equal-family; a tie-break/diagnostic, not the primary objective.
- Raw net corrections: across the same four conditions, count FC-wrong/student-right minus FC-right/student-wrong. Conditions remain correlated; it is a descriptive count, not independent sample size.

Also publish clean and each corruption separately; teacher-correct retention, harm counts, score margins and full200 NLL; actual TRAIN-observed versus zero-positive base classes; heldout class statistics only after lock. Do not select on heldout-class labels, old H or benchmarks.

Checkpoint diagnostic steps are0,250,500,1000,1500,2000. Step0 may verify FC/base identity but is never a trained nominee. If a forward is nonfinite, its checkpoint is invalid with a technical reason, not a zero accuracy silently omitted from the denominator.

## 2. R qualification and deterministic state machine

A checkpoint passes the R foundation gate iff:
1. `A >= A_FC8 + .005 - 1e-10` (half a percentage point in fraction units);
2. `M >= M_FC8 - 1e-10`;
3. `C >= C_FC8 - 1e-10`;
4. raw net corrections versus same-input FC8 is positive.

These are predeclared engineering research gates, not confidence intervals. Exact teacher copying cannot pass item1. No individual-category all-nondecrease claim is implied.

Within each R arm choose its best **passing** checkpoint; if none passes, record its best diagnostic checkpoint. Ordering is A descending, then M descending, C descending, CE ascending, step ascending. Round only for human tables; ties within1e-10 use the next key, and exact final ties choose earlier step. Across passing R arms use the same order then fixed simpler-arm order R0,R1,R2,R3. The selected architecture is Rstar.

| State | Mandatory action | Expected training work | `all` exit |
|---|---|---:|---|
| No seed17 R passes | Publish all4 R results/selected diagnostic heads and prior audit; no G/new maps/H2 | 8,000 | 0, COMPLETE_NO_2D_FOUNDATION |
| Seed17 R passes, same-architecture seed29 fails | Publish Rstar repeat and full R comparisons; no G/new maps/H2 | 10,000 | 0, COMPLETE_2D_NOT_REPEATED |
| R passes in both | Freeze both bases; train fiveG in both seeds; activate transfer/confirmation/maps | 30,000 maximum | 0 if all activated stages complete |
| Required activated stage technically blocked | Finish valid independent work and publish partial evidence | actual | nonzero |

Do not train a second seed of every R arm. R0–R3 mechanism contrasts are initially single-seed; the qualified architecture is repeated. If publication claims every R contrast is replicated, that would exceed the evidence.

## 3. G selection and direct-pair interpretation

G0 is same-seed frozen Rstar; it has no new training. Each G arm chooses its best checkpoint using the same A/M/C/CE/step order. A proposed G nominee needs A at least0.005 above Rstar and nondecreasing M,C, plus positive net corrections against Rstar. If none qualifies, retain Rstar as DEV nominee. G1 is a valid learned simple alternative; do not force G4/G5 to win.

Before H2/real-proposal/map results, freeze seed17's nominated architecture and all selected checkpoints. Seed29 trains all fiveG with its separately qualified same-architecture Rstar; it does not pick a new architecture for the final story. Keep all repeated direct contrasts regardless of sign:
- G2 minus G1: physical grouping versus view grouping.
- G3 minus G2: additional membership supervision without direct use.
- G4 minus G3: direct routing of the same supervised membership.
- G5 minus G4: extra correspondence loss.
- each G minus Rstar and same-input FC8: absolute value, not merely recovery from a damaged control.

G1/G2/G3/G4/G5 have equal trainable layer shapes and within-seed starting parameters. H2 and map comparisons use same observations and per-seed base. Two seeds demonstrate sensitivity/repetition, not independent scenes or a formal significance result. If the direct comparison lacks repeated positive evidence, do not call its mechanism validated.

## 4. Mechanism diagnostics after selection

Run small deterministic interventions on fixed preselected DEV objects, then report old H/H2 descriptively:

**Membership route:** save raw predicted group rho, quality weights, object alpha, null weight, and pre-output local delta. At fixed keys/values, set all rho to1 (ablation), multiply by0.1 and0.01 (sensitivity), set all rho0 (null test), and lower one chosen group's rho. Predictions under interventions are diagnostics, never new contenders. Do not choose intervention values by AP.

**GT diagnostic purity:** after prediction locks, use annotation membership to compute non-target attention coefficient mass. Also report coefficient*value-norm mass, making clear this is a proxy contribution, not a causal decomposition of the normalized final representation. Report target/negative/unknown tokens and effective groups, member counts and missing groups. Do not compute purity only over easy matched cases without showing excluded counts.

**Correspondence:** deterministically rotate/permutate site correspondences in each nonanchor view (seed hash of object identity) while preserving each complete token's features, metadata and validity; do not change the VIEW partition. No GT chooses a harmful permutation. Measure output/class changes and recognition, not only embedding distance. A trained model being sensitive to wrong inputs alone does not prove it beats matched valid-input controls.

**Preservation:** compare teacher-correct and teacher-wrong TRAIN/DEV strata; report new correction/harm counts, clean/corrupt scores and prototype competition. Distinguish pretrained FC weights staying unchanged from the learned readout retaining their behavior.

For real proposals, record generated/lifted/multiview/matched counts and both proposal and unique-GT averages. For H2, disclose full200 competition and base/heldout support. H-old is exposed; H2 is only new-family **proposal** evaluation.

## 5. Whole-map outputs and gates

The main scientific output on the normal path contains nine methods: parent D2/G1/FC8, selected Rstar, and fiveG at seed17. Repeat contains six methods: Rstar and fiveG at seed29. Thus390 scene-method rows and30 ordered pools at most. Parent FC2/4 can appear in a historical appendix; they are not extra new scientific evaluations.

Use the same parent G1 exclusive owner array, surface, valid vocabulary, nearest target projection, per-object eligibility and recovery labels. Only selected existing labels change. Use the exact parent's N/Q grouping with new F, no post-result mixing-grid search. Ordinary FC8 baseline must reproduce its actual previous F/new-rank outputs; residual step0 is not necessarily G1.

Reuse `disagreement_query.outputs.build_payload` and LR prediction/evaluation primitives through an adapted registry; do not call hardcoded9/286/22 completion logic with fabricated rows. Enumerate actual positive G1 owners when constructing the evaluator mask registry, including unchanged recovery owners. Recompute current-class official area ranks after relabeling. Instance AP and semantic mIoU consume the **same** output. Do not claim reconstructed geometry/recall changed when the owner array is fixed.

Exact evaluation aliases require identical owner/semantic arrays, official serialized ranks, target projection and scorer context. An embedding hash or same rounded AP is insufficient. Ordered pooled AP is the released joint computation, not an average of per-scene AP. Preserve all five metrics, per-class scores, matching events and whole-scene confusion matrices.

Final map gate, applied separately to each seed method:
- Replica8 all five metrics >= parent G1 minus1e-10.
- CF18 all five >= parent G1 minus1e-10.
- CF18 APall > D2+1e-10; CF18 AP50 >= D2-1e-10.
- Material flag: CF18 APall-D2 >=0.001-1e-10 (0.10pp).

No latency/memory criterion. Development R gates cannot relax this final target. No cohort-dependent choice, best-seed ensemble, truncation of weak classes, reduced vocabulary AP or GT selection of which objects to update.

Report:
1. `DEV_NOMINEE` selected before transfers;
2. `EXPOSED_BENCHMARK_CANDIDATE` best passing seed17 result, if any (lexicographic mean APall gain then minimum cohort APall gain then mean mIoU gain then simpler method ID);
3. same-architecture seed29 outcome, never mixed with seed17;
4. `MECHANISM_EVIDENCE` direct matched contrasts and limits;
5. deployment `N0_UNCHANGED`.

A map winner different from DEV nominee is exploratory, not a confirmation-picked result. H2 result never picks the winner. If all fail, retain G1 and publish all negative results.

## 6. Three compact table families

**Table1 — full maps:** D2, G1, FC8, Rstar and fiveG; Replica8/CF18 APall/AP50/mIoU main, AP25/mAcc in companion columns/appendix; seed29 separate. Early stop has an explicitly NOT_TRIGGERED table, not blank values presented as measured zeros.

**Table2 — why learning helps or hurts:** fourR factorial on DEV/oldH/H2; learned-vs-FC, SURFACE–VIEW, ROUTED–AUX_ONLY and CORRESP–ROUTED; Top1/macro/NLL, correction/harm, seed and supervision budget. Parent negative results are historical, not current retrained rows.

**Table3 — actual route/robustness:** target/neighbor attention mass, null usage, correspondence validity, proposal corruption and real proposal recognition/coverage; training/input counts and conditional compute as supplemental evidence. Runtime is not the headline.

Machine-readable TSV/JSON/CSV plus compact Markdown/LaTeX fragments suffice. Do not create another enormous spreadsheet of signature counts. Retain raw numerical precision in JSON/NPZ. Store/readout dimensions and labels must remain typed rather than string-only table artifacts.

## 7. Evidence files and actual publication

Publish code, spec, method definitions, locked role lists, teacher-sidecar summaries, per-branch `updates.jsonl` or a lossless compressed copy, each `dev_*.json`, selected checkpoint/last metadata, RNG/sampler description, concise per-object scores/decisions, official pools, interventions and warnings. Unlike the previous release, component curves must be accessible in the evidence package; include effective supervision denominators, not just summed losses.

Selected inference weights need trainable MA/local modules and every required small frozen initialization/base component. Manifest deduplicates shared frozen Rstar/reference files. FrozenFC/text/backbone weights remain external. Load with strict state dict checks and exact config; sample inference must match before/after export. Report trained-head params separately from frozen MA reference/base and full FC backbone. Do not advertise no-extra-model memory when the residual uses a frozen MA reference.

Respect third-party licenses. No raw ScanNet scans, private GT maps, foundation weights or credentials on GitHub. Large feature grids and training optimizer checkpoints may stay in the recorded capacity store; selected small inference heads and the required compact evidence must actually be pushed, not represented only by server paths.

Run the full CLI once with resume, observe its exit, then commit and push normally. After push, compare complete local HEAD with the exact remote branch via ls-remote. Record exit status, head hashes, result identity, stage state, parent identity and working-tree status in **external** `publication/final.json`. A push-verified negative run is complete scientific execution, not accuracy success. Publication failure is separate from scientific failure.
