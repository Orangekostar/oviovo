# Evaluation, selection, workload and publication rules

All accuracy gates consume finite **fractions**, not percentage strings. Main metrics, in order: `apall, ap50, ap25, miou, macc`. Undefined metrics are null with provenance, never zeros. An undefined required gate metric cannot pass. Percent/pp conversion occurs only in reports.

## 1. Phases and coverage accounting

### Diagnostic records (not additional main methods)
- D-Surface: four screening scenes and their selected prediction-side owners; three visibility representations.
- D-Vocabulary: all 26 parent scenes, original available score vectors; nested class subsets and two diagnostic score treatments. No new visual/text inference and no reduced-vocabulary AP table.
- Fixed-set duplicate/permutation/withdrawal checks: a small representative set plus deterministic unit fixtures; do not inflate experiment counts by individual classes or array elements.

### Screening A
Fixed four scenes, ordered as `replica_probe2=[office1,room0]` and `cf_probe2=[scene0011_00,scene0050_00]`.

Six methods: DQ00_D2, DQ01_G1, AREA_MEAN, COVERAGE_MEAN, VERIFY_MEAN, DISAGREEMENT_MEAN. **24 logical scene-method rows, 12 ordered pools.** Baselines must be pooled for these exact two-scene sets; the complete Replica8/CF18 values cannot substitute. Whole-scene output includes all baseline objects, not only queried owners.

Choose one simple query QS from AREA/COVERAGE/VERIFY. For each method m define, from the two screening pools relative to G1,
```
R(m) = (min(delta_APall_rep, delta_APall_cf),
        mean(delta_APall_rep, delta_APall_cf),
        mean(delta_mIoU_rep, delta_mIoU_cf),
        -logical_FULL_reads, -logical_extra_probe_heads, -fixed_method_order)
```
Maximize this tuple on unrounded finite numbers. Use order AREA, COVERAGE, VERIFY. This is development selection across both pools, not a dataset-specific switch. Archive every competing result. There is no need for QS to pass the final upgrade gate.

### Screening B
Reuse all observations; do not obtain third views. Add AREA and SUPPORT updates on QS and on DISAGREEMENT (4 methods x 4 scenes = 16 rows, 8 pools). Thus **40 rows and 20 pools total**, at most 10 distinct method identities. If paths are identical, keep logical rows but use strict scoring-identity reuse. Update degeneracy is a valid scientific result.

Choose US from MEAN/AREA on QS only, using the same tuple; exact tie prefers MEAN. Record explicit QS and US in `screen_selection.json`. Now the six candidate paths needed for a 2x2 comparison all exist: QS and DISAGREEMENT crossed with MEAN, AREA, SUPPORT (some are inherited from Screening A).

### Screening extension gate
Evaluate these **three fixed** candidate-reference contrasts, with the now-frozen US:
1. DISAGREEMENT+US vs QS+US.
2. QS+SUPPORT vs QS+US.
3. DISAGREEMENT+SUPPORT vs DISAGREEMENT+US.

A contrast has an actionable screening signal only if ALL conditions hold:
- Candidate actual whole output differs from the reference for at least one scene.
- Candidate-reference APall delta in each screening cohort is >= -0.001 (minus tolerance); mean of the two APall deltas is >= +0.0005 (minus tolerance).
- Candidate vs G1 in EACH screening cohort: APall >= G1-0.001, AP50 >= G1-0.001, mIoU >= G1-0.002, with tolerance 1e-10. These are screening loss budgets, **not** the final upgrade criterion.
- Across the four scenes, the contrast has at least one net additional fixed-original-support correct-class outcome (`wrong_to_right - right_to_wrong >=1`) OR at least one net **class-aware unique** GT50 match (`gained - lost >=1`). A raw score-entry count does not satisfy this condition. Use the same old support/reference mapping for candidate and control; mark ambiguous/undefined mappings.

If at least one contrast passes, run the complete regression automatically. If none passes, stop new acquisition after the completed screening and publish `SCIENCE_COMPLETE_SCREEN_ONLY / COMPLETE_NO_SCREEN_SIGNAL / NOT_RUN_NO_SCREEN_SIGNAL`. Do not ask for another confirmation, add thresholds, increase view count, or pick a favorable new scene. This prescribed negative branch is task completion; resource or scorer failure is not.

This gate recognizes limited exploratory signals while bounding damage. It does not claim confidence intervals or unseen generalization. The final strict gate is unchanged.

## 2. Complete regression: fixed six conditions
After freezing QS, US and all operator hashes, run the original **ordered** Replica8 and CF18 lists in the JSON:
```
DQ00_D2
DQ01_G1
QS_US
QD_US                 # QD = DISAGREEMENT
QS_SUPPORT
QD_SUPPORT
```
These are **156 logical records, 12 complete pools**, including 52 baseline rows. The four screening scenes are included in the original order and may reuse exact scored outputs. The additional 22 scenes are not permitted to change QS/US or algorithm parameters. They are historically exposed, so call the outcome regression, not independent confirmation.

If only one proposed component helps, the corresponding single-component row can win. Even the simple QS_US can win. Do not make the combination mandatory. No new methods from the vocabulary diagnostic enter this table.

Main scientific requirements: actual fixed source geometry and one exclusive G1 partition; homogeneous current classes; original nearest-neighbor target projection and evaluator runtime threshold grid; original minimum 100 points; all confusion entries including unknown predictions; full ordered pooled matching with current-class area ranks. Keep all five metrics and per-class/per-scene traces, not just APall/mIoU.

The six contrasts to report are: QD_US-QS_US, QS_SUPPORT-QS_US, QD_SUPPORT-QD_US, QD_SUPPORT-QS_SUPPORT, final-G1, final-D2. If the last two have no new final candidate, report each tested row instead of declaring a gain. Different class changes can affect global AP without changing GT50 totals; include rank/class differences.

## 3. Final strict selection
Candidate set: all four non-baseline regression rows. A candidate passes iff:
1. All five metrics on BOTH complete cohorts are finite and >= the current G1 metric minus 1e-10.
2. CF18 APall > D2 APall + 1e-10.
3. CF18 AP50 >= D2 AP50 - 1e-10.

Material target additionally requires CF18 APall - D2 APall >= 0.001 - 1e-10 (0.10 percentage points). This is an engineering effect threshold, not a significance test.

If multiple pass, rank by `(CF18 APall, Replica APall, CF18 mIoU, Replica mIoU, -logical FULL reads, -extra probe heads, -fixed order)`. Fixed order: QS_US, QS_SUPPORT, QD_US, QD_SUPPORT. For actual output-identical methods prefer the simplest/lower-required-cost path. If none passes, retain DQ01_G1.

Show separately whether proposed query/aggregation beats its strongest matched simple control. A method beating G1 does not automatically prove that both new components are necessary. No per-object or per-dataset hindsight routing, no confirmation-set parameter fitting, no production deployment update.

## 4. Mechanism analysis
Produce one owner-level table linking:
- target selected / query eligible / required FULL success / applied change;
- common anchor and selected second frame for each policy;
- c0/c1, full and tile contrasts, real sign-conflict vs mere magnitude heterogeneity;
- area, coverage novelty, common physical support J, view angle and interest H;
- query divergence, fallback frequency, newly read versus duplicate evidence;
- AREA/SUPPORT weights, overlapping area, exact weight and output identity;
- fixed old geometry's GT assignment for diagnostics only, wrong-to-right/right-to-wrong and undefined mapping;
- class-aware unique GT50/75 changes, raw tied score entries and current-class rank changes;
- actual logical and physical model costs.

New source geometry and owner partition are immutable, so class-agnostic GT matching and geometric surface quality must not improve. If they differ, diagnose an output/evaluation bug rather than making a geometric claim. A scene/owner with no second qualified view remains in the baseline/no-query accounting; don't remove it from denominators.

After current-stage prediction locks, compute fixed original-support class references using the same released eligibility as the parent. Do not adapt the predictor to those references. Distinguish a maximum matching's non-unique identity choice from robust count changes; retain tie warnings. Do not claim each matching-set difference is a uniquely identified physical correction when maximum matchings are non-unique.

## 5. Resource bounds and what 'budget matched' means
Science excludes engineering probes, failures and conditional cold repeats, which have their own counters:
- Screening: 4 scenes x <=32 frames = <=128 distinct FC image encodings if none cached. <=64 objects; a shared anchor plus at most four distinct second views = <=320 FULL pools, <=256 anchor probe pools, **<=576 total successful region/head calls**.
- Additional full-regression scenes: 22 x16 objects x (anchor + two possible second views + four anchor probe tiles) = **<=2464 additional region/head calls**.
- Combined screening + full science: <=832 distinct FC image encodings (26x32); **<=3040 region/head calls**. Pilot aliases and shared frames reduce actual work. Do not count each logical method as independently encoding its images.
- Engineering: at most two real objects, <=4 FULL and <=8 probe pools, <=4 new image encodings outside normal science; record them separately.
- CPU geometry workers <=2, evaluation workers <=3, four threads each; one FC worker per chosen GPU. Preflight 20 GiB free in the new task store; record actual growth. No new model, training or segmentation.

Each operational query policy reads two FULL object observations. QD may additionally use four anchor subregions from the same FC image tensor. Thus FULL read/encoding budget is matched but head compute is not. The screen shared union need not physically repeat work, while reports/timing disclose policy-specific required work. For a deployable policy, account for geometrical candidate screening and anchor scoring, not only its final softmax.

An unrequested candidate's already-cached FC score is still unavailable to the policy before its second-view choice. Geometry and archived camera/depth are allowed for all candidates. Exact shared anchor records may be reused; every later allowed read is recorded.

## 6. Conditional timing
Do not run a large cold-timing campaign for a candidate that failed the strict full target. Always report science wall/CPU/GPU work and available conditional CPU aggregation measurements with their limited boundary.

If a candidate passes, measure FINAL and strongest matched simple QS_US (if FINAL=QS_US, choose the closest distinct simple operational comparator, AREA_MEAN; if its output/configuration is identical, measure once and alias explicitly) on Replica8 with two repeats, max 32 calls. Keep both repeated outputs and compare to the frozen scientific predictions.

Boundary: fixed G1 map, p0, raw input file locations and FC model/text are resident at the start. Time all required post-G1 physical site and observation preparation, candidate construction, RGB/depth reads, FC encoding, used FULL/probe pooling, policy selection, aggregation, ranked payload and required output writing. Derived per-call site/view/feature/result caches are empty. A single call may share frame encodings across its objects. Do not clear PyTorch's allocator cache as a substitute for feature-cache coldness. Keep OS page cache uncontrolled and say so.

Exclude original mapping/G1 production, initial model load, evaluation and optional archival verification. Report total wall and disjoint stage wall; GPU events aid attribution but are not added a second time. Peak allocated/reserved is for this process, not all upstream models. Same GPU and thread count, reverse scene/method order on repeat two; no cherry-picking fastest samples.

Worst-case cold envelope: 32 calls x32 encoded image frames =1024 inputs; 32x16x(2 FULL+4 probes)=3072 region/head calls. It is an upper bound, not a work target. Do not compare this incremental stage to OVI-MAP's online keyframe latency or infer 30 FPS.

## 7. Tables and completion criteria
Table 1: pilot 10-row result, clearly titled four-scene exposed screen. If regression executes, show its fixed six rows separately on full 8/18 pools. All five metrics in accompanying CSV.
Table 2: query and update matched contrasts, effective intervention/divergence, new correct vs damaged objects and exact-degeneracy rates.
Table 3: targets/eligible/updated counts, logical FULL/probe reads, actual cache/encoding/head work and costs. Add genuine conditional cold measurements only if executed.
Supplement: D-Surface and D-Vocabulary, source/eligibility coverage, all per-scene/per-class metrics, failure/unknown/tied events. No need for more backbone tables or a diagram-generation phase.

Screen-only expected science: 40 records and 20 two-scene pools, all diagnostic results. Full expected science: those screen results PLUS a separately labeled 156-record/12-pool regression (some actual score calls reused). Do not add overlapping logical datasets and call that many independent scenes. A real execution block must keep affected rows/pools missing and status non-complete.

Final reports distinguish implemented/executed/selected/published. CLI exit 0 requires realized-branch coverage, prediction access checks and reports; a successful empty run is prohibited. Normal Git push and verified external receipt finish publication even for COMPLETE_NO_SCREEN_SIGNAL or COMPLETE_NO_TARGET_GAIN.
