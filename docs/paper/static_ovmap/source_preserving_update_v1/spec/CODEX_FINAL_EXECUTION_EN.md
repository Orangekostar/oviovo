# Final execution instructions: source-preserving semantic updates

## 0. Mandate and scientific objective

Implement, run, evaluate, select, document, and publish the bounded study below in `Orangekostar/oviovo`. Do not stop at a plan, smoke test, or newly written functions. The deliverable is a real nine-condition comparison on the existing complete Replica-8 and ScanNet-CF18 cohorts, plus a reproducible GitHub publication, whether or not any candidate improves performance.

The question is: **when the same newly read object evidence is used, can preserving the original N/Q and/or F distributions recover the CF18 improvement of incumbent rereading without damaging Replica?** Separate the influence of new observations, ordinary AnyUp versus coarse FC, and the way old/new sources are combined.

This is the first, evidence-injection stage of the broader research direction. Do not train an update-risk classifier, collect new scenes, build an adaptive query policy, or restart structural repair in this task. A later stage may be recommended, but it is not an unfinished requirement of this one.

The authoritative constants and IDs are in `PROTOCOL_SPEC.json`; algorithm details are in `IMPLEMENTATION_CONTRACTS.md`. This English file orchestrates execution. If these files truly conflict, record the conflict and resolve it explicitly before the affected scientific run; do not silently invent a different experiment. The supplied reference tests are not production validation. No scientific accuracy improvement is asserted by delivering this package.

## 1. Isolate the task and preserve the parent

Use base commit `b000355eb8f91491e002df1499bd0170d91c35ef`, target branch `research/ovimap-source-preserving-update-v1`, and output root `/mnt/shared/ww/ovimap-source-preserving-update-v1/attempt_001`.

The actual parent is `/mnt/shared/ww/ovimap-minimal-instance-repair-v1/attempt_001`; its known physical-storage hint is `/home/ww/ovimap-minimal-instance-repair-v1-storage/attempt_001`. Resolve a symlink or explicitly supplied path map and preserve both logical and resolved paths. Do not assume any of these server paths exists in a different machine. Bind what is present.

Inspect `git status`, `git worktree list`, and the exact base commit. Create a new worktree from the base if necessary. Leave unrelated working trees, user edits, previous freezes, datasets and artifacts untouched. Do not reset, clean, force-push, upgrade environments, or merge into the deployment branch. If the target worktree already exists, verify it belongs to this task and resume it. A newer parent branch does not silently change the base.

Copy this specification package into `docs/paper/static_ovmap/source_preserving_update_v1/spec/`, excluding bytecode. Copy `PROTOCOL_SPEC.json` to `configs/static_ovmap/source_preserving_update_v1.json` verbatim. Record the base, current branch, initial dirty-file list and source-package digest.

Use the inherited controller environment `/home/ww/miniconda3/envs/ovimap-map/bin/python`, and obtain the actual FC worker Python, weight identity, text assets and dense cache roots from the parent binding/assets. Use one GPU worker and at most three CPU evaluation workers, each with at most four BLAS threads. Do not use the FC environment to run an incompatible released evaluator.

Check free space once before acquisition and at chunk boundaries during writing. Use an explicit `--storage-root` if needed. There is no authorization to delete older experiments. A genuine disk or dependency failure is recorded and handled locally; it must not cause successful unrelated leaves to be rerun.

## 2. Read these production interfaces before editing

Read the symbols listed in `SOURCE_EVIDENCE.md`, especially:

- `minimal_instance_repair/binding.py`: parent data shape, `load_scene`, frozen N/Q/F and D2 probabilities.
- `minimal_instance_repair/recognition_plan.py`, `recognition_worker.py`, `reread.py`: selected incumbents, FULL masks, saved per-region scores, original IR06/IR07 decisions.
- `backbone_wave1/readouts.py::fuse_readout` and `paired_evidence_study/residuals.py`: real D2 grouping and stable probability/residual utilities.
- `cvpr_compact/area_fallback.py::region_vector`: the exact same-view coarse representation control.
- `minimal_instance_repair/outputs.py::construct_partition`: actual raw-zero class-change cancellation, and official rank recomputation.
- `minimal_instance_repair/evaluation.py` and `m2_reviewer_study/evaluation.py`: actual-mask registries, evaluation identity, exact ordered pooling.

Do not call the old `all` controller or rerun the old structural preparation. Its mandatory proposal/observer dependencies are outside this study. Build a narrow new adapter that loads the necessary immutable inputs without rewriting the old protocol guards. Do not weaken old assertions to fit this task.

## 3. Bind actual evidence, not report numbers

Produce `source_binding.json` with actual paths, hashes and relevant identities. Confirm that the parent has 234 scene-method records and 18 complete pools, and that the reference predictions are the final post-correction parent version. The manuscript's rounded numbers are sanity checks, not inputs to scoring or selection.

For all 26 scenes bind:

1. The actual D2 and G1 payloads, complete masks/labels/ranks, evaluation receipts and ordered baseline pools.
2. Parent `recognition/<scene>/plan.json`, `decisions.json`, and the corresponding successful frame receipts/vector references. Published equivalents under `artifacts/.../scenes/<scene>/` can recover small records only when content/parent identity verifies; a similarly named file is not a substitute.
3. The unchanged N/Q/F score distributions, positive temperatures, ordered `valid_ids`, class text identity, and FC model identity from the common context.
4. Each parent's selected incumbent list and exact FULL views, including original support hash, frame/bank, RGB identity, mask, bbox, sizes and per-region AnyUp score vector.
5. G1 recovered-owner classes, support and all unselected incumbents that must stay identical.
6. Raw owner values needed for the inherited protected-raw-zero cancellation rule, and frozen evaluation nearest/matched arrays for ranking and scoring only.

Reconstruct D2 through the actual `fuse_readout` and compare its full probabilities with the parent at absolute tolerance `1e-12`; labels must be identical. Validate full vocabulary ordering, not merely vector lengths. N and Q remain distinct evidence streams even though they use the same physical visual model. Do not average their visual vectors with FC vectors.

Reconstruct the parent IR06/IR07 decisions from all of their saved region records using the original `reread_decision()` and original parent thresholds. Reconstruct their final arrays/ranks with the raw-zero cancellation and verify the actual historical payloads. This is a lineage check, not a new parameter search. If reconstruction is wrong, locate the mismatch before interpreting a new result. It does not require rerunning AnyUp or rerunning historical evaluation when exact identity is proven.

Never pretend that a missing successful parent AnyUp record is an ordinary no-view case. Try the exact bound published/physical copies once. If it is still unavailable, mark dependent scenes blocked and continue unaffected work; do not fabricate scores, silently downgrade the input, or relaunch all historical inference.

## 4. Lock the shared observation set

The object selection is exactly the parent's at-most-16 original incumbents per scene. Do not rerank after seeing new scores or replace blocked objects with easier objects.

For every selected incumbent, take all distinct parent-successful `FULL` AnyUp observations, at most one from each original bank and at most two in total. This set is V_i. One successful FULL observation is permitted; zero means keep G1. CORE and FRONTEND_INTERSECTION observations are read only to replay IR07; they do not enter the new A/C aggregates.

Create `evidence_manifest.json` before coarse scoring. The exact same FULL RGB/mask pairs will be used for coarse C and AnyUp A. No new projection, view search, cropping, erosion, segmentation or mask expansion is authorized.

The study deliberately uses a common valid-evidence domain for its matched comparisons. For an incumbent to enter it, require original selection, reconstructed D2 probability, available historical F, at least one successful parent A FULL view, successful C on every view in V_i, and no protected raw-zero row. All seven matched/new arms keep the exact G1 class outside this domain. No GT is involved.

**SU02/SU03 are matched replays, not automatically the historical IR06/IR07 values.** If the common domain differs from the parent domain, compute the matched outputs and their actual metrics. Show the historical-to-matched change separately, without relabeling old numbers. Retain every selected incumbent and its exclusion reason in the coverage ledger. The main cohort denominator remains the complete scene, never just the eligible objects.

The matched-domain restriction is for causal comparison of evidence injection. SU06 uses C for its decision, but belongs to a domain originally established using available A/C observations. Do not claim this matched experiment alone proves a standalone FC-only acquisition policy.

## 5. Obtain only the missing same-view coarse FC evidence

Implement `coarse_worker.py` in the inherited FC environment. Before scheduling a coarse pool, skip objects already outside the potential common domain because of absent D2/F, zero successful A FULL views, or protected raw-zero rows; retain their exclusion records. For each remaining required FULL region, reuse exactly the parent RGB conversion, `image_tensor`, `signed_mask`, original FP32 FC dense features and frozen visual prediction head. Call `cvpr_compact.area_fallback.region_vector`, including its v2 area fallback. Convert to per-view cosines through the same `cosine_record` convention. Do not substitute nearest-pixel pooling or raw dot products before the visual head.

Group requests by actual image tensor/model identity. Prefer the exact parent frame `dense_receipt`/`dense_arrays`. Validate tensor/model/RGB identity, dtype and shape. Use the inherited ContentCache for matching read-only caches; recompute only a missing dense leaf from the bound FULL RGB, and store it in the new task root. Never invoke AnyUp: all A scores already exist. An owner ID alone is never a region-cache key.

The number of successful new coarse region pools is at most `26*16*2=832`; new FC encodings cannot exceed the number of required distinct FULL image identities (and therefore cannot exceed 832). Report the actual smaller bound after inventory. New AnyUp Q/K, N/Q, frontend and mapping calls must be zero.

A completed valid input whose pool returns `EMPTY_AREA_MASK_SUPPORT` or `INVALID_REGION_FEATURE` is an unavailable C observation. Keep the object outside the common domain; do not discard a paired A view just to obtain a more favorable mean. Nonfinite/corrupt inputs, missing files, OOM or unfinished workers are execution failures, not scientific abstentions. At most one identical transient retry; then log and repair the cause. Do not repeatedly retry a deterministic representation failure.

Seal per-region scores and statuses, then seal the common domain. Count logical requested regions, actual successful pools, dense-cache hits, actual new FC image inputs, failed attempts and wall time independently. Do not report all cache hits as new inference or all the study as zero inference merely because A was reused.

## 6. Implement and execute the exact nine-condition slate

Use the equations and edge cases in `IMPLEMENTATION_CONTRACTS.md` and the reference mathematical functions. All new operations use float64 CPU probabilities, the original F temperature, the complete ordered vocabulary, and the same per-view equal-cosine average. There is no temperature fit, weight sweep, top-k class restriction, softmax logit-scale multiplication, or extra acceptance threshold.

| ID | Operation |
|---|---|
| SU00_D2 | Exact parent D2 output, as the no-recovery comparator |
| SU01_G1 | Exact current complete G1 output |
| SU02_HARD_MATCHED | Original IR06 class decision, applied only on the common domain |
| SU03_STABLE_MATCHED | Original IR07 class decision, applied only on the common domain |
| SU04_F_REPLACE | Keep original N/Q; replace dense group F with A |
| SU05_F_BLEND | Keep original N/Q; dense group becomes `(F+A)/2` in probability space |
| SU06_F_COARSE | Same as SU05, but use paired coarse C instead of A |
| SU07_GLOBAL_BLEND | Mix whole D2 with A, with beta equal to SU05's A mass |
| SU08_PAIRED_DELTA | Update only F with half the paired A-minus-C log-score residual; keep N/Q |

The point of SU08 is to test whether using only the same-observation representation change is better than adding all A evidence. It is not a proven Bayesian likelihood-ratio method. Its helper already exists in this repository; do not claim the formula is a new invention.

Do not impose the IR06 raw 0.01 margin or IR07 stability checks on SU04–SU08; that would create undeclared combinations. These five arms use full-distribution argmax in the original category order. Missing-domain keeps and the inherited raw-zero cancellation are the only shared safeguards. Apply the same rules to Replica and CF18. Log any proposal whose winner is neither the previous class nor the original A top class.

Implement `decisions.py` without evaluator imports. Save each old distribution, paired new distributions, intermediate dense group, final distribution, proposed class, final applied class and reason, or their deduplicated references. Also save equality flags between arms and logit/margin changes. A change in a probability vector with no class change is not a changed map.

## 7. Construct real outputs and evaluate them correctly

Start from the actual G1 arrays, with `operations=[]`. Only the selected common-domain original D2 owners may be uniformly relabeled. Preserve owner IDs, mesh, source rows, faces, TSDF, point projection, every G1 recovered-owner class, and all noneligible incumbent classes exactly.

Retain the parent whole-incumbent abstention when a proposed class change would modify any `raw==0` row. This restriction is inherited for comparison, not a newly validated semantic principle. Report its impact; do not silently remove it or partially relabel an owner.

Use a narrow builder or the unchanged `construct_partition` with no structural operations. Recompute original current-class, target-area ranks for the entire output; preserve six-decimal released rank serialization. Old ranks may legitimately change when classes change even with fixed masks. Use `PredictionPayload(branch='COMBO')`, and lock complete predictions before opening GT labels.

Use an actual G1 partition registry, including recovered owners, not an incumbent-only N0 source dictionary. Reuse the reviewed `partition_evaluator` logic or a narrow equivalent. Do not rerun parent freeze/structural phases. Baseline and identical-content aliases require exact owner, semantic and rank arrays, matched evaluator context, vocabulary, target projection and ordered inputs. Equal headline metrics are never an alias proof.

Evaluate all nine methods on all 26 complete scenes, use the unchanged released threshold vector and official pool, and compute all five metrics. Keep APall = .50:.05:.90, AP25 = .25, strict released comparisons, ignored-region logic and minimum region size unchanged. No averaging per-scene AP to stand in for a dataset pool. Semantic confusion must cover the complete valid target set, including small/unknown/recovered support.

Produce 234 scene-method rows and 18 ordered pools. Two main baselines contribute 52 reusable rows; the seven matched/new arms contribute at most 182 new actual-scoring or identity-proven rows. Reuse calculations where full content matches, while retaining all logical conditions. A blocked scene prevents that method's full pool; never rescale a subset into a full result.

## 8. Diagnose why an update helps or hurts

After prediction lock, reuse parent geometric-matchability and released trace machinery. Compute new diagnostic matches only for the actual changed labels; do not change masks in diagnosis.

For each matched/new arm versus SU01, and the six predefined method pairs, report:

- Selected, common-domain, unavailable, protected, proposed-change and applied-change counts by cohort and scene.
- Wrong-to-right, right-to-wrong, wrong-to-wrong and unchanged outcomes for geometrically matchable original incumbents, with GT matching definition stated. Keep geometric-insufficiency cases separate. Matchable proposals are not necessarily new unique GT matches.
- Unique released GT50 and GT75 gained/lost; definite versus ambiguous tied TP/FP score entries; changed old ranking and class-level pooled differences.
- Whether each changed object had available N/Q/F, its old-source margins for old/new classes, and A/C paired score differences. Do not treat same-encoder agreement as independent evidence.
- Recovered-owner array/class parity, and the number of new changes prevented by the inherited raw-zero rule.
- Common-domain replay versus original IR06/IR07: changed coverage and resulting output identity; do not attribute this coverage difference to fusion.

Do not route predictions by these diagnostic labels. Do not train from this ledger in this task. Conclusions must distinguish an input/aggregation change, a changed posterior that does not change a label, a changed label that does not create a match, and a genuine matching improvement.

## 9. Selection and stop decisions

Use unrounded fractions. A candidate passes only if all five metrics in both complete cohorts are no lower than SU01 within `1e-10`, CF18 APall is strictly greater than D2 plus tolerance, and CF18 AP50 is at least D2 within tolerance. The additional material marker requires CF18 APall minus D2 >= .001 fraction; it does not replace the pass gate or assert significance.

SU02–SU08 are all eligible to win, including simple matched controls. Apply the specified lexicographic metric order and fixed simplicity tie-break. With no passing candidate, retain SU01. No dataset-specific routing, best-class mosaic, post-hoc thresholds, or hidden rescue fitting. Use separate statuses for execution completion, scientific target, selection, and publication. Keep deployment `N0_UNCHANGED`.

Generate `next_stage_assessment.json`: state whether source preservation helped relative to matched hard replacement, whether AnyUp beat same-view C under the same update, whether F-group updates beat equal-A-mass global mixing, and whether paired residual adds value. Give measured quantities. These observations can motivate a later risk model or calibration study, but do not start it.

## 10. Cost accounting: no second expensive cold-timing campaign

This task measures conditional CPU decision overhead with all scores resident, on Replica-8, two opposite-order rounds. Warm once, then time 50 applications of the unchanged decision function per scene/method and divide by 50. Keep both round values and report the mean in milliseconds/scene. Exclude loading, FC, AnyUp, view selection, ranking, payload copying, evaluator and file writes. Separate an optional payload-build wall time from decision time; never relabel it as end-to-end runtime.

For acquisition, retain actual per-frame pool/encoding counters and wall time. Report inherited A evidence and source acquisition as **reused but not computationally free at deployment**. Existing 16.23/81.10/etc. seconds are prior-task references only, not new timing measurements. Do not reuse them as this task's new performance claims.

There are zero newly authorized end-to-end cold calls. Once a real precision candidate exists, an independent full-boundary timing protocol can be defined. This keeps the present study focused and avoids measuring many unsuccessful methods with repeated GPU inference.

## 11. Limited, relevant validation

Run approximately ten focused production tests once before the main decision freeze; rerun only affected tests after a necessary bug fix. Run two real pilots on the already fixed office1 and scene0011_00 evidence. Pilot outputs may later be reused by exact identity. No whole-repository test suite, fuzzing campaign, recursive artifact rehash on each phase, or three redundant audits.

Test the properties enumerated in `IMPLEMENTATION_CONTRACTS.md`. Crucially verify actual source reconstruction, matched-domain keeps, raw-zero behavior, common A/C view identity, F-only and missing-source behavior, residual identity, same-partition scorer coverage and real historical replay. The supplied local synthetic tests cannot certify these production properties.

Finish implementing all phases and freeze source/config before reading new full pooled outcomes. Existing parent results are already exposed. If an objective code defect is discovered later, record the correction and invalidate only affected descendants. Preserve superseded records. Do not present the repaired protocol as untouched prospective confirmation.

## 12. Controller and executable commands to implement

Implement the following new entry point and phases (these are deliverable paths, not existing files):

`src/static_ovmap/source_preserving_update/` with responsibilities `binding`, `evidence`, `coarse_worker`, `decisions`, `outputs`, `evaluation`, `analysis`, `selection`, `reporting`, and resumable orchestration. Responsibilities may share utility files, but all phases must work.

`scripts/evaluation/run_ovimap_source_preserving_update.py`

Accepted flags: `--spec`, `--parent-root`, `--output-root`, `--storage-root`, `--path-map`, `--gpu`, `--phase`, `--resume`. Phase order:

`bind -> prepare -> coarse -> pilot -> freeze -> predict -> evaluate -> diagnose -> select -> costs -> tables -> publish`.

`all` runs this sequence. `--resume` skips only content-validated successful leaves; it does not rerun finished acquisition or parent controllers. `pilot` runs new production decisions/build/evaluation on the two existing pilot scenes, with no extra AnyUp and no parameter changes. `freeze` records actual implementation/config identity. Metadata keys must respect the existing payload validator; GT-bearing diagnostics remain separate from predictor metadata.

Final actual command:

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python \
  scripts/evaluation/run_ovimap_source_preserving_update.py \
  --spec configs/static_ovmap/source_preserving_update_v1.json \
  --parent-root /mnt/shared/ww/ovimap-minimal-instance-repair-v1/attempt_001 \
  --output-root /mnt/shared/ww/ovimap-source-preserving-update-v1/attempt_001 \
  --phase all --resume
```

Honor optional explicit path/GPU/storage overrides throughout nested structures and subprocesses. A pilot with no applicable edit is a valid KEEP case, not proof that a nonidentity path was exercised. Verify at least one actual C-readout path among the already-required acquisition records; if the entire common domain is empty, complete the supported baseline/KEEP outputs and publish `NO_APPLICABLE_PAIRED_UPDATES` rather than inventing a success or selecting extra objects. No extra interactive confirmation is required for the authorized ordinary branch publication. Read dependencies to resolve paths before asking the user anything. For truly unavailable resources, publish the exact partial/block status, not a fabricated completion. The `all` orchestrator must attempt unaffected leaves and reach reporting/publication for a partial run. Its final exit status is zero only when all required scientific outputs and publication are complete; a scientifically negative but fully executed study can exit zero, while an unresolved dependency or failed push must exit nonzero with an explicit summary.

## 13. Publication and completion

From one canonical result store generate three tables, machine data, compact evidence, and four reports:

- `docs/paper/static_ovmap/SOURCE_UPDATE_RESULTS.md`
- `docs/paper/static_ovmap/SOURCE_UPDATE_HANDOFF.md`
- `docs/paper/static_ovmap/SOURCE_UPDATE_SELECTION.md`
- `docs/paper/static_ovmap/SOURCE_UPDATE_CLAIMS.md`

Publish compact results under `artifacts/static_ovmap/source_preserving_update_v1/`: main tables in MD/CSV/JSON/LaTeX, scene/pool metrics, selected-object and paired-view manifest, source-score bundle, decisions, diagnostics, coverage, source identity, execution/cost logs and explicit dependency manifest. Do not regenerate or publish licensed RGB-D, checkpoints, full TSDF, dense feature maps or enormous redundant per-method arrays. Large predictions stay in shared storage with exact hashes, relative identifiers and regeneration commands. Small real scores/results must be on GitHub, not just server links.

Include the original IR06/IR07 historical figures in a clearly labeled contextual panel or appendix with their actual provenance. Do not paste them into matched replay rows. The package's reference math does not justify claiming a new Bayesian method or probability calibration.

Run the delivered full command to a real terminal exit status, then commit code and actual small results to the named branch and ordinary push it. Verify `git rev-parse HEAD` equals `git ls-remote origin refs/heads/research/ovimap-source-preserving-update-v1`. Write `publication/final.json` outside the commit with full local/remote SHA, UTC timestamp and statuses. Do not try to embed a commit's own hash recursively inside itself. If a final report amendment requires a new commit, push and verify the new SHA again once.

The final answer to the user must give scope completed, exact target result, passing candidates or none, preserved deployment, three table summaries, actual incremental compute/cost boundary, remaining limitations and full verified publication SHA. Never call a zero-exit run or passing test suite a scientific improvement. Do not promise CF18 gains or Replica nondegradation before measuring them.
