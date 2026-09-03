# CROVE Benchmark Alignment Handoff

Date: 2026-09-03

Repository: `git@github.com:Orangekostar/oviovo.git`

Base: `research/crove-localized-current-ownership` at
`668aefc49034ef97d090b811b1c3f291ecafbd66`

Working branch: `research/crove-benchmark-alignment-audit`

Frozen B0-B8 evidence HEAD before this report: `4443e7f`

## 1. Is the current TESSE official metric equivalent to the RSS 2024 paper protocol?

No equivalence is proven. The RSS release commit contains no `khronos_eval`
implementation, while the available evaluator/configuration is post-release;
the original paper data/evaluator closure is not content-addressed. Status:
`BLOCKED_PAPER_ASSET`. RSS Table I values are reproduction targets, not a local
same-protocol baseline.

## 2. Is the local summarizer equivalent to upstream aggregation?

Yes, within the frozen latest-public post-release scope. A6 and P5 row grids,
duplicate collapse, available slices, and Object/Dynamic/Change/Background F1
match with maximum absolute delta `1.3877787807814457e-17`, below the frozen
`1e-12` tolerance. This does not prove RSS evaluator identity.

## 3. Is the CROVE-versus-Khronos semantic frontend comparison fair?

Not as currently presented. CROVE C0 uses an end-to-end open-vocabulary
frontend, whereas the RSS Table I K0 target uses simulator GT semantics. K0,
K1, C0, and C1 must remain separate conditions; oracle rows are not ranking
eligible.

## 4. How much of the CROVE gap does the GT-semantics oracle close?

No percentage is identifiable. K0 is `BLOCKED_PAPER_ASSET`; C1 and K1 are
`INCONCLUSIVE_MISSING_CONDITION`. The preregistered 50% strict-majority rule
cannot run without at least two finite, matched gaps. Missing values were not
converted to zero.

## 5. What temporal protocol do the current official artifacts contain?

They are a post-release, diagonal-equivalent variant. Each has 946 raw rows but
only 43 unique `(map, query)` states, all satisfying belief time equals robot
time; the inner historical loop duplicates the diagonal and does not create a
true off-diagonal history grid. They are neither a verified RSS Table I output
nor evidence of genuine full historical 4D behavior.

## 6. Does the current-state slice change method ranking?

No on the available diagonal-only artifacts. Current and paired post-release
aggregates are numerically identical, with zero reversals across six finite A6
pairwise comparisons. This rules out a favorable reversal claim; it does not
test how a genuine off-diagonal historical grid would rank the methods.

## 7. Did exact Dynamic attribution succeed?

Yes for the frozen P5 Apartment map under the latest-public post-release
evaluator. The reviewed patch emits 1,135,450 dynamic and 118,859 object/change
events; every row-level TP/FP/FN mass matches, and all three patched/unpatched
official CSV pairs are byte-identical. This is a debug diagnostic, not an RSS
result. The earlier P6-E finding remains valid for unpatched final CSVs alone:
they expose 0% exact runtime provenance.

## 8. Can 3RScan localize identity and change errors exactly?

The metadata and evaluator contract can: fixed IDs, session order,
rigid/nonrigid/removed metadata, evaluator-derived additions, ID switches,
false Re-ID, reactivation, stale-object FP, and type recall are explicit. The
real pilot cannot yet demonstrate this because the selected 44 visits lack
runtime RGB-D/mesh/annotation members. Status: `BLOCKED_DATASET_ACCESS`.
Community t-AP/t-REC and custom exact-ID diagnostics remain separate.

## 9. Is Flat more directly sensitive to signed visibility and current ownership?

The pre-result literature/GT contract says it is a more direct controlled
mechanism test than aggregate TESSE history: it offers two causal runs, changed
objects, complete current geometry, and GT/predicted panoptic conditions.
Empirical sensitivity is unmeasured because the archive is inaccessible from
this provider. Status: `BLOCKED_DATASET_ACCESS`; no synthetic score is promoted.

## 10. Is CROVE still weak on aligned benchmarks?

That broad claim is not identifiable. The matched oracle matrix and both
alternative real-data pilots are incomplete. On the available Apartment
evidence, P5 improves Change F1, current mIoU, and Ghost over A6 but lowers
Object F1; P6-C does not repair that tradeoff, and no tested variant passes all
frozen gates. This is confirmed method-side weakness on one tradeoff, not a
general aligned-benchmark verdict.

## 11. What is the final benchmark decision?

`KEEP_TESSE_HEADLINE_WITH_SPLIT_PROTOCOL`. Retain TESSE for continuous
short-term dynamics, but separate RSS targets, post-release aggregation,
current-diagonal diagnostics, common-v2 current metrics, and semantic input
conditions. Do not replace TESSE with unexecuted pilots.

## 12. Was the decision supported before method pilot results were viewed?

Yes. The pre-result scorecard was frozen at SHA-256
`f4de8e629dacc3a8e13a1123636179fe3c20ec0a145bbcdf1aff87c64a607694`.
Its totals are TESSE official 16, TESSE current diagonal 18, 3RScan 19, and Flat
16. The decision uses their complementary coverage and later reproducibility
gates; it does not reinterpret totals after seeing method results.

## 13. How should the next paper's tables be arranged?

Main Table 1 should retain Replica/ScanNet200 static open-vocabulary
calibration. Main Table 2 should be a split TESSE end-to-end table with matched
inputs and separate current Object/Dynamic/Change and common-v2
mIoU/Ghost/background/recovery groups. Main Table 3 should report the
A6/c553/P5/P6-C Apartment Pareto ablation. RSS targets, exact attribution,
oracle fairness, censored events, and 3RScan/Flat readiness belong in the
supplement until their missing conditions are executed.

## 14. Which work still requires Office or held-out evaluation?

Office is `NOT_RUN_HELD_OUT`. Before a paper claim, run the frozen selected
CROVE variant and every locally rerun comparator on Office under the same
post-release/current/common-v2 contracts; run K0/K1/C1 only after their source
identities exist. The frozen 3RScan 10-environment set and both Flat visits also
require real-data execution. No threshold, scene selection, or benchmark role
may be revised from held-out outcomes.

## 15. Which conclusions are diagnostic rather than paper claims?

The following remain diagnostic: RSS target comparisons without exact
reproduction; latest-public current-diagonal results; common-v2 current
metrics; GT-semantics/GT-panoptic oracle rows; the P5 attribution patch;
custom 3RScan exact-ID metrics; all synthetic adapter tests; and the Flat
mechanism hypothesis. Apartment-only variant deltas cannot establish
generalization or SOTA. The paper may claim only the source-bound measured
values and their explicit protocol boundaries.

### Large Artifact Inventory

| Artifact path | SHA-256 | Bytes | Source/config identity |
| --- | --- | ---: | --- |
| `/home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p5/khronos_apartment_candidate/map/final.4dmap` | `2e3609cb3caa84adc8779e1b358463eafe63b05deeb8a25a4a6daef6730c69ec` | 12,434,728,742 | P5 manifest `3cfa3891...`; artifact evaluator closure remains unproven |
| `/tmp/crove-b4-p5-patched/sidecars/object_associations.jsonl` | `1b7939562e97472cc2ca9398d0c49b0ca44ac163642ab3137bc341e1e9368f6d` | 27,863,620 | Khronos `63faadde...`; patch `4d5bd903...`; config `82a3a087...` |
| `/tmp/crove-b4-p5-patched/sidecars/dynamic_associations.jsonl` | `3f1849c24390bfc43b633019378fbfcd81fec572485a236563e4ae137943ecef` | 262,053,340 | Khronos `63faadde...`; patch `4d5bd903...`; config `82a3a087...` |
| `/tmp/crove-b4-p5-attribution-receipt-v2.json` | `cfd3869722003c8e2757d8f4add88770d878c15dbd5ea102038a664b30da1457` | 1,957 | Audit of the preceding map, patch, CSVs, and sidecars |
| `/tmp/crove-b5-p5-current-v3.json` | `13e9b569bd8afb523982817a708dfe7199bb88e15eaf0204b40fb68b878951af` | 28,716 | `TESSE_CURRENT_DIAGONAL`; all P5 input/sidecar hashes embedded |
| `/home/ww/vv/dataset/3RScan/3RScan.json` | `674a00f50f76b198b9de44efd86c390fea3da37ba8f12cf8ccd00045e265fa64` | 3,155,995 | Official 3RScan metadata; selection manifest `f72febc9...` |

The map is a frozen B4 input, not regenerated by this audit. The sidecars were
generated with:

```bash
/tmp/crove-khronos-attribution-build2/khronos_eval/exp_pipeline \
  /tmp/crove-khronos-attribution-clean/khronos_eval/config/pipeline/apartment.yaml \
  /tmp/crove-b4-p5-patched true true false
```

The receipt was generated with:

```bash
python scripts/evaluation/audit_khronos_exact_attribution.py \
  --unpatched-results /tmp/crove-b4-p5-unpatched/results \
  --patched-results /tmp/crove-b4-p5-patched/results \
  --object-sidecar /tmp/crove-b4-p5-patched/sidecars/object_associations.jsonl \
  --dynamic-sidecar /tmp/crove-b4-p5-patched/sidecars/dynamic_associations.jsonl \
  --input-map /home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p5/khronos_apartment_candidate/map/final.4dmap \
  --input-map-sha256 2e3609cb3caa84adc8779e1b358463eafe63b05deeb8a25a4a6daef6730c69ec \
  --input-map-byte-count 12434728742 \
  --source-checkout /tmp/crove-khronos-attribution-clean \
  --source-commit 63faadde6ed92220e78fb2f6ca86dcc54bb5cf9e \
  --patch external_patches/khronos_eval_exact_attribution.patch \
  --patch-sha256 4d5bd9033b92758e0062e873f9ed19b45a4c7516cb8945930d30c85a660b38c7 \
  --output /tmp/crove-b4-p5-attribution-receipt-v2.json
```

The exact final commit cannot be embedded in the commit that creates it. The
release contract is `UPLOAD_STATUS = VERIFIED` only when local `HEAD` equals
the remote branch SHA after the final push; the terminal result is reported by
the B9 release command and final response.

### Verification Evidence

- historical evaluation regressions: `168 passed in 13.75s`;
- B0-B9 audit/adapter/report tests: `55 passed in 0.99s`;
- T1 protected-source verifier: pass;
- Ruff on the new evaluation surface: pass;
- `python -m compileall -q src scripts tests`: pass;
- full repository: `5803 passed, 10 skipped, 5 warnings in 1262.73s`.

The skips are explicit missing-external-asset guards. The five warnings are
pre-existing NumPy 2.5 deprecations in dense-semantics race tests; no test
failed.
