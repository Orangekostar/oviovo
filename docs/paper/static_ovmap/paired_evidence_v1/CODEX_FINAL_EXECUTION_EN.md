# Final Codex execution instruction: paired residual and lineage-aware semantic evidence

**Issued:** 2026-09-29. **Type:** develop, execute, select, analyze and publish a bounded research study. This document does not report new benchmark results.

## 0. Authority, scope and end state

Execute this instruction in `Orangekostar/oviovo`, starting from the verified wave-1 commit
`9c35333088551a9d4a67937936d4d9c347b2b207`, on the new branch:

`research/ovimap-paired-evidence-v1`

The primary research baseline B is `AW_E03_FC_FROZEN_A7`, not old M2_CAL. Preserve the original geometry, owner registry, RGB-D observations, projection, frozen model outputs and published experiments. Implement and measure two families:

1. **R:** extract the incremental information of OVR relative to the matched frozen FC model, rather than replacing FC wholesale.
2. **D:** test whether class-pair evidence and physically grounded observation-dependence proxies improve on ordinary probability/logit fusion.

Run the prespecified simple controls as seriously as the proposed methods. A simpler winner, a measured negative result, or a demonstrated mathematical degeneracy is an acceptable scientific outcome. An executable smoke or a table of null metrics is not completion of an executable experiment.

**Do not execute the third, conditional-information acquisition idea in this study.** It requires its own causal candidate/feedback interface and validation of the uncertainty model. Report it as `QUEUED_NOT_EXECUTED`, not implemented, failed, or validated. Do not add E06, prompt engineering, new models, segmentation, mapping, geometry edits, test-time optimization, neural training, new data collection, or a model-download workaround.

The current study is **offline, frozen-evidence semantic refinement** on exposed research scenes. It is not an online mapping demonstration or fresh generalization confirmation. Deployment remains `N0_UNCHANGED`; the research nominee is recorded separately.

Deliver the implemented runner, small numerical modules, all prescribed feasible predictions and evaluations, a CAL-frozen selection, object-level mechanism analysis, related small results, handoff, and a verified normal GitHub push. Preserve previous worktrees and unrelated uncommitted files.

`PROTOCOL_SPEC.json` is the machine-readable registry and budget. This document fixes algorithm semantics; `MATH_AND_EDGE_CASES.md` makes the equations explicit. Resolve accidental discrepancies by these semantics and the fixed registry before inference/evaluation, recording the correction. Do not silently redefine an algorithm to get a favorable result.

## 1. Verified starting evidence and what it does NOT establish

At the pinned revision, Replica official-current-class / released-dataset-pool results are:

| Original method | APall % | AP50 % | mIoU % |
|---|---:|---:|---:|
| N0 | 8.685272 | 21.476863 | 27.261462 |
| RV_A7_COS_REFIT | 9.217907 | 21.236029 | 27.720221 |
| AW_E03_FC_FROZEN_A7 (B) | 10.491020 | 24.323075 | 30.013672 |
| AW_E03_FC_FROZEN_DIRECT | 10.498388 | 20.287022 | 24.091640 |
| AW_E03_OVR_A7 | 10.232525 | 22.023194 | 28.623663 |
| AW_E03_OVR_DIRECT | 10.939594 | 22.006838 | 26.180430 |

These numbers locate immutable controls, not targets to hard-code into result generation. OVR directly recognized more of the common 157 identifiable objects than FC (94 versus 82), but OVR fusion was worse. FC fusion corrected 10 and harmed 2 relative to old A7 in the 161 uniquely geometry-identifiable owners; those counts are not AP summands. This supports studying incorporation of incremental evidence. It does not establish that temperature or correlated errors are the cause.

Current fusion (`m2_reviewer_study/fusion.py::fuse`) averages softmax probabilities of available sources. The FC/OVR worker uses the same original static request manifest and saves per-request `feature` vectors, original/resized/dense mask support, text prototypes and used request IDs. Only changed-slot temperatures were fitted on two CAL scenes. The old wave-1 runner and selection have their own strict method/branch guards; **do not remove these guards to add the new study**.

Full source bindings and precise upstream citations are in `SOURCES.md`. Review these files before modifying anything:

| Existing file/function | Use in this study |
|---|---|
| `a7_evidence_upgrade/binding.py::bind` | Understand parent/reviewer path relationships; implement a new binder, not a call that requires the old branch. |
| `a7_evidence_upgrade/region_worker.py::run` | Recover FC/OVR source bundles, per-view arrays and text prototypes. Never rerun it merely to change the readout. |
| `a7_evidence_upgrade/region_adapter.py` | Verify the paired model, mask, normalization, 14-template and weight boundaries. |
| `a7_evidence_upgrade/calibration.py::base_temperatures`, `fit_source` | Reuse frozen fold/final temperature values. Do not refit them. |
| `a7_evidence_upgrade/e01.py::generate_sources` | Read how actual final Q retention and raw six-crop means were recovered. Reuse existing arrays when present. |
| `composition_study/object_evidence.py::native_request_ids` | Resolve native observations by unique paid frame/bbox/area/pose, not owner-number equality. |
| `module_validation/scannet_study.py::native_readout` | Native saved order, minimum two observations, last eight of retained top ten, FP32 arithmetic. |
| `m2_reviewer_study/scores.py::read_scene` | Load frozen full scores and the N0 cosine source without inventing logits. |
| `m2_reviewer_study/fusion.py::fuse` | Reproduce B and OVR-base exactly; retain as immutable comparison. |
| `module_validation/scannet_study.py::{relabel_prediction,save_prediction}` | Export a locked whole-map semantic relabel with original geometry/ranks. |
| `m2_reviewer_study/evaluation.py::{SceneEvaluator,pool}` | Released per-scene matching, both ranking views, and real dataset pooling. |
| `a7_evidence_upgrade/diagnostics.py`, `m2_reviewer_study/attribution.py` | Reuse the definitions of geometry-only corrections and actual released matching, separately. |

## 2. Fixed data, working paths and bounded recovery

CAL scenes: `scene0056_00`, `scene0534_00`.
Replica scenes, in this order: `office0`, `office1`, `office2`, `office3`, `office4`, `room0`, `room1`, `room2`.

All ten scenes are already exposed development/regression data. Do not invent a new held-out split by subdividing frames, objects, views or class pairs. A class pair is not an independent training sample.

Default parent:
`/mnt/shared/ww/ovimap-a7-evidence-upgrade-wave1/attempt_001/binding.json`

Default new output:
`/mnt/shared/ww/ovimap-paired-evidence-v1/attempt_001`

Coordinator default, subject to the real parent runtime receipt:
`/home/ww/miniconda3/envs/ovimap-map/bin/python`.

Resolve arrays and text from the parent binding, its reviewer binding, immutable receipts, source manifests and exactly referenced local paths. Do not assume GitHub's compact archive contains the full NPZ arrays. Do not scan the whole machine, inspect credentials, or reinstall environments. If roots moved, accept a single explicit `--path-map` old-root/new-root JSON and verify the referenced input content before reuse.

Published wave-1 files are under
`artifacts/static_ovmap/a7_evidence_upgrade_wave1/attempt_001/`.
Sources may be gzip-compressed in the published copy and plain JSON in shared storage. Implement explicit `read_json_or_gzip`; do not pass gzip bytes to the old plain-JSON reader. Prefer the complete shared run. The compact sources/locked/calibration files can support score-only controls when local per-view arrays are unavailable.

**Resource failure must be local to its dependency.** Missing lineage arrays do not block R0/R1/D1/D1_ANCHORED/D2 or a residual whose common-view score pairing is already proven. Do not fabricate arrays from scores. Do not invoke a vision model to replace missing caches under this instruction. State a real file or permission failure in the handoff.

## 3. Evidence contract and baseline reconstruction

Use internal source symbols `N`, `Q`, `F`, `O`:

- N = N0 **cosine** evidence reconstructed by `read_scene(...).cosine_native`.
- Q = original, unchanged `Q_GAIN` evidence.
- F = `AW_E03_FC_FROZEN` region source in the former S2 slot.
- O = `AW_E03_OVR` region source, not an additional segmentation map.

Load the full label vocabulary in its original `valid_ids` order. It is not safe to equate a label ID with a column index. Scores must be genuine complete vectors, and their argmax must reproduce each available source label. A source with `available=false` contributes no probability, no vote, and no fabricated one-hot evidence.

Preserve the six original control method IDs in the specification. For every scene reconstruct:

- B = arithmetic mean of available `softmax(s_N/T_N)`, `softmax(s_Q/T_Q)`, `softmax(s_F/T_F)`.
- OVR-base = the corresponding N,Q,O fusion.
- FC_DIRECT and OVR_DIRECT = original genuine source label, otherwise original N0 fallback.

Use the exact wave-1 **fold temperatures on each CAL scene**, and final temperatures on Replica. `T_N` is the A7 cosine-refit N0 value, NOT old canonical-relative N0 temperature. `T_Q` is the original Q temperature. `T_F` and `T_O` are the changed-slot wave-1 files. Reuse all existing fitted values; do not refit at this stage.

Check original whole-map labels, source availability, used requests, owner registry, prediction keys and control metrics through the same established code paths. Cache each input hash once per invocation. Do not rehash all old published files at every phase.

Save a `FrozenEvidence` bundle per scene/owner containing source scores, class order, available flags, baseline full probabilities, original labels, source/fold/model identities, final used request IDs, areas, and references to relevant per-view arrays. No annotations belong in this bundle. Store diagnostic annotations separately and only open them after predictions or label-free evidence descriptors are locked.

**No original probability defined:** if all N/Q/F are genuinely absent, retain the exact original fallback label, mark `NO_BASE_DISTRIBUTION`, and do not invent a uniform B distribution. Unavailable rows remain in complete-map metrics. R1 may use a real O source even in this case; other base-dependent methods retain B's label.

### 3.1 FC/OVR pairing

For each owner, join FC and OVR by the original static request identity, source image identity, original global recognition mask, bbox, area, target lineage and common operator/prompt configuration. Use the intersection of genuinely successful requests and the same original `visible_target_pixels` weights, in the original order. Never include a view that was evicted or unavailable in the underlying source.

If both sources use the exact same successful request list and weights, their frozen aggregated scores are valid paired scores; validate that fact once. If lists differ, reconstruct both aggregates from the common-view `feature` arrays and each model's own frozen text matrix. If the common set is empty or its necessary arrays are unavailable, set `NO_PAIRED_RESIDUAL` and return B for that owner in R2/R3 variants. Do not call a fallback N0 label a paired FC observation.

Keep B based on **all of its original F evidence**, even when residual scores use only a common subset. Record the subset restriction. Text matrices must have the same class names/templates/order, but do not assume their numeric vectors are identical across checkpoints; save the measured equality/difference. A residual may reflect all differences in the verified paired region pipeline, not automatically visual weights alone.

## 4. R family: ordinary controls and paired residuals

All computations use float64 stable log-sum-exp/softmax. Let `s_Fcap,s_Ocap` be the paired cosine scores. The MAIN residual uses a common scale:

`r_shared = (s_Ocap - s_Fcap) / T_F`.

This is the difference of common-temperature log probabilities up to a class-independent constant. It prevents different temperature values alone from producing a residual. **Do not normalize or subtract latent vectors across models.**

The calibration-confounded diagnostic is
`r_cal = log_softmax(s_Ocap/T_O) - log_softmax(s_Fcap/T_F)`.

The latter deliberately includes both representation and old calibration effects; do not describe it as a pure weight-tuning increment.

### R0 — `PE_R0_BLEND`: ordinary final-prediction interpolation

`p = (1-lambda) * p_B + lambda * p_OVR_base`, lambda in `{0,.25,.5,.75,1}`. If OVR-base has no genuine distribution, keep B; do not create another fallback vote. At lambda=0 return B exactly. This is a strong cheap control, not a proposed novelty.

### R1 — `PE_R1_FOURWAY`: ordinary fourth-source ensemble

Average the individually calibrated probabilities of genuinely available N,Q,F,O, once each. Use old source temperatures. It is not `(3*B+O)/4` when fewer than three base sources are available. Do not count O twice or require four sources when fewer are present. Preserve N0 fallback if all four are absent.

### R2 — `PE_R2_GLOBAL`: whole-vocabulary ratio update

`p = softmax(log(p_B) + eta * r_shared)`, eta in `{0,.25,.5,1,2}`. A zero eta or absent pair returns B exactly. No clipping of residuals or confidence threshold is allowed in this version.

### R3 — `PE_R3_LOCAL`: local, mass-preserving ratio update

Before applying the residual, fix `C_i = Top5(p_B) union {argmax(s_Ocap)}` (use all classes if fewer than five). Resolve equal scores by the original `valid_ids` order. C depends only on frozen predictions, never GT, a new method's output or the eventual selected eta.

Let `q = sum_{c in C} p_B[c]`. Keep `p_new[c]=p_B[c]` outside C, and inside C assign

`p_new[C] = q * softmax(log(p_B[C]) + eta*r_shared[C])`.

Eta has the same five choices as R2. Do not renormalize the whole vector after this assignment: correct rounding, if necessary, inside C only. The intended invariants are original outside probabilities and original inside total mass, not guaranteed label preservation. Eta=0, a constant residual within C, or an empty pair must return the original base exactly, including its tie decision.

### R3CAL — `PE_R3_LOCAL_CALDELTA`

Same C, paired views, operator and eta grid as R3, replacing only `r_shared` with `r_cal`. This is necessary to identify whether apparent residual benefits are mainly due to unequal source calibration. Do not tune temperatures in addition to eta.

R0/R2/R3/R3CAL parameter choices are global per method, not per scene, owner or category. Preserve every CAL grid result, even when eta=0 wins. Report both the selected configuration and the best active configuration as diagnostics; an active configuration is not silently promoted over the selected one.

## 5. D family: dependency hypotheses with explicit, computable assumptions

Use only the baseline's available N,Q,F sources. O is not used here. Preserve each source's original calibrated score `z_m=s_m/T_m`. Full class-pair differences are `d_m[a,b]=z_m[a]-z_m[b]`, for all a<b in the frozen vocabulary. No GT shortlist.

### D1 — `PE_D1_LOGPOOL`

`p = softmax(mean_m(z_m))`. This is ordinary logit/logarithmic opinion pooling; no novelty claim.

### D1A — `PE_D1_ANCHORED`

Recover class potentials with the same graph regularizer used by D3/D4, but use uniform pair weights and equally averaged differences. With all class pairs and ridge lambda=K (K=number of classes), this is exactly

`u = .5*center(mean(z_m)) + .5*center(log(p_B)); p=softmax(u)`.

Use both the closed form and graph solution in the numerical check. Without this control a D3/D4 improvement could simply be the benefit of anchoring to B rather than evidence-dependence modeling.

### D2 — `PE_D2_GROUPED`

Average available N/Q probabilities within their group, then average that group probability with F if F is available. If a group is absent, the other group receives all mass. This tests whether a simple prior grouping of the two original SigLIP streams is sufficient. It is not learned reliability and must not be presented as source independence.

### 5.1 Recover per-view evidence for D3/D4

For N, use `saved[owner]['feat'][-8:]` and the corresponding saved areas/order only when native has at least two retained observations. Resolve these views with `native_request_ids`; do not use all ten retained views. For Q use only `decisions['retained_features'][owner]` and the original raw six-crop means; wave-1 E01 `features.npz` already exposes `retained_<owner>` and `areas_<owner>`. For F use each successful saved normalized region vector and the original static view area weights. Preserve each pipeline's feature norm convention.

The frozen scores stay untouched. For derivative calculations only, compute a float64 surrogate `a_m=sum_e alpha_me*f_me`, `v_m=a_m/||a_m||`, with areas normalized to sum one. Native original FP32 arithmetic must reproduce the original baseline through its original function; a separately recorded float64 surrogate error up to `1e-6` is not a reason to change the native prediction. Larger reconstruction error requires resolving the mapping, not relaxing the tolerance or inventing features.

If any available source lacks its required per-view data, D3/D4 return B for that owner and explicitly record `MISSING_DEPENDENCE_INPUT`; do not silently remove that source. Scores-only D1/D1A/D2 still run. Do not rebuild lost visual evidence.

### 5.2 Define a positive-semidefinite observation kernel, not a claimed true covariance

Define an observation atom at the actual view-representation level. Keep two identities:

1. **Exact computation identity:** model weight content, actual image/processor/input geometry and precision, mask, bbox, six-crop policy or region policy. An identical request name alone is insufficient. Identical frozen sources with identical evidence and calibrated scores are collapsed once before D computations; a duplicated file is not an additional measurement.
2. **Dependence-proxy support:** original image content ID, actual vision-checkpoint/embedding-space ID and full-resolution recognition support. N/Q use their real union mask; F uses its actual global target mask. Never equate these different masks.

For atoms e,f, set

`Kobs[e,f] = 0` if original RGB identities or vision-checkpoint coordinate spaces differ;
otherwise `Kobs[e,f] = |M_e intersection M_f| / sqrt(|M_e|*|M_f|)`.

This normalized binary-support Gram kernel is PSD within each image/model block. It is **not mask IoU** and does not claim geometric overlap equals error correlation. F and native embeddings remain separate blocks; do not compute cross-model latent dot products. Same-scene but different-frame observations have zero modeled covariance in this first study; unmodeled temporal/model correlations must be disclosed.

Diagonal entries are 1 for each valid atom. If masks needed to establish a non-diagonal relation are genuinely missing, retain independent atom features, set that relationship unmodeled through its own isolated block, and report `UNKNOWN_SUPPORT`, not observed independence. Do not impute mask contents. Use actual bound arrays; no new raycast/map capture is authorized. Kernel construction and its coverage are prediction-side and label-free.

Within a source, a repeated reference to the exact same request is idempotently deduplicated only when content and weight agree; conflicting duplicate weights are an input error. Distinct observed frames remain distinct even if scores happen to coincide.

### 5.3 Class-pair sensitivity covariance proxy

For class a,b in source m with dimension d_m and unit text prototypes t, define

`J_me = alpha_me/(T_m*||a_m||) * (I-v_m*v_m^T)*(t_ma-t_mb)`.

Use a fixed relative isotropic feature-perturbation scale:

`L_me = ||f_me|| / sqrt(d_m) * J_me`.

Build

`Sigma0[m,n] = sum_{e in m,f in n} Kobs[e,f] * dot(L_me,L_nf)`

only within compatible embedding spaces. Interpret this as a **sensitivity/overlap covariance proxy** under the stated perturbation model, not an empirical semantic-error covariance. The norm factor avoids silently assigning the same total perturbation energy to 768- and 1024-dimensional features or to unit vectors versus six-crop means.

Let `vbar=max(mean(diag(Sigma0)),1e-12)` and
`Sigma = Sigma0 + 0.1*vbar*I`.

The fixed diagonal floor encodes unmodeled error and gives positive definiteness; it is not learned from CAL labels. Do not claim to have estimated category-specific bias from 22 objects. The proxy is PSD by its Gram construction; arbitrary independent pairwise correlation entries would not give that guarantee.

### D3 — `PE_D3_DIAGONAL`

Use `diag(diag(Sigma))`. This is the matched class-pair reliability model without cross-source dependence.

### D4 — `PE_D4_LINEAGE`

Use Sigma with modeled cross-source terms. If the actual kernel yields no shared support, equality with D3 is a measured structural degeneration, not a failure to run and not proof that dependence correction helped.

For both, solve the small nonnegative minimum-variance problem

`alpha*=argmin alpha^T Sigma alpha; alpha>=0, sum(alpha)=1`.

At most three genuine source classes exist. Enumerate nonempty active subsets (at most seven); solve the equality-constrained system on each subset and keep the feasible minimum. This is faster and more deterministic than launching a general-purpose optimizer for each pair. Use no raw inverse when a linear solve suffices. Roundoff negatives within 1e-10 may be clamped and renormalized; real negative solutions are infeasible. Use the numerical reference implementation/tests as a specification check, not as proof of improved accuracy.

Set `dhat_ab=sum_m alpha_m*d_m_ab`; let `V_ab=alpha^T Sigma alpha`. Form inverse variance weights with a relative floor:

`w_ab = 1/max(V_ab, 1e-6*median(V_all_pairs), 1e-12)`, then normalize all pair weights to mean 1.

Recover class potentials by solving

`min_u sum_{a<b} w_ab*((u_a-u_b)-dhat_ab)^2 + K*||u-u0||^2`,
`u0=center(log(p_B))`.

The linear system is `(L_graph+K*I)u = b_graph+K*u0`. Center the solution to remove numerical drift and output `softmax(u)`. The positive ridge gives a unique solution; neither that fact nor minimum variance implies improved AP. D3 and D4 use the same ridge, weight floor, vocabulary and solver.

Use all class pairs, not pairs selected using GT. Compute in chunks (default 256 pairs) and reuse class Gram products when useful. Do not allocate an array of size owners × K² × views × feature dimension. Limit CPU BLAS threads to 4 by default.

### D4S — `PE_D4_SHUFFLED`

This is a mechanism-falsification control, not another tuned candidate. In the actual N/Q/F model only N/Q can have nonzero cross-source covariance. For each owner, derive `rho_ab=Sigma_NQ/sqrt(Sigma_NN*Sigma_QQ)`. Permute this rho vector across the ordered class-pair list using a seed derived from SHA256(`1729|scene|owner`). Reinsert permuted rho with each pair's **original diagonal variances**; F cross terms stay zero. This preserves positive definiteness and variance marginals but destroys the class-pair alignment of N/Q dependence. If N or Q is absent, or rho is constant, record an identity control.

Use the same QP, inverse-variance weights and graph solve as D4. Do not claim this shuffles the physical images; it is specifically a permutation of the class-pair dependence signal. D4 should beat D3 and this control before attributing a gain to meaningful dependence modeling.

## 6. Mathematical properties and finite tests

Read `MATH_AND_EDGE_CASES.md` and run `math_reference.py` once. It checks synthetic operator properties only; it does not consume user scene data or evaluate a model. Required production checks are compact:

- eta/lambda zero returns B exactly; local mass/outside probabilities and constant-residual identity;
- common-temperature residual is zero for identical paired scores; the CALDELTA diagnostic need not be;
- class-order permutation equivariance and genuine-source fallback;
- finite-difference check of J on the surrogate feature aggregation;
- observation kernel PSD, diagonal/cross covariance handling and simplex optimum against a tiny exhaustive grid;
- anchored equal-pair solution equals the closed form; no-overlap D4 equals D3;
- an exact duplicate source alias does not acquire extra weight in D methods;
- immutable geometry and same-source known-control metric parity;
- one end-to-end real scene through prediction + original evaluator, then required complete pools.

Do not add a target number of tests, whole-CROVE regressions, red-team testing, repeated archive audits or stress fuzzing. Fix an actual failing boundary and rerun the affected checks. Do not run a GPU smoke: no new neural interface is introduced.

## 7. CAL-only selection and frozen Replica regression

Selection rules are prospective for THIS study. Do not alter wave-1's nomination of N0. Family-level parameter selection and final research nomination use official-current-class **released CAL pooling**, never arithmetic scene averages. All source temperatures stay their original out-of-fold values for CAL and final values for Replica.

For each of R0/R2/R3/R3CAL, execute all five parameter values on both CAL scenes. For other new conditions execute the single fixed configuration. Store all predictions, including unchanged ones, and reuse original evaluator receipts when the complete evaluation identity is truly identical. A reused receipt needs an alias record; do not claim an independent rerun.

Use this deterministic **practical-tie** ranking, with metrics in percentage points:

1. Retain rows within **0.05 pp APall** of the maximum.
2. From those, retain rows within **0.10 pp mIoU** of their maximum.
3. From those, retain rows within **0.10 pp AP50** of their maximum.
4. Prefer lower prespecified compute tier, then smaller distance from identity (smaller lambda/eta), then registry order.

These bands express a research preference, not a confidence interval or statistical equivalence claim. They avoid treating a 0.001 pp AP difference as proof of practical superiority. Report strict metric differences as well. Compute tiers are fixed in the JSON: old/base < arithmetic/logit < graph < paired extra-OVR methods; these tiers are tie-break preferences, **not measured FLOPs**. Always publish the real heterogeneous operation ledger separately.

Choose ONE parameter per method across both CAL scenes; no per-scene or per-class parameters. Then lock all 11 method configurations before any Replica method evaluation or diagnostics. The default single-configuration methods are never silently retuned. CAL results are exposed selection evidence, not an unbiased performance estimate.

Research nomination candidates: B, N0, the measured whole-map new methods except D4_SHUFFLED, FC_DIRECT, OVR_DIRECT and OVR-base. DIRECT controls are allowed to win this new research nomination; do not exclude an effective simple method merely because the old study excluded it. Old A7 remains a reported control, not a preferred new deployment.

### One prespecified optional combination

After the R/D CAL sweep, choose the best **active** nonzero eta of R3_LOCAL using the same ranking. Use D4_LINEAGE unchanged. If both are executable and each changes at least one CAL label versus B, execute exactly one combination `PE_COMBO_D4_R3` on CAL:

- use D4 probabilities as the starting distribution;
- apply the R3_LOCAL common-temperature residual with the selected active eta;
- retain the original R3 candidate C built from B, not a newly selected D4 shortlist;
- conserve D4's probability mass inside that fixed C, not B's old mass.

Run this combination even when one component has negative CAL score change; single-module dominance is not a prerequisite for testing interaction. If R3 is active only at an eta that lost the family selection, label this explicitly as a prespecified interaction probe. No alternate pair, order or parameter search is allowed. Include the measured combination in final CAL research nomination, then lock it for Replica. A missing/identity component yields `NOT_TRIGGERED_NO_ACTIVE_PAIR`, not invented combination metrics.

Evaluate **all 11 frozen new methods and all six controls on all eight Replica scenes**, not just the nominee. Only external input failures may produce a partial method; report denominators and fallback mechanisms. Do not retune, re-nominate or switch fusion type after seeing Replica. Report its descriptive best method and Pareto frontier separately from the CAL-frozen nominee. Deployment remains unchanged.

## 8. Implementation plan and actual execution graph

Create task-local code, proposed names below. These are NEW interfaces to implement, not asserted existing files:

```
src/static_ovmap/paired_evidence_study/
  binding.py          # resolve old bytes, new branch/spec, compact availability
  evidence.py         # N/Q/F/O distributions, common-view scores, per-view features
  residuals.py        # R0/R1/R2/R3/CALDELTA
  dependence.py      # support kernel, J, covariance, active-set weights
  pair_graph.py      # D1/D1A/D2/D3/D4/shuffled and graph solver
  selection.py       # fixed scalar grids, practical ties, combination, freeze
  diagnostics.py     # prediction-side mechanisms, then evaluation-only outcomes
  evaluation.py      # task wrapper around existing released evaluator
  workflow.py        # genuinely calls all leaves and handles partial dependencies
  reporting.py
  publication.py
scripts/evaluation/run_ovimap_paired_evidence.py
```

Avoid a generic experiment platform. Group small modules if useful, but keep readout and annotation functions separate and maintain the specified behavior. Implement a CLI with `--phase bind|recover|cal|freeze|replica|diagnose|report|publish|all`, `--wave1-binding`, `--output-root`, `--spec`, `--resume`, optional `--path-map` and `--threads`.

Suggested final command, after resolving the real interpreter:

```
/home/ww/miniconda3/envs/ovimap-map/bin/python \
  scripts/evaluation/run_ovimap_paired_evidence.py \
  --phase all --resume --threads 4 \
  --wave1-binding /mnt/shared/ww/ovimap-a7-evidence-upgrade-wave1/attempt_001/binding.json \
  --output-root /mnt/shared/ww/ovimap-paired-evidence-v1/attempt_001
```

Use the protocol location committed under
`docs/paper/static_ovmap/paired_evidence_v1/PROTOCOL_SPEC.json` as default `--spec`.

| Phase | Must actually do | Completion condition |
|---|---|---|
| Bind | Resolve parent, source and evaluator identities; new output/branch | Real readable source table; absent dependencies listed precisely |
| Recover | Reconstruct B/OVR-base, paired scores, masks/kernel inputs, prefix of mathematical diagnostics | Known controls match; per-owner capability flags, no vision calls |
| CAL | Predict and evaluate all frozen grids/controls; measure proposed and simple operators | Real complete CAL pool for every feasible configuration |
| Freeze | Select parameters; run at most one active combination; lock nominee and all transfer settings | Immutable transfer manifest independent of Replica labels/results |
| Replica | Execute every fixed method on the eight scenes | Whole-map labels, both ranking rows and complete official pools |
| Diagnose | Join locked predictions to GT and released matcher; explain gains/losses/dependencies | Mechanism and object ledger, no prediction rewrite |
| Report | Read measured artifacts, not placeholders | Tables and status truthful; no fixed `measured={}` or hard-coded COMPLETE |
| Publish | Commit only this task's related code/config/results and handoff; normal push | Local/remote full SHA match or explicit publication error |

`all` must execute this graph and return normally, not a shell of receipt reads. On a real leaf failure, finish independent feasible leaves and report `PARTIAL`; do not create nominally COMPLETE untested rows. Resume should not rerun vision, fit temperatures again, or rehash every old archive.

## 9. Evaluation, units and measurement limits

Primary metric view: `OFFICIAL_CURRENT_CLASS` plus `RELEASED_DATASET_POOL`. Use the actual released evaluator namespace and runtime overlap vector; current APall/uAP averages the implemented thresholds 0.50–0.90, AP25 is separate. Do not label it COCO AP through 0.95. Copy the actual vector into every protocol receipt.

Auxiliary view: `FROZEN_N0` rankings, and per-scene results. Source geometry/ranks remain frozen in the prediction payload; the official exporter recalculates class-relative ranks for that evaluation view only. Do not change owner masks, projection tolerance, ignore handling, valid classes or minimum size.

`pool()` must send the ordered complete scene set to the released evaluator and sum semantic confusion matrices. Do not average per-scene AP or mIoU and name that result official pooling. Reuse matched trace utilities for actual added/lost GT matches and per-class AP changes.

Save labels before constructing an evaluator (which opens GT). All candidates must retain the complete map and every owner, including objects outside identifiable diagnostic correspondences. Source failure is not scientific rejection. Positive results on partial effective coverage must state that coverage.

For the frozen research nominee versus B, calculate eight leave-one-scene-out **seven-scene released pools** on Replica, reusing existing scene receipts. This is a descriptive sensitivity analysis, not a bootstrap confidence interval or eight independent trials. Do not feed repeated absolute GT paths to the evaluator to fake a bootstrap. No new model inference is needed.

## 10. Required diagnostic questions

Produce both annotation-free evidence descriptors and a later outcome join.

1. **Probability-pool blocking:** on genuine three-source objects, for every identifiable wrong B prediction w and correct source label g, compute `(p_N(w)-p_N(g))+(p_Q(w)-p_Q(g))`. Values >1 make it impossible for any single region probability difference (bounded by 1) to beat that competitor under equal arithmetic pooling. Report actual counts and margins; do not assume this explains OVR failures.
2. **Paired information:** on the common F/O view subset, enumerate F-wrong/O-right, F-right/O-wrong, both-right and both-wrong. For each method report which extra O corrections survived and which introduced harm. Separate pair restriction effects from residual effects.
3. **Local update behavior:** candidate-set coverage, mass q, eta=0, no-pair, no-new-label, full-support versus changed-label counts, and outside-mass drift. No model is credited for an N0 fallback.
4. **Dependence coverage:** exact shared computation atoms; same-frame/mask overlap; modeled N/Q covariance coverage; unknown support; rho distribution; effective source weights; active QP sets; D4=D3 identities; residual cycle energy of pairwise differences and graph-fit residual. These quantify the actual hypothesis, not an a priori causal story.
5. **Matched controls:** R3−R2, R3−R0, R3−R1, R3−CALDELTA; D4−D3, D4−D1_ANCHORED, D4−D2, D4−SHUFFLED. Explain whether any advantage comes from source weighting, the anchor, or meaningful dependence.
6. **Object outcomes:** corrected/harmed relative to B and separately N0, with unique geometry IoU>0.5; distinguish from official matching at AP25/AP50 and higher implemented thresholds. Preserve missed correct source cases and false positives/ignore events. Do not translate correction counts into AP changes by addition.
7. **Adverse examples:** include the first two corrected and first two harmed owners in stable owner-ID order per scene; do not select only attractive examples. Existing RGB can be referenced; no new segmentation is needed.
8. **Calibration:** NLL/Brier on the same fixed common-identifiable subset, with support count. Improved confidence or fitting loss is not proof of improved official map quality.

Fit no classifier, covariance regressor or neural selector in this study. The only new supervised selection is the finite CAL choice of lambda/eta and research nominee. D proxy constants are fixed. State that old Q and scalar temperatures already used supervision.

## 11. Resources, method cost and reporting

New image forwards, image-model loads, text-model forwards, training jobs, geometry reconstructions and new downloads: **zero**. CPU access to cached float arrays, dot products, small QPs, graph solves and the original evaluator are permitted. Reading a model-free text matrix is not text-model inference.

No more than the four declared five-value parameter grids; no tuning on Replica. Do not lower an evidence requirement by forging a source. New compact publication target: <=100 MiB; gzip JSON/CSV, avoid duplicating old prediction NPZs, full masks, old 16,430-file publication archives or old released traces. Large generated arrays stay in shared storage with one content manifest and reconstruction commands.

Method cost is not zero merely because this invocation has zero image inference. D variants logically require the original N,Q,F evidence and their stored observation support; residual variants generally require **both** F and O. Report necessary model/input-identified operation unions separately from physical operations reused this time. Only exact computation identities permit deduplicating cost. Dense-image, region-pool and crop counts stay separate; no fake latency by adding heterogeneous counts.

Measure CPU read/recovery, operator, parameter sweep, evaluation, reporting and publication time separately. Record failed attempt time where available, and explicitly mark missing historical times. Report memory peaks as actual observations, not sums of per-worker maxima. Do not claim online speed, parameter-free learning or zero-cost deployment.

## 12. Mandatory deliverables

Under `docs/paper/static_ovmap/`:

- `PAIRED_EVIDENCE_RESULTS.md`: complete metrics, mechanism contrasts, coverage, failures, costs and limitations.
- `PAIRED_EVIDENCE_HANDOFF.md`: environment/paths, actual commands, old/new code mapping, input/output identities, resume behavior and remaining tasks.
- `PAIRED_EVIDENCE_SELECTION.md`: every CAL grid, practical-tie ranking, selected/no-op and active configurations, combination condition and frozen nominee.
- `PAIRED_EVIDENCE_CLAIMS.md`: for each proposed claim, `SUPPORTED_IN_THIS_STUDY`, `NOT_SUPPORTED`, or `UNTESTED`, with exact evidence. Distinguish algebraic property from empirical accuracy and inherited mathematical tools from a candidate contribution.

Under `artifacts/static_ovmap/paired_evidence_v1/attempt_001/`, publish small source index/capability manifest, protocol and source lock, scalar choices, rows and pooled metrics, compact per-owner labels/probabilities as size permits, matched contrasts, diagnostic summaries, actual execution ledger, tight test result and new-file publication manifest. Keep exact stable IDs so the full sources can be rebuilt from shared caches.

Required summary tables:

A. All frozen methods against B and N0: APall, AP50, AP25, mIoU, mAcc; both official pool and auxiliary fixed-rank pool.
B. Mechanism contrasts and one composition interaction, not just the best row.
C. Input/paired/dependence coverage and mathematical degeneracies.
D. Per-object correction, damage and actual matcher changes.
E. Physical invocation costs versus full-method logical costs.
F. CAL nomination, Replica descriptive frontier, and justified next research step.

A best result need not improve every scene. Conversely an official mean improvement does not justify hiding severe scene losses or claiming statistical significance. If simple R0/D2 wins, recommend the simpler result; do not preserve the proposed mechanism's name by changing it after evaluation.

## 13. GitHub publication and final completion

Before coding, check actual `git status`, worktree and remote; make the isolated task branch from the pinned base (or an explicitly verified descendant containing it with no scientific changes to locked inputs). Do not overwrite unrelated changes or force-reset. No force push.

Before committing, check only this task's staged diff for accidental weights, datasets, secrets and unrelated modifications; run the compact relevant checks once and regenerate reports from the locked results. Commit related code, configs, reference documentation and actual small results. Use a normal command equivalent to:

```
git push -u origin HEAD:refs/heads/research/ovimap-paired-evidence-v1
local_sha=$(git rev-parse HEAD)
remote_sha=$(git ls-remote origin refs/heads/research/ovimap-paired-evidence-v1 | cut -f1)
test "$local_sha" = "$remote_sha"
```

Record an external publication receipt containing exact branch, full commit, push result and comparison. A receipt claiming the hash of its own containing commit would be circular; do not generate a new commit merely to embed its own final SHA. Reference the external receipt in the handoff and return the verified SHA to the user. If authentication/push fails, preserve the local commit and report the actual error; never use `PUSH_VERIFIED` before the comparison succeeds.

Final user response must state engineering completion, measured/blocked methods, selected frozen research nominee, actual Replica benchmark differences, evidence-coverage limitations, new physical inference count (must be zero), publication status, full SHA and links to RESULTS/HANDOFF/SELECTION. Do not end with a promise to execute later.

## 14. What authorizes the next research wave?

No additional online query experiment is automatically authorized. The next proposal may prioritize conditional acquisition only after this study shows whether class-pair reliability/dependence has predictive value. If D4 merely equals D3 because useful overlap is absent, do not develop a conditional-information query model on the premise that dependence removal was validated. If R3 helps but D4 does not, advance the residual mechanism without attaching an unsupported dependence story. If no mechanism beats strong simple controls, report that plainly and retain the current FC baseline.
