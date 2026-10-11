# Diagnostics that test information and routing, not a new score-search grid

## D0. Previous R correction/harm audit

Use the latest PR selected FC8/R1/R3 scores, embeddings and object records, bound to the same original keys and condition order. No new training, no alternate checkpoint choice. For TRAIN, DEV and the already exposed old-H, record full-vocabulary baseline margin, winner/runner-up scores, L2 representation drift, teacher/student correctness, actual-positive-base/zero-positive-base/heldout strata and family. Keep family names out of neural inputs.

Produce distributions and fixed margin bins [(-inf,0),[0,.005),[.005,.01),[.01,.02),[.02,.05),[.05,inf)] using GT margin only for descriptive diagnostics. Also report the GT-free top1-top2 baseline gap separately. They are not the same variable. Do not train a selector on PR's correction list, add a protected-object list, or use old-H to choose this task's checkpoints.

D0 is existing-result analysis; inability to retrieve a genuinely absent detailed old artifact is disclosed. It does not change FG settings. Restore the exact published counterpart when available before marking a field unavailable.

## D1. Do current local features have a target-filtering opportunity?

Required on all622 TRAIN base and69 DEV base originals, four original conditions, same8-view prefix. No old-H or H2 reads here. Predictor data and GT target sidecars remain separately loaded.

For each original/condition, form observed physical-site groups. Within each group use equal valid-view weights on raw features, then **unit(phi(mean raw))**. No learned weights. Compute two diagnostic object vectors:

```
ALL:  unit(.75*z0 + .25*mean(u_j for all observed sites))
TARGET_ONLY: unit(.75*z0 + .25*mean(u_j for known-target sites))
```

A site is known-target if at least one visible membership=1 and none=0. If a site has both1 and0, preserve records and flag a target-sidecar inconsistency; do not majority-vote it away. Unknown=-1 does not become target or negative. No known-target sites -> exact z0 fallback with reason; report fallback-denominator and shared-nonfallback comparisons separately. ALL includes unknown sites and they are separately counted.

Total local weight is .25 in both conditions whenever supported. Values may include useful context even when not target; this is not a theorem that TARGET_ONLY should win and is not a strict global upper bound. Neither branch is a real model. Label every output `GT_DIAGNOSTIC_ONLY`. These values cannot be exported into a map prediction or nominate a checkpoint.

Report by family/category/clean/corruption: original and variant counts, target/non-target/unknown local observations, sites, full200 Top-1/macro/NLL, net corrections vs FC8, and TARGET_ONLY minus ALL. This diagnostic informs interpretation only; run the four planned training arms regardless.

## D2. Post-lock routing interventions on selected FG4

Run after all activated seed nominations, on TRAIN base, DEV base and old-H all; repeat on H2/real proposals only if those stages activate. Use the same FG4 forward's base, group values, learned quality, group keys, energies and original predicted rho. No new weights or data are selected. FG4 may be nonqualifying; its diagnostic still runs.

For each query k, original alpha has mass `r_k=sum(alpha_kj)`. Compute:

1. **MATCHED_MASS**: `aMM_kj = r_k * softmax(e_k)_j`. Null mass remains1-r_k. This removes which-group membership selection while preserving query-wise total probability mass.
2. **MATCHED_MASS_AND_DELTA_NORM**: compute original delta_R and delta_MM before final object normalization. If both nonzero, `delta_NM = delta_MM * norm(delta_R)/norm(delta_MM)`, then unit(z0+delta_NM). If delta_R=0, return z0. If delta_R>1e-12 but delta_MM<=1e-12, mark that paired cell undefined (keep original FG4 result); do not pretend a zero correction is a norm match.
3. **ALL_RHO_ONE**: set all effective rho to1, same e and u, recompute original absolute_route. This can change both total mass and distribution; it is not the matched-mass control.
4. **ALL_RHO_ZERO**: zero effective rho; exact z0 identity.

MATCHED_MASS does **not** preserve the correction vector norm because values can cancel differently. The norm-matched intervention isolates direction more closely but is a post-hoc intervention, not a deployable new trained method. It does not enter checkpoints, selection, full-map method counts or runtime comparisons. Mask undefined pairs symmetrically for paired metrics and disclose the reduced denominator. Do not reduce the denominator for ordinary main model evaluation.

Report full-vocabulary outcomes, exact alpha/null mass and delta-norm checks, and FG4 minus each intervention. A worsened intervention is evidence that the forward used its routing, not sufficient proof that FG4 beats FC or improves maps.

## D3. Does membership actually suppress pollution?

For each query/local token, effective aggregation coefficient is `alpha_kj * a_j(token)`. Define known non-target probability mass by summing these coefficients over membership0; target1 and unknown-1 mass are separate. Query-average and then object-average, never treat repeated tokens as independent samples.

For diagnostics using GT, also report `eta*alpha_j*||u_j-z0||` as a group-contribution norm proxy, explicitly not a causal attribution (group values include nonlinearity and vector cancellation). For VIEW grouping, groups can contain target and context: use their actual member weights rather than assigning the entire group one purity label.

Purity diagnostics requiring site-membership sidecars apply to TRAIN/DEV/oldH/H2, where those exact sidecars exist. Real-proposal diagnostics require only locked-score classification and the mass/norm interventions; do not invent per-site GT purity from class matching. If no exact independent post-lock site-target mapping is available, mark real-proposal purity metrics NOT_COMPUTED.

Record membership classification/Brier/AUROC only if both classes are present; undefined denominators remain null. Attention mass and membership accuracy need not agree. Test a finite intervention reducing one group's rho by0.5 and verify its own alpha decreases while others/null can increase. Do not claim the total correction norm or AP must decrease monotonically.

## How to read the four scientific contrasts

| contrast | supported interpretation only if accuracy also helps |
|---|---|
| FG2-FG1 | grouping identical tokens by physical site has added value |
| FG3-FG2 | additional TRAIN membership supervision helps even without direct route |
| FG4-FG3 | directly using membership helps beyond shared supervision |
| FG4-MATCHED_MASS | group-specific selection matters beyond a scalar reduction of mass |
| FG4-MATCHED_MASS_AND_DELTA_NORM | direction matters beyond simply shrinking the update norm |

Also retain ordinary FC8 and current G1. A method improving on a damaged control is not automatically a successful new representation. Do not manufacture a best-per-object oracle from these interventions and call it a real method.
