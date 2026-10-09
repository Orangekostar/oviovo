# Final Codex execution instruction: disagreement-conditioned re-observation

## 0. Mission and success
Implement, execute, analyze, select, and publish the exact study in this folder. Do not return only a plan, smoke test, fabricated table, or proposal for later work. The experimental aim is to improve the current D2 + G1 complete map without losing Replica performance, by changing **which additional FC observations are acquired** and **how those observations are aggregated**, not by changing the visual backbone or geometry.

The final upgrade gate is fixed: both complete cohorts must preserve all five G1 metrics, CF18 APall must exceed D2 and CF18 AP50 must reach D2. Use original unrounded fractions and the JSON tolerance. No improvement is guaranteed. Publish negative results and keep G1 if no candidate passes. Deployment remains `N0_UNCHANGED`.

Use one implementation/configuration for both datasets. Four-scene screening is exploratory; the 26-scene regression is also on previously exposed data, not untouched confirmation.

## 1. Repository and inputs: read before editing
Base: `9188e16d4ee7d0460de43413dfe73c225a400f41`. Branch to create: `research/ovimap-disagreement-query-v1`.

Use a separate worktree. Record original branch/status first; never reset, clean, force-push, or delete another task's files. The preferred worktree is `/mnt/shared/ww/ovimap-disagreement-query-v1/worktree`. On an existing target branch, inspect and resume its compatible work rather than overwriting it. Resolve GitHub remote from the actual repository, not a hard-coded credential.

Read the pinned files listed in `SOURCE_EVIDENCE.md`, especially:
- `source_preserving_update/{binding,decisions,outputs,evaluation}.py`;
- `minimal_instance_repair/observations.py`;
- `samv_local_probe/{geometry,query_plan,fc_worker}.py`;
- `cvpr_compact/{projected_views,area_fallback,region_worker,evaluation}.py`;
- `backbone_wave1/readouts.py` and the released scorer adapter.

Full baseline/data parent: `/mnt/shared/ww/ovimap-source-preserving-update-v1/attempt_001` (release `a95c24d...`). The later SAM-V parent supplies optional four-scene cached observations, not the full 26-scene baseline. Follow the full parent's `source_binding.json`, `result_store.json`, context records, payload receipts, assets, and path maps. `SOURCE_UPDATE_HANDOFF.md` identifies exact compact score and decision bundles. Read original external artifacts when they are required; a markdown table is not a replacement for an actual payload or score vector.

Create a new binding adapter. Old `bind()` and evaluation runners hard-code branch, method IDs, 234/18 coverage and old producer freezes. Do not invoke them to mutate their original roots; reuse low-level read/FC/scorer utilities with an explicitly remapped new output root. Resolve D2/G1 against actual `SU00_D2`/`SU01_G1` parent records. The input binder may still encounter `IR00_D2`/`IR01_G1` lineage inside that parent: map these identities explicitly rather than renaming bytes.

Before new work, reconstruct D2's full class distributions with `fuse_readout`, compare original labels and stored distributions, and check G1's original D2 support/classes. Check the actual evaluator, vocabulary, projection and baseline confusion/match artifacts. Copy common inputs to a predictor-only namespace: no GT labels, target matching or nearest-to-GT information is available to query policies. Evaluation projection is reserved for output ranking/scoring after predictions lock.

Inherited controller Python is `/home/ww/miniconda3/envs/ovimap-map/bin/python`; use the parent's actual FC Python and fixed FC model/text. Do not upgrade existing environments or download a new model. Use `--storage-root`, `--path-map`, `--gpu` for genuine location/device changes. Check free disk once before material acquisition and monitor task output growth; reuse content-addressed caches, never delete other tasks to solve ENOSPC.

## 2. Implement the new task, not another old-task patch
Add:
```
src/static_ovmap/disagreement_query/
  binding.py             # new immutable input adapter
  physical_support.py    # canonical sites, quadrature, visibility diagnostics
  query_plan.py          # common target/anchor/candidate plan
  acquisition.py         # staged ordinary-FC requests and access-controlled evidence
  fc_worker.py           # inherited exact FC operators, new request schema
  policies.py            # AREA/COVERAGE/VERIFY/DISAGREEMENT
  updates.py             # MEAN/AREA/SUPPORT, exact content deduplication
  diagnostics.py         # vocab/physical/prediction-level attribution
  outputs.py             # fixed-G1 semantic-only payloads
  evaluation.py          # actual output, ordered released pooling
  selection.py           # deterministic pilot/full gates
  runtime.py             # bounded phase costs and conditional cold measurement
  reporting.py           # tables, four reports, compact bundles
  runner.py              # dependency-aware resume, phase dispatch
scripts/evaluation/run_ovimap_disagreement_query.py
configs/static_ovmap/disagreement_query_v1.json
```
The layout may combine small files only where interfaces remain clear; do not create unused abstractions. Copy this package's JSON into the config without numerical edits; include the original instruction/spec under `docs/paper/static_ovmap/disagreement_query_v1/spec/`.

New entry point:
```
/home/ww/miniconda3/envs/ovimap-map/bin/python \
  scripts/evaluation/run_ovimap_disagreement_query.py \
  --spec configs/static_ovmap/disagreement_query_v1.json \
  --parent-root /mnt/shared/ww/ovimap-source-preserving-update-v1/attempt_001 \
  --output-root /mnt/shared/ww/ovimap-disagreement-query-v1/attempt_001 \
  --phase all --resume
```
Implement individual phases `bind`, `diagnose`, `screen`, `expand`, `time`, `report`, `verify`; `all` runs the dependency chain with the defined conditional skips. `expand` consults the frozen screening decision; it is not a manual override to run unapproved configurations. Publishing is performed after a real completed CLI run and is not recursively invoked by resume.

## 3. Stage P0/P1: fixed diagnostic and acquisition inputs
First create an immutable prediction-side plan for the four screening scenes, without reading new FC candidate scores or labels. Use existing 32-frame pose-bank convention. Keep source-row raycast masks as the ordinary-FC FULL-mask definition; directly projected canonical physical sites provide independent support measurements. They do not repaint the map.

Run two diagnostics defined in the contracts:
- D-Surface on the four screening scenes: compare representative-row observations, exact-coordinate aggregation and direct depth/BVH-verified surface visibility. Quantify physical area/sampling convention, not only raw row counts. Reuse parent observations only when camera, depth, map and operator identity actually match.
- D-Vocabulary on original parent score vectors from all 26 scenes: nested predetermined label subsets containing the same two classes, current probability fusion versus fixed scaled-score fusion. No GT-chosen distractors, no new text encoder, no reduced-vocabulary AP. This diagnostic cannot select the main fusion rule or change its temperature.

Do not stop the query study merely because a diagnostic is negative. Its function is explanation. A demonstrable coordinate/scorer bug must be corrected as an engineering issue with scoped invalidation and amended provenance, not declared a novel method. Numerical choices otherwise remain fixed.

## 4. Stage P2: make the scientific query causal
Each selected incumbent gets at most eight geometrically qualified candidate views and a shared maximum-area anchor. The competition is G1's current class versus D2's strongest other class. No candidate FC score, predicted class or GT may be used to build the eight-view set.

Acquire the shared anchor FULL FC vector. DISAGREEMENT additionally requests up to four fixed quadrant-intersection regions on that anchor. Their scores localize a **proxy** for conflicting evidence; they are not a learned estimate of information gain or ground-truth reliability. Exactly define the proxy as in the contracts.

Lock each policy's second-view choice using the policy-specific evidence visible at that moment. Only then acquire the union of selected second FULL regions. An acquisition service may share actual image encodings, but a policy must not inspect another policy's unrequested region scores or future frames. Store `visible_evidence_keys`, selected frame, geometry scores, interest weights, fallback reason and decision hash.

Ordinary policies need two FULL reads. DISAGREEMENT has the same two-FULL-read budget but up to four extra anchor subregion/head calls. Report this explicitly: equal FULL-image-read budget is not equal FLOPs. Do not force cheap controls to run unused subregion heads in standalone timing.

Technical execution failures do not become successful empty observations. Scientifically unavailable/zero region vectors use the specified keep/fallback behavior and remain in coverage denominators; no replacement object or retry on an easier view.

## 5. Stage P3: small screening, with non-redundant controls
Stage A executes D2, G1, and the four query policies with MEAN updating: 6 methods x 4 scenes = 24 logical rows and 12 ordered two-scene pools. Choose the best simple query among AREA/COVERAGE/VERIFY using the fixed ranking, not per-dataset selection.

Stage B adds AREA and SUPPORT updating on that one simple query and on DISAGREEMENT: four more methods, 16 rows and 8 pools, with no new visual evidence. Total screening: **40 rows, 20 pools**. If different methods produce identical outputs, retain separate logical rows and an equality proof; evaluate once only when complete scoring identity matches.

Do not implement pose-group averaging as the third main updater. With exactly two equally treated observations, it always reduces to the same mean whether they share a group or form singleton groups. AREA is a genuine non-redundant control. SUPPORT may legitimately equal AREA or MEAN on some support sets; report its effective weights and equality rate, not an invented gain.

Select the simple updater US among MEAN/AREA using only the already selected simple query QS. Freeze QS and US. Evaluate the three predefined novelty contrasts and the exact screening extension gate. If no signal passes, conclude `SCIENCE_COMPLETE_SCREEN_ONLY`, keep G1, generate all diagnostics/tables/handoff and publish; do not expand or start another hyperparameter grid.

## 6. Stage P4: conditional complete regression
If screening passes, automatically execute the fixed six-line factorial regression:
D2; G1; QS+US; DISAGREEMENT+US; QS+SUPPORT; DISAGREEMENT+SUPPORT.
Use all 8 Replica and 18 CF18 scenes in the original fixed order. Freeze algorithm and both selected simple baselines before inspecting the additional 22 scenes. These scenes are historically exposed, not a pristine test set. Pilot outputs may be reused only when query/feature/output/scorer identities match exactly.

Expected regression coverage: **156 logical rows, 12 complete ordered pools**. Every new method constructs a real whole-scene G1 payload, changes only selected D2 incumbent classes, recomputes official current-class area ranks, and uses the same exclusive partition for all metrics. Recovered G1 owners, unknown regions and unselected labels remain unchanged.

Read the actual released IoU threshold grid/minimum region settings; assert they equal the inherited protocol (.50:.05:.90 plus AP25, minimum 100 points). Do not silently call another AP implementation. Never substitute mean scene AP for an ordered pool.

Apply the full gate; allow the strongest simple control to win. A positive pilot is not a deployment upgrade. If all fail, retain G1 and stop.

## 7. Stage P5: attribution and bounded costs
Report all five metrics and current-G1 / D2 deltas, query divergence, actual class changes, fixed-original-support wrong-to-right/right-to-wrong, class-aware unique GT50/75 changes, tied score-entry caveats, and all excluded/unavailable objects. Since masks are fixed, class-agnostic geometry matching should be invariant; it cannot be claimed as a new structure gain.

Split costs into logical per-policy requirements and actual union/cache execution. Count FULL pools, anchor-probe pools, FC image inputs, head calls, visibility rays, cache hits, model loads, failed attempts and measured wall time. Unknown failed-attempt timing remains null. Do not count phases/checks/array elements as scientific experiments.

Only if a complete candidate passes the strict upgrade gate, measure it and the strongest matched simple policy in eight Replica scenes x two repeats: at most 32 cold calls. Fresh derived observation/feature/result state, model resident, common G1/source distributions resident, reverse order on repeat two. Time all post-G1 preparation through ranked output and required writing; evaluate outside. This is incremental semantic re-query cost, not full reconstruction or online FPS. No target pass => no mandatory cold-timing campaign; report actual acquisition costs instead.

## 8. Proportionate implementation validation
Run the 12 small supplied reference tests, then at most 12 directed production tests/invariants covering the genuine new failure modes. Two real engineering objects maximum, deterministically the first query-eligible selected owner in office1 and scene0011_00. If one has no query-eligible owner, record NOT_APPLICABLE and do not select a GT-informed substitute. Direct FC versus adapter check on the same anchor FULL; zero-update payload parity; policy evidence access; original ordered scoring parity. Small actual tests may reuse the same physical forward only with accurate accounting. Pre-outcome engineering repair does not permit tuning on AP.

Do not run the whole repository suite, fuzz/safety campaigns, blanket rehashes of every model in every loop, thousands of duplicated requirement assertions, or extensive speed sweeps. Hash immutable large inputs once with file-state memoization; verify changed leaves only. Write one meaningful requirement-to-artifact matrix.

Budget engineering extras separately (max four FULL + eight probe pools, max four image encodings). Runtime retry is once per failed leaf after a concrete fix, not repeated attempts to obtain a better score.

## 9. Resume, freezing, and status semantics
Use immutable keys for geometry, camera, FULL/subregion mask, model/text, pooling, support sites, current visible evidence and selected rule. New operator versions must not hit old incompatible region records. Common text-independent dense features may be shared across both vocabularies only when model and image tensor identity match; region/class vectors include text identity.

Lock query choices before second-view scores, predictions before current-stage GT evaluation, pilot selection before expansion, and source code/config before outcome inspection. A resumable interrupted run reuses complete leaves, reports failures honestly, and invalidates only new-task descendants whose inputs changed. Do not alter old experiments.

`all` returns 0 only when every required phase for its realized branch is complete; a defined screen-stop is complete, an unresolved input/model/scorer failure is not (nonzero exit). Separate implementation, science, expansion, selection, reporting and publication statuses.

## 10. Deliver tables and publish the actual work
Under `docs/paper/static_ovmap/` create:
- `DISAGREEMENT_QUERY_RESULTS.md`
- `DISAGREEMENT_QUERY_HANDOFF.md`
- `DISAGREEMENT_QUERY_SELECTION.md`
- `DISAGREEMENT_QUERY_CLAIMS.md`

Under `artifacts/static_ovmap/disagreement_query_v1/` publish a compact result store, per-scene metrics, query/evidence ledgers, content-restorable per-object distributions, diagnostic counts, costs, frozen choices, table CSV/LaTeX/Markdown and a concise completion matrix. Large RGB-D, TSDF, dense feature caches and weights stay in shared storage with location and content identity. Small artifacts must actually be in GitHub, not merely referenced by a server path.

Keep exactly three main table families: performance; query/update contrasts; evidence coverage/correction/cost. Put geometry/vocabulary diagnostics in supplements. Do not blend four-scene pools with 8/18-scene pools. A skipped regression table must explicitly state NOT_RUN_NO_SCREEN_SIGNAL, not show predicted numbers. Do not require a PDF/plot-production pipeline for completion.

Run the exact full CLI successfully and save its real exit and logs. Audit against this instruction and fix genuine omissions. Inspect `git diff --check`, actual staged paths and file sizes; avoid blanket `git add .` of large assets. Commit code/results/handoff, normal push to the new branch, and verify:
```
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git ls-remote origin refs/heads/research/ovimap-disagreement-query-v1 | awk '{print $1}')
test "$LOCAL" = "$REMOTE"
```
Only then write external `publication/final.json` with full SHAs, CLI exit, result identity and timestamp. Do not put the final commit's own future SHA into a tracked file. If a report amendment is needed, commit/push again and update the external receipt. Do not claim remote success on a failed push.

Final Codex response: actual completion status; strict/material target status; selected method; complete cohort table if run; mechanism verdicts; real new compute; unresolved limitations; exact branch and full remote SHA; clickable report/hand-off locations. Do not promise future background work or claim a gain from a simple control as a validated new mechanism.
