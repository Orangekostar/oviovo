# FG2 Branch Utility Wave2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. The primary executes scientific and semantic work. Only deterministic formatting leaves may use a mechanical_worker.

**Goal:** Execute the frozen 1014 branch admission protocol through real measurements, conditional training/maps, evidence review and verified C1/C2 publication.

**Architecture:** Stream one original/condition at a time through prefix2/4/8 and the two frozen heads. During the baseline capture, retain local decompositions only in memory, emit factorial/static/action outputs, and share each subset head across L/B. Subsequent phases aggregate these sealed units; training sees only cached 24-dimensional features and utility targets. Predictions for every activated map arm are locked together before scoring.

**Tech Stack:** Existing FP32 PyTorch FC environment, NumPy, native ScanNet/Replica scorers, Python subprocess observers, normal Git branch pushes.

**Spec:** `docs/paper/static_ovmap/branch_utility_wave2_v1/prompt/00_下一步实验设计与Codex任务.md` and byte-identical `configs/static_ovmap/branch_utility_wave2_v1.json`.

## Global Constraints

- Source0f96c5e09c3ced9262105c1352cc428643f5676e; scientific FG parent bec4176d9a6784a89ea293a9803e5cf6fab178ec; branch research/fg2-branch-utility-wave2-v1; N0_UNCHANGED.
- Exactly622 TRAIN/69 DEV base originals; 24/4 families; inherited160/40 split; conditions clean/truncate/append/truncate_append; prefixes2/4/8.
- FC/text/two FG2 heads frozen; new RGB/SAM/geometry0. Labels enter offline scoring/utility only, never seven-tensor forward or 24-feature creation.
- One GPU; CPU scorers≤3; dense+sparse cache≤32GiB; pre-map native head attempts≤85000; initialfree≥10GiB/reserve≥5GiB.
- Exactly six focused test groups. No full suite, fuzz, stress, alternative seeds/thresholds/architectures or legacy whole-task entrypoints.
- Logical action count3n+1, n<3 KEEP; retain support-proved aliases. Controller24→32→16→1 GELU, zero final layer,1345 parameters,8000 conditional updates maximum.
- Git new files≤40MiB each, total≤150MiB; do not publish raw dense grids, RGB or all local U. Complete action summaries/features, curves, candidates, weights and locked prediction receipts are published in bounded chunks.
- Runtime roots from task_paths.json; parents and previous output are read-only. Verify sealed parent JSON before resolver rewrites. Native worktree attempt failed outside Git; manual isolated fallback already established.
- User authorizes inline execution without confirmation and normal C1/C2 pushes. Primary retains design, experiments, debugging, review and integration.

---

### Task1: Contracts, source binding and focused red tests

**Files:** new common.py, binding.py, subsets.py, decomposition.py; tests/static_ovmap/test_branch_utility_wave2_v1.py.

**Interfaces:** `validate_inputs(x)` validates exactly7 tensors with a common view axis; `subset_inputs(x, indices)` slices that axis in every tensor; `state_key(x, condition)` includes condition/frame IDs; `compose(base, result, eta=.5)` uses correct(base,U-base,alpha,eta), empty returns suppliedbase; `local_statistics(result)` returns mass/g/null/groupcount; `effects(h00,h10,h01,h11)` returns symmetric base/local/interaction/total. `preflight(spec,parent_root,previous_root,output_root,path_map,gpu)` returns a newly sealed binding with independent task_source_base; `load_binding(root)` revalidates parents; `runtime_registry(binding,fc)` verifies frozen identities and loads two fixed exports.

- [x] Write six test functions for subset/prefix/condition/history; Compose cross-base/empty/KEEP; actions/dedup/short bank; feature isolation/normalizer/paired initialization/draws; selection/aliases/exact resume; NQF/rank/dynamic coverage/publication limits. Each imports its production APIs inside the test and asserts explicit numerical/structural invariants.
- [x] Run `PYTHONPATH=src /home/ww/miniconda3/envs/ovimap-map/bin/python -m pytest tests/static_ovmap/test_branch_utility_wave2_v1.py -q`; observe missing new modules before implementation.
- [x] Implement exact contracts without mutating inherited inputs. Validate nonfinite/zero valid vectors. Use one captured same-set original for diagonal and history comparisons; record real FP32 evidence separately from engineering fixtures.
- [x] Preflight verifies622/69,24/4,74/28 classes, original manifests/feature bytes/head SHA/steps, minimum FC dependencies, free disk and previous history. Copy manifests, readonly symlinks, never invoke old prepare/H2/training.

### Task2: One streaming native-state capture

**Files:** new action_bank.py, features.py, capture.py.

**Interfaces:** `build_actions(x, full, evaluate_subset)` returns ordered KEEP/G/L/B records with kind, frame_id, edited, noop_proof, embedding, base and local state. `action_features(x,full,action,view_index)` returns finite FP32 length24 without targets. `capture.run(binding,fc,loader,heads)` writes sealed original/condition units, baseline.json and raw factorial/static/action blocks. `factorial(root)` and `action_bank(root)` aggregate those units without new native calls.

- [x] Load requestedprefix with the original producer before statistics; subsetsevenkeys after truncation; reject duplicate frame IDs or future frames. Reset native state cache per object/condition/seed. Actual short-prefix aliases use support keys; requestedprefix is still recorded.
- [x] Fullstates2/4/8 plus leave-one-out4/8 at most15/head/unit. L/B share subset; G uses full Local. Empty-local drop reuses Local and L is support-proved noop. Keep original native FP32 diagonal; rebase cross cells using U-newB. Stream native attempt ledger, completed units and regeneration commands.
- [x] Emit full200 cosine/margin/top1 separately from CE160 utility atT.07. Store raw action embeddings/cosines OUT; publish action utility/full200 margin/predictions/features in compressed bounded blocks. Do not store all group vectors.
- [x] Compute exact24 features including original quality, same-site semantic disagreement, metadata directions, occupancy/depth; normalize copies only, popstd, no-support zeros, exclude zero shared means, no labels/XYZ/target membership.
- [x] STATIC bases original FCmean, original maskpixelareaweighted, cosine medoid tie frameID; γ0/.5/1 with eta=.5γ. Compose residual relative to each suppliedbase. Meanγ1 uses native FG2, Meanγ0 native FC.
- [x] Compare all available historical DEV embeddings at tolerances1e-5/1e-6 and labels/A/M/C exactly. Main denominator all originals; report actual n4/n8 paired coverage separately. Fixed example categories use effect_base<−1e-10 and effect_local≥−1e-10, reverse, or both>1e-10, earliest key.

### Task3: TRAIN gate, exact conditional paired learning, DEV lock/repeat

**Files:** new utility_model.py, training.py, selection.py.

**Interfaces:** `Normalizer.fit(features)`/transform; `UtilityRegressor(seed)`; `make_draws(objects,seed,steps=2000,batch=16)`; `select_action(actions,scores,tau,coupled=False)`; `compare_candidates(a,b)`; `promotion(candidate,reference)`; `training_gate(root)`; `train(binding)`; `select(root)`; `repeat(root)`.

- [x] Gate per original over fourconditions atprefix8: ≥40 positive originals/8families,40 negative/8families,20 branch-preference/6families, each frozen seed, fixed±.01; no DEV influence. Failure means zero new optimizer steps and STATIC only.
- [x] Fit one TRAIN-only normalizer/seed over all nonKEEP actionrows; stdfloor1e-6; control zero final8 afterstandardization. Shared init(seed+2009) and precomputed draws(seed+3001) across paired arms. Drawclass→original, prefixes4/8 equiprobable, conditions .5/1⁄6/1⁄6/1⁄6; batch16; no resampling empty groups.
- [x] Train each2000 steps FP32 AdamW1e-3,wd1e-4,betas.9/.999,eps1e-8,constantLR, clip1; SmoothL1β.1 meanpernonKEEP thenper16group; loss0 empty. Training worker imports no FC/FG2 model. Save500/1000/1500/2000 and exact resume100 with model/optim/RNG/drawindex/norm/dataidentity, all scalar curves.
- [x] Selectseed17 STATIC9 and, iftrained, three16cell methods; prefix2alwaysKEEP; rank A8/M8/C8/A4(desc),editfraction over equally weightedlogical DEV4+8 conditionrows,stepasc,tau desc,fixedorder;STATIC tiesMEAN/AREA/MEDOID thenγ1/.5/0;tol1e-10. Proven noops edits0, STATIC discrete action edits0.
- [x] Lock each method and strongest control C_STAR before candidate seed29 readouts. Seed29 only lockedstep/tau, no reselection. Promotionbothseeds ∆A8≥.005, ∆M8/∆C8/∆A4≥−.005, ordinary netcorrections8>0. WITHpasses→all4 available methods; otherwise onlylocked C_STAR ifpasses. No threshold changes.

### Task4: Existing conveyor, conditional maps and native evaluation

**Files:** new prediction.py, evaluation.py, conveyor.py.

**Interfaces:** `conveyor.run(binding)` analyzes existing FG2/FC8 readout→Ftop1→NQFclass→fixedowner/scorer changes. `prediction.run(binding,fc,heads)` writes allactivated26scene predictions/seed and global lock. `evaluation.run(binding)` requires thatlock, max3 CPU workers, dynamicrows/pools and exactreferencealiases. `validate_coverage(cohorts,methods,rows)` validates26×methods, no old fixeddispatcher.

- [x] Conveyor is mandatory even if no maps promoted; verify original readouts/decisions/store/scorer artifacts; perstage changedcounts, matched support/metric deltas, precise dependency reason if absent. No new oldM1 maps or rank tuning.
- [x] Newmaps maximum4methods×2seeds×26=208. Replace F only for eligible fixed G1 owners; preserve N/Q/temperatures,PE_D2_GROUPED pooling,background/restoredowners and native ranking; Replica contexttext vsScanNetfull200. Feature enumeration costs allsubsetheads+phi+transfers, notonlyMLP.
- [x] Lock all new predictions before GT. Reuseverified D2/G1/FC8/FG2s17/s29 refs26each; validate exactprediction/scoring-inputaliases. Exactlytwo additional native FG2s17 scorer checks, first scene/cohort. Poolofficial instance confusion/semantic totals, no average AP. Publish fourcohort×seed metrics/deltas, base/heldout absentnull, per sceneedit/conveyor.

### Task5: Observer, reporting, publication and final audit

**Files:** new runner.py, reporting.py, publication.py; entrypoint scripts/evaluation/run_fg2_branch_utility_wave2_v1.py; RESULTS/HANDOFF/CLAIMS/REVIEW.md and artifacts/static_ovmap/branch_utility_wave2_v1/.

**Interfaces:** CLI `--spec --parent-root --previous-root --output-root --gpu --path-map --phase --resume`. Phases fixedpreflight/baseline/factorial/action_bank/training_gate/train/select/repeat/map_predict/map_evaluate/report/publish plus all. `reporting.run(root)` always renders observed state; `publication.run(root)` C1verify→C2verify→externalreceipt. Native GPU phases and small-training phase use independent observed FC-Python subprocesses and inheritedsingleGPUlock.

- [ ] Real subprocess exit receipts only after wait; on failure preserve error/log/partialunits then report/publish truthfully in finally, returnnonzero. Terminalresume verifiesremote/source/config/runtime/artifacts, launchesno new worker/commit if unchanged.
- [x] Sixcore tables inputs/baselines,factorial,utility/branchcoverage,static/learned DEV+repeat,officialmaps or precise fixedactivationfailure,cost with forward/phi/action/transfer/MLP/encodings/update/wall/memory boundaries. Additional family/condition/examples mandatory. CSV LF, JSON null, noinventedzeroresults.
- [x] Publish every finitecandidate, controller/norm/lockedprofiles, allaction summaries/features, curves, proof and rawblocks bytes/hash/regeneration commands within40/150MiB. Scoring/selection reports distinguish diagnosticGT-best from deployable learned policies. No claims about unseen scenes/newgeometry/onlineFPS.
- [ ] Run onlysixgroups; record actualinvocation/exit. Execute CLIall and exactresume, inspectstageexits/identities/counts/gates/tables. Primary prompt-by-promptreview maps eachrequirement to actual evidence, no delegation ofreview. Fix onlyaffectedunits/tests before C1.
- [ ] Normal C1 includescode/data/results/review, verifyremote; C2onlyC1receipt/HANDOFF, verifyremote, externalfinalreceipt hasactualexecution/repository/publicationstatuses andcontrollerobserverexit. No mainmerge/force/selfSHAloop. Markgoalcomplete onlyallrequiredwork verified.

## Plan self-review

Coverage: Tasks1–5 cover promptA0/A1/A2, TRAINgate,pairedconditionallearning,Bselection/repeat,Cconveyor/newmaps,resource ceilings,strict prefix/condition isolation,artifacts/normalGit andfinalreview. Numericalinterfaces definedabove are used consistently; frozenJSONunchanged. Output paths differ from read-onlyparents. Jointcapture rather than independent nativepasses is necessary to obey82920 forecast/85000 hardlimit; its actual cost is reported underbaseline capture, not hidden. Selection editfraction interpretation and descriptiveexample criteria are explicit implementation choices within the frozen specification.

## Observed prepublication state

Science and all activated conditional phases are complete. Four paired arms ran exactly2000 steps;57 finite candidates and fixed seed29 profiles are recorded. No method passed both-seed promotion, so new maps/native extra scorer checks are correctly not activated. Native attempts74021, new RGB/SAM0; six focused groups passed the final correctly configured invocation. Primary evidence review is complete; CLIall/publication and terminalresume actual exits remain to be recorded externally before goal completion.
