# FG2 Structure and Confirmation Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. The primary owns architecture, scientific semantics, algorithms, debugging, core code, review and integration. Only explicit mechanical leaves may use mechanical_worker.

**Goal:** Execute the frozen 8-arm paired training, locked CONF12 confirmation, REG26 application, statistics and ordinary C1/C2 publication, preserving negative and missing results.

**Architecture:** A task-specific binder preserves the original FC context and exact object ordering. Seven-tensor sparse inputs are shared by the fixed models. Atomic per-arm checkpoints and immutable predictor/input/prediction locks separate DEV selection from confirmation annotations. GPU work is serialized under the inherited lock; dynamic scorers retain the released kernels. An external coordinator observes scientific exit before report and publication.

**Tech Stack:** Inherited Python/PyTorch FP32 environments, NumPy/SciPy, original FrozenFC/SAM2/ScanNet projector/scorer, deterministic JSON/NPZ/gzip, matplotlib, normal Git.

**Spec:** `docs/paper/static_ovmap/structure_confirmation_v1/prompt/00_给Codex的最终执行指令.md` plus files01–05; byte-identical `configs/static_ovmap/structure_confirmation_v1.json`.

## Global Constraints

- BASE=2f0d8a4f7043557462f05875eca8b7f71eb66761; FG_BASE=bec4176d9a6784a89ea293a9803e5cf6fab178ec; research/fg2-structure-confirmation-v1; deployment N0_UNCHANGED.
- Fresh S_LL,V_LL,G_LL,S_UL,S_LU,S_UU,S_PRE,MV_ATTN, seeds17/29,2000 updates,16micro per update; original order indexes622 base TRAIN/24families;69base DEV/4families;160base/40heldout; full200 inference.
- Original loss, optimizer, LR, eta and draws; permanent250/500/1000/1500/2000states; full resume every100 and before evaluation; preserve last2; replay<=400. No saving a partially completed update as completed.
- Training/DEV physical head<=1100000, confirmation<=300000, REG<=131072, engineering<=200,cost<=4000; 32000 nominal optimizer updates; every replay counted separately.
- FG-family shares parent initialization; membership frozen/unused; G joint valid-token permutation salt9173 with canonical SHA key; uniform outer retains its own differentiable mass; S_PRE uses raw phi without token L2.
- Seed17 A/M/C/earliest selects each method with tolerance1e-10; seed29 uses the same method step. Fixed1000 and terminal2000 remain separate. No positive-signal gate.
- CONF12 names from exact sealed Git blob41b2769612e89e6f9d4e67c26d29550d79d8c884. No replacement or additional families. Predictor lock precedes acquisition; all real predictions lock precedes annotation parsing.
- One inherited A40/GPU lock; CPU scorers<=3 and BLAS1; FP32/noTF32/AMP; no upgrade/global proxy changes. CPU dense8GiB+sparse24GiB total<=32GiB.
- Initial free>=80GiB; reserve20GiB; new runtime<=200GiB; downloads<=120GiB and <=2 concurrent; FC<=1152 unique confirmation images; SAM<=48images. New NQ/old-data encoding/maps0.
- 19selected slots, all2/4/8prefixes;16matched1000extra slots at8V; <=420 real AP rows/35pools. REG11slots×2readouts×26=572rows/44pools.
- Four fixed family bootstrap comparisons×1000 draws seed4314, same draws for GT and real, separate per-seed metrics then mean difference; never concatenate seeds or average scene AP.
- Exactly4 focused test groups; normal once, affected-only repairs and minimal real checks. No full suite or new experimental methods. Cases<=12; retained unknown/negative/null/abstention.
- Git model-only1000/2000/selected weights deduplicated; no base weights/raw RGB/mesh/GT/optimizer. Blob<=40MiB,total<=256MiB. C1 results then C2 receipt/HANDOFF only; actual remote SHAs and external terminal resume.

### Task1: Binding, tensor/model contracts and four test groups

**Files:** common.py,binding.py,data.py,grouping.py,models.py,selection.py; tests/static_ovmap/test_structure_confirmation_v1.py.

**Interfaces:** `bind(args,spec)` preserves inherited binding fields and writes roles; `SparseInputs(fc,manifest).load(obj,condition,prefix)` returns exact7keys plus audit, sharing full8; `permute_inputs(x,key,condition)` is external to head; `make_models(seed)` returns8heads; `static_model(name)` returns a common result dictionary. `choose_step(rows)` applies only A/M/C/earliest.

- [x] Run4 production-API test groups and implement grouping/model/budget/checkpoint APIs. The initial RED was an import failure, not a valid behavior RED; later DEV-key and completed-training resume repairs have actual behavior RED/GREEN evidence.
- [x] Verify ZIP/config bytes, origin/BASE/branch, fixed CONF blob, original TRAIN/DEV object ordering, all role family intersections and actual dependency visibility. Preserve missing inputs explicitly.
- [x] Build minimal FrozenFC context with original class_split bytecopy, upstream source link, path migration after verification, inheritedGPU0 and exact text/model identities.
- [x] Implement parent-exact S/V math, external G input transformation, differentiable own-mass factorial, raw phi-before-pool, controlled MV and3static methods; no targets enter forward.
- [x] Actual S/V/phi gradient/input parity checks use <=200 engineering calls. Record active/frozen/gradient parameters and real G correspondence retention; no forced permutation effect.

### Task2: Shared cached inputs, paired optimization, DEV and export lock

**Files:** training.py,checkpoints.py,recognition.py,export.py.

**Interfaces:** `train_arm(binding,fc,loader,objects,dev,schedule,seed,method)` writes permanent model/DEV states and safe resumable generations; `evaluate(head,...)` saves full200 scores without additional inference; `freeze(root)` exports all selected/1000/2000 model-only states and immutable predictor lock.

- [x] Reuse or materialize exact full8 sparse tensors once perobject/condition from existing grids; no old RGB encoding. Finite consumed-byte identities and total LRU budget apply.
- [x] Generate parent draws on all original objects in original order and lock schedule. Train all16arms from common FG initialization, independent seeded MV, reset shared RNG after construction.
- [x] Use parent optimizer/LR/loss with16micro, gradient clip1 and eta ramp; inspect effective gradients afterstep2. Log actual updates/forwards/phi/memory/time, including errors and replay.
- [x] Save only after successful updates; preserve before-validation resume,5permanent evaluation states and matching DEV arrays; interrupted incomplete validation reruns with actual calls counted.
- [x] Verify updated-state resume-next-step equality and same-seed initialization/draw contracts; select with seed17 only, then export exact seed29 same-step states. Baselines and all negative curves remain.

### Task3: Model-locked acquisition, label-free geometry/SAM/features and real predictions

**Files:** confirmation.py,sam_worker.py,prediction.py.

**Interfaces:** `acquire(root)` consumes predictor lock and exact prior authorization; `geometry(root)` produces label-free banks; `proposals(root)` locks every retained support; `features(root)` captures shared banks; `predict_real(root)` locks all model/object/prefix outputs before GT.

- [x] Resolve only declared data roots and exact five suffixes. Plan official size/ETag and storage before <=2 transfers; one retry/file; retain missing/404/authorization reasons without swapping names.
- [x] Decode/project finite96-frame banks in original unaligned coordinates using inherited SensorReader/SourceRowProjector. Store frame provenance, registration and sorted frame identities.
- [x] Run pinned SAM2 on fixed4seed frames/family, keep top16 by prescribed GT-free ordering. Lift raw supports, retain overlap/duplicates and unavailable proposals, derive qualified views/sites with no annotations.
- [x] Capture each bank FC image once using original numerical operations with task storage bounds. Build real seven-tensor inputs, fixed support and input lock, retaining NO_MULTIVIEW_SUPPORT unknowns.
- [x] Infer19selected×prefixes and16matched1000×8V slots; reuse only exact identical model/input identities; save full200cosines, masks/source ids/fallback, globally lock all12 real outputs.

### Task4: GT diagnostic, sparse official AP and fixed-map application

**Files:** gt_diagnostic.py,evaluation.py,prediction.py.

**Interfaces:** `prepare_gt(root,allow_annotation_parse=True)` requires real prediction lock; `predict_gt(root)` handles fixed diagnosis objects; `score_real(root)` uses independent overlapping masks and fullGT; `map_predict(root)` locks all26×22outputs; `map_score(root)` consumes that lock.

- [x] Parse annotations only after real lock. Hash-select <=64legal GT objects before visibility; preserve unsupported objects and4condition abstentions, separate GT support/target directories and role labels.
- [x] Compute clean ordinary/balanced Top1 and family/category A/M/C for base/heldout/all200 competition. CE160 heldout=null; NLLdefined counts/abstentions/H/null denominators explicit.
- [x] Use original ScanNet200 namespace, minimum100 and overlaps; identity mesh indexing, every retained positive mask independently, constant1rank and shared input ordering. FullGT remains even for missing support;35logicalconfiguration slots/12scene pools remain complete or explicit partial.
- [x] Keep descriptive maxIoU, uniqueGT balanced recognition and strict one-to-one geometry separate from official score entries/TP/AP. No new sparse semantic mIoU.
- [x] Existing REG26 eligible objects use identical cached inputs for11slots, original scene-specific text/NQ/temperature/G1; FONLY andD2 preserve B_FC noneligible/background/recovery. Global prediction lock precedes scoring;2baseline engineering checks only.

### Task5: Statistics, cost, real figures and evidence bounds

**Files:** statistics.py,costs.py,visualize.py,reporting.py.

**Interfaces:** `paired_statistics(root)` uses fixed draw matrices and original matches; `cost(root)` uses hash-first16DEV originals; `qualitative(root)` uses deterministic median cases; `report(root)` writes compact auditable artifacts and required documents.

- [x] Four fixed comparisons×1000family draws, same drawn families for both seeds/domains; GT primarymatched1000 and realprimaryselected, secondarypoint estimates. Actual full-sequence pool/namespace parity and defined-class ranges.
- [x] Compute same-seed/matched-step inner/outer-relative factorial effects and interaction, G correspondence retention, allseed/family/prefix data, exact null and limitedcoverage evidence status.
- [x] Cost19slots×16objects×3prefixes×4passes<=3648; GPU sync, separate overallregion/phi/transfer andhead timing, median/P90/memory; shared FC/SAM actualcost and cache hits separated.
- [x] Select <=12real S/G andS/MV success/failure/unchanged cases by median/ties/family rules, real camera/scale/colors. Generate4specified chart families from savedJSON; primary inspect rendered outputs.
- [x] Read the3attached primary external sources to define MV/PFE scope. Publish EXTERNAL_FULL_REPRODUCTION=NOT_RUN and distinguish reconstructed sparse CONF from exposed full-map REG; no CVPR success guarantee.

### Task6: Actual orchestration, compact release and final requirement audit

**Files:** runner.py,publication.py; scripts/evaluation/run_fg2_structure_confirmation_v1.py; required docs/artifacts.

**Interfaces:** publicCLI all02.phases and specified path flags; science child observed by coordinating parent before report/publish; `publish(root)` real ordinary C1/C2; `terminal_resume(root)` verifies finite consumed inputs/sources/remote and starts0workers.

- [x] Implement explicit dependency errors and independent continuation; actual negative outcomes exit0, unresolved required dependencies nonzero. No legacywhole train/all/proposals pipeline dispatch.
- [x] Run help and4focused groups normally once, affected-only repairs; execute all with external actual exit records, perarm/stage receipts and complete resource ledger.
- [x] Publish all contracts/spec/source evidence, complete prediction scores and releasedPR sufficient statistics with originalAP parity, allcurves/locks/coverage/negative/null/alias/weights; rawlarge records retain privateSHA/regeneration metadata.
- [x] Primary reviews00§0–15 and01–05 againstactual source/data/gradients/checkpoints/predictionlocks/metrics/figures/counts/roles/model exports and artifactbudget. Correct affected items only.
- [ ] Complete ordinaryC1 code/results push with fullremote SHA, C2receipt/HANDOFFonly push, externalfinalC2proof and actual publicall resume0workers/head/commits. Markgoalcomplete only actual fullcontract proven; otherwise preserve explicitPARTIAL with exact dependencies.

## Plan self-review

All main sections0–15 and the method/data interfaces are mapped above. Mandatory new training and confirmation are execution tasks, not substituted with old selected weights or a future plan. Annotation barriers and preserved original sampling indexes are explicit. Statistical independence is by family and original object, not seed/view/condition. Scientific negatives do not gate confirmation or publication. User authorization supplies inline execution, worktree creation and ordinary research-branch pushes without renewed skill approval; the package's two-commit transaction overrides generic per-task plan commits.

## Measured continuation, 2026-10-15

All16arms and80permanent DEV states completed; fixed1000/selected/terminal2000 retained. Original training receipt c95eb2e163aee593097633de4b580af138e9ce9c8cdcc447572ecd62e1bb95af is preserved during resume. SAM completed48image encodings/192decoder batches, yielding337retained proposals,327with multiview inputs;10unavailable proposals remain unknown. The initial SAM import failures occurred before main/model access and encoded0images; original logs are archived. A task-scoped SAM nvJitLink preload fixes controller library precedence without changing either environment. Removed only forbidden payload metadata.GT_used; GT-free provenance remains outside payload. All26map predictions are locked (572logical rows,1436eligible objects,16236actual head calls including440failed-metadata-attempt calls), with exactly2oldFCpayload parity checks. The actual fixed CONF bank contains635frames, all encoded once;1152is an upper bound, not a requested padding target. Cost3648calls and real FC gradient/parity checks completed. CONF predictions and official420rows/35pools completed. GT279originals (215base/64novel across12families),270supported/9retained abstentions. REAL337proposals/327supported; IoU>.5 matches18proposals/16uniqueGT across8families, including3novelGT across3families: LIMITED_REAL_OR_NOVEL_SUPPORT. GTmatched1000 clean Top1 seedmean: S38.710%,G39.427%,V37.993%,MV41.756%; MVnovel0.781% vsS53.125%. REALselected pooled APall seedmean: S0.727%,G0.649%,V0.742%,MV0.503%. These points do not prove physical correspondence necessity. Saved G mappings show physical-pair residue2.2%–2.6% at8V, with no additional inference. REG572rows/44pools, all4000fixedbootstrapdrawsets,4statisticalcharts,7comparisonfigures(6distinctproposals,5unavailableslots),actualscientificexit0 andprimaryprepublicationreview completed. NormalC1/C2 publication andterminalresume arethelastpost-C1 operations, recordedexternally inOUT/publication/final.json andterminal_cli_resume.json; thisC1plan snapshotisnotrewrittenbyC2. Cases use real depth/mesh projections without publishing originalRGB.

## Final prepublication primary review

Allscientific stages completed with actualexit0. Primaryreview maps00sections0–15and01–05 toconcreteevidence. All198automatedchecks and57finiteverificationmemos/11422entriespass;all2135scoreobjects/42weights/48profiles and44pooledsemanticconfusions exportedlosslessly,allindexedSHAschecked. FouractualstatisticalPNGs and7casePNGs inspected. Thephysicalcorrespondencemechanismisnotconfirmed; negatives andlimitedreal/novelcoverage remainvisible. C1/C2andterminalresume areverifiedonlyaftertheiractualexecution and recordedexternally toavoidathirdself-referentialcommit.
