# OVMap Contribution Attribution Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. The primary owns scientific interpretation, implementation, debugging, review and integration. Mechanical workers may only execute independently specified leaves.

**Goal:** Execute the complete frozen contribution-attribution contract, record actual E/R/S evidence and cost, nominate confirmation priorities, then verify normal C1/C2 publication.

**Architecture:** A new binder consumes immutable AREA/COMPACT/BB/FG/LR/Wave2 receipts and records role-specific dependencies. Probability-only predictions are globally locked before their scoring; frozen-head intervention is a separate single-GPU worker. Scene assignment, matches-only aggregation, crossed rank and clustered statistics retain the original released kernels. A coordinating all process observes the scientific child exit before reporting/publication; an external observer records its own actual exit.

**Tech Stack:** NumPy/SciPy, original released SceneEvaluator and pool kernels, FP32 PyTorch in the inherited FC environment, matplotlib, atomic JSON/gzip/NPZ, normal Git.

**Spec:** `docs/paper/static_ovmap/contribution_attribution_v1/prompt/00_给Codex的最终执行指令.md`; byte-identical `configs/static_ovmap/contribution_attribution_v1.json`.

## Global Constraints

- BASE f4245ec0654577634fefb60f3344c6ca9c9335e8; FG science bec4176d9a6784a89ea293a9803e5cf6fab178ec; research/ovmap-contribution-attribution-v1; N0_UNCHANGED.
- New training/RGB encoding/SAM/maps0; exposed results only; confirmation NOT_RUN_BY_CONTRACT and submission_evidence_complete=false.
- Replica8/CF18 exact order; CF18 has7 physical families. Fusion DEV scene0056_00,scene0534_00,scene0445_00,scene0626_00 is family-disjoint; S622/69 originals,24/4 families,160/40 classes,4 conditions, selected heads17/29.
- E0–E7 plus conditional E8 and J0/J1/J2: ≤12 methods,312 logical scene rows,24 pools;15 DEV weights/60 rows/15 pools;2 rank pairs/104 non-diagonal rows;2 extra engineering scene scores.
- Surface only DEV prefix8, selected heads,5 fixed shuffles101/211/307/401/503;3312 DEV forwards plus two TRAIN engineering originals under4000. Single inherited GPU/lock, FP32/no TF32/AMP; CPU scorers≤3 and BLAS1; total cache≤32GiB.
- OUT initialfree≥10GiB/reserve≥5GiB; no deletion of user files. Task Git blobs≤40MiB,total≤150MiB; no raw mesh/mask/RGB/GT or pretrained/optimizer weights.
- Exactly4 focused implementation test groups; no full suite, extra models, thresholds, whole-disk scans or legacy bind/train/all/evaluation dispatch.
- Source labels are independent of GT; unknown retains mask and semantic error. Official events, descriptive max-IoU and one-to-one geometry never share a name or denominator.
- Reporting producer digest is separate from numerical producer digest; immutable units only reuse identical direct source/spec/role/view/rank/scorer inputs.
- Normal C1 result push verified before C2 receipt/handoff; C2 proof external. All scientific exit is observed before publication, all-controller final exit observed outside it.

### Task1: Binding, numerical contracts and four red groups

**Files:** common.py,binding.py,inventory.py,sources.py,payloads.py,selection.py; tests/static_ovmap/test_contribution_attribution_v1.py.
**Interfaces:** `read_document(path,index)` validates original sealed content before path migration; `preflight(args,spec)` writes task binding/role lock/dependency visibility; `load_binding(root)` validates task/spec roots. `source_probabilities(sources,temperatures,ids,owner)` validates finite full-vocabulary available scores; `fuse(probabilities,names,operator,weights=None,native_label=0)` returns probability and class index or UNKNOWN/PARENT fallback. `fixed_owner_output(baseline,labels,ids,nearest,matched,method)` preserves all owner support/background, allows unknown, recomputes native ranks. `select_weight(candidates)` applies APall,mIoU,positive sources,L1,lex with1e-10; `freeze_weights(root)` writes immutable selected or defined null lock.

- [x] Write4 production-API tests before code, with expected probability examples, permutation/support invariants, factorial/rank/namespace and resume/budget/lock rules. Observe missing production modules rather than import-path error.
- [x] Bind new task without old binder; resolve only specified roots, verify role counts/scenes, record original parent identity/path arguments and partial dependency reasons. Defer FC dependencies to surface phase.
- [x] Implement exactly available source normalization, D2 original fallback, all8 availability patterns,15 weights. Test underflow handling without changing predictions, invalid available score raises.
- [x] Inventory actual original rows/pools and scoring context/projection; ledger contains all fields in spec4.1, original machine precision and alias proofs.

### Task2: Cached main predictions and official scores

**Files:** prediction.py,scoring.py,history.py,interaction.py.
**Interfaces:** `predict(root,role)` writes every runnable method/scene then global lock; `score(root,role)` consumes lock and dynamic rows, complete pools only. `score_explicit_view(evaluator,labels,ranks,method,prediction_identity)` preserves supplied six-decimal ranks. `historical(root)` reads A0/A1/A2/A3/A4/A5/U2/G3 and FG exact records; `factorial(p00,p10,p01,p11)` returns conditional increments, interaction and mean marginal closure.

- [x] Read four BB00 DEV sources/config/native contexts; if all available create15×4 predictions before scoring, select only DEV and freeze once. Defined missing branch excludes E8 only.
- [x] Lock all runnable26-scene E methods on A0 support plus eligible-only J fields on fixed FG0 geometry. Source score arrays pair by method/object/owner/vocabulary, not row position; missing whole J cell stays partial.
- [x] Reuse scoring only if full geometry/owner/labels/ranks/scorer/view are identical. Original context compatibility and overlap vector validated; all positive owners indexed even unknown/small; confusion valid-GT total checked.
- [x] Score dynamic independent scenes with≤3 CPU workers, count assign/score, matches-only and pool separately. Exactlytwo A1 engineering scorer checks. Cache/prune no formal cells; allpartial rows retained withoutshrunk completepool.
- [x] Derive E×R and S×fusion five-metric effects, per scene/family/class/fullpool and domain lineage. A5 explicitly F+R; N-only vs A0 differences measured.

### Task3: Surface and three attribution notions

**Files:** surface.py,recovery.py,attribute.py,rank.py.
**Interfaces:** `permute_valid(x,key,condition,shuffle_seed)` jointly permutes valid local_raw/local_unit/metadata perview; SHA task/key/condition/frame/seed RNG; `permute_slots(x,indices)` jointly reorders allslots including valid. `one_to_one(projected_owners,gt_instances,valid_ids,minimum,threshold)` maximizes cardinality thenIoU without semantic predictions. `error_path(reference,oldF,newF,oldfused,newfused)` applies spec9.1 priority. `rank_cross(root)` locks fixed pairs' twooffdiagonal explicitviews.

- [x] Read step1000 FG1/FG2 arrays for bothseeds, keys/classes/draws/init matched; missingrecord explicit, selected separate. No newTRAIN inference.
- [x] Build minimal frozen FC context original byteclasssplit/operatorlink/registry with verified text SHA and headSHA. DEV original+5shuffle/seed share baseline history check;2TRAIN工程对象共同slot等变，计实际head/phi/grid/time/memory，原输入不修改。
- [x] Validate R same mesh/incumbents/raw residual candidates; produce all source/evaluation/point/TPFPuniqueGT/tie counts from actual registry/trace, with official events separate from owners.
- [x] Compute max-IoU descriptive references and strict .25/.5/.75 one-to-one diagnostic including semanticunknown owners. FixedE geometry diagnostics identical; R may alter owners.
- [x] Classify everypositiveG1 owner including noteligible and all specified errorbins; retain source metrics/NLL/Brier/protection and allfourdenominators. Newlabels/probabilities never useGT.
- [x] Rank pair E6/E7 and FG0/FG2s17 diagonals explicitviewparity before reuse; unsupportedmissingrank/domaindifferent stoponlypair; offdiagonal semanticconfusion equalslabelendpoint.

### Task4: Family statistics, real visuals and confirmation nomination

**Files:** stability.py,visualize.py,nomination.py,reporting.py.
**Interfaces:** `namespace_matches(matches,token)` deepcopies scans and recursive predictionfilenames without score/count/intersection changes. `evaluate_cached(receipts,namespace,draw)` uses original evaluate_matches/compute_averages and confusion sum. `shortlist(candidates)` follows fixedfinite-positive cells/LFO/mean∆AP/fixedorder; `nominate_confirmation(queue,exposure)` hashes eligiblefamily names and selects lowestcapture.

- [x] Fivefixedpairs LFO by8/7families, fullorderedpool parity once. Fourfixedbootstrapgroups2000 each seed4314, sharedpaired draws; duplicatedcapture/filename namespaces verified; reportnull/excluded/GTclasssupport ranges.
- [x] Lock pure shortlist before qualitative selection; cases atmedianabsGTmargin/IoU withfixedties andfamilydiversity; ≤2modules×6cases, missingcategories explicit. Savedgeometry POINT_PROJECTION ifnoheadlessrender, noinventedspace/RGB; onePNG each.
- [x] Generate fourstatsfigures from savedJSON and three main tables plus all15candidates, family/perclass/availability/steps/shuffle/rank/cost/detaildata. Parent timings labelledhistorical, newtimingsactual/nonadditive.
- [x] Read exactlysix attached primarypapers' relevantmethod/abstract sections; boundedadditionalsearch≤4 onlyifneeded; E/R/S nearestcomparison tables with conservative novelty, exposure andprotocol differences.
- [x] Read officialqueue and finite exposurelocks only; noannotation/prediction access for nominees. Lock≤12families andcandidateconfigs/resources; confirmationexecutionNOT_RUN_BY_CONTRACT, submissionevidencecompletefalse.

### Task5: Real orchestration, bounded artifacts and publication audit

**Files:** runner.py,publication.py; scripts/evaluation/run_ovmap_contribution_attribution_v1.py; task docs/artifacts.
**Interfaces:** CLI exactlyspec3 phases/args, private scientificworker notpublicexperiment replacement; `all` scientificsubprocess→actualexecution_observed→report→publish. `publish(root)` consumescompleted/partial honestartifacts,C1/C2normaltransaction; `resume` verifiesinput/source/spec/remote andlaunchesnothingifidentical.

- [x] Implement independentstage dependencies and recordactual worker exits afterwait; partialinputs returnnonzero whileindependentstages/report/publishproceed. Explicitsinglephase failsmissingprerequisites, neveroldall. Separate numerical/reportdigest.
- [ ] Verify --help; runfourfocusedgroups once, affectedgroups onlyafterrepairs. Execute allwithexternalobserver; nooptimizers/encoders/SAM/geometry invocation; countallactualGPU/CPU/score/stat/render operations.
- [x] Publish compact sources/validids/availability/errorpaths/PRsufficientstatistics/confusions checkedagainstofficialAP; rawlargeOUT records bytes/SHA/regencmd. Fulltaskfilelimit/pathinventory once, nodroppingnegativecells.
- [x] Primary review every specsection1–16 againstactualfiles, stages,locks,tests,plots,counts,labels/null/exposure/lineage. Correctonlyaffecteditems; actualnovelty/positiveresultsnotprewritten.
- [ ] Observe scientificexitbeforeC1, publishnormalC1 andcheckfullremoteSHA; C2receipt/handoffonly,checkremote; externalfinalall/resumeactualexits andfinalaudit. Markgoalcompleteonlyfullcontractproven; missingmandatoryinputsremainpartial.

## Plan self-review

All numbered sections are assigned above. Dependencies allow optional E8 nonselection without suppressing fixed E/R/S; mandatory inputs remain explicit partial. Fixed source equality and score identity are separate. Statistical resampling uses unchanged scorer matches and unique namespaces, not averaged scene AP. The original prompt supplies the scientific design; no new hypothesis, train run, data expansion or deployment change is introduced. User authorized inline execution without renewed skill approval and C1/C2 normal pushes; no design-only early stop.

## Actual execution snapshot before publication

All 14 scientific phases are complete; the scientific-all process was externally observed with exit 0. Four focused groups passed once, with only affected groups repeated after recorded repairs. The public all coordinator and terminal published resume remain to be observed. 312 formal rows/24 pools, 60 DEV rows/15 pools, 100 physical rank assignments (including 7 retained from an interrupted prefix), 3324 frozen head calls, and 8000 paired bootstrap draws are recorded. No new training, RGB encoding, SAM proposals or map construction occurred. Primary review covers the actual result; C1/C2 publication and the final terminal-resume check will be recorded in the transaction receipts and external final review, without a third commit to rewrite this snapshot.
