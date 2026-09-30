# Final Codex execution instruction — OVI-MAP backbone wave-1

**Task:** implement, measure, select, document, and publish `ovimap-backbone-wave1-v1`.
**Repository:** `Orangekostar/oviovo`.
**Starting commit:** `1ce806b22c63943843300006b5b1034ec9e1cb0b`.
**Working branch:** `research/ovimap-backbone-wave1-v1`.
**Upstream:** `OVI-MAP/OVI-MAP@f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424` plus the existing native-v10 capture patch.

Read this file, `PROTOCOL_SPEC.json`, `IMPLEMENTATION_CONTRACTS.md`, and `SOURCE_EVIDENCE.md` before editing. JSON fixes numeric constants and experiment IDs; the contracts fix algorithm semantics; this instruction fixes orchestration. They are intended to agree. Resolve a genuine contradiction before launching affected jobs, record the correction, and do not improvise a result-dependent interpretation. The Chinese review and the earlier roadmap are explanatory, not overriding specifications.

## 1. Research objective and bounded scope

Improve **the formation of 3D instances** while retaining OVI's RGB-D/posed TSDF representation and the useful N/Q/FC semantic pipeline. This is not another cached-score fusion experiment. New front-end masks, association decisions, owner partitions, and sometimes surface coordinates are allowed to change. The final category rules are fixed.

Answer four questions:

1. Does simultaneous RGB-instance/depth fusion improve over the mutation-order-dependent native procedure?
2. Does object-level association improve over the native procedure and the much simpler activation of its already-computed overlap-ratio predicate?
3. Does genuinely directional mutual matching add value over the same object-level forward matcher?
4. Do temporally propagated masks, optionally filtered by past RGB-D consistency, improve the actual reconstructed map rather than only a recognition crop?

This wave executes **B00, B01, B05 and B03**. It does not execute the entire twelve-item roadmap. B04 full OVRCOAT, B02 small-fragment retention, B06–B12, SAM3, new learned quality heads, depth completion, Gaussian reconstruction, and conditional-information query selection are **DEFERRED_OUT_OF_SCOPE**. The already acquired SAM2 implementation is deliberately selected over an untested full OVRCOAT wrapper. SAM2 is a control/implementation tool, not relabeled as a new 2026 paper.

The seven independently screened configurations are:

| ID | Single change from the native map recipe |
|---|---|
| `BB00_NATIVE` | No scientific change; new task's reference map and semantic readouts |
| `BB01_SYNC` | Simultaneous depth-instance fusion, fixed original thresholds |
| `BB05_RATIO_GATE` | Activate the existing native overlap-ratio eligibility predicate only |
| `BB05_FORWARD` | Object-conditioned superpoint association with directional forward matching |
| `BB05_BIDIR` | Identical machinery to FORWARD, but retain only forward/backward agreed pairs |
| `BB03_SAM2_RAW` | CropFormer seeds and bounded SAM2 temporal propagation enter reconstruction |
| `BB03_SAM2_GEOM` | The same cached raw propagation, with the specified RGB-D consistency filter |

Allow exactly one additional configuration, `BBX_COMPOSE`, combining the development-selected structural and front-end candidates. No full Cartesian product, no new parameter sweep after looking at results.

Negative or unchanged results are valid. A smoke run, a changed config file, or a nonzero intermediate probability difference is not a completed scientific experiment. All seven configurations must produce real development maps and metrics, or have specific, truthful technical block records.

## 2. Read the actual code and bind the real runtime

Use the source map in `SOURCE_EVIDENCE.md`. In particular, read:

- User repo: `module_validation/scannet_runtime.py`, `native_capture.py`, `scannet_study.py`, `evaluation.py`, `query_study.py`, `query_pipeline.py`, `semantic_study.py`, `region_evidence.py`; `a7_evidence_upgrade/region_adapter.py`, `region_worker.py`, `recognition_worker.py`; `paired_evidence_study/residuals.py`, `evidence.py`; `m2_reviewer_study/evaluation.py`; the native-v10 patch and relevant handoffs.
- Upstream: `scripts/panoptic_mapping_.py`, `utils/common_scannet_nyu.py`, `view_selection.py`; `GlobalSegmentMap_py::integrateFrame`; `computeSegmentLabelCandidatesConfidence`, `decideLabelPointCloudsConfidence`, `getNextSegmentLabelPairWithConfidence`, `updateInstanceConfidence`, `updateLabelClassInstanceConfidence`, `IncreaseSegGraphConfidence`, `mergeLabelConfidence`; `SemanticInstanceLabelFusion::getInstanceLabel`; instance mesh export and the read-only raycaster.

### Corrections to the earlier roadmap — mandatory

**Keep `inst_association=4`.** Mode4 initializes and updates the segment graph, but its ordinary instance lookup uses label/class/instance counts. Mode3 is not the only graph-enabled mode. Changing to3 would change the baseline and the meaning of the intervention.

**Do not conflate superpoints and objects.** A depth fragment has a global superpoint label; several superpoints can belong to one object. The native graph already groups same-current-instance fragments when adding edges. The proposed change is a *whole-local-object association decision before/around fragment label assignment*, not a claim that the old graph never groups fragments.

**The native ratio predicate is currently unused.** In `getNextSegmentLabelPairWithConfidence`, `ratio_greater_than_min` is computed but not in the accepting `if`. Preserve that behavior in BB00 and all variants except BB05_RATIO_GATE. Do not silently repair every arm.

**Do not reuse the old paired runner.** It disables CUDA and accepts an old branch/scene registry. Likewise, old capture runners enforce the old patch hash and frozen-geometry identity. Implement task-local wrappers instead of deleting historical checks or presenting new geometry under an old identity.

### Binding paths

Resolve actual inputs from these receipts, not guessed filenames:

- `/mnt/shared/ww/ovimap-a7-evidence-upgrade-wave1/attempt_001/binding.json`
- `/mnt/shared/ww/ovimap-paired-evidence-v1/attempt_001/binding.json`
- `configs/evaluation/ovimap_module_scannet_runtime.json`
- `configs/evaluation/ovimap_module_scannet_study.json`
- `/mnt/shared/ww/ovimap-module-validation-v1/data/scannet/acquisition_lock.json`
- The wave-1 `assets/{sam2,fc_frozen,ovrcoat}/download_receipt.json`, `workers.json`, and `environments/*.json`.

The known old native extension is under `.../tooling/repairs-20260922/native-query-v10/lib/`; never overwrite it. Its historical upstream checkout is `/home/ww/crove/ovimap-module-validation-upstream`. The new experiment must use an isolated upstream checkout and a new native build location from the spec.

Create `resolved_inputs.json` once before measurements. Record scene schedules, RGB/depth/camera identity, original CropFormer and geometric-mask roots, source/config/code/model identities, loaded extension path, native modes, Q model/standardizer/budget, source temperatures, evaluation vocabulary, and runtime executables. A missing file is a concrete prerequisite failure; an unexamined path is not. Check direct paths first, then the explicit receipt chain and a bounded search within the known study roots. Never scan all user storage or open protected confirmation targets to find substitutes.

## 3. Data, compute and experiments to run

### Development cohort

Run the seven fixed configurations on:

`scene0056_00`, `scene0534_00`, `scene0445_00`, `scene0626_00`.

These four scenes are **historically exposed development data**, not new validation data. Use each scene's original 200-slot schedule and invalid-pose handling from the locked export/capture. No shortening difficult scenes, no choosing representative frames after GT inspection, and no extra frames for the SAM arm.

After development selection, run at most four distinct map configurations on the eight historical Replica scenes:

`office0`, `office1`, `office2`, `office3`, `office4`, `room0`, `room1`, `room2`.

The four possible configurations are BB00, the structural candidate, the front-end candidate, and BBX_COMPOSE. Use the original per-scene frame lists, calibration and vocabulary. Replica is a frozen **historical transfer/regression** phase; do not retune after seeing it. Do not open `scene0553_00` or `scene0064_00` or call an old confirmation leaf to enlarge this wave. No new-scene confirmation is authorized by this package.

### Scope and resource budget

The maximum is 64 successful map–scene configurations: 7×4 development, at most4 composition maps, and at most4×8 Replica maps. Each has three final readouts, so at most192 primary official scene/readout rows; secondary ranks, diagnostic traces, imported historical controls, retries and aliases do not increase the independent experiment count.

Use two CPU mapping jobs with eight native threads each by default; evaluate with three CPU jobs/four BLAS threads. One neural worker per designated free GPU. Preserve existing shared GPU locks and other users' processes. Runtime concurrency may be reduced to fit memory without changing the scientific recipe; record it. No network training, Q retraining, source temperature refit, new backbone pretraining, or unbounded model search.

New GPU inference is expected for changed masks and new temporal predictions. Share exact raw SAM predictions between RAW and GEOM. Reuse actual content-identical model requests; do not report reused work as new physical inference. A missing persistent FC dense feature map cannot be reconstructed from an already pooled vector.

For an installation/runtime failure, perform one targeted repair/retry per failure signature, record it, then isolate the affected family. Do not spend the task repeatedly rebuilding environments or auditing all historical artifacts. Genuine core failures remain blocked; completing other branches must not turn them into COMPLETE.

## 4. Implementation layout and phase interface

Create `src/static_ovmap/backbone_wave1/` with clear separation of:

`binding.py`, `runtime.py`, `fusion.py`, `association.py`, `frontend_sam2.py`, `capture.py`, `semantic_readout.py`, `evaluation.py`, `selection.py`, `reporting.py`.

Names are new task-local files, not claims about pre-existing APIs. Add `scripts/evaluation/run_ovimap_backbone_wave1.py` with phases:

`bind, prepare, bridge, diagnose, screen, compose, freeze, transfer, report, publish, all`.

Required arguments: `--spec`, `--output-root`, `--phase`, `--resume`, `--gpu`, `--mapping-workers`, `--mapping-threads`, `--evaluation-workers`. An optional explicit path-remapping JSON may relocate proven roots; it must not change scientific file identities or overwrite old paths.

`all` executes the full available DAG through reports and publication, even when measurements show no gain. `report` and `publish` consume completed receipts and never relaunch inference. `resume` reuses exact completed jobs and restarts an interrupted TSDF scene from the beginning after preserving its failed attempt; do not invent a native checkpoint capability.

Create a scoped native patch under `third_party_patches/ovimap/backbone_wave1_v1/`, layered on native-v10. Build a new extension against the already available compatible dependencies. Set import/library paths explicitly and record `consistent_gsm.__file__` plus binary identity in each map receipt. An unused source edit or a new binary that is not loaded fails intervention verification.

Commit implementation/configuration before the main measurements. Record implementation commit(s) and actual consumed code hashes. A later bug fix invalidates only affected downstream jobs; keep earlier failed/retired attempts distinct.

## 5. Required map interventions

Implement the algorithms exactly as defined in `IMPLEMENTATION_CONTRACTS.md`.

### BB00 / B00

Reproduce native depth fusion, association mode4, TSDF parameters and frame schedule. A task-local probe/capture with all scientific switches off must not update map evidence. Capture raw geometric segments before the100-pixel gate, fused fragments, current local-group membership, native superpoint/object decisions, current map owner projections, final raw owner surfaces and official-export surfaces.

Run one shared order diagnostic on the first three valid scheduled frames of every development scene: native order, reversed candidate visitation, and bijectively relabeled two-dimensional IDs. Compare canonical spatial partitions, grouping and confidence, not the numeric IDs. This is a diagnostic, not three extra benchmark methods.

Measure *where* candidate support is lost or merged, after outputs are locked. Never use final GT overlap to admit a fragment or force an instance match.

### BB01_SYNC

Use immutable depth-segment support and area to decide all split-qualified 2D intersections. Emit those disjoint intersections together, subtract their union once, recompute the remaining candidates on the actual residual, and apply the original .2 assignment rule. Keep .9/.5 thresholds, score values and initial100-pixel gate. Do not add a new minimum-size gate to each carved intersection. Make tie/order handling spatially deterministic and document this as part of the simultaneous procedure.

An ID permutation alone is not geometry improvement. If canonical support is unchanged, report that rather than manufacturing an intervention from renamed labels.

### BB05_RATIO_GATE

Keep the native greedy fragment matcher and every other component. Add the already-computed `ratio_greater_than_min` to the existing accepting predicate. Use the original bound configuration value; do not substitute .2 for this native parameter. This arm answers whether a minimal eligibility change is enough.

### BB05_FORWARD and BB05_BIDIR

Construct local objects by grouping current retained fragments that share a nonzero input instance label. Query **the map before inserting the current frame** using the new read-only, depth-conditioned probe. Use mode4 count-based owner lookup at factor0, consistent with the ordinary raycaster; do not substitute mode3 graph owner IDs or final-surface IDs.

Compute actual directional coverages, solve positive-benefit bipartite assignments with dummy unmatched nodes, and use either forward pairs or the mutual intersection. Forward and backward differ in their denominators. Solving a symmetric cost matrix and its transpose is not an acceptable implementation of the bidirectional condition.

Enforce the chosen object decision through superpoint candidate generation/recomputation, the mode4 object-count update and compatible alias merging. Keep separate superpoint labels for distinct depth fragments. Allocate one native new object ID per unmatched local object. Do not silently omit unmatched objects, reuse a conflicting historical superpoint, or merely change an exported color table. Do not overwrite historical count tables to force a result.

Keep the native within-local-instance graph update once per frame. Structural controls remain comparable: both directional arms use identical probe/candidate machinery and merge compatibility. The added coupling itself is part of FORWARD; BIDIR minus FORWARD isolates mutual agreement. Log planned pairs, actual assigned labels, count-owner membership, blocked candidate/alias operations, and eventual realized partitions.

### BB03_SAM2_RAW / BB03_SAM2_GEOM

Use the existing author SAM2 checkpoint and pinned `facebookresearch/sam2@2b90b9f5ceec907a1c18123530e92e794ad901a4` code/config from the wave-1 receipt. Use the actual video predictor, not repeated independent box prediction.

Process fixed chunks of five valid scheduled frames. Seed all nonzero CropFormer masks on the first frame only; output the original CropFormer raster on that first frame. For later frames, propagate forward to the current frame only, using `max_frame_num_to_track=0` at the pinned API. No future masks, reverse propagation, GT/human prompts, final labels, or extra camera frames. Bound state by resetting at chunk boundaries. RAW/GEOM share identical predictor inputs and raw neural outputs.

Retain genuinely uncovered current CropFormer detections according to the fixed rule, so seed-only propagation is not confused with new-object discovery. GEOM applies a past-depth/pose consistency filter and specified current-frame CropFormer fallback to the same RAW predictions; it does not feed corrected masks back into SAM. Feed the final exclusive instance rasters into `frameToSegmentsCropFormer` and reconstruct the map. Keep foreground `inst_score=1.0` in this wave; passing SAM confidence into TSDF would be an additional intervention.

If SAM assets/API remain unusable after the bounded repair, finish and publish the structural measurements with the SAM arms marked blocked and the full-wave scientific conclusion incomplete. Do not silently run the untested full OVRCOAT wrapper as a replacement or call the entire requested wave complete.

## 6. Fresh geometry requires fresh semantic evidence

For **each** reconstructed map, produce a new native anchor and then the three readouts:

- `NATIVE_READOUT`: original SigLIP selection, six-crop features, top10 saved order/last8 aggregation and original canonical-relative category decision.
- `FC_EQ`: original cosine N, Q_GAIN and FC probabilities averaged across genuinely available sources.
- `D2`: average available N/Q inside their group, then average available groups with FC; all-present weights .25/.25/.5.

Do not attach D2 to old G0 geometry and describe it as a new backbone. Do not copy per-owner scores, view lists or old projected masks by owner number. A task-local map→capture→requests→features→sources DAG is required.

Native observations may be acquired through a deferred worker after CPU mapping if the exact selection metadata, request images/masks/boxes, ordering, final color lookup and failed-request handling reproduce native behavior. The native skip-feature path does not save a completed feature pickle; add an explicit metadata product and reconstruct the pickle, rather than inventing the missing file. The bridge phase must compare this deferred readout with historical native evidence. If equivalence is not established, use the real native worker under the isolated map recipe instead; do not change the classifier to make numbers match.

Run the frozen Q_GAIN policy with budget200 attempted requests, its bound predictor and standardizer, fresh per-map causal lineage and state. Debit logical requests before feature access, even on a physical cache hit. Preserve current-frame-only decision inputs. Reconcile final ownership using actual alias/segment evidence, not a best-IoU match to GT.

FC uses the frozen `convnext_large_d_320`, original14 templates, original mask pooling, area aggregation, at most3 static requests per target and the original128-target policy. Re-run selection on the **new** native lineage. Keep the actual historical static-manifest contract: follow its generator/source metadata, including any recorded fresh-mask route. For new development scenes use that same recipe, not an alternate top-k heuristic. Record cap exclusions and view availability; do not pretend a cap/ID-order artifact is pure geometry gain.

Reuse original leave-one-parent-CAL-scene-out scalar temperatures for scene0056_00/scene0534_00; use original final scalars on scene0445_00/scene0626_00 and Replica. Do not fit temperatures to the new geometries. The frozen Q model, source encoders and text prototypes are not model-selection knobs in this wave.

Preserve the native triangle-first-color surface paint and eligibility rules in the official main experiment. Export numeric raw owners separately for geometry diagnosis. Otherwise a switch from native painting to numerical labels can create an unacknowledged export improvement. A G0 bridge exposes such discrepancies.

## 7. Metrics and model/configuration selection

### Official primary results

Use `m2_reviewer_study.evaluation`'s released evaluator, `official_view`, confusion construction and `pool` semantics through a task-local adapter. Instantiate it against the **new** map's anchor/configuration and unique mask root. Never use the old Evidence object's geometry for a new map.

Regenerate Open3D float32 1NN projection when xyz/order changes, with strict squared distance `<0.05**2`. Even if coordinates happen to be unchanged, masks and geometry provenance must be correctly rebound. Read the actual released overlap vector; it is historically .50–.90 plus separate .25, not an assumed COCO .50–.95 vector.

Report APall, AP50, AP25, mIoU and mAcc as fractions in machine data and percent/percentage-point deltas in reports. Pool the complete ordered scene files in the released evaluator and sum semantic confusion matrices. A mean of scene APs is a separate descriptive quantity, never relabeled official pooling.

Also report frozen-native ranking **within each new map** to show semantic ranking effects. It is not possible to apply G0's owner-rank table to an unrelated new partition. Main selection uses official current-class ranking only.

### Required geometric and intervention diagnostics

Use raw numeric owner surfaces, independently of whether native semantics admitted an owner. Report class-agnostic maximum-one-to-one recall at .25/.5/.75, per-GT best IoU, fragmentation and contamination summaries, and observed-surface precision/completeness/F-score at5cm. Define valid GT/ignore/min-size denominators from the released protocol; retain all eligible GT objects in recall denominators, including unobserved/missed objects. These are diagnostics, not replacements for official semantic AP.

Report front-end changed-pixel/candidate coverage, changed fragment supports and grouping, proposal/accepted/realized association counts, new/merged/split owners, cap/semantic availability changes, added/removed evaluator matches and false positives. Across different maps compare instances by predicted spatial overlap and separate GT-indexed diagnostics; never compare equal owner numbers as if they were persistent correspondence.

For BB01, report order sensitivity; for association, compare BIDIR→FORWARD→RATIO_GATE→BB00; for SAM, compare GEOM→RAW→BB00. An intermediate decision change that leaves the final map unchanged is `NO_EFFECTIVE_MAP_INTERVENTION`, not a gain.

### Frozen development selection

No continuous parameter fitting or extra grid is allowed. Select by D2 official **four-scene development pool**, not two CAL scenes alone and not Replica. All four are explicitly reused development scenes; this does not establish independent validation.

Apply the deterministic preference rule:

1. Keep candidates within .05 percentage points of maximum APall.
2. Among them keep candidates within .10pp of maximum mIoU.
3. Among them keep candidates within .10pp of maximum AP50.
4. Break remaining ties by fewer standalone-required image encodings, lower median attributable standalone map-plus-readout time, fewer altered functional blocks, then method ID. Include the shared raw SAM prerequisite in both RAW and GEOM's standalone accounting; a cache hit is not zero standalone method cost. Report this derived timing separately from actual invocation time.

These are declared engineering preferences, not confidence intervals or statistical equivalence tests. Always publish exact metric differences. Undefined metrics are null with reasons; do not replace them by0 or silently drop an entire unsuccessful scene.

Choose the structural candidate from the complete noncontrol structural arms, and the front-end candidate from complete SAM arms. A candidate may be negative relative to G0. If both have genuine canonical map interventions, run their one prescribed composition on all four development scenes. Structural and front-end candidates are frozen **before** the composition results; no second combination or reversed selection search.

Freeze the final research nominee among G0, the two candidates and the optional composition. Transfer that whole distinct set, up to4 configurations, to Replica. Publish all seven arms' development rows and all transferred rows. A negative transfer result does not authorize changing the earlier nominee. Deployment remains `N0_UNCHANGED`.

## 8. Execution order and stopping rules

- **bind:** resolve known inputs and explicit scene roles; create the manifest and job matrix.
- **prepare:** implement/refactor scoped code; layer and build the new native patch; one real SAM preflight; run targeted fixtures and record real loaded binaries/models. Do not access new GT for model readiness checks.
- **bridge:** run BB00 on scene0056_00 through map, source reconstruction, all3 readouts and official evaluation. Compare exact recipe/request behavior with historical evidence. Hash mismatches due to code identity are not output mismatches. Differences beyond .05pp trigger one targeted diagnosis of inputs, binary, ordering, export and precision. Do not silently refit. An unresolved incompatible baseline blocks *comparative claims* and must be visible; actual correctly specified remaining jobs may be archived, but cannot be reported as verified historical replication.
- **diagnose:** shared B00 frame-order and loss-path records for the development cohort; real annotations only after predictions are locked.
- **screen:** all seven maps × four scenes, all3 readouts; no need to wait for a positive diagnostic to run an already authorized arm. Reuse BB00's completed bridge job.
- **compose:** select the two candidates and run at most one composition as specified. Skip only for explicit missing prerequisite or no real intervention; not merely because a candidate loses by a small amount.
- **freeze:** commit the development decision/configuration identities before Replica results are produced.
- **transfer:** up to four fixed maps × eight Replica scenes, each with fresh sources. Do not shorten this to one favorable room.
- **report:** complete measured tables, mechanisms, limits, errors, actual costs and code provenance.
- **publish:** scoped commit, actual ordinary GitHub push and full SHA comparison.

When a job fails, mark the smallest dependent branch blocked, finish unrelated executable jobs, and publish partial results honestly. Never infer a result for a missing scene from a related run. Do not skip difficult scientific computation and spend the saved time on hundreds of safety tests.

## 9. Validation proportional to the task

Use at most about12 targeted new test functions, plus one real short native trace and the already required complete bridge scene. Parameterized cases are fine. Focus on:

- native OFF behavior; simultaneous partition/grouping and empty residuals;
- a real directional assignment counterexample and unmatched handling;
- read-only pre-insertion probe and active mode4 enforcement through recomputation/merges;
- no future SAM prompts and exactly-current-frame propagation;
- fresh geometry/owner/projection identities, legitimate cache aliases and readout equivalence;
- official pooled evaluator parity and dynamic report generation.

Do not rerun the whole old199-test suite, generic fuzz/security suites, all16k historic release-file hashes, or repeated repository-wide lint passes. Run lint/compile on modified Python and compile the actual native target. Read/hash each consumed large asset once per process/job and cache its result with metadata checks; do not waive identity verification, but do not turn it into the dominant workload.

The reference fixtures supplied with this instruction package are not native/GPU validation and do not count as completed experiments.

## 10. Deliverables and GitHub publication

Create these reports in `docs/paper/static_ovmap/`:

- `BACKBONE_WAVE1_RESULTS.md`: every completed development/transfer condition; official pools, scene deltas, geometric evidence, control comparisons, actual failures and scope.
- `BACKBONE_WAVE1_HANDOFF.md`: real paths, environments, native patch/build, source code commit(s), CLI commands, resumption instructions, missing prerequisites and external-artifact restoration.
- `BACKBONE_WAVE1_SELECTION.md`: pre-Replica candidates, composition and nominee, all selection inputs and tie steps; no rewritten historical nomination.
- `BACKBONE_WAVE1_CLAIMS.md`: supported, unsupported and untested mechanism claims; exposed scenes and extra compute explicitly acknowledged.

Under `artifacts/static_ovmap/backbone_wave1_v1/`, publish compact:
`resolved_inputs`, `experiment_matrix`, `bridge_parity`, per-scene metrics and pools, `selection`, `intervention_summary`, event ledgers, per-class summaries/confusion matrices, costs, scoped tests, code/model/patch identities and `external_artifacts.json`.

Keep raw RGB-D, foundation weights, full TSDF/surface arrays, dense maps and large per-frame probability tensors on shared storage. Record real paths, bytes, hashes and recreation commands. Compress repetitive JSON. Aim for a compact release around100MiB or less, but do not spend hours optimizing an arbitrary archive limit; do not put individual >95MiB files in ordinary Git.

Reports must compute label/partition/probability difference statements from outputs, not use fixed success prose. Separate implementation, experimental coverage, scientific conclusion and publication status. Use `COMPLETE_NO_NET_GAIN` for a fully measured negative study, not for a technically blocked study. Only positive actual geometric evidence supports a stronger-backbone claim; AP changes from semantic coverage/ranking alone must be identified.

Use ordinary scoped Git operations. Never commit credentials, datasets or unrelated local modifications, and never force push. Commit code/spec first and results/reports after measurements; a local-only code commit does not satisfy publication.

After the final normal push, compare:

```bash
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git ls-remote --heads origin refs/heads/research/ovimap-backbone-wave1-v1 | awk '{print $1}')
test "$LOCAL" = "$REMOTE"
```

Write the external publication receipt to `<output_root>/publication/final.json` with actual branch, full local and remote SHA, status, UTC check time, and release path. Do not commit a receipt requiring its own commit hash. If authentication/network blocks the push, record `BLOCKED_PUSH_AUTH` or the actual cause; never claim `PUSH_VERIFIED` from local success alone. Reuse existing connected credentials without printing them.

## 11. Final response required from Codex

Return the full remote-verified SHA and branch, actual scope completed/blocked, development nomination and frozen Replica results for NATIVE_READOUT/FC_EQ/D2, geometric and semantic deltas versus the new BB00 reference, which simple control each mechanism did or did not beat, actual new model forwards and wall/CPU/GPU timing categories, concise test results and links to all four reports. State explicitly that all study scenes were previously exposed and deployment was not replaced.

Finish the executable scientific work and publication in the current task. Do not stop after planning, smoke tests or synthetic kernels; do not invent missing results; do not continue an unlimited repair/tuning loop to obtain a positive result.


## 12. Bootstrap and exact phase commands

Discover the existing user repository through the verified historical worktree, then create the isolated worktree from the pinned commit. If the named branch/worktree already exists, resume only after its task/spec identity matches; do not reset or overwrite unrelated work. Use existing authenticated `origin`, not embedded credentials. Copy this package's normative MD/JSON files into `docs/paper/static_ovmap/backbone_wave1_v1/`; keep the package fixtures clearly separate from native study tests.

After implementing the new runner, the intended command is:

```bash
cd /home/ww/crove/ovimap-backbone-wave1
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_backbone_wave1.py \
  --phase all \
  --spec docs/paper/static_ovmap/backbone_wave1_v1/PROTOCOL_SPEC.json \
  --output-root /mnt/shared/ww/ovimap-backbone-wave1-v1/attempt_001 \
  --gpu 2 --mapping-workers 2 --mapping-threads 8 --evaluation-workers 3 --resume
```

For a completed/resumable task, use the same arguments with `--phase report` or `--phase publish`; neither phase may launch map/model jobs. Protect the coordinator with one nonblocking task lock; child jobs own their actual per-GPU/output locks. The command above is an interface to be implemented, not a claim that the runner already exists in the source commit.

Source edits made during a failed measurement require an explicit new code identity and rerun of affected dependencies. Resume completed independent variants; do not archive the whole study for a local report formatting fix. Include all failures, repairs and retries in actual costs and never let a restart erase their logs.
