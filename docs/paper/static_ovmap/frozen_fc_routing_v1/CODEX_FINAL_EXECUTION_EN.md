# Final execution directive — frozen FC + local 3D evidence routing

## 0. Mission, fixed scope and authority

Implement, run, evaluate, select, document and publish `ovimap-frozen-fc-routing-v1` in `Orangekostar/oviovo`. Branch from **76db115aa7df652a0b2dc4575df52e6a74a63549** into **research/ovimap-frozen-fc-routing-v1**. The previous PR experiment completed with no qualified R head; that is accepted evidence, **not a prerequisite failure for this task**.

The scientific question is whether local 3D evidence selection can improve an ordinary, frozen FC object readout, and whether target-membership routing improves on matched learned controls rather than merely reducing total update size.

Keep RGB-D inputs, object banks, fixed surface geometry, FC checkpoint/phi/text, N/Q and original D2 integration. No MA training or inference in the new FG forward, no R checkpoint as base, no 768x768 trainable value transform, no AnyUp, SAM-V, new backbone, view search, teacher-preservation loss, correspondence loss, geometry reconstruction, dataset-conditioned configuration or extra gate. SAM2 is allowed only for the already specified **post-lock real-proposal diagnostic**.

Read all contracts. Copy this complete pack into `docs/paper/static_ovmap/frozen_fc_routing_v1/` and unchanged PROTOCOL_SPEC.json into `configs/static_ovmap/frozen_fc_routing_v1.json`. Numerical choices are frozen experimental settings, not optimality claims. Filesystem relocation uses explicit CLI/path maps; it must not change method settings.

## 1. Non-destructive source and dependency binding

Use a new worktree; locate the user's repository through known worktrees/origin, not a disk-wide crawl. Verify origin and base ancestry, fetch normally, and preserve existing work. No reset/clean/stash/rebase/force push. An existing matching branch/output can resume; a different task identity is a conflict, not permission to overwrite.

`--parent-root` refers to the completed **PR** run, normally `/mnt/shared/ww/ovimap-preservation-routing-v1/attempt_001`. Its source_binding points to the **LR** run (`6767eb90...`) and then the **SU**/minimal/backbone inputs. PR has no new full-map results. Resolve the actual LR full baselines and banks; do not demand nonexistent PR whole-map results or successful R nominations.

Write a task-owned binder. Do not invoke old `bind`, `train_G`, `require_transfer`, `runner.all_status` or fixed-count evaluators. They enforce old branch names, R qualifications and 390-row protocols. Reuse verified read-only loaders and scoring kernels, with real aliases for the SAME parent baselines. Do not forge nomination receipts or weaken old checks. No old parent checkout is required to remain at an obsolete HEAD if its consumed immutable artifacts are verified.

Resolve one A40 from explicit/inherited GPU identity and test real CUDA. The parent reports an NVML mismatch; a failing nvidia-smi call alone is not reason to block a functioning CUDA worker. Keep controller, FC and conditional SAM2 environments isolated; no global package upgrade. Use the existing GPU file lock in a blocking queue, rather than terminating other jobs.

## 2. Work packages and concrete implementation

Create `scripts/evaluation/run_ovimap_frozen_fc_routing.py` and `src/static_ovmap/frozen_fc_routing/`. Keep modules small; responsibilities are more important than matching every suggested filename.

| Package | Actual implementation | Reuse / inspect | Required output |
|---|---|---|---|
| P0 binding | PR -> LR -> source-update inputs, fixed dataset/classes, real FC identity | PR binding/data; LR features; full LR receipts | source_binding.json, dependency_manifest.json, run_registry.json |
| P1 diagnostics/data | fixed parent R error audit, same-input FC bank, D1 label-assisted local filtering | PR recognition/selection; separate target sidecars | prior_audit/, diagnostic_D1/, data_manifest.json |
| P2 production model | FrozenFCBase and four identical-shaped FG heads, explicit eta, routing interventions | PR models.py numerical group structure; LR FrozenFC.phi | model.py, model_config.json, actual gradient/identity checks |
| P3 train/select | four fresh arms; exact same draws per seed; DEV gates | LR draw_schedule/optimizer/scheduler; PR summarize/compare | eight or four arm receipts, curves/checkpoints, seed nominations |
| P4 diagnostics/transfer | locked heads, old-H, matched-mass and matched-norm tests; conditional maps/H2/real proposals | PR transfer/evaluation/H2/proposals numeric routines | locked embeddings, complete evaluation, honest coverage |
| P5 release | actual weights, tables, handoff and observed normal push | parent publication conventions only | four reports, compact artifacts, external publication/final.json |

Use production numerical functions in tests, training and inference. The delivered reference implementation is an equation check, not a parallel scientific model. Do not silently replace `phi(weighted raw)` with weighted projected vectors.

## 3. Always-executed sequence

1. **bind/prepare:** resolve immutable parent assets; exact 24 TRAIN / 4 DEV / old-H roles and 160/40 class IDs; no resampling or new supervised data. Copy PR's H2 names and real-proposal frame/settings plan. Check only names/availability for H2 before outcome lock.
2. **prior_audit:** use stored PR R1/R3 and FC scores to decompose corrections/harm by margin, category support and family. Do not fit anything or create test-object whitelists.
3. **engineer:** use two TRAIN objects for finite CPU/CUDA, graph, phi gradient and output checks. Up to20 discarded updates. Reproduce actual ordinary FC8 vectors; verify literal zero-relabel G1 output and original scorer on two scenes. This is not new scientific map evidence.
4. **diagnostic_D1:** run the specified ALL versus known-target-only local support intervention on TRAIN/DEV base objects. GT is diagnostic only. Its result never changes the method matrix, hyperparameters or mandatory main training.
5. **train:** run FG1_VIEW, FG2_SURFACE, FG3_AUX, FG4_ROUTE, seed17, **2,000 optimizer updates each**. Do not wait for or load an Rstar. All four main arms run even if a previous arm is negative.
6. **nominate:** evaluate the fixed checkpoints, choose per-arm qualified best or best diagnostic, then choose one main DEV nominee. Do not inspect newly evaluated old-H/H2/maps before nomination.
7. **repeat:** if any main arm qualifies, train ALL FOUR arms with seed29, 2,000 each. Keep the seed17 architecture nominee unchanged; seed29 independently chooses its checkpoint by the same DEV rule. A different repeat winner does not replace the locked nominee.
8. **lock:** write exact selected heads, code/spec, dataset/banks and query/diagnostic plans. Full transfer activates only when the SAME nominee qualifies in both seeds.
9. **diagnose:** for every trained selected head, report TRAIN/DEV base performance, all-class post-lock strata, exposed old-H recognition, correction/harm and FG4 interventions. This executes even after a completed negative main screen. No labels become neural inputs.
10. **predict/evaluate:** only if full transfer is active, run actual 26-map predictions and official evaluation for the fixed 7 main plus4 repeat rows. Freeze all map predictions before new evaluation. No conditional per-scene switch.
11. **confirm:** only if full transfer active, execute exact H2 and real-proposal checks with frozen heads. Do not tune using results. A confirmation dependency failure must not suppress already valid map results; continue reporting with nonzero partial status.
12. **report:** publish completed positive or negative path, all executed branch results, interventions, counts, compact heads, component curves, exact command and unexecuted-stage reasons.

## 4. CLI to implement and actually invoke

Support `--spec`, `--parent-root`, `--output-root`, optional `--storage-root`, `--path-map`, `--gpu`, repeatable `--data-root`, `--sam2-root`, `--sam2-python`, `--phase`, `--resume`. Phases are the exact list in PROTOCOL_SPEC.json. `all` follows the graph above, including negative paths and final report. FC/training workers run in the bound FC environment; scoring runs in the inherited mapping environment.

Example **target CLI, not an already existing script**:

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python -u \
  scripts/evaluation/run_ovimap_frozen_fc_routing.py \
  --spec configs/static_ovmap/frozen_fc_routing_v1.json \
  --parent-root /mnt/shared/ww/ovimap-preservation-routing-v1/attempt_001 \
  --output-root /mnt/shared/ww/ovimap-frozen-fc-routing-v1/attempt_001 \
  --gpu 0 --phase all --resume
```

The handoff must replace paths/GPU with actually resolved values. Existing path-map/capacity aliases are valid; do not edit numerical settings for relocation. If `all` encounters an active mandatory stage failure, retain partial results and return1. If all applicable work completes with no signal, return0 with the correct negative status. A logger printing COMPLETE is not an observed subprocess exit.

## 5. Frozen model/training decisions

Use MODEL_AND_TRAINING_CONTRACT.md exactly. Same 216,194-parameter architecture in all four arms; inactive membership parameters are disclosed, not counted as evidence that supervision was applied. No global learned value/projection matrix. Shared fixed local vectors are formed by the real nonlinear frozen phi AFTER learned raw pooling. Maintain its input autograd.

Data draws: original LR uniform-present-class then object schedule, prefixes2/4/8, fixed corruption cycle, effective batch16. AdamW1e-4, original100-update LR warmup/cosine, clip1, FP32, no TF32/autocast. Fresh head init uses seed+1009 identically for all arms; membership final linear is zero initialized in all arms (initial rho0.5); queries Normal(0,.02). No extra training warmup arm.

Eta uses completed updates, starts0 and ramps to0.5 by200. Inference uses0.5. The diagnostic step0 uses eta0 by an explicit override and is never selected. The first scientific step has eta0 and lr0; zero classification movement then is expected, not a defect to repair by changing the schedule.

## 6. Selection and interpretation

DEV base labels only, full200-class competition. Qualification: A >= FC8 A + .005 -1e-10; M and C >= FC8 -1e-10; positive net corrections. Use the exact inherited A/M/C definitions. Best checkpoint prioritizes qualified candidates then A/M/C descending, CE ascending, earlier step; nomination ties prefer FG1, FG2, FG3, FG4. No speed objective, post-hoc H selection or best-seed averaging.

No seed17 qualification -> COMPLETE_NO_FG_SIGNAL (8,000 steps). Main signal without same nominee repeat -> COMPLETE_FG_NOT_REPEATED (16,000 steps). Both qualify -> full frozen regression and confirmation (16,000 steps). No statistical significance is implied by these deterministic gates.

Whole-map research upgrade requires both cohorts all5 >= current G1; CF APall > D2 +1e-10 and AP50 >= D2 -1e-10. Material signal is another .001 APall fraction above D2. Per-method flags are reported by seed. A method other than the DEV nominee that passes maps is an exposed exploratory result, not the preconfirmed nominee. Deployment stays N0_UNCHANGED.

## 7. Finite checks and completion

Twelve compact groups suffice: lineage/splits; real frozen-FC identity; phi input gradients; eta0/missing/zero-rho identity; group masking/permutations; matched init/draws; membership-loss isolation; mass/norm intervention math; selection/state graph; actual map output/rank/scorer parity; checkpoint/resume; exported load and artifact coverage. Do not run the entire repository, build an auditing framework, hash every minibatch tensor or count thousands of tensor entries as tests.

Do keep ordinary component logs and one-time expensive-input identities. Do not discard failed attempts or unknown timing fields. Corrupted inputs/cache identities require correction; actual negative metrics do not authorize tuning.

## 8. Required final publication

Reports: FROZEN_FC_ROUTING_RESULTS.md, FROZEN_FC_ROUTING_HANDOFF.md, FROZEN_FC_ROUTING_SELECTION.md, FROZEN_FC_ROUTING_CLAIMS.md in docs/paper/static_ovmap/. Three table families: map performance; mechanism/recognition; coverage/transfer/cost. Keep matched-mass and matched-norm interventions explicitly diagnostic, not new trained winners.

Publish code, configs, source bindings, selected head weights for every trained arm/seed, architecture/FC dependencies, full scalar training/component/DEV curves, compact embeddings/scores/decisions and results. No learned MA or frozen initial MA dependency belongs in an FG checkpoint. Licensed scans, GT maps, foundation weights and large dense caches remain external; never publish credentials.

Actually execute `all --resume`, observe the exit in a parent shell/process, and archive the terminal log. Commit the real artifacts, normal push to the new branch, query full local and remote HEAD, and only then write external publication/final.json. Its status is PUSH_VERIFIED only for equal full SHAs. Do not place a commit's own future SHA inside that commit. If push fails, report publication failure separately from scientific completion. Never claim a plan, smoke fit or cached inherited metric is a new trained scientific result.
