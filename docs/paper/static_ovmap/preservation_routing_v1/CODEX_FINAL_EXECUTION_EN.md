# Final execution directive — preservation and routing v1

## 0. Task, authority, and boundaries

Implement and actually execute `ovimap-preservation-routing-v1` in `Orangekostar/oviovo`, starting from commit `6767eb90c2fd6999621b356b87269c012821713a`, on **`research/ovimap-preservation-routing-v1`**. Deliver trained heads, selection, complete applicable evaluations, four reports and a verified normal GitHub push. Accuracy and interpretable mechanism evidence are the objectives; runtime is recorded, not optimized or used to reject a candidate.

This pack supersedes the *proposed next experiment*, not the numerical record of the completed LR run. Keep that run read-only. Do not alter LR's splits, trained heads, archived H results, or checks to make the new run appear to be a resumed old experiment. Do not claim three universally effective innovations in advance.

Read all three contracts before coding. Copy the pack under `docs/paper/static_ovmap/preservation_routing_v1/`, and copy the unchanged spec to `configs/static_ovmap/preservation_routing_v1.json`. Implement `scripts/evaluation/run_ovimap_preservation_routing.py` and task code under `src/static_ovmap/preservation_routing/`. Shared production kernels must implement training, inference and tests; no easier test-only substitute for the model.

### The three scientific questions

1. Can learned 2D region readout preserve and improve the same-input frozen FC, instead of damaging it?
2. On a qualified, **frozen** 2D base, does physical-site grouping help beyond matched view grouping?
3. Does explicit group-level routing make membership supervision reduce contamination and improve recognition, beyond the same supervision without direct routing?

A slower method can pass. A mathematically well-behaved but inaccurate method cannot pass merely because its property tests succeed.

## 1. Bind actual inputs before judging or training

Use a new worktree, normal fetch, and the exact base. Never reset/clean/rebase someone else's worktree, force push, overwrite parent artifacts, or search unrelated accounts/disks. Resume an existing task branch only when base, protocol and data identities match.

Default completed LR parent:
`/mnt/shared/ww/ovimap-learned-object-readout-v1/attempt_001`.
Resolve its capacity-store symlink/path map; known storage hint:
`/mnt/shared/capacity/node101/ww/ovimap-learned-object-readout-v1/attempt_001`.

Bind its source/result stores, split/class manifests, feature manifests, scene proposals, selected-head registry, DEV/H records and execution/publication receipts. Check observed parent 286/22 coverage and 10,000+6,000 completed updates from actual receipts. Do not infer current local cleanliness from a remote report. If a newer publication-only commit exists, require identical bound scientific input identities; never silently select newer trained outputs.

Follow the LR binding to full source-update/minimal-repair/backbone inputs for real D2/G1 maps. Do not use four-scene SAM-V or disagreement data as the full baseline. Reuse the actual read-only loaders and scoring primitives, but implement a new binding adapter: old `bind`, `fixed_spec`, runner and coverage functions hardcode older branches/registries.

Record one dependency manifest, one method registry, and ordinary stage receipts. Hash large immutable files once and reuse a validated stat/content memo. No signatures for every minibatch or thousands of per-element “requirements.”

## 2. Production work packages

| Work package | Files/responsibility in new module | Existing source to inspect/reuse | Required output |
|---|---|---|---|
| P0 binding + previous-run audit | `binding.py`, `parent_audit.py` | LR binding, reporting, training; parent receipts | `source_binding.json`, compact prior curves, missing-log report |
| P1 data + teachers | `data.py`, `teachers.py` | LR ObjectLoader, FrozenFC, data, inventory, sensor | exact parent TRAIN/DEV manifests, teacher scores/correctness sidecars, H2 plan |
| P2 2D preservation | `models.py`, `losses.py`, `training.py` | LR NONE numerical MA path and frozen phi | R0–R3 trained heads/curves, first-step gradients, Rstar nomination |
| P3 group routing | same model/loss kernels + `mechanisms.py` | LR local token metadata and grouping | five matched G arms per enabled seed, routing interventions |
| P4 transfer/evaluation | `recognition.py`, `prediction.py`, `evaluation.py` | LR recognition/prediction/evaluation, released scorer | old H, H2, real-proposal diagnostic, real map outputs/pools |
| P5 publication | `reporting.py`, runner | parent normal publication pattern | tables, checkpoints, all negative controls, reports, receipt |

Use the paths and symbols listed in SOURCE_EVIDENCE. New filenames are proposed responsibilities, not claims that those files already exist.

## 3. Stage P0: review the failed experiment without repeating it

Publish retained parent `updates.jsonl`, per-checkpoint `dev_*.json` and selected/last/warmup metadata in a compact derivative, with actual loss-component availability. Missing parent component logs are **not recoverable by inventing values**. A missing optional old log does not block new training; document it.

Using parent TRAIN and DEV only, evaluate the ordinary FC reference and, where available, the parent shared warmup and selected heads to locate the loss of recognition. Parent H is already exposed, not a new confirmation set. Do not let old H or the maps pick the new checkpoint.

A tiny frozen-key numerical intervention must demonstrate the old within-group common-shift cancellation and the new absolute routing response. This supports a mechanistic hypothesis, not a proof that it explains all previous accuracy loss.

## 4. Stage P1: exact data reuse and finite engineering checks

Keep exactly the parent 24 TRAIN families, four DEV families, 160/40 class IDs, camera conventions, eight-view banks, corruption suite, physical sites and FP32 feature values. Do not redraw the split or train on all785 proposals: only parent base-positive objects (reported622) enter category training. Count actual IDs, not augmented examples.

Bind teacher embeddings for clean and corrupt observations at prefixes2/4/8 using the existing ordinary FC vectors. Teacher correctness is computed on **TRAIN only** with the full200 frozen vocabulary and the clean input of the same prefix. The teacher is not the parent trained MA, not D2, and not a benchmark-specific GT lookup. Teacher masks affect training losses only; at inference they do not exist.

Reserve four H2 scene families by the deterministic rule in DATA_AND_EXECUTION, before outcomes; only inventory names/calibration then. No H2 semantic parsing or sample qualification until all model nominations are locked.

Check real tensor dimensions C1536/D768, preprocessing, v2 pooling, text row order, differentiability through frozen phi, and parent FC8 reproduction. Use two fixed TRAIN objects for at most20 discarded optimization updates; reset all scientific initialization/RNG afterward. Parent feature caches can be reused; no new basis model is required.

## 5. Stage P2: the four R arms

Use fresh MA initialization (same numerical upstream architecture), **no failed parent MA warmup**. All four arms have equal trainable shapes and common object draws:

- `PR_R0_DIRECT`: direct MA, common base loss.
- `PR_R1_RESIDUAL`: function-preserving FC residual, same base loss.
- `PR_R2_KEEP`: direct MA, conditional teacher-preservation loss.
- `PR_R3_RESIDUAL_KEEP`: FC residual plus conditional teacher preservation.

The exact residual is `normalize(z_FC + (z_MA_trainable - z_MA_initial_frozen))`. The frozen initial MA is an exact initialization copy; it is not updated. This preserves FC at initialization without a zero output multiplier that blocks all MA gradients. Direct arms use the same trainable MA but do not use the FC skip or subtractor. Frozen-reference state is checkpointed and reported as inference state, not hidden trainable capacity.

Run each seed17 R arm for2,000 optimizer updates, total8,000. Save losses and full200 DEV top1, macro recall, clean accuracy, correction/harm counts and base160 CE at fixed checkpoints. Step0 is an identity diagnostic, never an eligible trained nomination.

Nominate Rstar only by the exact DEV gate in EVALUATION_AND_RELEASE. No viable R arm -> finish R/parent diagnostics, report `COMPLETE_NO_2D_FOUNDATION`, publish; do not start G, H2 acquisition, real proposals or new whole-map evaluation.

If a seed17 Rstar passes, freeze its architecture and train only that architecture from fresh seed29 initialization for2,000 updates. Select its checkpoint by the same DEV rule. If it fails, report `COMPLETE_2D_NOT_REPEATED` and publish. Do not swap to another seed29 architecture or label it a runtime block. These are intentional completed negative paths, exit0.

## 6. Stage P3: five matched geometry arms, only on a viable base

Only after Rstar passes independently in both seeds, freeze each seed's Rstar and train the five G arms for2,000 updates each. All use identical token sets, quality/membership head shapes and draw schedules within the seed:

- `PR_G1_VIEW`: view grouping, no membership/correspondence supervision, no membership routing.
- `PR_G2_SURFACE`: site grouping, otherwise same.
- `PR_G3_AUX_ONLY`: site grouping plus membership supervision, but membership output does not directly gate a group.
- `PR_G4_ROUTED`: same asG3 plus explicit group-level membership routing.
- `PR_G5_CORRESP`: G4 plus the predeclared same-site correspondence loss.

Every arm uses the same count-normalized null route, bounded object-attention logits and zero-initialized linear residual output. Ungated controls set effective rho to1. Only G4/G5 use predicted rho. This isolates the use of membership from extra layer count and group count.

Do not put LayerNorm/L2 normalization or a nonzero bias after summing the gated local residual: it would undo amplitude rejection. Missing local support returns the frozen base exactly. Freeze the base in both eval mode and requires_grad; never update it behind the controls.

Seed17 chooses one G architecture on DEV, or chooses Rstar if none improves. Repeat **all five** G architectures in seed29 so VIEW–SURFACE and AUX_ONLY–ROUTED have paired repeats. Never choose a different proposed architecture because seed29/H2/maps prefer it. Total path cap is30,000 scientific updates:8,000 R-main +2,000 R-repeat +20,000 G.

## 7. Freeze, then perform real diagnostics and transfer evaluation

After all selected checkpoints/nominations and hashes lock:

1. Evaluate all selected trained heads and FC8 on old H as an exposed regression, not selection.
2. Apply finite routing/teacher/purity/correspondence interventions; report no-op and unknown denominators.
3. Generate real, GT-free SAM2 proposals on exactly two predetermined frames from each existing DEV family, at most8 images/128 retained proposals. This is a **diagnostic generator**, not a new mapping method or training set. Use the specified pinned AMG API; do not use GT clicks, GT boxes, oracle unions or choose masks using GT. Lock proposals and view inputs before matching to annotations.
4. Materialize/parse the four prelocked H2 families, run the unchanged GT-derived proposal generator, and evaluate all selected heads. H2 is proposal recognition, not independent whole-map AP.
5. Construct all applicable full-map predictions on the same existing G1 partition. Only eligible existing incumbent labels may change. G1 recovery labels, unknown rows, raw-zero protected owners, geometry and target projection remain unchanged.
6. Recompute current-class official ranks; evaluate one actual output for instance and semantic metrics; perform released ordered pooling.

H2 technical unavailability must not cancel valid training/maps: finish what is available and publish `PARTIAL_CONFIRMATION_BLOCKED`, with nonzero `all` exit and null missing outcomes. Sparse genuine H2 novel-class support is a limitation, not a technical failure or permission to replace families. Real-proposal generator errors similarly cannot be reported as successful empty predictions; genuinely zero or few proposals are a valid coverage outcome.

## 8. Evaluation coverage and selection

On the full path, seed17 main map paths are D2,G1,FC8,Rstar,and fiveG =9 methods =234 rows/18 pools. Seed29 contributes Rstar+fiveG =156 rows/12 pools. **Maximum390 rows/30 pools.** G0 is Rstar, not another scientific head; aliases must be explicit. Other three R-arm results belong to proposal-level tables and are never erased; their whole-map expansion is outside this fixed study. Parent FC2/FC4 and LR05–08 may be shown separately as historical results with correct protocol labels.

The original whole-map gate remains: both full cohorts' five metrics nondecreasing vsG1; CF APall>D2 and AP50>=D2, fraction tolerance1e-10. Material CF APall gain is0.001 fraction aboveD2. Report DEV nomination, exploratory benchmark winner and seed-repeat status separately. No dataset-conditioned methods, teacher-oracle output, seed ensemble, post-hoc temperature sweep or changed evaluation vocabulary.

A negative G mechanism is a valid result even when Rstar passes. Passing a numerical map gate does not automatically establish novelty. Direct pair evidence and repetition must support each claim.

## 9. Minimal validation, real resumability

Use at most12 focused test groups specified in MODEL_CONTRACT and the checklist. Include real two-object CUDA checks and two-scene output/scorer integration. Reuse relevant parent evaluator tests; do not rerun whole repositories, redo official model training, or build a new auditing framework.

Resume model/reference/base, optimizer, LR step, sampler position and all RNGs exactly. Save last every100 updates and on a recoverable interrupt; retain selected/last, small DEV curves and initial reference, not a checkpoint for every minibatch. If a crash loses uncheckpointed work, record discarded/replayed update counts separately from successful scientific updates. Unrecorded failure time is null, not zero. No deletion of parent data after ENOSPC.

A required input missing, nonfinite model output or exception is not a no-op method fallback. Record stage-specific `BLOCKED_DATA`, `BLOCKED_MODEL_INTERFACE`, `BLOCKED_RUNTIME`, `PARTIAL_EVALUATION` or `PARTIAL_CONFIRMATION_BLOCKED`. Valid completed negative scientific paths use exit0; missing mandatory activated stages use nonzero.

## 10. CLI and actual publication

Support `--spec`, `--parent-root`, `--output-root`, `--storage-root`, `--path-map`, `--gpu`, repeatable `--data-root`, `--sam2-root`, `--sam2-python`, `--phase`, `--resume`. Phase order is the spec's registry. Old parent FC/controller/SAM2 environments are separate; never globally upgrade them. An example command to implement:

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python -u \
  scripts/evaluation/run_ovimap_preservation_routing.py \
  --spec configs/static_ovmap/preservation_routing_v1.json \
  --parent-root /mnt/shared/ww/ovimap-learned-object-readout-v1/attempt_001 \
  --output-root /mnt/shared/ww/ovimap-preservation-routing-v1/attempt_001 \
  --phase all --resume
```

Write the **actual resolved command** to HANDOFF. Observe complete CLI exit in a subprocess; never infer it from individual receipts. Commit source, spec, tables, compact decisions/embeddings/curves, selected trained weights including necessary frozen initial/base dependencies, and:

- `PRESERVATION_ROUTING_RESULTS.md`
- `PRESERVATION_ROUTING_HANDOFF.md`
- `PRESERVATION_ROUTING_SELECTION.md`
- `PRESERVATION_ROUTING_CLAIMS.md`

Include negatives and missing-stage reasons. No raw licensed scans, GT maps, huge dense grids, or foundation-model weights in Git. Retain external absolute paths plus portable hash manifests. Run a finite checkpoint-load/sample-score roundtrip before release. Normal commit and push to the requested branch; read full local HEAD and `git ls-remote` for that exact branch, compare, and only then write external `publication/final.json` as PUSH_VERIFIED. A commit must not purport to contain its own future SHA. If push fails, preserve deliverables and report the actual failure; do not claim upload completed.
