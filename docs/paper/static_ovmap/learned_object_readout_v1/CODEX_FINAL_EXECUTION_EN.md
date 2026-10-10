# Final execution directive: learned multi-view object readout

## 0. Mission and authority

Implement, actually train, evaluate, select, document, and publish `ovimap-learned-object-readout-v1`. Work on `Orangekostar/oviovo`, from commit `30f59c1fa3d783054dc53fc6693e1fd68e8c2a6d`, on a new branch `research/ovimap-learned-object-readout-v1`. The intended improvement is better object semantics, NOT a lower runtime number. A slower candidate may pass. A fast candidate without accuracy/mechanism gains does not pass.

Read this file and the three contracts before coding. Copy this entire pack into `docs/paper/static_ovmap/learned_object_readout_v1/` and copy the unchanged numerical spec to `configs/static_ovmap/learned_object_readout_v1.json`. Resolve filesystem locations through explicit CLI overrides, not by altering scientific parameters. Record versions, effective paths and input identities once at stage boundaries.

This is a **new supervised experiment**, not another training-free inference wrapper. Do not claim the complete system is training-free. Do not introduce SAM-V, AnyUp, a new visual backbone, a trainable text encoder, geometry reconstruction, a new association pipeline, or a learned update-risk gate in this phase. A gate requires a separate supervised old/new decision corpus; the present new training families do not already have N/Q/D2/G1 predictions. Do not fabricate such a corpus or silently imitate it with ground truth.

The source is inspected but the server training inventory is not known. Data discovery/materialization is mandatory. If the required licensed data truly cannot be obtained, finish useful code and dependency documentation, publish a clearly blocked run, and return nonzero from `all`. Never turn missing training into training on CF18, random weights, copied baseline metrics, or a fake successful run.

## 1. Preserve the existing work

Use a new worktree. Resolve the local repository from the known parent worktrees/remotes. Fetch the exact base commit normally; do not reset, clean, stash, rebase or force-push anyone's existing work. Verify the remote is `Orangekostar/oviovo`. If the requested new branch already exists, resume that task only if its base/protocol identities match; otherwise stop with a concise conflict report.

Existing trained/frozen components, benchmark geometry and parent artifacts are read-only. Use the full source-update parent as the 26-scene source, not the four-scene disagreement or SAM-V probe as the baseline. Default parent is `/mnt/shared/ww/ovimap-source-preserving-update-v1/attempt_001`. Follow verified bindings to the minimal-repair and backbone captures, FC assets and original scoring context. Cache root hints include `/mnt/shared/ww/ovimap-disagreement-query-v1/attempt_001` and its capacity-store target. Honor existing `PathResolver` maps/symlinks.

The old `bind()` functions enforce older branch names, publications and counts. Build a task-owned binding adapter. Reuse their proven read-only `load_scene`, baseline validation, source identity and output/scorer primitives only with the structures they actually require. Do not weaken old checks or feed them invented metadata to bypass their schema.

## 2. Work packages and files to implement

Implement the public entrypoint `scripts/evaluation/run_ovimap_learned_object_readout.py` and a task module under `src/static_ovmap/learned_object_readout/`. Suggested files are responsibilities, not a demand for a large framework:

| Package | Production responsibilities | Existing source to inspect/reuse | Mandatory output |
|---|---|---|---|
| P0 Binding/inventory | Bind G1/D2, FC/text, benchmark cohorts; locate independent training assets | `disagreement_query/binding.py`, `source_preserving_update/binding.py`, parent HANDOFF | `source_binding.json`, `data_inventory.json`, `dependency_manifest.json` |
| P1 Data/splits | New-family roles and base/novel class lists; calibrated RGB-D export; GT-derived training proposals | `datasets/scannet200.py` only for its stated format; official ScanNet sensor and ScanNet200 annotation routines | `split_manifest.json`, `class_split.json`, `data_profile.json`, example manifests |
| P2 Features/observations | Frozen FC raw dense and pointwise projections, fixed views, masks, physical sites, train-only supervision | `region_adapter.py`, `area_fallback.py`, `disagreement_query/physical_support.py`, `minimal_instance_repair/observations.py` | immutable features and schemas, per-view identities, support statistics |
| P3 Models/train | Actual Mask-Adapter architecture port, matched VIEW/SURFACE models, optional-by-method auxiliary losses; fixed schedule | upstream pinned `mask_adapter_head.py`, `convnext.py`, `fcclip.py`; frozen numerical FC head | checkpoints, optimizer logs, DEV metrics, parameter/gradient checks |
| P4 Prediction/eval | F replacement on real G1; ordered benchmark pooling; standalone proposal recognition | `source_preserving_update/decisions.py`, `disagreement_query/outputs.py`, `m2_reviewer_study/evaluation.py`, `cvpr_compact/evaluation.py` | real complete payloads, 234+52 results and 18+4 pools, correction diagnostics |
| P5 Selection/report/publish | Freeze DEV nominations; preserve both seeds; three table families; normal push | parent publication patterns, not its old hardcoded counts | four MD reports, compact results, small learned weights, external publication receipt |

Use common numerical kernels in training, inference and the small tests. Do not maintain a “test-only model” different from the scientific model.

## 3. Fixed methods

Main seed 17 has nine benchmark paths:

1. `LR00_D2`: exact existing D2.
2. `LR01_G1`: exact existing full G1 v2.
3. `LR02_FC_2`: ordinary frozen FC, up to two views.
4. `LR03_FC_4`: same ordinary FC, up to four prefix views.
5. `LR04_FC_8`: same ordinary FC, up to eight prefix views.
6. `LR05_MA_8`: real Mask-Adapter head architecture, retrained on our data; independent per-view pooling, then equal feature aggregation.
7. `LR06_MV_VIEW`: same per-view head plus the specified hierarchical aggregation grouped by view.
8. `LR07_MV_SURFACE`: identical aggregation parameters/inputs, grouping the same local tokens by physical site instead.
9. `LR08_MV_AUX`: LR07 plus explicitly specified token-membership and same-site losses.

All nonparent methods replace only the historical F contribution, with identical N/Q/missing-source treatment. Same output eligibility, fixed 8-view bank and input features across methods. Do not add method-specific confidence vetoes or tune a new fusion coefficient. Test learned representation quality separately from its whole-map F replacement.

Repeat seed 29 is compulsory after DEV selection: train LR05 and the DEV-best proposed branch among LR06/07/08, from a shared seed-29 warmup. Identify that proposed branch before H or benchmark result access. No benchmark-conditioned seed selection, ensembling or switched dataset configurations.

## 4. Actual stages and ordering

### P0 — Bind, inventory and obtain data

First verify full parent baseline identities and real feature dimensions. Read reports to understand previous failures, but do not use their GT correction lists to choose new objects or fit parameters. Recover the original FC environment path from parent assets; use one available A40 with the existing GPU lock. Accuracy is not constrained by a seconds-per-scene limit.

Inventory data under explicit CLI roots, parent dataset roots, environment variables and their documented immediate `scans`/`scans_train` children. Do not crawl all disks. Read only scene filenames, calibration, split membership, file existence and completion until roles are locked. Materialize the required scenes from already-authorized raw exports or `.sens` and annotations. An existing authorized downloader may be used for the frozen scene list; do not accept terms on the user's behalf, invent access, or download the whole dataset.

The exact data procedure, 24/4/4 family split, withheld classes, camera mapping and failures are in DATA_AND_SPLITS.md. No smaller profile is authorized. Missing data is a dependency failure, not negative evidence about the method.

### P1 — Freeze splits and data generation

H annotations, H-derived masks and H features are first constructed only after both seed nominations, inside the `holdout` stage. Initial `data`/`features` can inventory its files but cannot parse its annotations.

Write family exclusions, fixed role lists and class lists before producing model outputs. Record annotation supervision explicitly. All models train on exactly the same examples and corruption schedule. On DEV/H, the target label is separate from the model input; their proposals are GT-derived diagnostic inputs and must be labeled that way. They are NOT new independent full-map benchmarks.

The initial `features` stage processes TRAIN/DEV only (and may prepare unlabeled regression inputs); it does not open H annotations. For existing Replica8/CF18 build masks, views and local sites from the predicted G1/D2/capture only. No target annotations, target-mesh semantic labels, best-IoU lookup, or already-known TP lists enter their inference manifests. The evaluator's frozen nearest-target mapping is used only for evaluation/output ranks as already required by the protocol, never to sample neural inputs.

### P2 — Establish feature and gradient correctness

Use the exact existing FP32 FC preprocessing and checkpoint. Capture raw `clip_vis_dense` with `torch.no_grad()`; persist ordinary CPU tensors, not autograd-restricted inference tensors. Save projected dense U using the actual upstream `_2d` normalization/drop/head helper, and sparse token raw values if needed, with code/checkpoint/dtype/tensor-content identity. Share each image across all objects and methods.

Check actual raw channel dimension C and projected text dimension D. Do not trust stale source comments (e.g. “640”) to size layers. D must equal the bound text matrix width and the upstream large-head expected width; any adaptation must be recorded before training, not covered by `strict=False`.

Test a real new head backward pass through the frozen FC visual projection: new-head gradients must be finite/nonzero, frozen encoder/head/text parameters must remain unchanged. `.requires_grad_(False)` on the FC head is correct; wrapping its use during training in `no_grad()`/`inference_mode()` or converting pooled tensors to numpy is not.

Port only Mask-Adapter numerical head and ConvNeXt blocks, preserving notices; registry/config wrappers can be removed. Compare the port's output and input/parameter gradients once against the pinned numerical source initialized with the same weights on a small tensor. Do not run the full upstream segmentor or substitute unrelated pretrained Mask-Adapter weights.

Run an engineering fit on two TRAIN objects for 20 optimizer steps, discarded afterwards. Validate actual parameter movement, finite gradients/loss and a lower end loss on this tiny set. Do not use this engineering fit to choose hyperparameters. A failure justifies debugging implementation or data; a real valid negative scientific result does not.

### P3 — Fixed training, checkpoint choice and seed repeat

Execute the schedule in MODEL_AND_TRAINING_CONTRACT.md: seed-17 MA warmup 2,000 steps, then four 2,000-step branches; seed-29 warmup and two nominated branches add 6,000. Total 16,000 scientific optimizer updates, with the same effective batch and draw schedule within each seed. Resume optimizer, sampler, RNG, step and data identities exactly after interruption.

At fixed validation steps choose each checkpoint only by equal-family mean base-class DEV loss. Select the proposed branch and make both seed nominations before H and existing benchmark annotation access. Store the full ranking and tie-breaks. Do not extend training or switch learning rate because a Replica/CF result is disappointing.

OOM engineering responses may reduce microbatch and increase accumulation while retaining effective batch (default already microbatch one), or use activation checkpointing/offload unchanged arithmetic. Do not shrink resolution, views, sites, data or network after outcomes. A real unsupported resource error is documented rather than silently changing the method.

### P4 — Holdout recognition and complete exposed benchmark regression

Evaluate all seven nonparent methods on H clean/corrupted proposals using the same full 200-class frozen text table; report base, heldout-class and all-class metrics with exact denominators. H labels are accessed once after DEV nominations; do not retune afterward. Unavailable heldout-category examples yield null subgroup metrics, never invented zero-shot evidence.

Freeze all scientific predictors and construct every main whole-map output before evaluating Replica8/CF18. Evaluate all nine main paths even if a trained branch looks unimpressive on DEV: this first supervised phase must show whether learned representations transfer. Run the two repeat branches on the same 26 maps separately. There are 234 main scene-method records/18 pools and 52 repeat records/4 pools, 286/22 if complete. Aliasing truly identical scorer inputs is allowed but every logical condition must have an identity proof. No means of scene AP substituted for pooled AP.

Report standalone F classification, final map metrics, and per-object new/old errors; a better learned feature can be hidden or harmed by its D2 integration and these are different findings. Diagnose, do not add a post-hoc integration sweep.

### P5 — Accuracy-first conclusion and real publication

Write the strict/material gates from unrounded full metrics. A passing seed-17 method is a research upgrade candidate; the DEV nomination and the seed-29 repeat are reported independently. A claimed proposed-mechanism success additionally needs improvement over LR05 / matched VIEW grouping, not just over damaged controls. A slower result is not rejected for speed.

Collect actual new encodings, model/step counts, stage wall times and allocated/reserved peaks during executed work. No new large cold-timing grid, online FPS claim or cross-hardware ratio. Make exactly three main table families, plus machine-readable supplementary detail.

Execute the complete `--phase all --resume` CLI and observe its terminal exit. Publish real source, configs, compact data/metric/decision manifests, both seed nominations, learned small-head checkpoints and four reports. Exclude raw licensed scans, GT maps, frozen backbone weights and huge dense tensors. Normal commit/push; compare full local and remote SHA afterward; write external `publication/final.json`. Do not put a commit's own future SHA inside that commit.

## 5. CLI to implement

Support `--spec`, `--parent-root`, `--observation-root`, `--output-root`, `--storage-root`, repeatable `--data-root`, `--path-map`, `--gpu`, `--phase`, `--resume`. The phases are `bind`, `data`, `features`, `engineer`, `train`, `nominate`, `repeat`, `holdout`, `predict`, `evaluate`, `report`, `all`.

After resolving authorized data roots, the actual reproducible command must be written to HANDOFF, for example:

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python \
  scripts/evaluation/run_ovimap_learned_object_readout.py \
  --spec configs/static_ovmap/learned_object_readout_v1.json \
  --parent-root /mnt/shared/ww/ovimap-source-preserving-update-v1/attempt_001 \
  --observation-root /mnt/shared/ww/ovimap-disagreement-query-v1/attempt_001 \
  --output-root /mnt/shared/ww/ovimap-learned-object-readout-v1/attempt_001 \
  --phase all --resume
```

The controller may spawn training/FC workers in the recorded FC environment, and evaluation in the original mapping environment. The example is an entrypoint to implement, not an assertion that it already exists. Resolved data paths go into the binding and the printed reproduction command, not an invented universal `/data/scannet` path.

## 6. Finite validation, no audit overbuilding

Required test groups: family/class separation; RGB/depth and axis transforms; normalized feature sampling; upstream-head port parity; frozen-head gradient; clean/corrupt supervision isolation; permutation/group distinction; missing-view masking; real whole-output/rank parity; resume/checkpoint roundtrip; full ordered-pool coverage. One or two small tests per group suffice. Reuse relevant parent evaluator checks, do not run the entire repository or create thousands of per-element “requirements”.

Keep one dependency manifest, one split manifest, one run registry, ordinary per-stage receipts and checkpoints. Do not build a new framework for signing every minibatch. Count scientific experiments, training updates and tests separately.

Failure status is precise: `BLOCKED_TRAINING_DATA`, `BLOCKED_CALIBRATION`, `BLOCKED_RUNTIME`, `PARTIAL_RESULTS`, or completed `NO_TARGET_GAIN`/`TARGET_MET`. A completed negative result is publishable and uses exit 0; missing mandatory stages use nonzero. Missing metrics are null with a reason. Never mark a training dry-run as a scientifically trained head.
