# Data, lineage and conditional transfer

## 1. Three parent levels, different roles

1. **PR immediate parent**, publication76db115...: completed four R arms/8,000 updates and no new scientific whole-map results. Use its diagnostics, path maps, H2 names and real-proposal settings. Default logical root `/mnt/shared/ww/ovimap-preservation-routing-v1/attempt_001`; documented physical root `/mnt/shared/capacity/node101/ww/ovimap-preservation-routing-v1/attempt_001`.
2. **LR data/map parent**, publication6767eb90...: full24/4/4 family data, FC grids, predicted-map8-view banks, 286 rows/22 pools. Resolve from PR source_binding.lr_parent_root. Its full-map FC8 is the exact reference for this task's same-input baseline.
3. **SU/minimal/backbone ancestors**: original G1/D2 payloads, N/Q/F scores, frozen FC/text/temperatures and released evaluators. Resolve through the actual LR binding. Never substitute a four-scene probe.

Validate the immediate parent's status, observed exit and external publication receipt using the actual current schema. A scientific-negative status is accepted. Verify equal local_HEAD/remote_HEAD and matching publication's result identity, not merely a README SHA. If external paths moved, use PathResolver/CLI overrides and check immutable contents; do not require that old checkout HEADs remain frozen or demand a nonexistent PR full-map table.

Write identities into the NEW root. Avoid helpers that could rewrite parent manifests or memo files. Exact read-only copies/symlinks of data and feature manifests are permitted; their embedded file paths must resolve correctly. Old `capture_dataset` may reject new task IDs even when features are identical; use a task-owned read-only cache adapter rather than fabricating old binding identities.

## 2. Required reuse

Copy exact LR split_manifest/class_split and consumed feature identities. Verify:

- TRAIN:24 families,785 total originals,622 base originals,74 actually observed positive categories.
- DEV:4 families,87 total,69 base originals,28 base classes.
- old_H:4 families,101 originals,already exposed.
-160 base and40 heldout class IDs, ordered200-ID frozen text table, no rehash or redefinition.

Exact role names are in PROTOCOL_SPEC.json and must equal LR records. Per-object four condition masks/sites and2/4/8 view prefixes are unchanged. No resampling sites, masks, new family acquisition for training, changed perturbation severity or different classification crop.

Use parent registered depth-grid RGB for TRAIN/DEV/oldH/H2. The map bank retains its original image preprocessing/convention. Record the distinction; do not silently realign old benchmark imagery. All training heads face the same distinction.

Feature keys include raw source image/tensor, checkpoint/phi, dtype, dimensions and original coordinate transform. Existing cached values can be reused after identity verification. A missing immutable cache may be restored or recomputed on the SAME source input with the SAME operator, with real cost counted. Do not make a different image, altered view or another checkpoint look like a cache hit. A missing true source is a dependency failure.

The new head does not need dense projected U. Reusing the old loader is allowed for minimum implementation risk; a sparse per-object cache optimization is allowed only after real output and phi-input-gradient parity, and must be shared by all arms. It is not a new experiment.

## 3. Neural inputs vs supervision

The forward API is tensor-only, with the eight named keys in the spec. `class_id`, family, owner ID, proposal condition label, target mask/purity, GT surface ID and teacher correctness never enter it. Data selection follows existing geometry/mask rules, never GT correction lists.

Membership sidecars enter the TRAIN loss only for FG3/FG4. In diagnostics, GT can compute target-only filtering, purity and correction labels, but those methods are tagged `GT_DIAGNOSTIC_ONLY`, excluded from nomination and never exported as deployed predictions. D1 must not choose a different set of objects or alter the training plan.

H/DEV proposal recognition uses GT-derived support as an openly declared diagnostic input; it is not a fully predicted map pipeline. Existing map inference uses predicted supports only. Benchmark nearest-target mapping is used for frozen scoring/ranks, not for neural sampling or object selection.

## 4. H2 plan and data access

Copy the exact PR `H2_plan.json` (including selected names and real_proposal_frames). Verify it was not consumed for training/tuning by the known parent chain or another discoverable run in the finite lineage. Do not claim global absence from all possible experiments. If names were previously exposed, retain them and mark confirmation as exposed; do not search for a favorable replacement.

Before main training, bind names and source-file availability only. Do not parse H2 annotations or build its semantic inputs. Full transfer activates ONLY after seed17 nominee and its seed29 instance qualify and all checkpoints are locked. Then execute four selected families via the same calibrated LR `prepare_family` generator, with up to96 frames/family and64 objects/family. No scene replacement based on object counts, labels or results.

Existing authorized ScanNet acquisition primitives may obtain exactly these names; do not accept new terms for the user or download the whole dataset. The parent authorization/path records are evidence to verify, not a promise that every file is currently available. Required raw suffixes and calibration remain those in LR inventory. Reuse exact available data; copy only small manifests to the new root.

H2 recognition evaluates FC8 and all eight trained heads using full200 text over the same objects and four conditions. Report base/heldout/all, original counts, families, actual views, no-op conditions, per-condition Top-1/macro/NLL and corrections. Fewer than32 originals or fewer than20 novel originals across2 families is insufficient support, not a reason to replace scenes. Metrics on present objects remain real; strength-of-confirmation status is limited.

H2 is **new-family proposal recognition**, never new whole-map AP. Missing requested input/calibration means an active-stage failure (nonzero partial); successful acquisition with sparse eligible objects is a completed but limited measurement. Continue full-map reporting even when confirmation dependencies fail.

## 5. Real predicted-proposal diagnostic

Activate on the same full-transfer lock. Reuse the exact eight prelocked DEV images (two per family), SAM2.1 Hiera-L checkpoint and parent PR numerical `data.real_proposal` settings. This is an input-transfer check, not the new backbone, a training set, or a benchmark selector. Preserve the SAM2 environment and source pin2b90b9f5... separately from FC. Use PR sam2_worker and numerical lifting routines, NOT its R-gated dispatcher.

Automatic grid prompts only; no GT boxes/points. Retain at most16 masks per image according to the original generator-quality ordering/tie breaks, before GT matching. This is a retained mask cap, not a cap on raw generated masks. Lock masks and all target-free lifted inputs first. Project predicted masks to mesh rows using the existing routine; require two qualified reprojection views. Do not repair poor matches by an oracle union, new GT grouping, or hand-picked points.

Run FC8 and all eight heads on every common supported proposal. Only afterward match to official eligible GT by class-agnostic best IoU, with existing min100 vertices and strict IoU>.25 for defined class targets; also report >.5 separately. Undefined or unsupported proposals stay in coverage statistics, not silently dropped. This matched-subset classification is not detector AP or independent full-map evaluation.

The parent exact-row lifting can produce few multi-view proposals; record that limitation. Zero successful supported proposals from a successful generator is valid limited evidence, not fake zero accuracy or permission to change the lifting method here. Generator crashes, missing weights and nonfinite outputs are actual failures. Ground-truth-free input construction and model predictions must finish before label matching.

H2 permits at most384 new FC images; the real-proposal context bank permits at most384, normally fewer because parent data are reused. These counts exclude restored parent caches, retries and engineering, which are reported separately. Do not hide real extra work inside a bound labeled only scientific.

## 6. Parent stages not to call blindly

- `preservation_routing.bind`: expects a completed LR publication as immediate parent and PR schema.
- `geometry_training.train_G`: requires two qualified Rstar receipts and loads an R checkpoint.
- `holdout2.require_transfer`: checks PR R/G nomination schema.
- `transfer.lock`: has390-row expectations and qualified-R logic.
- `evaluation.run/evaluate_scene`: fixed15 methods and390/30 counts.
- `real_proposals.run`: dispatches only after PR transfer and loads PR heads.

Create thin task-owned orchestration adapting these contracts to real FG identities, while retaining tested numerical sensor, projection, pooling and scorer functions. Never create dummy R receipts to pass these conditions.
