# Observation-Query Real-Training V2 Handoff

## Status

- Branch: `research/ovi-rescene-observation-query-train-v2`
- Evidence base and evaluated Git head: `9879915092f8b0f90d89d64b86c74f36aedc7a17`
- Evaluated state: base commit plus exact source bindings in checkpoint manifests
  and evaluation summaries
- Delivery commit: recorded by the post-push local/remote SHA check
- Implementation: `IMPLEMENTATION_READY`
- Smoke: `REAL_TRAIN_SMOKE_COMPLETE`
- Experiment: `THREE_WAY_PILOT_COMPLETE`
- Scientific signal: `SCIENTIFIC_SIGNAL_NEGATIVE`

## Data

TRAIN is official-train environment
`02b33dfb-be2b-2d54-92d2-cd012b2b3c40`, pair
`scene0001_00-scene0001_01`. Both official `sequence.zip` files were downloaded,
materialized to 143/114 RGB-D frames, mapped independently with native OVI-MAP,
and converted into a 61,359-point model bundle, 385-region ObservationBank, and
source-bound training targets. DEV is different validation environment
`20c993b7-698f-29c5-847d-c8cb8a685f5a`, pair
`scene0109_00-scene0109_01`, and is explicitly exposed.

Raw 3RScan RGB-D, processed arrays, frozen third-party checkpoints, backbone
caches, optimizer/RNG state, and prediction NPZ files remain local. The Git
delivery contains receipts, per-run CSV/JSON evidence, compact trainable states,
and deterministic visualizations, but no restricted raw data.

## Implementation

- Full-GT targets are no longer clipped to reconstruction support; zero-TP F1 is
  numeric zero. Full, common, camera-diagnostic, and raw domains remain separate.
- Observation preparation accepts an explicit role and pair rather than using a
  hard-coded DEV source. The thin training-preparation CLI reuses the existing
  label-transfer implementation.
- TRAIN provenance is validated independently from DEV model input and
  ObservationBank identities, enabling legitimate TRAIN-to-DEV evaluation.
- Cumulative resume restores optimizer and RNG/sampler state, emits periodic
  snapshots and curves, preserves failed-run evidence, and checks checkpoint
  replay with explicit floating-point tolerances.

## Training And Evaluation

`OBS_BASE_TUNED`, `OBS_FUSE`, and `OBS_FULL` each reached 1,000 real optimizer
updates with seed 45. FULL used a verified 200-update smoke plus 800-update
resume. The public model folders contain both terminal comparison checkpoints
and the selected DEV checkpoints. All ten frozen/200/500/1000 DEV evaluations
are `PASS` under `OVI_OBSERVATION_QUERY_INSTANCE_V2`.

Each model folder also contains the literal training config used by the run,
with SHA-256 `b5ff6ed04c8b922fcd34498deaadd2d199ec9e737a954c443ebda03c8c10f405`.
The top-level `pilot_training_v2.json` preserves the mathematical training
configuration but updates provenance, split, and metric naming; use the archived
file when exact input-byte provenance matters.

The primary `t1` Common F1@.50 is 0.1379 frozen, 0.1429 for best BASE at update
200, 0.1379 for best FUSE at update 200, and 0.1379 for best FULL at update 500.
At the common 1,000-update endpoint it is 0.1379 / 0.1290 / 0.0690 / 0.0690.
Full-GT F1@.50 and identity TP remain zero throughout. See
`observation_query_training_v2_results.md` and `run_index.json` for the complete
claim boundary and artifact map.

One initial 200-update FULL diagnostic completed training but failed an overly
strict exact-equality reload gate. The maximum replay difference was
`3.814697265625e-6`; it is retained locally and excluded from formal results.
The verified rerun uses `rtol=1e-5`, `atol=1e-6` and passed.

## Reproduction

Create `configs/observation_query/runtime.local.json` from the portable example,
then prepare the TRAIN target:

```bash
PYTHONPATH=. /home/ww/miniconda3/envs/persist4d/bin/python \
  scripts/training/prepare_ovi_observation_training.py \
  --runtime-config configs/observation_query/runtime.local.json \
  --pilot-config configs/observation_query/pilot_training_v2.json \
  --split-manifest configs/observation_query/splits_training_v2.json \
  --pair-id scene0001_00-scene0001_01 \
  --model-bundle /path/to/train/model_bundle \
  --observation-bank /path/to/train/observation_bank \
  --output-root /path/to/train/training_targets
```

Run a fixed-budget method and evaluate its checkpoint on DEV:

```bash
PYTHONPATH=. RESCENE_DEVICE=cuda:0 /home/ww/miniconda3/envs/persist4d/bin/python \
  scripts/training/train_ovi_observation_query.py \
  --config configs/observation_query/pilot_training_v2.json \
  --runtime configs/observation_query/runtime.local.json \
  --method OBS_BASE_TUNED --stage pilot \
  --target-total-updates 1000 --run-id <train-run-id>

PYTHONPATH=. RESCENE_DEVICE=cuda:0 /home/ww/miniconda3/envs/persist4d/bin/python \
  scripts/evaluation/run_ovi_observation_query.py \
  --config configs/observation_query/pilot_training_v2.json \
  --runtime configs/observation_query/runtime.local.json \
  --method OBS_BASE_TUNED --checkpoint /path/to/checkpoint \
  --role DEV --pair-id scene0109_00-scene0109_01 --run-id <eval-run-id>
```

## Continuation Boundary

There is no unfinished task in this bounded V2 pilot after upload verification.
The highest-value next experiment is not another checkpoint on the exposed DEV
pair: expand to several official TRAIN environments and reserve an untouched
confirmation environment, then decide whether the observation branch merits
architectural revision based on multi-environment evidence.
