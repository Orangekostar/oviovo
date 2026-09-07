# Observation-Query Development Handoff

## Version And Status

- Repository: `Orangekostar/oviovo`
- Branch: `research/ovi-rescene-observation-query`
- Evidence base: `4354df9df093628eaa7b3470319944914db85711`
- Final delivery commit: recorded by the external local/remote SHA check after this file is committed
- T00-T05: `IMPLEMENTED` with real bounded DEV evidence
- T06: `IMPLEMENTED_TRAIN_ASSET_GATED`
- T07-T08: evaluator implemented; frozen baseline real; trained comparisons unavailable
- T09: `INSTANCE_CONSTRUCTION_PASS_MAP_EFFECT_NOT_EVALUABLE`
- Confirmation set: not selected or run
- Model asset: `NOT_CREATED`

## Data And Contracts

TRAIN is environment `02b33dfb-be2b-2d54-92d2-cd012b2b3c40`, pair `scene0001_00-scene0001_01`. DEV is environment `20c993b7-698f-29c5-847d-c8cb8a685f5a`, pair `scene0109_00-scene0109_01`, and was previously inspected. The base checkpoint overlaps the dataset's training/validation provenance; this pilot is engineering evidence, not an untouched generalization claim.

Raw processed columns are XYZ 0:3, RGB 3:6, normals 6:9, segment 9, semantic 10, and instance 11. Processed points already use the shared reference frame; no second global transform is applied. GT columns never enter model inputs. The input remains fixed D, fixed native M, fixed Q=100, and T=2.

The ObservationBank uses 16 frames per visit, 490 regions (286 parents and 204 depth fragments), frozen SigLIP features `[490,1024]`, 51,825 same-visit CSR edges, and 30,131 supported M rows. Unknown depth/label rows are not negatives.

## Implementation

- `src/oviv2/observation_query/contracts.py`, `observations.py`: source-bound region bank, depth relations, CSR incidence, and camera/reference conventions.
- `src/training/ovi_observation_data.py`: environment split checks, label transfer, label-valid masks, class mapping, and official temporal identities.
- `src/oviv2/observation_query/model.py`: observation fusion/attention after native 3D cross-attention, optional mask feedback, and exact disabled fallback.
- `src/oviv2/observation_query/losses.py`: masked Hungarian, native mask/class/Dice, soft region supervision, and bidirectional consistency.
- `src/oviv2/observation_query/training.py`, `scripts/training/train_ovi_observation_query.py`: frozen backbone cache, bounded mode contracts, optimizer groups, updates, safetensors checkpoint, resume, and replay.
- `src/oviv2/observation_query/dense_adapter.py`: raw/final/dense ownership, source D-row preservation, conservative temporal candidates, and VisitMap/PairRelation conversion.
- `scripts/evaluation/run_ovi_observation_query.py`: long-table raw/final/identity evaluation and current-map adapter evidence.

Parent OVI embeddings/labels remain a stated `parent_ovi_control`; this version does not claim clean semantics for merged parents or unsupported no-parent surface discovery.

## Real Commands

```bash
PYTHONPATH=. /home/ww/miniconda3/envs/persist4d/bin/python \
  scripts/evaluation/prepare_ovi_observations.py build-model-bundle \
  --pair-id scene0109_00-scene0109_01 \
  --output-root /home/ww/oviovo_baseline_runs/observation_query/20260907_scene0109_dev_v1/model_bundle

PYTHONPATH=. /home/ww/miniconda3/envs/ovimap-map/bin/python \
  scripts/evaluation/prepare_ovi_observations.py prepare-bank \
  --model-bundle /home/ww/oviovo_baseline_runs/observation_query/20260907_scene0109_dev_v1/model_bundle \
  --output-root /home/ww/oviovo_baseline_runs/observation_query/20260907_scene0109_dev_v1/observation_bank \
  --pair-id scene0109_00-scene0109_01 --device cuda:0

PYTHONPATH=. RESCENE_DEVICE=cuda:0 /home/ww/miniconda3/envs/persist4d/bin/python \
  scripts/evaluation/run_ovi_observation_query.py \
  --config configs/observation_query/pilot_v1.json \
  --runtime configs/observation_query/runtime.local.json \
  --method OBS_BASE_FROZEN --role DEV \
  --run-id 20260907_scene0109_obs_base_frozen_v2

PYTHONPATH=. /home/ww/miniconda3/envs/persist4d/bin/python \
  scripts/training/train_ovi_observation_query.py \
  --config configs/observation_query/pilot_v1.json \
  --runtime configs/observation_query/runtime.local.json \
  --method OBS_FULL --stage smoke \
  --run-id 20260907_scene0001_t06_asset_gate_v2
```

The last command exits 2 by contract and writes `TRAINING_ASSET_GATED`. Do not run the official downloader until the user explicitly confirms the 3RScan Terms of Use. After both TRAIN `sequence.zip` files and derived TRAIN artifacts exist, rerun the same command with a new run ID; the fixed smoke budget is 200 optimizer updates.

## Results And Assets

Scientific numbers and limitations are in `docs/paper/observation_query_results.md`. Machine-readable per-domain rows, gated rows, prediction arrays, and summaries are under `configs/evaluation/results/observation_query/runs/`. The compact asset policy and local-only cache hashes are in `configs/evaluation/results/observation_query/artifact_manifest.json`.

No training curve or trained state exists because training never passed the legal/source asset gate. The Git prediction package is 14,154,684 bytes and contains masks, classes, query scores, and both visits' dense owner/source arrays. Large reproducible preprocessing and frozen-backbone caches remain local-only and are not reported as uploaded.

`OBS_PER_VISIT` is retained as an evaluation contract, but this delivery does not implement or claim a joint-pair training checkpoint for it.

The strongest simple trained control cannot be identified until TRAIN is available. The frozen baseline itself is weak and has zero identity recall; map effect is therefore unproven. The single highest-value continuation is to materialize the declared TRAIN RGB-D pair legally and run the fixed `BASE_TUNED`/`FUSE`/`FULL` pilot without changing thresholds or the DEV selection rule.
