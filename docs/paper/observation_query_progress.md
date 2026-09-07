# Observation Query Progress

- Current task: T00-T10 delivery complete; final local/remote SHA equality is reported externally to avoid a self-referential commit.
- Completed: T00-T10 implementation, bounded real evidence, verification, commit, and push.
- Branch/base: `research/ovi-rescene-observation-query`, based on `4354df9df093628eaa7b3470319944914db85711`.
- Real DEV pair: `scene0109_00-scene0109_01`; it was previously inspected and is not an untouched confirmation set.
- Real TRAIN pair: `scene0001_00-scene0001_01`; both official `sequence.zip` files are absent. No Terms-of-Use acceptance was inferred, DEV was not substituted, optimizer updates are 0, and `MODEL_ASSET=NOT_CREATED`.

T01-T02 produced a source-bound ObservationBank with 490 regions, 51,825 region-to-M edges, and support for 30,131/40,076 M rows. Three fixed overlays and the camera-visible diagnostic are recorded under `configs/evaluation/results/observation_query/`.

T03 transfers high-resolution TRAIN-only labels to M through same-visit radius-neighbour consensus. Unknown rows remain outside `label_valid`; unknown semantic classes keep mask supervision with class target `-100`; cross-visit identity expansion requires official metadata.

T04-T05 implement the decoder-internal observation branch and masked joint criterion. The current-source GPU rerun gives exact native fallback at equal RNG state (maximum mask/class difference 0), finite FULL outputs `[40076,100]`, `[1,100,19]`, `[490,101]`, and finite total loss 37.7818 at update 0 / 38.3439 at update 200. The 0.168 s figure is cached decoder time and excludes the frozen backbone.

T06 implements source-bound frozen-backbone caching, optimizer groups, accumulation, clipping, compact safetensors checkpoints, resume checks, and reload equality. The current real TRAIN gate remains `TRAINING_ASSET_GATED` with two missing `sequence.zip` files.

T07-T08 implement `BASE_TUNED`, `LATE`, `FUSE`, `ATTN`, `FULL`, `NO_FEEDBACK`, and `NO_CONSISTENCY` mode contracts plus raw/final/dense/identity evaluation. Only `OBS_BASE_FROZEN` has a real DEV metric run. All trained modes are `NOT_RUN_MISSING_TRAIN_ASSETS`, never zero-filled.

For `OBS_BASE_FROZEN`, visit 0/1 full-GT F1@0.50 is 0.0889/0.1250, common-M F1 is 0.1053/0.1379, camera-visible F1 is 0.2727/0.4000, raw best-IoU mean is 0.1778/0.1349, and identity recall is 0. The current runner is internally repeatable, but it is not bit-exact to the older historical raw-prediction cache; the mismatch is recorded rather than assigned a speculative cause.

T09 converts OBS dense outputs into two geometry-conserving `VisitMap` values plus source-D row mappings and candidate-only `PairRelation` values for the existing t1-first composer. The real DEV output has 0 candidate relations after conflict filtering and 0 identity recall, so status is `INSTANCE_CONSTRUCTION_PASS_MAP_EFFECT_NOT_EVALUABLE`; no Ghost improvement is claimed.

Machine-readable status: `configs/evaluation/results/observation_query/experiment_status.json`.
Artifact inventory: `configs/evaluation/results/observation_query/artifact_manifest.json`.
Next scientific dependency: legally materialize the declared official TRAIN pair, then run the fixed smoke and trained comparison protocol without substituting DEV data.
