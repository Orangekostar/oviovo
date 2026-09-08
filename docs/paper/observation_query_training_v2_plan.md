# Observation-query real-training pilot V2

## Question

This pilot tests whether raw multi-view regions and depth support improve the
formation of temporal instance queries beyond (1) native decoder fine-tuning
and (2) simple observation-feature fusion. It does not test a new backbone,
long-term lifecycle model, or a full benchmark claim.

## Frozen design

- OVI-MAP provides each visit's dense surface and raw CropFormer regions.
- Frozen SigLIP encodes region appearance; frozen Concerto encodes the joint
  two-visit model input.
- The temporal window is 2, the query count is 100, and dense readout uses the
  same score and mask thresholds for every method.
- `OBS_BASE_TUNED` trains the native decoder/head without observation input.
- `OBS_FUSE` trains the same native parameters plus simple observation fusion.
- `OBS_FULL` trains observation attention, mask feedback, and consistency in
  addition to the shared native trainable parameters.

## Data and budget

The official TRAIN pair is `scene0001_00-scene0001_01`, from environment
`02b33dfb-be2b-2d54-92d2-cd012b2b3c40`. The evaluation pair is the previously
inspected DEV pair `scene0109_00-scene0109_01`, from a different physical
environment. All three trained methods use seed 45, AdamW, gradient accumulation
2, the same base checkpoint, one TRAIN pair, and 1,000 optimizer updates.
`OBS_FULL` first performs a 200-update smoke run and resumes cumulatively to
1,000 updates; the other methods start from the common base.

## Evaluation

The primary development measure is current-visit (`t1`) final-instance F1 at
IoU 0.50 on `COMMON_INPUT_SUPPORT_V2`. `FULL_GT_V2`, both raw-query domains,
IoU 0.25, TP/FP/FN, and temporal identity are reported as diagnostics. The
zero-update frozen result is recomputed under the same V2 domains.

The result is a single-environment transfer pilot. DEV is exposed, there is one
seed, and no untouched confirmation environment is selected; therefore the
result cannot support a generalization or SOTA claim.
