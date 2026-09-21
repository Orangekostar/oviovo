# OVI-MAP Module Validation Results

## Table A. Branch metrics and cost

| Method | Branch | Status | uAP | mIoU | Cost | Evidence |
|---|---|---|---|---|---|---|
| N0 | N0 | REUSED | null | null | null | /mnt/shared/ww/ovimap-t1-attribution-v1/evaluation |
| S_NATIVE_AREA | S | BLOCKED | null | null | null | /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/semantic_summary.json |
| S_NATIVE_VOTE | S | BLOCKED | null | null | null | /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/semantic_summary.json |
| S_SIGLIP2_AREA | S | BLOCKED | null | null | null | /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/semantic_summary.json |
| S_SIGLIP2_VOTE | S | BLOCKED | null | null | null | /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/semantic_summary.json |
| S_WOW_VOTE | S | BLOCKED | null | null | null | /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/semantic_summary.json |
| S_SIMPLE | S | BLOCKED | null | null | null | /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/semantic_summary.json |
| S_NO_CONTEXT | S | BLOCKED | null | null | null | /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/semantic_summary.json |
| S_PAIRED | S | BLOCKED | null | null | null | /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/semantic_summary.json |
| G_ORIGINAL | G | BLOCKED | null | null | null | /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/geometry_summary.json |
| G_AGREEMENT | G | BLOCKED | null | null | null | /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/geometry_summary.json |
| G_QUALITY | G | BLOCKED | null | null | null | /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/geometry_summary.json |
| Q_COMBINE | Q | BLOCKED | null | null | null | /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/query_summary.json |
| Q_AREA | Q | BLOCKED | null | null | null | /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/query_summary.json |
| Q_UNCERTAINTY | Q | BLOCKED | null | null | null | /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/query_summary.json |
| Q_GAIN | Q | BLOCKED | null | null | null | /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/query_summary.json |
| COMBO_GS | COMBO | NOT_REQUIRED | null | null | null | null |
| COMBO_Q_REFINEMENT | COMBO | NOT_REQUIRED | null | null | null | null |

## Table B. Semantic suggestion, adoption, and effect

| Method | Suggestions | Adoptions | Corrections | Damage | Status |
|---|---|---|---|---|---|
| S_NATIVE_AREA | null | null | null | null | BLOCKED |
| S_NATIVE_VOTE | null | null | null | null | BLOCKED |
| S_SIGLIP2_AREA | null | null | null | null | BLOCKED |
| S_SIGLIP2_VOTE | null | null | null | null | BLOCKED |
| S_WOW_VOTE | null | null | null | null | BLOCKED |
| S_SIMPLE | null | null | null | null | BLOCKED |
| S_NO_CONTEXT | null | null | null | null | BLOCKED |
| S_PAIRED | null | null | null | null | BLOCKED |

## Table C. Geometry partitions and preservation

| Method | Hypotheses | Changed partitions | Preserved objects | Status |
|---|---|---|---|---|
| G_ORIGINAL | null | null | null | BLOCKED |
| G_AGREEMENT | null | null | null | BLOCKED |
| G_QUALITY | null | null | null | BLOCKED |

## Table D. Query acquisition and readout

| Method | Available | Attempts | Successes | Crops | uAP | Status |
|---|---|---|---|---|---|---|
| Q_COMBINE | null | null | null | null | null | BLOCKED |
| Q_AREA | null | null | null | null | null | BLOCKED |
| Q_UNCERTAINTY | null | null | null | null | null | BLOCKED |
| Q_GAIN | null | null | null | null | null | BLOCKED |

## Table E. Selection, combinations, and confirmation

| Final candidate | Science | Confirmation | Publication evidence |
|---|---|---|---|
| N0 | INCONCLUSIVE_PREREQUISITES | NOT_REQUIRED_NO_RETAINED_CANDIDATE | /mnt/shared/ww/ovimap-t1-attribution-v1/evaluation |

Supporting execution evidence:

- These are implementation-boundary smokes on historical Room0, not SELECT measurements.
- Semantic adapters: native SigLIP crop vectors [9, 1024]; legacy six-crop max error 8.120649263448909e-08.
- Geometry smoke: 12288 surface rows, 4262 native leaves, 8 complete bounded hypotheses.
- Query smoke: 35 current-state candidates with exact request/mask/bbox parity and 31 native combine selections with exact request IDs; Q_AREA logical cost {'attempts': 18, 'crop_inputs': 108, 'failures': 0, 'successes': 18}; Q_UNCERTAINTY logical cost {'attempts': 18, 'crop_inputs': 108, 'failures': 0, 'successes': 18}; physical cost {'cache_hits': 36, 'crop_inputs': 0, 'inference_seconds': 0.0, 'model_forwards': 0, 'model_loads': 0, 'tiles': 0}.
