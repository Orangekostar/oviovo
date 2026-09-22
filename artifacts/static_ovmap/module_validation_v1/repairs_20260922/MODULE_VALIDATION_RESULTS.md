# OVI-MAP Module Validation Results

## Table A. Branch metrics and cost

| Method | Branch | Status | uAP | mIoU | Cost | Evidence |
|---|---|---|---|---|---|---|
| N0 | N0 | BLOCKED | null | null | null | null |
| S_NATIVE_AREA | S | BLOCKED | null | null | null | null |
| S_NATIVE_VOTE | S | BLOCKED | null | null | null | null |
| S_SIGLIP2_AREA | S | BLOCKED | null | null | null | null |
| S_SIGLIP2_VOTE | S | BLOCKED | null | null | null | null |
| S_WOW_VOTE | S | BLOCKED | null | null | null | null |
| S_SIMPLE | S | BLOCKED | null | null | null | null |
| S_NO_CONTEXT | S | BLOCKED | null | null | null | null |
| S_PAIRED | S | BLOCKED | null | null | null | null |
| G_ORIGINAL | G | BLOCKED | null | null | null | null |
| G_AGREEMENT | G | BLOCKED | null | null | null | null |
| G_QUALITY | G | BLOCKED | null | null | null | null |
| Q_COMBINE | Q | BLOCKED | null | null | null | null |
| Q_AREA | Q | BLOCKED | null | null | null | null |
| Q_UNCERTAINTY | Q | BLOCKED | null | null | null | null |
| Q_GAIN | Q | BLOCKED | null | null | null | null |
| COMBO_GS | COMBO | BLOCKED | null | null | null | null |
| COMBO_Q_REFINEMENT | COMBO | BLOCKED | null | null | null | null |

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
| N0 | INCONCLUSIVE_PREREQUISITES | NOT_RUN_PREREQUISITES | null |

Supporting execution evidence:

- capture: verify existing native capture payload and frame schedule; status COMPLETE; evidence /mnt/shared/ww/ovimap-module-validation-v1/attempt_003/capture_smoke.json.
- semantic: one native captured request per visual adapter; status COMPLETE; evidence /mnt/shared/ww/ovimap-module-validation-v1/attempt_003/semantic_smoke.json.
- geometry: first_4096_native_faces; status COMPLETE; evidence /mnt/shared/ww/ovimap-module-validation-v1/attempt_003/geometry_smoke.json.
- query: native candidate parity on all captured frames; frame-0 cached policy replay; status COMPLETE; evidence /mnt/shared/ww/ovimap-module-validation-v1/attempt_003/query_smoke.json.
