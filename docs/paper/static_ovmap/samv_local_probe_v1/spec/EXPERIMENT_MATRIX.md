# Fixed experiment matrix

## Main arms (24 scene-method records; 12 two-scene pools)

| ID | Partition | Object classes | Comparison role |
|---|---|---|---|
| SV00_G1 | Actual parent G1 | Actual parent G1 | No intervention |
| SV01_SAM2_GEOM | SAM2.1-L local repair | All retained owner IDs inherit G1 classes | Strong segmentation control |
| SV02_SAMV_GEOM | SAM-V local repair | Same inherited classes | Mask/partition effect |
| SV03_OLDMASK_FC | Actual parent G1 | Same-view original masks, frozen FC, fixed update | Additional-view/readout control |
| SV04_SAMV_FC | Actual parent G1 | Same-view SAM-V masks, same FC/update | Mask quality in semantic readout |
| SV05_COMBINED | Exact SV02 owner array | Exact SV04 owner-to-class decisions | Fixed geometry × semantics combination |

REF_D2 is a reference-only parent output: 4 additional scene records and 2 subset
pools. All logical coverage is therefore **28 records / 14 pools**. There are only
20 potentially new main-arm predictions, since SV00 and REF_D2 are inherited.
An exact local content alias is legitimate but does not equal a physical evaluator run.

## Fixed scenes

- replica_probe2: office1, room0
- cf_probe2: scene0011_00, scene0050_00

These names are frozen before the new model outputs. They are previously exposed
scenes and the CF scans belong to two different physical scene families. Do not
rename these pools Replica8, CF18, ScanNet200 validation, or independent testing.

## Fixed contrasts

1. SV02 − SV01: learned multi-view segmentor under the same prompts/views/lifter.
2. SV02 − SV00: structural change with inherited classes.
3. SV04 − SV03: new mask versus old projected mask under paired FC observations.
4. SV03 − SV00: new observations/readout without SAM-V masks.
5. SV05 − SV02: semantic change on the repaired support.
6. SV05 − SV04: structural change with the same class decisions.

No AnyUp, new N/Q, new temperatures, additional prompt search, confidence-score
replacement, or per-dataset recipe selection is included.
