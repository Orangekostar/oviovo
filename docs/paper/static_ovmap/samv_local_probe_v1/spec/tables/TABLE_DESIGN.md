# Three compact table families — generated from actual result_store

No numeric results are supplied by this package. Use `—` with a reason when unmeasured;
never use a parent full-cohort metric as the value of the two-scene probe.

## Table 1 — full-map outcome on fixed probe subsets

| Method | Replica-probe2 APall | AP50 | mIoU | CF-probe2 APall | AP50 | mIoU |
|---|---:|---:|---:|---:|---:|---:|
| D2 (reference only) | — | — | — | — | — | — |
| G1 | — | — | — | — | — | — |
| SAM2 geometry | — | — | — | — | — | — |
| SAM-V geometry | — | — | — | — | — | — |
| Old-mask same-view FC | — | — | — | — | — | — |
| SAM-V-mask same-view FC | — | — | — | — | — | — |
| SAM-V geometry + same class decisions | — | — | — | — | — | — |

Caption identifies the four scenes, inherited fixed surface, actual exclusive
partitions, two-scene ordered pooling, AP threshold range and previous exposure.
AP25/mAcc go to the complete supplement; gates use all five unrounded values.

## Table 2 — mechanism, not just counts of new owners

Two blocks:
- Structure: original/SAM2/SAM-V, same frozen targets; anchor adherence; raw/admitted
  support; class-agnostic GT50/75 gained/lost; mean fixed-reference support IoU;
  conflicting rows, protected rows, donor harms. Raw proposals are not deployed masks.
- Semantics: SV04−SV03 and SV05−SV02, common successful target count, wrong→right/
  right→wrong, unique released GT50/75 gained/lost, AP/mIoU differences.

Do not force undefined 2D GT metrics into this table. Do not call output counts recall.

## Table 3 — actual cost scopes

Block A: all first-run scientific work, new/cache-hit image inputs, joint VGGT windows,
SAM2 tracks and FC region pools, separated stage wall times and real failure costs.
Block B: paired segmentation-window microtiming, 2 windows × 2 models × 2 repetitions;
median/mean and all raw times, peak allocated/reserved, actual window length and dtype.
Do not combine Block B with old G1 recovery latency or call it end-to-end speed.

Standalone complete-map latency is `NOT_MEASURED_THIS_PROBE`. Table snippets must
be generated from the canonical store, with per-cell source identities. No new fourth
main table or baseline training campaign is required.
