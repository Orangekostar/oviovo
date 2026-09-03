# P6-A Counterfactual Attribution Report

## Scope And Contract

- `CODE_EVIDENCE`: CF0, CF1, ten leave-one-out CF2 variants, ten only-one
  CF3 variants, and two preregistered P2 feature groups were recomposed from
  the frozen P5 source. The source mapper was not rerun.
- `CODE_EVIDENCE`: every variant is marked `diagnostic_only=true` and
  `promotion_eligible=false`; anchor IDs are evidence coordinates, not a
  deployable rule.
- Scene: Apartment only. Office was not accessed.

## Execution Integrity

`MEASURED_EVIDENCE`: all 24 variants completed 1,745 frames and 43 official
states. Each gate receipt binds an unchanged official evaluation and two
byte-identical common-v2 repetitions.

| Artifact | SHA-256 | Bytes |
| --- | --- | ---: |
| Frozen plan | `3a43ab9fc8bb3aebd1c5b880f8df3baf4d08c84f86a1228f223a443ef200bb89` | 5,083 |
| Collection manifest | `41585bcfa8b3a48c8df0c88faf8307701b045c185dd658cae64b5c52d8a7f9f0` | 6,336 |
| Metrics CSV | `6053b069dc18e0a8589f81949fa3c2431298bc857cf5cc778c577c88032aa049` | 8,478 |
| Anchor features CSV | `c157ef4dec7fbecf50ca5d9b76e28fd4e6293a36835e40b0165bc8c7a8702f67` | 1,519 |

The collection is under
`/home/ww/oviovo_baseline_runs/20260903_crove_localized_current_ownership/p6a/collection`.
Its manifest binds all 24 per-variant gate receipts.

## Formal Results

| Variant | Obj F1 | Dyn F1 | Chg F1 | Current mIoU | Ghost |
| --- | ---: | ---: | ---: | ---: | ---: |
| CF0, no suppression | 0.348472 | 0.069225 | 0.088458 | 0.149560 | 0.443760 |
| CF1, full P5 | 0.367952 | 0.069225 | 0.088088 | 0.149563 | 0.441677 |
| G1, P2 Ghost-dominant five | 0.348472 | 0.069225 | 0.088088 | 0.149568 | 0.441677 |
| G2, complementary five | 0.367952 | 0.069225 | 0.088458 | 0.149563 | 0.443760 |

Across the full matrix, Object F1 ranges from `0.348472` to `0.367952`,
Ghost from `0.441677` to `0.443760`, and current mIoU from `0.149549` to
`0.149570`. Dynamic F1 is exactly `0.06922505723328032` in every variant.
No variant reaches the frozen Object floor `0.372762`, and 0/24 pass all hard
gates.

## Attribution Finding

`MEASURED_EVIDENCE`: the five later/short-latency transitions selected by P2
recover the complete P5 Ghost reduction (`-0.002082` versus CF0) but no Object
gain. The complementary five earlier/long-latency transitions recover the
complete P5 Object gain (`+0.019480`) but no Ghost reduction. Only-one and
leave-one-out results reproduce this separation: the earlier group contributes
Object F1, while the later group contributes Ghost.

`MEASURED_EVIDENCE`: current-mIoU changes are small and variant-specific;
Change F1 takes only two values. The evidence therefore rejects a simple
single-anchor or scalar-threshold explanation, and confirms that the P5 trade-off
is at least two-factor and non-promotable from IDs.

## Decision

`MEASURED_EVIDENCE`: `NO_GO / DIAGNOSTIC_ONLY`. CF1 remains the best measured
trade-off in this matrix but still misses the A6 Object floor by `0.004810`.
No manual anchor-ID filter, threshold sweep, or Office run is authorized.

`HYPOTHESIS`: transition timing and geometry may be useful features in a
multi-scene ownership model, but ten transitions from one scene cannot establish
a general rule. P6-C supplies the required non-ID spatial test and is evaluated
separately.
