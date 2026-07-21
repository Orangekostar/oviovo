# OVIV2 Route 1 Layered Pareto Plan

**Date:** 2026-07-21  
**Development scene:** Replica `room0`, 200 frames, source stride 10  
**Starting snapshot:** unchanged Stage 3 baseline at
`/home/ww/oviovo_final_outputs/oviv2_replica8_stage3_s049_200f_5f272a8/room0`

## Starting Evidence

Route 2 produced no paper-facing winner. The best `sam_labeled` run raised AP25/AP50
from `0.2803600107/0.0497723659` to `0.3274767331/0.1506332888`, with F5 exactly
unchanged, but reduced mIoU/mAcc/f-mIoU by `0.0018183/0.0016927/0.0004340`.
The accepted starting snapshot therefore remains the Stage 3 baseline. Route 2 caches
are retained as diagnostic evidence, not as the mapper input for Route 1.

The baseline room0 diagnostics are 31 overmerged entities and 18 fragmented ground-truth
instances. Existing semantic scale evidence shows that `fusion_entity_weight_scale=0.49`
is the best tested scalar: `0.45` improves mAcc only, while `0.60` and `0.65` improve
f-mIoU only and all other tested values fail the three-metric gate.

## Global Constraints

- Mapping and hypothesis generation must not read ground-truth labels, meshes, matches, or evaluation outputs.
- The Stage 3 snapshot, entity registry, geometry, ownership, and dense evidence are immutable inputs.
- Every candidate is written to a fresh output root with source/config/code hashes.
- A stage is accepted only when all six headline metrics are finite and non-decreasing against the accepted parent.
- A semantic-only stage must keep AP25/AP50/F5 byte-identical; an instance-only stage must keep mIoU/mAcc/f-mIoU/F5 byte-identical; a geometry-only stage must keep the other five metrics non-decreasing.
- The composed Route 1 result must strictly improve all six metrics before Replica-8 promotion.

## Stage A: Protocol-Aligned Instance Head

Create an evaluator-side instance head. Do not change the mapper or snapshot.

1. Read raw persistent entities and their mesh vertices from the schema-3 registry/snapshot.
2. Emit one raw entity hypothesis per valid non-structural entity.
3. Split each entity mesh into deterministic triangle-adjacency connected components.
4. Keep the parent and add child hypotheses when projected support is at least 200 vertices.
5. Score parents from accepted-view count and semantic margin; score children as parent score times the squared child-to-parent support ratio, with multiplier 2.0 as the initial room0 hypothesis.
6. Project each hypothesis independently under the existing class-agnostic protocol; do not create one mutually exclusive predicted-label map.
7. Deduplicate only near-identical predicted hypotheses with deterministic score/ID ties.
8. Preserve the legacy evaluator output as an audit sidecar and report raw/child/deduplicated counts.

Required tests cover component determinism, parent-plus-child retention, score finiteness,
independent projection competition, class-agnostic semantic independence, and GT-free
hypothesis generation. The first gate is AP25/AP50 strict improvement with semantic and
F5 byte-identical.

## Stage B: Semantic Replay

Use the same immutable geometry/ownership/entity state and regenerate only semantic labels.

1. Add a semantic-only evaluation mode that records the fixed geometry/instance checksums and excludes snapshot-specific checksums from protocol comparison.
2. Reproduce the existing scale sweep at `0.46`, `0.47`, `0.48`, and `0.49`; stop if no value strictly improves mIoU, mAcc, and f-mIoU together.
3. If scalar fusion is insufficient, implement a dense-evidence replay contract. Sweep one frozen parameter at a time in this order: entropy power, minimum quality, minimum probability.
4. Keep entity IDs, projected instance masks, geometry vertices, AP25/AP50, and F5 byte-identical in every semantic candidate.

No semantic candidate is promoted on a single aggregate. All three semantic metrics must
strictly improve together.

## Stage C: Geometry Precision

Start only from the accepted A+B snapshot. Use a separate mesh derivation head over the
immutable TSDF snapshot.

1. Measure weight/support distributions and vertex-to-input-depth residuals without GT at runtime.
2. Sweep mesh extraction support thresholds and low-support component removal one parameter at a time.
3. Prefer candidates that increase F5 through recall recovery; reject any candidate that lowers precision, semantic metrics, or AP metrics.
4. Keep geometry thresholds out of mapper integration and preserve the raw mesh as an audit output.

## Gates and Artifacts

Each stage requires a 20-frame smoke/evaluator check, then a fresh 200-frame room0 run or
snapshot derivation. Store metrics, protocol audit, source hashes, timing, entity counts,
overmerge/fragmentation diagnostics, and reproduction hashes. Use the existing Pareto
comparator with a Route 1 stage mode for exact unchanged-head checks and final mode for
the composed result.

The first implementation target is:

- `src/evaluation/oviv2_instance_head.py`
- `scripts/evaluation/evaluate_oviv2_instance_head.py`
- focused instance-head and CLI tests

After a composed room0 winner is reproduced, freeze its config and run the same code and
thresholds on all eight Replica scenes. Report per-scene regressions even if the macro
gate passes, then fill the benchmark table from the verified Replica-8 aggregate.
