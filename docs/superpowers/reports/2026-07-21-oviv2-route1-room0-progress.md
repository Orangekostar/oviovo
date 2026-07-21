# OVIV2 Route 1 Room0 Progress

**Date:** 2026-07-21  
**Scene:** Replica `room0`, 200 frames, stride 10  
**Baseline:** Stage 3 fused snapshot, entity weight scale `0.49`

## Stage A: Protocol-Aligned Instance Head

Frozen implementation commit: `9ebb561`

Frozen configuration:

```json
{
  "minimum_component_vertices": 20,
  "child_score_multiplier": 8.0,
  "deduplication_iou_threshold": 0.7,
  "view_count_exponent": 2.0,
  "semantic_evidence_exponent": 1.0,
  "maximum_projection_distance_m": 0.05
}
```

The head retains all raw entities without predicted-semantic filtering, adds same-entity
triangle connected components, scores hypotheses only from map-time evidence, projects
each hypothesis independently, and deduplicates near-identical projected masks. Ground
truth enters only the evaluator after hypothesis generation.

| Metric | Stage 3 baseline | Stage A | Delta | Gate |
| --- | ---: | ---: | ---: | --- |
| mIoU | 0.3828341530 | 0.3828341530 | 0 | exact |
| mAcc | 0.4337307984 | 0.4337307984 | 0 | exact |
| f-mIoU | 0.6646659615 | 0.6646659615 | 0 | exact |
| AP25 | 0.2803600107 | **0.3827375951** | **+0.1023775844** | strict pass |
| AP50 | 0.0497723659 | **0.1305074808** | **+0.0807351149** | strict pass |
| F@5cm | 0.9160999937 | 0.9160999937 | 0 | exact |

The frozen run produced 81 parent and 54 child hypotheses. Runtime was 78.07 seconds,
peak RSS was 1,313,404 KiB, and no swap was used. The `instance` Pareto audit status is
`PASS`.

Artifacts:

- `/home/ww/oviovo_experiments/20260721_route1/instance_head_frozen_9ebb561/metrics.json`
- `/home/ww/oviovo_experiments/20260721_route1/instance_head_frozen_9ebb561/instance_head_audit.json`
- `/home/ww/oviovo_experiments/20260721_route1/instance_head_frozen_9ebb561/pareto_audit.json`

Hashes:

- metrics: `a261f3371e680c433c006de70d1a3ca83d70f75f45aaefab3da9c0e1829af42f`
- algorithm: `eb81c20d8637240fdcc7edad3127316abe5b6028fa85d95ec8d05da0b5a495bd`
- instance head source: `072bc5fcb2b3f836461994b7f606cb6c89bc27a2044ee4108edda30dd063a42b`
- CLI source: `054639f1e72128fd47e3903fc2332c909a7d6d21e1ab70224aab3aae6f051925`

## Stage B: Scalar Semantic Sweep

The frozen-snapshot scalar sweep cannot improve all three semantic metrics together:

| Entity scale | mIoU | mAcc | f-mIoU | Decision |
| ---: | ---: | ---: | ---: | --- |
| 0.46 | 0.3827614907 | 0.4338253861 | 0.6645420315 | reject |
| 0.47 | 0.3827751338 | 0.4336889299 | 0.6645439695 | reject |
| 0.48 | 0.3828213708 | 0.4337212355 | 0.6646387593 | reject |
| 0.49 | **0.3828341530** | 0.4337307984 | **0.6646659615** | retained baseline |

AP25, AP50, and F@5cm remain byte-identical across the scalar sweep. Route 1 therefore
stops scalar tuning and proceeds to dense-evidence semantic replay over the immutable
geometry, ownership, and entity registry.
