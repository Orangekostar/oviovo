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

## Stage B: Auditable Semantic Replay

Frozen replay implementation commit: `d1ff610`

The accepted candidate keeps global entropy power 16 and uses entropy power 32 for
wall, floor, ceiling, and table. The replay binds all 200 RGB/depth/pose inputs, dense
cache files, frontend manifest, source files, base snapshot checksums, and structure
replay configuration. The semantic Pareto audit is `PASS`.

| Metric | Stage 3 baseline | Stage B | Delta |
| --- | ---: | ---: | ---: |
| mIoU | 0.3828341530 | **0.3831077036** | **+0.0002735506** |
| mAcc | 0.4337307984 | **0.4343641210** | **+0.0006333226** |
| f-mIoU | 0.6646659615 | **0.6648088665** | **+0.0001429050** |
| AP25 | 0.2803600107 | 0.2803600107 | 0 |
| AP50 | 0.0497723659 | 0.0497723659 | 0 |
| F@5cm | 0.9160999937 | 0.9160999937 | 0 |

## Stage C and Composed Winner

Frozen composition implementation commit: `6c9c104`

The geometry head lowers mesh extraction weight from 1.0 to 0.5. Newly recovered
low-support vertices retain geometry, while their semantic label is stabilized from the
nearest weight-1.0 reference vertex within 0.075 m. The transformation is GT-free.

| Metric | Stage 3 baseline | Composed winner | Delta |
| --- | ---: | ---: | ---: |
| mIoU | 0.3828341530 | **0.3837512496** | **+0.0009170966** |
| mAcc | 0.4337307984 | **0.4354314683** | **+0.0017006699** |
| f-mIoU | 0.6646659615 | **0.6660187076** | **+0.0013527460** |
| AP25 | 0.2803600107 | **0.3889825273** | **+0.1086225166** |
| AP50 | 0.0497723659 | **0.1305074808** | **+0.0807351149** |
| F@5cm | 0.9160999937 | **0.9178952033** | **+0.0017952096** |

The frozen `composed` Pareto audit is `PASS`; the base snapshot checksums are exact.
Metrics SHA-256: `da3e091c61731fc84c63468f11b11bedb41d31030bf60dda78b87a674fba7715`.

## Route 3 Stage D: Auxiliary Instance Ensemble

Implementation commits: `045ec10`, `59c6089`, and provenance fix `24a8b5e`.

The auxiliary stream uses the independently mapped, frozen Route 2 `sam_labeled`
snapshot only to generate instance proposals. The primary Route 1 snapshot remains the
sole source for semantic and geometry metrics. Auxiliary parents are scored from their
own mesh support with weight 6, support exponent 1.5, and bias 0.05. The primary head
uses child multiplier 12, view exponent 2.5, semantic exponent 1.5, and final
cross-source deduplication IoU 0.9. No score or proposal reads GT geometry, labels, or
matches.

| Metric | Route 1 composed | Route 3 Stage D | Delta | Gate |
| --- | ---: | ---: | ---: | --- |
| mIoU | 0.3837512496 | 0.3837512496 | 0 | IEEE-identical |
| mAcc | 0.4354314683 | 0.4354314683 | 0 | IEEE-identical |
| f-mIoU | 0.6660187076 | 0.6660187076 | 0 | IEEE-identical |
| AP25 | 0.3889825273 | **0.5039730208** | **+0.1149904935** | strict pass |
| AP50 | 0.1305074808 | **0.2688231250** | **+0.1383156442** | strict pass |
| F@5cm | 0.9178952033 | 0.9178952033 | 0 | IEEE-identical |

The ensemble contains 136 primary and 125 auxiliary raw hypotheses. Of 261 total,
253 pass the projection-support floor and 243 remain after deduplication. Recall rises
to 0.676471 at IoU 0.25 and 0.441176 at IoU 0.50. The `instance` Pareto audit is
`PASS`.

The evaluator was repeated into a fresh directory; `metrics.json` and
`instance_head_audit.json` are byte-identical. Frozen hashes:

- metrics: `088b7ca9bb53584b52c603b2b2b60b402253d2959263c41c390443e39fd9887b`
- instance audit: `45880578846d18b83bfed88d910816ea0564d6079fe1090ecd0a9e3c9b4a88ad`
- Pareto audit: `3e71e632b368fe542df0a068699cc0dcff6c184ba1aa0f47c790706af67e2e28`
- auxiliary algorithm: `9b9369cede2f46cd8821733253e8e75a6c69819aa7dc9ce73624f5d5486b50b5`

Artifact root:
`/home/ww/oviovo_experiments/20260721_route3/room0_aux_ensemble_frozen_59c6089`.

This is an accepted room0 instance-stage result, not a replacement for the current
Replica benchmark row. Promotion requires strict six-metric improvement over the
current Route 1 result and one identically configured auxiliary stream for every
Replica scene.
