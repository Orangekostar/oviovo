# P6-D Geometry-Gated Dense Readout Report

## Scope

- `CODE_EVIDENCE`: the rejected path emits exact temporal-compact geometry and
  adds only three audit metadata fields.
- Scene: Apartment only; Office remained held out.
- Candidate: translated dense OVI-MAP geometry for moved bound objects, accepted
  only by the preregistered bidirectional 5 cm agreement gate.
- Source: frozen P4A temporal-compact composition; localized ownership disabled.
- Gate config SHA-256: `9cd0933b0418ebe2391830aaa468e5d2954e8fc168eb97355ba13077f67333c8`
  (399 bytes).

## Geometry Result

`MEASURED_EVIDENCE`: the gate evaluated 104 moved-object checkpoint records and
accepted none. All 104 records therefore used the exact temporal-compact fallback. The dominant
rejection was compact-to-template coverage (104/104); template-to-compact
coverage failed for 99/104 and the y-axis extent residual failed for 99/104.

At the final checkpoint, the five moved objects contain 3,502 compact points.
Their pooled within-readout nearest-neighbor spacing is 0.032167 m median and
0.048584 m p90. The per-object measurements are:

| Anchor | Points | NN median m | NN p90 m | Points / bbox m3 |
| --- | ---: | ---: | ---: | ---: |
| `ovimap:131` | 818 | 0.032545 | 0.039747 | 197.957 |
| `ovimap:179` | 154 | 0.028651 | 0.038864 | 45.558 |
| `ovimap:2` | 177 | 0.030830 | 0.050770 | 327.196 |
| `ovimap:6` | 2,234 | 0.032951 | 0.049312 | 196.857 |
| `ovimap:85` | 119 | 0.029305 | 0.043214 | 2,525.988 |

The 43 checkpoint snapshots occupy 649,613,149 bytes. The final snapshot is
13,146,647 bytes and the final entity readout is 53,438 bytes. The serialized
bridge contains 3,195 PLY files totaling 3,318,492,028 bytes; its complete
bridge input is 3,362,192,461 bytes.

## Exact Fallback

All 43 snapshot SHA-256 values are byte-identical to the frozen P4A compact
control, including the final snapshot
`a718a689d5569b2eb631880e35d87606511dcd2b0420352e96e6464028b5eace`.
Entity records are semantically identical after removing only the three
additive audit fields `geometry_gate_id`, `geometry_gate_accepted`, and
`geometry_gate_rejection_reasons`.

## Metrics And Decision

| Metric | P6-D | Hard gate | Result |
| --- | ---: | ---: | --- |
| Object F1 | 0.348472 | >= 0.372762 | FAIL |
| Dynamic F1 | 0.069225 | >= 0.069225 | PASS |
| Change F1 | 0.088458 | >= 0.088088 | PASS |
| Current mIoU | 0.149560 | >= 0.149563 | FAIL |
| Ghost | 0.443760 | <= 0.441677 | FAIL |
| Frames / states | 1,745 / 43 | 1,745 / 43 | PASS |

`MEASURED_EVIDENCE`: `NO_GO / REJECTED_RETAIN_A6`. No dense geometry passed the
fixed agreement gate, and the exact compact fallback fails the Object, Current, and
Ghost promotion thresholds. The combined P6-C + P6-D candidate is therefore
`NOT_RUN_STANDALONE_GATE` regardless of the P6-C result.

`HYPOTHESIS`: a future dense method would need deformation-aware template
support rather than centroid translation, but P6 provides no evidence to relax
the registered gate.

## Bound Artifacts

| Artifact | SHA-256 | Bytes |
| --- | --- | ---: |
| Composition manifest | `1a3098140282257f247f835307f6b3727a3055c67ee84a297d65899f2609a9c7` | 200,619 |
| Official metrics | `ef16da07ccd5c2ff1ee8b9f5ebca95cc99ff036596b3ce21823c5969237f9b9c` | 1,146 |
| Gate decision | `0f02a566e95d904ce6381e196e36f4408340ca05b0e9ca3742071e7d77bd7fb6` | 5,092 |

Artifacts are under
`/home/ww/oviovo_baseline_runs/20260903_crove_localized_current_ownership/p6d/apartment_hybrid`.
