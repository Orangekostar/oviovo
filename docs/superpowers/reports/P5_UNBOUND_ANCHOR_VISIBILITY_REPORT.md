# P5 Unbound-Anchor Visibility Report

## Scope

- Branch: `research/crove-dense-current-state-recovery`.
- Frozen base: `c55396b24d7706016a569c951914c79c9f641408`.
- Development scene: Apartment only; Office outputs were not accessed.
- Backbone: frozen OVI-MAP geometry and semantics with CROVE temporal state.
- Candidate: causal visibility suppression for unbound OVI-MAP anchors using
  compact moved-object geometry.

## Preregistered Policy

The single Apartment candidate used 5 cm deterministic voxel sampling capped
at 1,000 points per anchor, a 0.1 m depth tolerance, at least 10 valid projected
samples, at least 0.8 visible-absence fraction with zero present samples, and
at least six strong absence observations from three camera positions separated
by 0.25 m. Occluded and depth-unknown observations preserve state; two
consecutive strong-present observations reactivate a dormant anchor.

The policy is causal: it processes RGB-D frames in chronological order after
the frozen anchor cutoff at frame 262 and publishes the state available at each
checkpoint. It does not modify bound anchors, temporal association, background,
or the evaluator.

## Measured Mechanism

The runner processed all 1,745 Apartment frames and emitted ten active-to-
dormant transitions. Five are the P2-attributed dominant stale anchors:
`ovimap:5` at frame 636, `ovimap:81` at 1,214, `ovimap:129` at 1,215,
`ovimap:102` at 1,217, and `ovimap:117` at 1,218. The other transitions are
`ovimap:28`, `ovimap:34`, `ovimap:17`, `ovimap:26`, and `ovimap:33` at frames
355--356. No bound anchor was eligible for suppression.

## Apartment Decision

| Metric | P5 candidate | Registered baseline | Delta |
| --- | ---: | ---: | ---: |
| Object F1 | 0.367952 | 0.372762 | -0.004810 |
| Dynamic F1 | 0.069225 | not gated | -- |
| Change F1 | 0.088088 | 0.060853 | +0.027235 |
| Current mIoU | 0.149563 | 0.142897 | +0.006666 |
| Ghost | 0.441677 | 0.646883 | -0.205206 |

**REJECTED_RETAIN_A6.** P5 validates the measured stale-anchor mechanism and
improves Change F1, Current mIoU, and Ghost. It nevertheless misses the
preregistered Object F1 floor by 0.004810, so the candidate is not promoted or
frozen. Office remains held out and was not run. A6 remains the formal
official-metric candidate.

## Artifact Binding

| Artifact | SHA-256 |
| --- | --- |
| Composition manifest | `3cfa389183a930ac5fb10aa34093cf63e64e1c8206f1f4e9ccffef1490369fc3` |
| Visibility diagnostics | `7ee9a287a3484147bf8daeb02105a2829046bb357dee4aec1fcfb66ac090ddf3` |
| Temporal manifest | `2e90dc7ce98555c594016ac2bac0673ec426b69a67184ef775fdc780b6fe5841` |
| Bridge manifest | `c6e693e169f1adfdec111be0c949b61446ee2f7ed842e8a066324a98eb9668cf` |
| Bridge run status | `6e2e2f8d135256fd776c4323fd38cfe7561cac700a5234ea1688475713fdccee` |
| Official metrics | `901fe7587f2b224819d3ed4838a4bfe225ed3c361e51914e4acd18093bda7a21` |
| Gate decision | `b3b4db3ea564e5c4aec821c37d3362cae0fbee677bea0238df226c29123d6f12` |

## Verification

- Pure visibility policy tests: 11 passed.
- Composition runner tests: 16 passed.
- Core and runner integration tests: 50 passed.
- T1 exactness/non-interference tests: 48 passed.
- Official evaluation: PASS with 43 states; repeated summaries are
  byte-identical.
- Common-v2 gate: `REJECTED_RETAIN_A6` on `object_f1_floor`.

