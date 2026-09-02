# P4A Dense Moved-Anchor Shadow Report

## Scope

- Branch: `research/crove-dense-current-state-recovery`.
- Frozen base: `c55396b24d7706016a569c951914c79c9f641408`.
- Development scene: Apartment only; Office outputs were not accessed.
- Candidate: OVI-MAP dense anchor geometry translated to CROVE's latest causal
  current centroid.
- Role: visualization shadow only. No official metric improvement is claimed.

## Bound Runs

| Artifact | SHA-256 | Bytes |
| --- | --- | ---: |
| Anchor manifest | `b6a3bdb18216d2e2b0a43d76f00a57a769fddb845a8c6e310379fe6a455d84af` | 3,914 |
| Compact-control manifest | `10bb0edcd2d89175b606e25ba465b931795ab7b68347914357695b27d22e5e82` | 93,806 |
| Dense-shadow manifest | `71d03749ca162711fdada6af7ed33dccd452fa6ef51d7e51a7a9dc6d7dd397a5` | 93,822 |
| Final anchor snapshot + entities | `5088204a565311bc67c5b776edf55ac8e00bbbcad59b1c6c062d254ba752241f` + `d17dd78a101d8482bc7c4f2223d90297214072e4e511d603c85b5293e046799c` | 18,800,925 |
| Final compact snapshot + entities | `a718a689d5569b2eb631880e35d87606511dcd2b0420352e96e6464028b5eace` + `83378c646b2f00de83ec22235cffe1b645e6d9145028e20b9a9d9ece5314ba16` | 13,198,402 |
| Final dense snapshot + entities | `2d7135f63365665f7bec6b3ed348bb15ca162b452ae6ce31b0fe7ce6b129424c` + `0d39087b6520e4fa8bb3f0752ee0ec6373aca077a0540b3b8fe3693130a4fc65` | 17,836,796 |

The newly generated compact control has the same frame inventory and identical
`snapshot`, `entities`, `checkpoint_status`, and `diagnostics` records as all
43 checkpoints in the previously frozen compact composition. The only new
run-level field is the explicit readout contract.

## Representative Moved Entities

Each row is the first checkpoint where the target entity is emitted as moved.
Coverage is dense-to-compact nearest-neighbor coverage at 5 cm.

| Entity | Frame | Anchor pts | Compact pts | Dense pts | NN median m | NN p90 m | Coverage | Rigid residual m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ovimap:131` | 613 | 2,912 | 379 | 2,912 | 0.031593 | 0.063236 | 0.786745 | 0 |
| `ovimap:179` | 711 | 7,253 | 357 | 7,253 | 0.044494 | 0.101103 | 0.566386 | 0 |
| `ovimap:6` | 713 | 241,957 | 3,093 | 241,957 | 0.027704 | 0.194856 | 0.688647 | 4.31e-7 |
| `ovimap:85` | 908 | 3,666 | 333 | 3,666 | 0.041817 | 0.056651 | 0.802782 | 0 |
| `ovimap:2` | 1,208 | 1,681,524 | 880 | 1,681,524 | 1.408461 | 2.655144 | 0 | 1.10e-7 |

At frame 1,472 the same five moved entities contain 1,937,312 anchor/dense
points versus 3,502 compact points. All semantic, lifecycle, identity, and
existing metadata fields match the compact readout. The largest measured
translation residual is `8.81e-7 m`, below the preregistered `1e-6 m` limit.

Exact anchor and dense bounds, translations, source hashes, and serialized
sizes are recorded in the six source-bound JSON audits under
`/home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p4a/audit/`.

## Decision

**PASS for a separately named Apartment evaluation candidate; not promoted to
the formal readout.** The shadow removes the measured dense-to-sparse point
collapse without changing CROVE state or the compact control. However,
`ovimap:2` has zero 5 cm agreement with compact current geometry and spans a
large anchor template, so centroid translation alone may move an incorrectly
bound or non-rigid template. The official evaluator must determine whether the
dense candidate improves the registered metrics. P4B remains blocked, and no
Office run is authorized before final Apartment configuration freeze.

## Verification

- Dense audit: 6/6 checkpoints PASS.
- Core/runner/audit tests: 45 passed.
- Temporal association/identity/runtime tests: 261 passed.
- T1 exactness/non-interference tests: 48 passed.
