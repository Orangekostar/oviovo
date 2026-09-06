# OVI-MAP Backbone x ReScene Real-Evidence Handoff

Date: 2026-09-06

Branch: `research/ovi-rescene-real-evidence`

## Outcome

The OVI-MAP-backed path is executable without changing the frozen OVI geometry.
ReScene is used only for cross-visit identity evidence. The real Apartment forward,
six-pair 3RScan development matrix, downstream ownership readout, and one
evidence-gated resolver adaptation all passed their source-binding checks.

The selected adaptation is a query-consistent fragment-union resolver with a
prediction-confidence threshold of `0.3`. This threshold was selected on the six
frozen 3RScan development pairs and is not an unseen-test result.

## 3RScan Inventory

The bound `3RScan.json` SHA-256 is
`674a00f50f76b198b9de44efd86c390fea3da37ba8f12cf8ccd00045e265fa64`.
Across all 478 environments:

| Visit condition | Scene count |
| --- | ---: |
| exactly T = 2 | 194 |
| T >= 3 | 284 |
| T >= 4 | 124 |
| T >= 5 | 56 |
| T >= 6 | 25 |

The current result uses six complete, validation-only, exactly-T=2 pairs selected
without method output. Their manifest SHA-256 is
`f6224c10ea7891dac50baf1a814a44f3a3e54210daaf76fdc6a7df993af67b91`.

## Apartment Evidence

The supported V3 input retains `97.43%` of source points, `95.58%` of adapter
tokens, `95.70%` of model tokens, and `99.34%` of entities. The native forward took
`3.814 s` and peaked at `10.97 GB` allocated GPU memory.

The frozen B4 geometric path produces 120 unique relations. The conservative
Apartment ReScene resolver produces eight unique 1:1 relations, four static and
four moved; four overlap B4 topology. Conditional query coverage is `0.08`, so this
result establishes executable learned evidence, not Apartment identity accuracy.

## 3RScan Development Results

Official native ReScene metrics, computed by the bound `stmetrics` evaluator:

| Metric | Value |
| --- | ---: |
| t-AP | 0.1931 |
| t-AP@0.50 | 0.3499 |
| t-AP@0.25 | 0.5491 |
| t-REC | 0.3781 |
| t-REC@0.50 | 0.5091 |
| t-REC@0.25 | 0.6062 |

Custom cross-visit association on the common 5 cm evaluator grid:

| Method | P@0.50 | R@0.50 | F1@0.50 | TP | FP | Rigid recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G full geometry | 0.0118 | 0.0265 | 0.0163 | 4 | 336 | 0.0000 |
| F independent features | 0.0133 | 0.0066 | 0.0088 | 1 | 74 | 0.0000 |
| R legacy query union | 0.0465 | 0.1060 | 0.0646 | 16 | 328 | 0.1176 |
| R one-to-one segment | 0.0000 | 0.0000 | 0.0000 | 0 | 49 | 0.0000 |
| R tuned fragment union | **0.4531** | **0.1921** | **0.2698** | **29** | **35** | **0.2941** |

The threshold-zero resolver exactly reproduces the recorded R legacy counts and
metrics before selecting the threshold. The result supports a resolver/fragment
organization diagnosis: raw ReScene queries contain signal, while forcing a whole
object query onto one atomic mesh segment destroys endpoint IoU.

## Ownership and Completion

A0, A1, and A2 share geometry SHA-256
`863e13b499e31c5cbd3e0c3f704e0a03558b806593f66579aa133ab33fa5e496` and
have zero geometry mutations. A1 applies 119 B4 relations in the supported domain;
one source-only removed candidate has no supported token and is explicitly excluded.
A2 applies eight ReScene relations.

Apartment has no protocol-compatible instance identity GT, so AP and PQ are
`NOT_COMPUTED`, not zero. C1 B7-G accepted five of 18 registrations and recovered
1,792 historical points, but all evaluated map metrics equal C0 B3:

| Metric | C0 B3 | C1 B7-G |
| --- | ---: | ---: |
| current mIoU | 0.1359 | 0.1359 |
| ghost | 0.0000 | 0.0000 |
| background F1@5 cm | 0.3664 | 0.3664 |
| surface F1@5 cm | 0.4313 | 0.4313 |
| surface precision@5 cm | 0.7344 | 0.7344 |
| surface recall@5 cm | 0.3053 | 0.3053 |

C2 is `NOT_COMPUTED_MISSING_R_BOUND_RECOVERY_ARTIFACT`. O1 and O2 are unavailable
because the Apartment pair lacks instance identity and object-transform GT. D1
native sensor support and D2 actual 3RScan OVI reconstruction are also
`MISSING_ASSET`; none is represented as zero.

## Claim Boundary

Supported claims:

- OVI-MAP can remain the immutable geometry/semantic backbone while ReScene supplies
  cross-visit identity evidence.
- On the frozen 3RScan development pairs, query-consistent fragment union with a
  `0.3` confidence gate substantially improves custom association over the two
  preregistered ReScene readouts.
- Identity regrouping does not move or mutate OVI geometry.

Unsupported claims:

- unseen 3RScan generalization or SOTA;
- Apartment instance AP/PQ improvement;
- C2 learned completion gain or oracle completion ceiling;
- D1/D2 domain robustness or Office generalization.

No value from this package was inserted into `benchmark_tables_baselines.md` because
the available metrics do not match a final held-out table protocol.

## Reproducibility

Primary small results:

| Artifact | SHA-256 |
| --- | --- |
| `apartment_native_b5_v3.json` | `72e5c5b91d99956ceec17784371208eeb8afba4bffd698bd69b657e6208d18c4` |
| `3rscan_t2_matrix_v1.json` | `2bb719c6d511387e07228cb0ba3f2faacbf334321f51408b0e7b01c99e1167b4` |
| `apartment_ownership_completion_v1.json` | `c7f2ea84dc8b4337af47f1dc0aeaba904f97e139c9393598ae6fedd6305f1575` |
| `adaptation_decision_v1.json` | `47bb9c889c96e6ee880148f5345f4141439a8086f45758cd55a6827429b8dc3e` |
| `resolver_tuning_v1.json` | `53d0ef99ee234592e6c1a17c7e756943976b1a528a3ff9b2c485224158713c61` |

The complete command, evaluated commit, input hashes, runtime, device, status, and
artifact root are in
`configs/evaluation/results/ovi_rescene_real_evidence/experiments.csv`.
