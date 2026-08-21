# CROVE Evidence-Routing Apartment Development Result

Date: 2026-08-21
Status: development-only; promotion failed
Scene: TESSE-CD Apartment
Method display name: CROVE
Artifact method key: OVIV2

## Run Identity

| Field | Value |
| --- | --- |
| Capture commit | `c069510e30cd31412fb9171482df39aa8c042d30` |
| Algorithm hash | `1935e744f36c37a7648f1baf9474fa5d654e73aacc89c95f59b55c28a423c632` |
| Config SHA-256 | `961e4ef9371f58c1662d23ae553f66a9f0250f2291c0886cbbf30f931a78d0e1` |
| Input SHA-256 | `a67114700d4fb00c40036a0586d541626c007abb83f298f32c4c18a060967f74` |
| Capture mode | `causal_table_metrics_only` |
| Capture status | `PASS` |
| Publication eligible | `false` |

The capture mode deliberately excludes cumulative and occlusion evaluation. It
is valid for Apartment selection but cannot be published as a formal Office or
final-table run.

## Metrics

| Metric | A6 baseline | Routed candidate | Delta | Gate |
| --- | ---: | ---: | ---: | --- |
| Official Obj. F1 | 0.372762 | 0.331880 | -0.040882 | **FAIL**: regression exceeds 0.01 |
| Official Dyn. F1 | unavailable | 0.026204 | finite | diagnostic gain |
| Official Chg. F1 | 0.060853 | 0.092390 | +0.031537 | PASS |
| Official background F1@0.2 | 0.665124 | 0.663438 | -0.001686 | diagnostic |
| Common-v2 current mIoU | 0.142897 | 0.164668 | +0.021770 | PASS |
| Common-v2 ghost rate | 0.646883 | 0.591129 | -0.055754 | PASS |
| Common-v2 background F@5cm | 0.096979 | 0.109195 | +0.012216 | diagnostic gain |
| Recovery latency (frames) | 450 | 450 | 0 | no gain |

The official result contains 43 finite states. Repeated official aggregation
produced byte-identical metric files.

## Promotion Decision

**REJECTED.** The candidate satisfies the preregistered dynamic/change,
current-mIoU, and ghost gates, but Obj. F1 regresses by 0.040882. The paper
tables must retain A6 until a candidate passes every hard gate. Identity-switch
precision is not emitted by the current artifact contract, so that gate also
remains unavailable rather than being inferred.

## Failure Evidence

The loss is dominated by extra object hallucinations, not missed-object recall:

| Diagnostic | A6 | Routed candidate | Delta |
| --- | ---: | ---: | ---: |
| Bridge symbol assignments | 231 | 376 | +145 |
| Latest dynamic assignments | 47 | 126 | +79 |
| Latest static assignments | 184 | 250 | +66 |
| Epoch-reset triggers | 189 | 376 | +187 |
| Proposal triggers | 514 | 1223 | +709 |
| Re-ID opportunities | 4067 | 14658 | +10591 |
| Re-ID triggers | 205 | 419 | +214 |
| Motion rejections | 38 | 126 | +88 |

Across the 43 official states, the candidate adds true detections but adds far
more hallucinated objects. The largest Obj. F1 losses occur in states 16-18,
where hallucinations rise from 76 to 116 while detections rise only from
28-29 to 30-31.

The implementation also routes identity-qualified motion into the same
`qualifies_as_motion` decision used for geometry epoch resets. This conflicts
with the design matrix, where qualified identity may update dynamic state but
not geometry epoch. The next candidate must separate dynamic-state
qualification from geometry-epoch qualification. A second diagnostic variant
may restore legacy active-prototype adaptation to test whether prototype
isolation is the remaining fragmentation source.

## Artifact Bindings

| Artifact | SHA-256 |
| --- | --- |
| Capture status | `7dd14105264f414863ee965430b13873d16dcb0e7b253a820b3f5764c66135b8` |
| Run manifest | `e7a179d0ee5f901875db3cffaacfcbd255d47401a9e703b9bbbd5c0d39a42899` |
| Temporal manifest | `2432a6e30972a1dfa6e328d395b91b7e73dae71a68c6ec2dc378a605f2d0287b` |
| Common-v2 summary | `dbdf3a2eede4c7f4eb4368af615fb9e4a26f70661eeafa2f880becf5fdeb196e` |
| Bridge manifest | `107de927bd1ffe5bf9095919248ffb2c9a7ee087892c5c86f877834e1be340b6` |
| Khronos run status | `0b24fcb812831e6272f8ee85a45050d6b2ae9728db6c20c49419d4530c3af5f7` |
| Official metrics | `726250261f2fbf72883f5aded8cb18db10024b8dc783840dd7a5f088137cc5f6` |
| Official metrics repeat | `726250261f2fbf72883f5aded8cb18db10024b8dc783840dd7a5f088137cc5f6` |
| Evaluation status | `4de31d49dd76b240e73639cdacf4109f0f3b07ebbb6f8147b741981d483da83f` |

No result was invented or estimated in this report. Every numeric cell above
comes from the source-bound A6 or routed Apartment artifacts. No main paper
table was changed.

## Component-Qualified Follow-up

The geometry/identity qualification correction was evaluated at commit
`8301984bcf82966ed6d6eb7b3320d869cd7604d3`. A fixed diagnostic ablation that
restores weak-match prototype adaptation was evaluated at commit
`21e74e388ed10018819156e152781fe3a48c5edf`. Both captures processed all 1,745
Apartment frames and produced 43 finite official states. The repeated official
metric files are byte-identical for both runs.

Both variants use algorithm hash
`1935e744f36c37a7648f1baf9474fa5d654e73aacc89c95f59b55c28a423c632`.
The main capture input SHA-256 is
`a67114700d4fb00c40036a0586d541626c007abb83f298f32c4c18a060967f74`.
The ablation capture input SHA-256 is
`51f4f55960b42750b9b4be3e7be060afef9e858e6659602cac5866da877662c6`;
it differs because node101 used the checked relocated asset paths. Its
algorithm configuration remains identical to the canonical Apartment config.

### Measured Metrics

| Metric | A6 | Component-qualified | Delta | No-prototype-isolation | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official Obj. F1 | 0.372762 | 0.350702 | -0.022060 | 0.320662 | -0.052101 |
| Official Dyn. F1 | unavailable | 0.070726 | finite | 0.115604 | finite |
| Official Chg. F1 | 0.060853 | 0.070489 | +0.009635 | 0.094201 | +0.033348 |
| Official background F1@0.2 | 0.665124 | 0.663640 | -0.001484 | 0.667956 | +0.002832 |
| Common-v2 current mIoU | 0.142897 | 0.117990 | -0.024907 | 0.095379 | -0.047518 |
| Common-v2 ghost rate | 0.646883 | 0.627961 | -0.018922 | 0.571717 | -0.075166 |
| Common-v2 background F@5cm | 0.096979 | 0.081008 | -0.015971 | 0.075476 | -0.021503 |
| Recovery latency (frames) | 450 | 450 | 0 | 450 | 0 |

### Gate Decision

**BOTH REJECTED; RETAIN A6.** The component-qualified variant misses the
required change gain by 0.000365, regresses Obj. F1 by 0.022060, and regresses
current mIoU by 0.024907. The prototype ablation passes the change and ghost
gates but regresses Obj. F1 by 0.052101 and current mIoU by 0.047518. Dyn. F1
is finite for both variants, but no finite A6 Dyn. F1 exists for a preregistered
delta comparison. Neither variant is eligible for an Office run.

Fragmentation diagnostics also prevent promotion:

| Diagnostic | A6 | Component-qualified | No-prototype-isolation |
| --- | ---: | ---: | ---: |
| Bridge symbol assignments | 231 | 259 | 249 |
| Latest dynamic assignments | 47 | 84 | 94 |
| Latest static assignments | 184 | 175 | 155 |
| Epoch-reset triggers | 189 | 148 | 147 |
| Proposal triggers | 514 | 916 | 1114 |
| Re-ID opportunities | 4067 | 3369 | 5512 |
| Re-ID triggers | 205 | 230 | 167 |
| Motion rejections | 38 | 132 | 210 |

The correction substantially reduces epoch resets and avoids the routed
candidate's 376-symbol failure, but it does not restore A6 object identity or
static semantic quality. Restoring weak-match prototype updates increases the
dynamic/change scores and lowers ghosting, while further damaging Obj. F1 and
current mIoU. This isolates prototype adaptation as a precision/recall tradeoff,
not a valid replacement for A6.

### Component-Qualified Artifact Bindings

| Artifact | SHA-256 |
| --- | --- |
| Capture status | `b9b6978c3531752cb3edda05af004cbbb30f7a6c3ee67ff1d1e67fd8024d0a1c` |
| Run manifest | `241ff61ca6e6e041eb94c6749c0ded5fdae35b667e1ffc4478c7062bc0e21a4c` |
| Temporal manifest | `c17fc21b5aeb976e37a596ba1881af71871fbb9b0cb6803bf60d9f5bd89f2120` |
| Common-v2 summary | `4c16f437615a6ecbf5a6eb684968c5fb9b14f546ceffde4103a11723001555c7` |
| Bridge manifest | `95a096e5a31235dc67ef02b58c794a37dc5bdbfcfbaf38913c62f0d3ed074ecf` |
| Khronos run status | `5880e27b139469eaba9e8b494d1bf2a1c9fba885690305d158799586bc942e65` |
| Official metrics and repeat | `26b66416f5346dfefd10cbc7af3c8f5811e24110e4639a5d035bcfdf4bf910cb` |
| Evaluation status | `a284cba00ce3021b913a5ced09bdd2700e8c2f827cb4548534397b862167481c` |

### Prototype-Ablation Artifact Bindings

| Artifact | SHA-256 |
| --- | --- |
| Capture status | `5d23e443a27c146cdf0d010b0e1c00c5efade3fda4e045ac782a904c5370ce53` |
| Run manifest | `35da1f312960fd06fc05aa2d572c71dc41391ce8e762ab684e809ac6879ece4f` |
| Temporal manifest | `ffe172b0bbfc07bda3c1713968e8fbf1523f842263ec50f3f01495320883a843` |
| Common-v2 summary | `fa0952dd6bf9a2f84357292e47d5b3e704cf06211f155f457c95d5ddca38331d` |
| Bridge manifest | `2ffe541a09e73054d74340f07ace61f54a0d646026116455eebf880a18796ee3` |
| Khronos run status | `905bd43fc31d4a62f840bf8a3da3c18cc3a1cbaa908f18f5725d5b76454b11f4` |
| Official metrics and repeat | `21c48ceb4875e9a26f7b992160790d123503cd4964b0f644847875ead34d0feb` |
| Evaluation status | `d1369c1ab5348153d4a0239c8ecddb57039f473edafbc3450d49998ba3ae660d` |

No main-paper or supplementary table was changed by this follow-up.
