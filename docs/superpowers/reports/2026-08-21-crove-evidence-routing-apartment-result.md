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
