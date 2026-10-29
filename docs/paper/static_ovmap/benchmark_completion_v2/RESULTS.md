# Current evidence coverage

This snapshot records measured coverage; completion dimensions are reported independently.

| Completion dimension | Complete |
|---|---|
| software_complete | false |
| calibration_complete | true |
| legacy_complete | false |
| standard_coverage_complete | false |
| external_comparisons_complete | true |
| publication_complete | false |
| claim_supported | false |

| Standard scope | Scored scenes | Scored cells |
|---|---:|---:|
| replica48 | 8/8 | 32/64 |
| scannet200_val | 312/312 | 2496/2496 |

Calibration: 1331 logical candidates; original279 objects and9 abstentions retained. No new training.
Legacy eligible substitutions and standard GLOBAL predictions use separate pools. Partial standard scores stay out of T1/T2.
Author-reported values are references, never counted as local model executions.
Only dependencies listed in the current remaining_jobs.json are unresolved; completed inputs and method scopes are retained.
Current CONF12 calibration and LOFO use exposed novel labels. Neither proves unseen-class calibration or the old physical grouping hypothesis.
All displayed values and per-seed aggregation are generated from result_store.json; see tables/S2.md for partial AP and tables/S6.md for full-GT proposal coverage.
