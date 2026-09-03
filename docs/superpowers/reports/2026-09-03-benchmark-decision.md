# CROVE Benchmark Alignment Decision

Date: 2026-09-03

Status: `FROZEN_EVIDENCE_DECISION`

## Decision

`KEEP_TESSE_HEADLINE_WITH_SPLIT_PROTOCOL`

TESSE remains the only validated route in this audit that covers continuous
short-term dynamics. It must no longer appear as one undifferentiated
comparison: the RSS paper target, the post-release public aggregation, the
current-diagonal diagnostic, common-v2 current-state metrics, and semantic
input conditions have different evidence identities. The evidence does not
authorize replacing or demoting TESSE, because paper-protocol reproduction and
input fairness are incomplete and both alternative real-data pilots are
blocked.

## Frozen Decision Basis

| Gate | Evidence | Outcome |
| --- | --- | --- |
| RSS Table I identity | RSS release contains no evaluator; original paper evaluator/data closure unavailable | `BLOCKED_PAPER_ASSET` |
| Local/upstream aggregation | A6 and P5 match latest-public aggregation within `1.39e-17` | `PARITY` |
| GT/OpenSet fairness | K0 and C1 absent; no oracle-gap closure can be computed | `INCONCLUSIVE_MISSING_CONDITION` |
| Current versus official ranking | 43 unique current-diagonal states; 0/6 finite A6 pairwise reversals | no material mismatch on available artifacts |
| Exact error identity | Patched/unpatched official CSVs byte-identical; P5 sidecars conserve all row mass | `REAL_REPLAY_PASS` |
| 3RScan pilot | metadata/selection/protocol pass; all 44 selected visit asset sets absent | `BLOCKED_DATASET_ACCESS` |
| Panoptic Flat pilot | adapter/metric protocol pass; official archive unavailable to this provider | `BLOCKED_DATASET_ACCESS` |

The suitability scorecard was frozen before pilot results at SHA-256
`f4de8e629dacc3a8e13a1123636179fe3c20ec0a145bbcdf1aff87c64a607694`.
Totals remain TESSE official 16, TESSE current diagonal 18, 3RScan 19, and
Panoptic Flat 16. These are suitability scores, not method scores. The highest
total does not override 3RScan's zero short-term-dynamics score or its missing
runtime assets.

## Root-Cause Classification

| Candidate cause | Classification | Supported boundary |
| --- | --- | --- |
| Local summarizer arithmetic/duplicate handling | ruled out | Exact parity with frozen latest-public aggregation on A6/P5 |
| RSS evaluator/protocol adaptation | unresolved | Current artifacts cannot be called an RSS Table I reproduction |
| Open-vocabulary frontend gap | unresolved | K0/C1 matched oracle matrix is incomplete |
| Mapper/backend gap | unresolved | It cannot be separated from frontend/protocol with current inputs |
| Full-history versus current-state mismatch | not demonstrated | Available output is diagonal-equivalent and has no rank reversal |
| Method-side maintenance tradeoff | confirmed | P5 improves Change/current mIoU/Ghost but loses Object F1 versus A6 |
| Alternative-benchmark advantage | unmeasured | No real 3RScan or Flat CROVE result exists |

The root cause is therefore `PARTIALLY_IDENTIFIED_MULTIFACTOR`, with a
confirmed method-side object-quality versus stale-state tradeoff and unresolved
paper-identity and frontend-fairness components. It is not defensible to assign
the whole gap to either the benchmark or the method.

## Transparent Pareto Report

All values below are frozen Apartment measurements. N/A means unavailable, not
zero. Dynamic F1 is aggregate dynamic surface mass, not an identity metric;
P5 alone also has a debug-only exact association sidecar.

| Method | Object F1 ↑ | Dynamic F1 ↑ | Change F1 ↑ | Current mIoU ↑ | Ghost ↓ | Khronos BG F1@0.2 ↑ | Common BG F@5cm ↑ | Recovery frames ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A6 | **0.372762** | N/A | 0.060853 | 0.142897 | 0.646883 | **0.665124** | N/A | N/A |
| c553 static-anchor | 0.348472 | **0.069225** | **0.088458** | 0.149560 | 0.443760 | 0.650180 | 0.100717 | 450 censored |
| P5 | 0.367952 | **0.069225** | 0.088088 | **0.149563** | **0.441677** | 0.650180 | 0.100717 | 450 censored |
| P6-C | 0.367307 | **0.069225** | 0.087246 | 0.149449 | 0.441689 | 0.650180 | 0.100717 | 450 censored |

The required research axes are kept separate:

- current object quality: Object F1, led by A6;
- current-state semantic quality: current mIoU, led narrowly by P5;
- stale-state suppression: Ghost, led narrowly by P5;
- dynamic identity: not measured as an identity score; aggregate Dynamic F1
  ties the three finite variants and exact identities are available only as a
  P5 diagnostic;
- background/geometry recovery: no candidate recovers either observable event
  inside 450 frames, and the common background F@5cm values are unchanged.

No variant Pareto-dominates all others. Historical all-or-nothing gates remain
unchanged; future selection should publish this vector rather than hide the
Object/Ghost tradeoff in one weighted score.

## Paper Benchmark Layout

1. Main static table: Replica and ScanNet200 open-vocabulary calibration.
2. Main dynamic table: matched end-to-end TESSE input conditions, with current
   Object/Dynamic/Change and common-v2 current mIoU/Ghost/background/recovery in
   explicitly separated column groups.
3. Main ablation table: A6, c553, P5, and P6-C Pareto dimensions, with missing
   values shown as N/A and Apartment-only scope stated.
4. Supplement: RSS Table I targets and the protocol-identity block; exact
   attribution; K0/K1/C0/C1 fairness; source-bound 3RScan/Flat protocol
   readiness; all failure and censored-event details.

Do not publish 3RScan or Flat as numerical main-table benchmarks until their
frozen real assets, baselines, and CROVE runs exist. Do not place RSS paper
numbers beside open-vocabulary CROVE as a same-input local baseline. Oracle
rows remain mechanism diagnostics and are never ranking eligible.

## Evidence Boundary

This decision preserves every failed or blocked stage. It does not claim that
TESSE is fully validated, that CROVE is competitive with a matched Khronos
system, that current and genuine full-history rankings are equal, or that the
alternative benchmarks favor CROVE. Those cells remain unavailable, not zero.
