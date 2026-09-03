# TESSE Input Fairness Audit

Date: 2026-09-03

Status: `INCONCLUSIVE_MISSING_CONDITION`

## Frozen Matrix

| Condition | Input condition | Status | Ranking eligible |
| --- | --- | --- | --- |
| K0 | Khronos paper-like, simulator GT semantics | `BLOCKED_PAPER_ASSET` | no |
| K1 | Khronos public OpenSet input | `INCONCLUSIVE_MISSING_CONDITION` | no |
| C0 | CROVE A6, open-vocabulary frontend | `PASS` | historical result only |
| C1 | CROVE, simulator GT semantics | `INCONCLUSIVE_MISSING_CONDITION` | no |

K0 remains blocked by the B3 evaluator/data identity gate. The available
post-release artifacts cannot be relabeled as the RSS Table I GT-semantics
condition. No source-bound public OpenSet artifact establishes K1.

C0 is bound to the frozen A6 Apartment result: Object F1 `0.3727621554`,
Dynamic F1 N/A, and Change F1 `0.0608534569`. C1 was not assigned a score. The
configured source database and a per-frame semantic stream proving RGB-D/GT
alignment are absent from the current asset mount. The fail-closed oracle
builder therefore rejected C1 before any mapper run.

## Diagnosis

The preregistered rule evaluates only positive finite
`Khronos_GT - CROVE_open` gaps and requires at least 50% closure on a strict
majority of at least two available headline metrics. K0 and C1 are both
missing, so no valid gap or closure value exists. The result is
`INCONCLUSIVE_MISSING_CONDITION`, not `FRONTEND_DOMINATED`,
`MAPPER_OR_PROTOCOL_DOMINATED`, or zero performance.

The audit receipt SHA-256 is
`a4fa8b88d7c63b8a0498080330997d523db4df1b974d5b3439ade9f20165a325`.
The C1 builder failure is retained as missing-input evidence; GT semantics are
always `ORACLE_DIAGNOSTIC` and `NOT_RANKING_ELIGIBLE`.

## Decision

Do not attribute the TESSE gap to the frontend or backend from the current
matrix. Do not compare C0 directly with paper K0 as though their perception
inputs were matched. Reopen this diagnosis only after source-bound K0 and C1
conditions exist.
