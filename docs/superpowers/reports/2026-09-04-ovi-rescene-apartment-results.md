# OVI-MAP x ReScene4D Two-Visit Apartment Results

Date: 2026-09-04

Status: `DETERMINISTIC_INFRASTRUCTURE_COMPLETE`

Learned status: `RESCENE_LEARNED_RESULT_BLOCKED`

## Bound Execution

| Item | Identity |
| --- | --- |
| Source commit | `96ba693b2996aa19b3e695beeb874f835c4375a9` |
| Matrix SHA-256 | `08dc52143b6c976d2b9f516fb312aa9e52ad42befe47e4fbb99cfaee6f07ee4d` |
| OVI input manifest SHA-256 | `1996a0008649cdeb58daa920a05d30366a971b2fef77ae2e6a4ae1ca8a3f7b94` |
| Matrix summary SHA-256 | `7aa21465fcf6870a94448b2c34022251bbad79e5dadaed58dbd7ed1d4af26dab` |
| Small receipt | `configs/evaluation/results/ovi_rescene_two_visit/apartment-96ba693.json` |
| Large artifact root | `/home/ww/oviovo_baseline_runs/20260904_ovi_rescene_two_visit/apartment-b0-b6-entity-lift-96ba693` |
| Large artifact bytes | 575,211,024 |

The matrix summary binds every row receipt; every row receipt binds its command,
logs, current-map manifest, and metric receipt. The run used Apartment frames
766--1021 and 1217--1472 and records the exact runtime source commit.

## Pareto Results

`N/A` values are unavailable under the frozen two-visit final-snapshot protocol
and are not zero-filled.

| Variant | Object F1 | Dynamic F1 | Change F1 | Current mIoU | Ghost | BG F@5cm | Surface F@5cm | Unobserved recall | Final points | Adapter s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 OVI t0 only | N/A | N/A | N/A | 0.095801 | 0.999846 | 0.000000 | 0.414514 | 0.075068 | 14,621,367 | 0.000 |
| B1 OVI union | N/A | N/A | N/A | 0.129288 | 0.999846 | 0.157900 | 0.468438 | 0.084523 | 26,958,216 | 0.000 |
| B2 OVI t1 only | N/A | N/A | N/A | 0.120858 | 0.000000 | 0.445280 | 0.420144 | 0.018912 | 12,336,849 | 0.000 |
| B3 visibility composer | N/A | N/A | N/A | 0.135886 | 0.000000 | 0.366360 | 0.431335 | 0.075646 | 15,622,601 | 0.000 |
| B4 geometric pairing + visibility | N/A | N/A | N/A | 0.135886 | 0.000000 | 0.366360 | 0.431335 | 0.075646 | 15,622,601 | 143.514 |
| B5 OVI + ReScene + visibility | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| B6 OVI + ReScene, no visibility | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

B5 and B6 are `BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT`; neither row is
ranking-eligible.

## Gates And Attribution

All executable preregistered gates pass:

- B2 practical Ghost: 0.000000 <= 0.20.
- B3/B4 Ghost relative to B2: 0.000000 <= 0.02.
- B3/B4 unobserved recall gain over B2: 0.056735 >= 0.05.

The initial full-label B3/B4 diagnostic retained 12,056 changed-region points
from two old Couch entities, of which 12,010 were Ghost. Exact point-group
provenance showed that over 90% of both entities had already been suppressed by
signed visible-free evidence; only thin occluded/unobserved residues remained.
The frozen correction lifts that robust evidence to the OVI object-entity
boundary. The corrected B3/B4 Ghost count is zero, while total current-surface
coverage changes only from 0.305597 to 0.305331 and observed-region stale
precision improves from 0.677428 to 0.930963.

## Decision

B3 is the selected deterministic candidate. It has the same measured Pareto
vector as B4 without B4's 143.514-second pairing cost. The result establishes
that OVI dense per-visit maps plus signed revisit visibility can match the
t1-only Ghost floor while recovering 0.056735 absolute unobserved-region
recall. It does not establish ReScene value, dynamic tracking F1, change F1, or
cross-visit identity accuracy.

ReScene verdict: `RESCENE_BLOCKED_EXTERNAL_ASSET`.
