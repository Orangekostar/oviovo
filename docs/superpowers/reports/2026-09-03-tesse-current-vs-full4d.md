# TESSE Current-Diagonal Versus Available Khronos Output

Date: 2026-09-03

Status: `NO_MATERIAL_RANK_MISMATCH_ON_AVAILABLE_DIAGONAL_ONLY_ARTIFACT`

## Protocol Boundary

`TESSE_CURRENT_DIAGONAL` requires an explicit map-to-robot timestamp binding
and selects only rows with `belief_time_ns == robot_time_ns`. It preserves the
same GT, predictions, association thresholds, and semantic condition as each
paired post-release Khronos result. It is a diagnostic protocol, not an RSS
Table I result.

The public evaluator's inner historical loop does not use its loop variable.
Each method therefore has 946 raw object/dynamic rows but only 43 unique
`(map_name, query_time)` states, all on the current diagonal. Conflicting
duplicates, future belief times, future trajectory times, and divergent exact
attribution sets are rejected before aggregation.

## Results

| Method | Object F1 | Dynamic F1 | Change F1 | Raw rows | Unique diagonal states |
| --- | ---: | ---: | ---: | ---: | ---: |
| A6 | 0.372762 | N/A | 0.060853 | 946 | 43 |
| c553 static-anchor | 0.348472 | 0.069225 | 0.088458 | 946 | 43 |
| P5 | 0.367952 | 0.069225 | 0.088088 | 946 | 43 |
| P6-C | 0.367307 | 0.069225 | 0.087246 | 946 | 43 |

For every available cell, the current-diagonal value equals the paired
post-release official aggregate. P5 additionally validates 118,859
static/change and 1,135,450 dynamic exact-attribution events against the
official row and time domains.

The output receipt SHA-256 values are:

| Method | Receipt SHA-256 |
| --- | --- |
| A6 | `808aa57b770271ada892e3c207e9f8cc180b9405b319e5cbf9182dc289c01dd8` |
| c553 static-anchor | `a70c55259b85b6f6885a5150e9094a1eb999bebb3cf250e1df4647df1d2d20bf` |
| P5 | `13e9b569bd8afb523982817a708dfe7199bb88e15eaf0204b40fb68b878951af` |
| P6-C | `d9aca14d6ba72117d60b4e23882c6c247a1cae5c6fc693a002fc9f6354b24d99` |

## Rank Analysis

Official and current ranks are identical:

- Object: A6, P5, P6-C, c553 static-anchor.
- Change: c553 static-anchor, P5, P6-C, A6.
- Dynamic: P5, P6-C, c553 static-anchor; A6 is N/A.

Across the six finite A6 pairwise comparisons there are zero direction
reversals. No candidate has multiple current gains paired with official
non-response, and the available rows contain no retrospective off-diagonal
mass. The preregistered material-mismatch decision is therefore `false`.

Separate common-v2 diagnostics still show current-state mechanism effects:

| Method | Current mIoU | Ghost |
| --- | ---: | ---: |
| A6 | 0.142897 | 0.646883 |
| c553 static-anchor | 0.149560 | 0.443760 |
| P5 | 0.149563 | 0.441677 |
| P6-C | 0.149449 | 0.441689 |

These values remain separate because their point-level protocol is not the
Khronos object/change association protocol.

## Decision

The available artifacts cannot demonstrate a full-4D-versus-current task
mismatch: they contain the current diagonal only. This rules out a favorable
rank reversal claim, but it does not prove that a genuine historical 4D grid
would be aligned with CROVE's bounded current-state claim. That question
remains untested rather than negative.
