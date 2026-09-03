# Causal 3RScan Pilot

Date: 2026-09-03

Status: `BLOCKED_DATASET_ACCESS`

## Frozen Sources and Selection

The official `3RScan.json` metadata is bound at SHA-256
`674a00f50f76b198b9de44efd86c390fea3da37ba8f12cf8ccd00045e265fa64`;
the validation scan list is bound at SHA-256
`002229133d4dbc311a01994b8ea31ecac0b662511864493efed9b4febcb0a4e8`.

The visit distribution over all 478 environments is:

| Visits | Exact environment count | At least this many visits |
| ---: | ---: | ---: |
| 2 | 194 | 478 |
| 3 | 160 | 284 |
| 4 | 68 | 124 |
| 5 | 31 | 56 |
| 6 | 8 | 25 |
| 7 | 7 | 17 |
| 8 | 7 | 10 |
| 10 | 1 | 3 |
| 12 | 2 | 2 |

The preregistered rule keeps validation environments whose reference and every
rescan are in the frozen validation list and that contain at least one
rigid/nonrigid/removed change. It sorts by reference UUID and takes the first
10. Forty-seven environments are eligible; the frozen 10 contain 44 visits.
No method output participates in selection.

Selection manifest SHA-256:
`f72febc96538b298323e5b5ad9e99e1d6b472c948d13060f3f1dd953aa5c0ad2`.

## Protocol Validation

The adapter preserves metadata session order, canonical scan UUIDs, fixed
instance IDs, registered change types, and evaluator-only global/object
transforms. Exact duplicate metadata records are normalized; distinct transform
hypotheses remain explicit. Added instances are derived by comparing reference
and current annotations inside the evaluator and are never exposed to the
mapper.

Predictions are strict session prefixes. The synthetic tests cover future-input
rejection, ID switches, false Re-ID, reactivation, current precision/recall,
stale-object false positives, and recall by rigid/nonrigid/added/removed type.
Community stage AP, t-AP, and t-REC remain a separate output namespace and are
not synthesized from the custom exact-ID fields.

## Runtime Gate

Only the metadata and validation list are present. None of the 44 selected
visit directories provides the required RGB-D sequence, refined mesh, instance
annotation, and semantic annotation members. Consequently:

- selection and protocol validation: `PASS`;
- CROVE/session-prefix execution: `BLOCKED_DATASET_ACCESS`;
- community and exact-ID real-data metrics: not run, not zero;
- benchmark promotion evidence: unavailable.

## Decision

Keep 3RScan as a source-bound long-term identity candidate, but do not use it
to replace or demote TESSE from this pilot. Reopen only when all required assets
for the frozen 10 environments are content-addressed and available.
