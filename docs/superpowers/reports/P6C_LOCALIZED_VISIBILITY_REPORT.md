# P6-C Localized Visibility Report

## Scope And Contract

- `CODE_EVIDENCE`: L1 changes only P5's spatial aggregation from whole anchor
  to signed 5 cm per-voxel ownership. P5 thresholds, source mapper, movement
  state, semantics, and dense native anchor coordinates remain unchanged.
- `CODE_EVIDENCE`: the immutable anchor is filtered only at composition time;
  bit-packed ownership masks are separate hash-bound sidecars and later PRESENT
  evidence restores the original points.
- Scene: Apartment only. Office was not accessed.

The frozen policy SHA-256 is
`53f4d808bda65e12a88496e2d27c887b24d9b1c5b47699b300261f4f1a67ec3c`
(518 bytes).

## State And Coverage

- `MEASURED_EVIDENCE`: 1,745/1,745 frames and 43/43 official states.
- `MEASURED_EVIDENCE`: 36 unbound anchors, 41,797 full 5 cm voxels, and
  15,938 evidence-sample voxels.
- `MEASURED_EVIDENCE`: implemented state is 2,487,978 bytes, below the 8 MB
  L2 trigger; L2 is `NOT_RUN_NOT_NEEDED`.
- `MEASURED_EVIDENCE`: 43 mask sidecars total 225,363 bytes. Across all
  checkpoint-anchor states, 570 are unchanged, 746 partially suppressed, and
  232 dormant. The final checkpoint has 10 unchanged, 18 partial, and 8
  dormant anchors.

The composition manifest SHA-256 is
`259f181e2e562cc6546e8ca08727049c48e71bde84f3bb05ba11bfa525e73bad`
(1,094,732 bytes). The final snapshot is 12,031,464 bytes with SHA-256
`1d4f7999be877f2b71503f4580f2ee8c339d5874a0e46a933428c4c521ac81af`;
the final entity file is 51,857 bytes with SHA-256
`878fc1835281249079963565a7f30d25484c43b864d0215b488035495f00f207`.

## Formal Metrics

| Metric | P5 | P6-C L1 | Delta | P6 hard gate | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Object F1 | 0.367952 | 0.367307 | -0.000645 | >= 0.372762 | FAIL |
| Dynamic F1 | 0.069225 | 0.069225 | 0.000000 | >= 0.069225 | PASS |
| Change F1 | 0.088088 | 0.087246 | -0.000842 | >= 0.088088 | FAIL |
| Current mIoU | 0.149563 | 0.149449 | -0.000114 | >= 0.149563 | FAIL |
| Ghost | 0.441677 | 0.441689 | +0.000011 | <= 0.441677 | FAIL |

`MEASURED_EVIDENCE`: both common-v2 repetitions are byte-identical at SHA-256
`1c778edfc8fd039ca2dd0c34a76967d0f001f7e24ec4912459c2e36117bbfca5`.
The official metric file is SHA-256
`eabb8f26ca478197acca4a89129a4089061cb4255778d09a15798e0c0b3d805c`
(1,147 bytes), and the final gate receipt is SHA-256
`05145bd2ccf9bc0d8a9e88142f48b78b729dcc34d2fd18aca70a1713ed53192b`
(5,073 bytes).

## Decision

`MEASURED_EVIDENCE`: `NO_GO / REJECTED_RETAIN_A6`. L1 does not recover the
missing Object F1 and slightly regresses P5 Change, Current, and Ghost. The
primary hypothesis is rejected for this frozen policy and state scale; no
threshold tuning or L2 run is authorized from these metrics.

`HYPOTHESIS`: partial geometric preservation alone is insufficient because the
remaining error also depends on semantic/instance identity quality. P6-A may
identify generalizable evidence features, but its ID-level diagnostics cannot
be promoted as a method rule.

Artifacts are under
`/home/ww/oviovo_baseline_runs/20260903_crove_localized_current_ownership/p6c/apartment_l1`.
