# P2 Root-Cause Report: CROVE Apartment Dynamic/Ghost Recovery

## Scope and Evidence Contract

- Branch: `research/crove-dense-current-state-recovery`.
- Frozen base: `c55396b24d7706016a569c951914c79c9f641408`.
- Development scene: Apartment only. Office outputs were not accessed.
- Runtime evidence is causal. Ground truth is used only by the offline evaluator
  and failure-attribution analyzer.
- The official object Ghost partition is exact over 40 common-v2 event-frame
  evaluations. Background rows are extended diagnostics and are excluded from
  the official Ghost metric.

## Bound Sources

| Source | SHA-256 |
| --- | --- |
| Source run manifest | `14265c0cb9358c7936cd9fb1baf1924bc8f55d156859d60eabfbd7c6bc4cb337` |
| Source runtime diagnostics | `4e3eef84499bc5ad991d553fc514a071b0000b9cb62ce812e34de0165c9efdc5` |
| Official dynamic-object CSV | `61ffb22a2d95f702517abaa4621f0eadd512e50c1cc7c9a7952058c457ac19ff` |
| Official metrics | `9ceb7024d68857a020f8ac059f5863530d51de3a7e58cf8d592eeda34d8f3855` |
| P2 attribution JSON | `2e3d47cbdb2bc627a886e843d843f95d6a7bdd806db93d66a9c6a8da1382b7bc` |
| Apartment config | `961e4ef9371f58c1662d23ae553f66a9f0250f2291c0886cbbf30f931a78d0e1` |

The source run processed 1,745 frames. Its algorithm hash is
`1935e744f36c37a7648f1baf9474fa5d654e73aacc89c95f59b55c28a423c632`
and its input hash is
`51f4f55960b42750b9b4be3e7be060afef9e858e6659602cac5866da877662c6`.

## Frozen Apartment Result

| Metric | Value |
| --- | ---: |
| Obj F1 | 0.348472 |
| Dyn F1 | 0.069225 |
| Chg F1 | 0.088458 |
| Current mIoU | 0.149560 |
| Mean official Ghost | 0.443760 |

The official dynamic-object CSV contains 43 unique state rows: 188 detected,
19,556 missed, and 22,431 hallucinated point matches. These evaluator counts do
not expose a one-to-one runtime entity identity, so this report does not relabel
them as association, motion, or proposal failures without additional evidence.

## Exact Ghost Attribution

`MEASURED_EVIDENCE`: all 3,955,583 official object predictions and all
2,152,474 official ghost matches are partitioned exactly.

| Authority | Predicted points | Ghost matches | Ghost contribution |
| --- | ---: | ---: | ---: |
| `ovimap_anchor_unbound` | 2,974,600 | 1,592,890 | 74.002752% |
| `ovimap_anchor_bound_unchanged` | 978,381 | 557,657 | 25.907723% |
| `crove_temporal_new` | 2,421 | 1,879 | 0.087295% |
| `crove_temporal_moved` | 181 | 48 | 0.002230% |

Within the 557,657 bound-unchanged ghost matches, the current overlay state at
the same event frame gives an exact secondary partition:

| Bound-anchor state | Ghost matches | Total official Ghost |
| --- | ---: | ---: |
| `dynamic_state != dynamic` | 356,093 | 16.543% |
| `geometry_epoch_advanced == false` | 174,264 | 8.096% |
| All current moved gates pass but moved geometry is not emitted | 27,300 | 1.268% |

The last row is a readout-state mismatch, not a threshold failure. A moved state
can persist in `AnchorOverlayState` while the current temporal prediction is
missing; `compose_anchor_checkpoint()` then falls through to the old anchor.

Top object contributors are `ovimap:81` (845,748 ghost matches), `ovimap:5`
(215,058), `ovimap:117` (186,984), `ovimap:102` (173,052), `ovimap:110`
(168,600), and `ovimap:129` (164,418).

## Per-Event Evidence

Official rows use the ten common-v2 evaluation frames for each event. Runtime
mechanism counts use disjoint post-intervention intervals so a frame is not
assigned to two events.

| Event | Official detected | Missed | Hallucinated | Object ghost | Unbound ghost | Bound ghost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 69 | 2,611 | 1,908 | 352,328 | 216,214 | 135,340 |
| 02 | 41 | 4,377 | 4,529 | 40,237 | 0 | 40,035 |
| 03 | 71 | 5,153 | 6,738 | 3,553 | 0 | 3,064 |
| 04 | 7 | 6,029 | 7,938 | 1,756,356 | 1,376,676 | 379,218 |

| Event interval | Proposal opp./trigger | Re-ID opp./trigger | ICP accept/reject | Motion reject | Epoch opp./trigger |
| --- | ---: | ---: | ---: | ---: | ---: |
| 01, frames 263-710 | 333 / 333 | 3,659 / 93 | 3,087 / 1,707 | 65 | 357 / 108 |
| 02, frames 711-857 | 15 / 15 | 2,199 / 6 | 1,191 / 559 | 3 | 83 / 22 |
| 03, frames 858-1021 | 86 / 86 | 2,945 / 113 | 902 / 448 | 10 | 119 / 41 |
| 04, frames 1022-1472 | 542 / 542 | 5,481 / 207 | 4,057 / 1,748 | 36 | 362 / 121 |

`CODE_EVIDENCE`: re-ID opportunities are qualified dormant candidate pairs,
not missed ground-truth identities. The trigger/opportunity ratio therefore
must not be interpreted as re-ID recall. Likewise, ICP reject counts are
motion-estimator decisions, not official Dyn misses.

## Ranked Root Causes

### R1: Unbound frozen-anchor authority is the dominant Ghost bottleneck

- `MEASURED_EVIDENCE`: 1,592,890 ghost matches, 74.002752% of all official
  object Ghost; event 04 alone contributes 1,376,676.
- Representative IDs: `ovimap:81`, `ovimap:5`, `ovimap:117`, `ovimap:102`,
  `ovimap:129`.
- `CODE_EVIDENCE`: `compose_anchor_checkpoint()` retains any unbound anchor;
  only bound anchors can enter moved/removed state.

### R2: Bound-anchor dynamic/epoch gates retain stale geometry

- `MEASURED_EVIDENCE`: 356,093 bound-anchor ghost matches are currently
  non-dynamic and 174,264 have no advanced geometry epoch. Together they
  explain 95.10% of bound-unchanged Ghost.
- Representative dynamic-false IDs: `ovimap:13`, `ovimap:57`, `ovimap:63`,
  `ovimap:65`, `ovimap:90`.
- Representative epoch IDs: `ovimap:10`, `ovimap:132`, `ovimap:138`,
  `ovimap:61`, `ovimap:67`, `ovimap:71`, `ovimap:91`.
- `CODE_EVIDENCE`: moved authority requires dynamic state, an advanced epoch,
  and displacement at least 0.2 m before temporal geometry can replace the
  anchor.

### R3: Dynamic visual sparsity is an authority-resolution discontinuity

- `MEASURED_EVIDENCE`: five moved bound objects fall from 1,937,312 dense
  anchor points to 3,502 temporal points; 11 new objects contain only 8,734
  points in total. Across official changed regions, moved readout contributes
  only 181 predicted points.
- Representative moved IDs: `ovimap:2`, `ovimap:6`, `ovimap:85`,
  `ovimap:131`, `ovimap:179`.
- `CODE_EVIDENCE`: `integrate_object_submap()` stores one weighted coordinate
  per 5 cm local voxel. `compose_anchor_checkpoint()` replaces the full anchor
  object with that compact temporal prediction when moved.

### R4: Association/motion ranking is not yet proven dominant

- `CODE_EVIDENCE`: active admission uses current centroid; `motion_weight` is
  absent from `_score_identity_edge()` despite being configured as 0.1.
- `MEASURED_EVIDENCE`: the source diagnostics expose candidate and estimator
  mechanisms, but cannot map their counts exactly to the 41,987 official Dyn
  missed-plus-hallucinated counts.
- Result: this is a hypothesis, not authority to tune association weights.

## Coverage and Gate

- `explained_ghost_points / all_ghost_points = 2,152,474 / 2,152,474 = 100%`.
- `explained_dyn_failures / all_dyn_failures`: not reportable as a reliable
  ratio because the official CSV omits runtime entity correspondence.
- The enhanced causal association/motion trace is implemented and tested, but
  its full 1,745-frame replay is auxiliary: the frozen source run itself takes
  roughly 15.4 hours. Its completion may refine R4, but cannot overturn the
  exact R1/R2 Ghost partition or the P0 density result.

Phase decisions:

- **P3 NO-GO**: association/motion scoring is not proven to be the dominant
  failure source. Do not sweep `motion_weight`.
- **P4A GO**: run the dense moved-anchor template as visualization-only shadow,
  then as a separately named evaluation candidate.
- **P4B BLOCKED** until P4A passes visual, metric, and non-interference gates.
- **P5 GO** for one conservative `visibility-grounded suppression` pilot on
  unbound anchors. Late binding is not selected for the first pilot.
- **P6 NO-GO / NOT NEEDED**: identity is not the dominant measured bottleneck.
