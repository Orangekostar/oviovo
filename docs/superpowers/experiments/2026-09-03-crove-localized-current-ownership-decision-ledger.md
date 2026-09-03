# CROVE P6 Localized Current Ownership Decision Ledger

Date: 2026-09-03  
Base: `dd3247f2d9e81e31a7508b8b692480dbb8855246`  
Branch: `research/crove-localized-current-ownership`

## P6-0 Preregistration

**Hypothesis:** Whole-anchor suppression causes the residual Object-F1/Ghost
trade-off; a 5 cm signed reversible mask can preserve supported native geometry
while removing stale regions.

**Evidence before:** P5 Object F1 `0.3679520027114184`, Dynamic F1
`0.06922505723328032`, Change F1 `0.0880875002614133`, current mIoU
`0.14956262241133872`, Ghost `0.44167745295296196`; P2 attributes
`74.002752%` of object Ghost to unbound anchors.

**Implementation:** Design and hard gates preregistered in
`docs/superpowers/specs/2026-09-03-crove-localized-current-ownership-design.md`.
Frozen RGB-D replay compares 5/10/20 cm state granularity before method code.

**Artifacts:** No new runtime artifact. Baseline evidence remains under
`/home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery`.

**Measured result:** Apartment has 41,797 full 5 cm unbound-anchor voxels. Raw
5 cm state estimate is 2,382,429 bytes. Direct PRESENT/ABSENT flip rate is
`1.513045%`; mixed PRESENT/ABSENT block rates are `4.337291%` at 10 cm and
`8.955983%` at 20 cm. Nine required baseline groups pass: `154 passed`.

**Decision:** GO with L1 5 cm per-voxel ownership. L2 is
`NOT_RUN_NOT_NEEDED` unless implemented state exceeds 8 MB or replay violates
determinism.

**Reason:** 5 cm matches the evaluation/state scale, has acceptable memory, and
avoids measured block-level evidence conflicts.

**Commit:** Design `7006a46`; plan/ledger commit pending.

## P6-A Counterfactual Attribution

**Hypothesis:** Pending.

**Evidence before:** Pending.

**Implementation:** Pending.

**Artifacts:** Pending.

**Measured result:** Pending.

**Decision:** Pending.

**Reason:** Pending.

**Commit:** Pending.

## P6-B Localized Reversible Ownership

**Hypothesis:** Pending.

**Evidence before:** Pending.

**Implementation:** Pending.

**Artifacts:** Pending.

**Measured result:** Pending.

**Decision:** Pending.

**Reason:** Pending.

**Commit:** Pending.

## P6-C Apartment Localized Candidate

**Hypothesis:** Pending.

**Evidence before:** Pending.

**Implementation:** Pending.

**Artifacts:** Pending.

**Measured result:** Pending.

**Decision:** Pending.

**Reason:** Pending.

**Commit:** Pending.

## P6-D Geometry-Gated Dense Readout

**Hypothesis:** Pending.

**Evidence before:** Pending.

**Implementation:** Pending.

**Artifacts:** Pending.

**Measured result:** Pending.

**Decision:** Pending.

**Reason:** Pending.

**Commit:** Pending.

## P6-E Official Dyn Provenance

**Hypothesis:** Pending.

**Evidence before:** Pending.

**Implementation:** Pending.

**Artifacts:** Pending.

**Measured result:** Pending.

**Decision:** Pending.

**Reason:** Pending.

**Commit:** Pending.

## Office

**Status:** `NOT_RUN_HELD_OUT` pending a complete Apartment all-gates PASS.

