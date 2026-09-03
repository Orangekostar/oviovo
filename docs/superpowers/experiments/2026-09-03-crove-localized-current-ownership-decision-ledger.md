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

**Commit:** Design `7006a46`; plan and ledger `14fee8c`.

## P6-A Counterfactual Attribution

**Hypothesis:** The ten P5 suppressions contain separable Ghost-benefit and
Object-damage factors that can be identified by causal counterfactual replay.

**Evidence before:** P5 gains `0.0194799752` Object F1 and reduces Ghost by
`0.0020820903` versus no suppression, but remains below the A6 Object floor.

**Implementation:** Reused the frozen P5 source and recomposed CF0, CF1, ten
leave-one-out, ten only-one, and two P2 feature-group variants. All variants
are explicitly diagnostic-only and cannot define a runtime ID rule.

**Artifacts:** Report `docs/superpowers/reports/P6A_COUNTERFACTUAL_REPORT.md`;
collection root
`/home/ww/oviovo_baseline_runs/20260903_crove_localized_current_ownership/p6a/collection`;
manifest SHA-256
`41585bcfa8b3a48c8df0c88faf8307701b045c185dd658cae64b5c52d8a7f9f0`.

**Measured result:** All 24 variants completed 1,745 frames and 43 states.
Dynamic F1 is invariant at `0.06922505723328032`; Object F1 ranges
`0.3484720275`-`0.3679520027`, and 0/24 variants pass every hard gate. The P2
dominant five recover the full Ghost benefit but no Object gain; the
complementary five recover the full Object gain but no Ghost benefit.

**Decision:** `NO_GO / DIAGNOSTIC_ONLY`.

**Reason:** The trade-off separates into two transition groups, but no measured
variant crosses the A6 Object floor and the one-scene ID evidence cannot become
a general method rule.

**Commit:** Counterfactual implementation `f7c3b02`, execution `5a34fa3`;
report recorded by the ledger commit containing this entry.

## P6-B Localized Reversible Ownership

**Hypothesis:** A 5 cm signed per-voxel state can remove only unsupported anchor
regions and restore them after two later PRESENT observations while preserving
the immutable OVI-MAP anchor.

**Evidence before:** P5's whole-anchor policy improves Ghost but loses Object
F1, and P6-0 measured only 1.513045% direct 5 cm PRESENT/ABSENT conflicts with a
2.38 MB raw-state estimate.

**Implementation:** Added typed immutable ownership/evidence state,
copy-on-advance causal updates, native point-order filtering, deterministic
bit-packed masks, partial/dormant metadata, and the
`localized_visibility_candidate` composition contract. P5 behavior remains
unchanged when localized ownership is absent.

**Artifacts:** Frozen policy
`configs/evaluation/crove_ovimap_localized_visibility_l1_v1.json`; runtime masks
are separate hash-bound sidecars rather than embedded map payloads.

**Measured result:** The Apartment L1 composition covers 1,745 frames and 43
states, tracks 41,797 full voxels across 36 anchors, and uses 2,487,978 bytes of
implemented state, below the 8 MB L2 trigger.

**Decision:** GO to standalone P6-C evaluation; L2 remains
`NOT_RUN_NOT_NEEDED`.

**Reason:** The implementation meets the fixed 5 cm, reversibility, memory,
causality, and compatibility contracts without modifying source mapping.

**Commit:** `52cab77`, `ab006ba`, `212ef74`, `d041022`.

## P6-C Apartment Localized Candidate

**Hypothesis:** Replacing whole-anchor suppression with 5 cm localized
ownership will preserve valid object geometry, cross the A6 Object F1 floor,
and retain P5's dynamic/current/Ghost gains.

**Evidence before:** P5 misses the Object floor by `0.0048099973`; P6-0 shows
low direct 5 cm evidence conflict and P6-B fits within the memory bound.

**Implementation:** Reused the frozen source run and P5 thresholds, recomposed
L1 only, exported the temporal artifact, imported the unchanged Khronos bridge,
and ran official plus two byte-identical common-v2 evaluations.

**Artifacts:** Report `docs/superpowers/reports/P6C_LOCALIZED_VISIBILITY_REPORT.md`;
run root
`/home/ww/oviovo_baseline_runs/20260903_crove_localized_current_ownership/p6c/apartment_l1`;
composition SHA-256 `259f181e2e562cc6546e8ca08727049c48e71bde84f3bb05ba11bfa525e73bad`.

**Measured result:** Object F1 `0.367307250721078`, Dynamic F1
`0.06922505723328032`, Change F1 `0.08724574941966247`, current mIoU
`0.14944876084641232`, Ghost `0.44168885642814226`; 1,745 frames, 43 states,
2,487,978 state bytes, and 43 mask sidecars.

**Decision:** `NO_GO / REJECTED_RETAIN_A6`; Office not authorized.

**Reason:** Only Dynamic meets the P6 hard gate. Object, Change, current, and
Ghost all fail their frozen thresholds, so neither L2 tuning nor Office is
allowed.

**Commit:** Implementation through `d041022`; evaluation report recorded by
the ledger commit containing this entry.

## P6-D Geometry-Gated Dense Readout

**Hypothesis:** A fixed bidirectional geometry-agreement gate can recover dense
moved geometry only where compact temporal and translated anchor templates
agree, while falling back exactly elsewhere.

**Evidence before:** P4A improved Change/current/Ghost but failed Object F1;
its moved objects included zero-coverage and meter-scale template failures.

**Implementation:** Added typed bidirectional 5 cm coverage, NN median/p90,
centroid, extent residual/ratio measurements and a hybrid readout whose rejected
path preserves the exact compact snapshot plus additive audit metadata.

**Artifacts:** Report `docs/superpowers/reports/P6D_HYBRID_DENSE_READOUT_REPORT.md`;
run root
`/home/ww/oviovo_baseline_runs/20260903_crove_localized_current_ownership/p6d/apartment_hybrid`;
composition SHA-256 `1a3098140282257f247f835307f6b3727a3055c67ee84a297d65899f2609a9c7`.

**Measured result:** 0/104 dense decisions accepted and 104/104 exact compact
fallbacks. Object F1 `0.34847202749654466`, Dynamic F1
`0.06922505723328032`, Change F1 `0.08845787063178367`, current mIoU
`0.1495595491165195`, Ghost `0.4437595432978686`; 1,745 frames and 43 states.

**Decision:** `NO_GO / REJECTED_RETAIN_A6`; combined candidate
`NOT_RUN_STANDALONE_GATE`.

**Reason:** Every dense proposal failed the preregistered gate, and exact
fallback fails Object, current, and Ghost hard gates.

**Commit:** Implementation `aafde6b`, `1641f2e`; report `1761104`.

## P6 Combined Candidate

**Hypothesis:** Localized ownership and geometry-gated dense readout may be
complementary only if each mechanism independently passes all hard gates.

**Evidence before:** P6-C and P6-D both have complete standalone Apartment
measurements.

**Implementation:** None; the preregistered standalone gate was applied before
composition.

**Artifacts:** P6-C and P6-D reports and gate receipts.

**Measured result:** P6-C fails four hard gates. P6-D accepts 0/104 dense
decisions and fails three hard gates.

**Decision:** `NOT_RUN_STANDALONE_GATE`.

**Reason:** Both standalone mechanisms are `NO_GO`; running their combination
would violate the preregistered causal-ablation protocol.

**Commit:** Decision recorded with final ledger.

## P6-E Official Dyn Provenance

**Hypothesis:** Existing official visualization and bridge artifacts may bind
official Dynamic-F1 error mass to exact CROVE runtime mechanisms strongly
enough to select proposal, association, motion, or identity work.

**Evidence before:** Official Apartment Dynamic F1 is
`0.06922505723328032`, but the existing evaluator reports only aggregate
detected, missed, and hallucinated trajectory mass.

**Implementation:** Added a strict, hash-bound JSON/CSV sidecar over the
official dynamic table, official visualization tables, bridge node identities,
composition entities, and source trajectories. The sidecar conserves official
mass and marks a join exact only when the official output itself identifies the
prediction; it never estimates or modifies official metrics.

**Artifacts:** Report
`docs/superpowers/reports/P6E_DYN_PROVENANCE_REPORT.md`; published root
`/home/ww/oviovo_baseline_runs/20260903_crove_localized_current_ownership/p6e/p5_actual`;
JSON SHA-256
`78784d10bd7a247fa164fdb2aeb8ca3067196fa41b4ef6fc978ff3cb7517a8cd`;
CSV SHA-256
`40dcbb5d8b5d1357b4915daf2acd86ed361f120e751897670df80538d75cee84`.

**Measured result:** Across 43 official query events, exact mass is 0,
ambiguous mass is 42,175 (100%), and unavailable mass is 0. The conserved mass
is 188 detected, 19,556 missed, and 22,431 hallucinated. The 2,877
checkpoint-local candidate predictions cannot be assigned to individual
official error units. A pre-existing instrumentation run was excluded because
294/336 emitted checkpoint files differed from A6 formal output and it was
incomplete at 42/43 checkpoints when audited.

**Decision:** `NO_GO / ATTRIBUTION_AMBIGUOUS`; ReScene remains
`NO_GO / NOT_NEEDED`.

**Reason:** Zero official Dynamic mass has exact prediction provenance, so no
dominant proposal, association, motion, geometry, or identity mechanism is
proven. Runtime instrumentation cannot reconstruct an identity omitted by the
official evaluator without an unsupported heuristic join.

**Commit:** Sidecar `f2245bf`; report recorded by the ledger commit containing
this entry.

## Office

**Status:** `NOT_RUN_HELD_OUT` pending a complete Apartment all-gates PASS.
