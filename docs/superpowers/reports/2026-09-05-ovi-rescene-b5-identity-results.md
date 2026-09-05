# OVI-MAP x Persist4D ReScene B5 Identity Results

Date: 2026-09-05
Base commit: `2136865993033e35c44dac12444363a4788e8452`
Verdict: `RESCENE_B5_BLOCKED`

## Result

C0 checkpoint structure and C1 pristine-upstream model topology both pass. C2
fails before GPU because the frozen OVI representation does not retain the
source camera RGB or source-bound point-normal tensor required by the checkpoint,
and native Pointcept spatial resampling merges OVI entity tokens. No C3 backend,
B5 projection, B4 reconstruction, relation comparison, current map, learned B7,
or Office command was run.

This result means that the current OVI-to-ReScene bridge cannot preserve the
checkpoint's input and provenance contract. It is not evidence that ReScene
identity itself fails.

## Checkpoint And Compatibility

The source-bound local reproduction checkpoint is
`/home/ww/paper5/checkpoints/rescene4d_concerto_t2_repro.ckpt`, 754,917,862
bytes, SHA-256
`85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e`.
It is a regular non-symlink file. Persist4D commit
`1380c4b9f37bec7933126ccc9bd70067de166f6f` binds it to 154/154 validation
sequences, t-mAP 27.939, t-REC 40.849, and overall mAP 36.314; the paper target
t-mAP is 34.800. It is not an official author checkpoint or paper-parity result.

The Concerto initialization is 433,987,358 bytes with SHA-256
`845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`
at revision `c31f993a56129f2ba9c5d06a35957e3f05bff710`. Pinned pristine ReScene
`fb2fe42eb8f1e926567c48eea9acb874e608ee10` consumes all 796 model tensors
with exact shapes. No model key is dropped, renamed, ignored, or reinitialized.
The classification is `MODEL_TOPOLOGY_STRICT_COMPATIBLE`, not a full Lightning
restore claim.

## C2 Evidence

The native model feature is nine channels: shared-centered XYZ, source camera
RGB in `[0,1]`, and unit source geometry normals. Dataset-normalized RGB is
discarded by the pinned Pointcept collator. The frozen OVI PLY RGB is an
instance palette, not camera RGB, and frozen snapshots preserve neither camera
RGB nor point-normal provenance.

| Visit | Source points | Adapter tokens | Would-be model tokens | All merges | Cross-entity merges | Drop | Duplication |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t0 | 14,563,575 | 552,612 | 446,153 | 106,459 | 2,973 | 0 | 0 |
| t1 | 12,327,444 | 490,416 | 408,276 | 82,140 | 2,069 | 0 | 0 |

These mappings are not permutations. Passing duplicate sparse coordinates or
inventing a simple `token_indices` permutation would break the existing CSR
entity provenance contract.

## Required Comparison Table

| Field | B4 geometric | B5 local ReScene |
| --- | ---: | ---: |
| backend status | PASS | NOT RUN: C2 blocked |
| checkpoint SHA | N/A | `85ed1aba...1546e` |
| t0 tokens | null | 552,612 adapter / 446,153 would-be model |
| t1 tokens | null | 490,416 adapter / 408,276 would-be model |
| relations total | 120 | null |
| persistent_static | null | null |
| persistent_moved | null | null |
| appeared | null | null |
| removed_candidate | null | null |
| split | null | null |
| merge | null | null |
| uncertain | null | null |
| 1:1 persistent total | null | null |
| runtime s | 143.51442972477525 | null |
| peak GPU memory | null | null |

B4's frozen manifest proves 120 relation IDs, but the old run did not serialize
the `PairRelation` objects. Per-state and 1:1 counts therefore remain null
because the conditional Task 4 reconstruction was not authorized after C2
failed. B4 peak GPU memory was not instrumented. B5 token values are CPU
preflight counts, not successful backend output counts.

The exact topology intersection, B4-only, B5-only, B5-only
`persistent_static`, B5-only `persistent_moved`, and registration-accepted
counts are all null with `b5_not_run_c2_blocked`; none is zero or predicted.

## Scope And Claims

Allowed claims are limited to strict checkpoint topology compatibility and the
failure of the current frozen OVI artifacts to satisfy the source-verified
ReScene input/provenance contract. True identity accuracy, superiority over B4,
final-map benefit, Ghost reduction, Office generalization, official paper
parity, and ReScene method failure are not established.

Learned B7 is `NOT_RUN_BY_SCOPE`. Office is `HELD_OUT`, `attempt_count=0`.
Direct asset revalidation finds 27/44 selected 3RScan visits complete; all 17
incomplete visits lack only `sequence.zip`. Final 3RScan ranking is `NOT_RUN`.

## Artifacts

Compact results are in
`configs/evaluation/results/ovi_rescene_b5_identity/`. External evidence is
under `/home/ww/oviovo_baseline_runs/20260905_ovi_rescene_b5_identity/`.
The external Apartment pair receipt is 7,249 bytes with SHA-256
`2fc579636d02acfa1fa026e9bfd0fa6483770a14e13e19372046481b86a646d2`
and producer commit `bebe0b268b8ba9b125f8198ac0f1f6ef5ee00c28`.

Focused verification passed 55 tests. The single authorized full-suite run
passed 6,009 tests with 10 skips and no failures in 815.93 seconds. Five
existing NumPy deprecation warnings were reported.
