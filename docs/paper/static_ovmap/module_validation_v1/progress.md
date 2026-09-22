# OVI-MAP module validation progress

Latest (2026-09-22 21:38 UTC): `attempt_006` bind is COMPLETE and exactly matches the original 8/2/2/2 acquisition lock. All 14 raw captures and 2,400 development slots are present, including 95 invalid poses without backfill. Native-v9 capture has completed 5/12 development scenes on GPU 2. S/G/Q, gated combinations/confirmation and reporting/release drivers are implemented; their full real execution is in progress. Four FIT baselines have verified native export/evaluator parity; no SELECT gain or final candidate is claimed. Data authorization is already confirmed; do not request it again.

Next work: finish the running native capture and S/G/Q jobs, execute the frozen conditional gates, publish actual results and verify the remote branch SHA. CONFIRM remains unopened. Current implementation details and evidence: [scientific execution plan](../../../superpowers/plans/2026-09-22-scannet-scientific-drivers.md).

The following historical attempt is retained for provenance:

Attempt: `/mnt/shared/ww/ovimap-module-validation-v1/attempt_003`

| Phase | Status | Blockers |
|---|---|---|
| bind | BLOCKED_INDEPENDENT_SCENES | BLOCKED_INDEPENDENT_SCENES |
| capture | PARTIAL | BLOCKED_INDEPENDENT_SCENES |
| semantic | PARTIAL | BLOCKED_INDEPENDENT_SCENES |
| geometry | PARTIAL | BLOCKED_INDEPENDENT_SCENES |
| query | PARTIAL | BLOCKED_INDEPENDENT_SCENES |
| select | BLOCKED_PREREQUISITE | BLOCKED_INDEPENDENT_SCENES |
| confirm | BLOCKED_PREREQUISITE | SELECTION_NOT_FROZEN |
| report | COMPLETE | BLOCKED_INDEPENDENT_SCENES, PHASE_NOT_COMPLETE:bind, PHASE_NOT_COMPLETE:capture, PHASE_NOT_COMPLETE:confirm, PHASE_NOT_COMPLETE:geometry, PHASE_NOT_COMPLETE:query, PHASE_NOT_COMPLETE:select, PHASE_NOT_COMPLETE:semantic, SELECTION_NOT_FROZEN, UNIMPLEMENTED_DEVELOPMENT_SCENE_PIPELINE |

2026-09-22 data preparation: scoped download/resume and native frame export are implemented and tested. Public ScanNet splits and a provisional 14-capture acquisition plan are ready. Raw downloads await confirmation of existing ScanNet authorization and terms agreement. See [ScanNet preparation](SCANNET_PREPARATION_20260922.md).

The historical phase statuses above remain unchanged. Their former data-authorization blocker has been resolved by the newer execution recorded at the top of this document.
