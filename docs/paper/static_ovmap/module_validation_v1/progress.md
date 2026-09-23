# OVI-MAP module validation progress

Latest (2026-09-23 03:28 UTC): all 12 corrected native-v10 development captures are complete. Full causal-membership validation passed on all 2,305 valid frames and 22,307 candidate requests across 2,400 scheduled slots, including the former failure on scene0534_00. FIT/CAL baseline evaluation and geometry-request preparation are complete. S/G/Q inference is now running or queued on GPU 2. The full study remains IN_PROGRESS with no SELECT gain or retained candidate. Native-v9 results remain historical; the 8/2/2/2 split and 95 invalid-pose slots are unchanged. See [repair evidence](QUERY_MEMBERSHIP_REPAIR.md) and the [technical validation receipt](../../../../artifacts/static_ovmap/module_validation_v1/native_v10_capture_validation.json). Data authorization is already confirmed; do not request it again.

Next work: finish S/G/Q, execute the frozen conditional gates, publish actual results and verify the remote branch SHA. CONFIRM remains unopened. Current implementation details and evidence: [scientific execution plan](../../../superpowers/plans/2026-09-22-scannet-scientific-drivers.md).

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
