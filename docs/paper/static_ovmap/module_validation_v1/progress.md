# OVI-MAP module validation progress

Latest: `attempt_005`, executed code `e1b4efc06d234c574ddc45ba8f5a50a61e311a58`. ScanNet data is now READY: 14 raw captures, 12 development scenes / 2,400 exported slots, 95 invalid poses logged without backfill. `bind=COMPLETE` (10 assets, zero missing), `capture=BLOCKED_GPU_BUSY` (CUDA 2 occupied). The real capture driver is connected; independent-scene S/G/Q scientific drivers remain incomplete. No new scientific metrics exist. Data authorization is already confirmed; do not request it again. See [current evidence and commands](SCANNET_EXECUTION_20260922.md).

Next work: run native capture when a GPU is available, bind ScanNet200 native readout/evaluation, then connect and execute the existing S/G/Q cores. The driver has unit/CLI/resource-gate coverage but has not completed a real scene's CropFormer/native replay.

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
