# OVI-MAP Module Validation Handoff

- Branch: `research/ovimap-module-validation-v1`
- Latest executed data/capture code: `e1b4efc06d234c574ddc45ba8f5a50a61e311a58`
- Implementation: `PARTIAL`
- Experiment: `BLOCKED_PREREQUISITES`
- Science: `INCONCLUSIVE_PREREQUISITES`
- Confirmation: `NOT_RUN_PREREQUISITES`
- Publication: `PUSH_VERIFIED` for executed code `2cad816d98ea07270a032d22f47fe76005c278c0`; see [publication receipt](../../../artifacts/static_ovmap/module_validation_v1/repairs_20260922/publication_receipt.json).

This correction supersedes the earlier overall-completion claim. [Repair details and commands](module_validation_v1/ENGINEERING_REPAIRS_20260922.md) distinguish completed engineering fixes from the outstanding independent-scene driver and data requirements. The generated per-attempt report records publication as unchecked because reporting itself does not contact GitHub.

ScanNet update: all 14 locked raw captures are downloaded; 12 development scenes have 2,400 exported slots. `attempt_005` has `bind=COMPLETE`; capture reached `BLOCKED_GPU_BUSY`. Authorization is already confirmed. The independent-scene S/G/Q scientific drivers remain incomplete. See [current status and commands](module_validation_v1/SCANNET_EXECUTION_20260922.md).

Evidence:

- `/mnt/shared/ww/ovimap-module-validation-v1/attempt_003/method_matrix.json`
- `/mnt/shared/ww/ovimap-module-validation-v1/attempt_003/receipts`

Executed scope:

- capture: verify existing native capture payload and frame schedule; status COMPLETE; evidence /mnt/shared/ww/ovimap-module-validation-v1/attempt_003/capture_smoke.json.
- semantic: one native captured request per visual adapter; status COMPLETE; evidence /mnt/shared/ww/ovimap-module-validation-v1/attempt_003/semantic_smoke.json.
- geometry: first_4096_native_faces; status COMPLETE; evidence /mnt/shared/ww/ovimap-module-validation-v1/attempt_003/geometry_smoke.json.
- query: native candidate parity on all captured frames; frame-0 cached policy replay; status COMPLETE; evidence /mnt/shared/ww/ovimap-module-validation-v1/attempt_003/query_smoke.json.

Historical reproduction command (does not run the new development study):

- `/home/ww/miniconda3/envs/ovimap-map/bin/python /home/ww/crove/ovimap-module-validation/scripts/evaluation/run_ovimap_module_study.py --spec /home/ww/crove/ovimap-module-validation/docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json --phase all --output-root /mnt/shared/ww/ovimap-module-validation-v1 --resolved-config /mnt/shared/ww/ovimap-module-validation-v1/attempt_003/resolved_config.json`

Next action: use the new execution document's bind/capture commands when GPU 2 is free, then finish independent-scene S/G/Q integration before scientific selection. Authorized FIT/CAL/SELECT data are now present.
