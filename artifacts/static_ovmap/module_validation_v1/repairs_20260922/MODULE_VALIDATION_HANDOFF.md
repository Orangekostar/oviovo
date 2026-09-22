# OVI-MAP Module Validation Handoff

- Branch: `research/ovimap-module-validation-v1`
- Commit: `2cad816d98ea07270a032d22f47fe76005c278c0`
- Implementation: `PARTIAL`
- Experiment: `BLOCKED_PREREQUISITES`
- Science: `INCONCLUSIVE_PREREQUISITES`
- Confirmation: `NOT_RUN_PREREQUISITES`
- Publication: `NOT_CHECKED_BY_REPORT`

Evidence:

- `/mnt/shared/ww/ovimap-module-validation-v1/attempt_003/method_matrix.json`
- `/mnt/shared/ww/ovimap-module-validation-v1/attempt_003/receipts`

Executed scope:

- capture: verify existing native capture payload and frame schedule; status COMPLETE; evidence /mnt/shared/ww/ovimap-module-validation-v1/attempt_003/capture_smoke.json.
- semantic: one native captured request per visual adapter; status COMPLETE; evidence /mnt/shared/ww/ovimap-module-validation-v1/attempt_003/semantic_smoke.json.
- geometry: first_4096_native_faces; status COMPLETE; evidence /mnt/shared/ww/ovimap-module-validation-v1/attempt_003/geometry_smoke.json.
- query: native candidate parity on all captured frames; frame-0 cached policy replay; status COMPLETE; evidence /mnt/shared/ww/ovimap-module-validation-v1/attempt_003/query_smoke.json.

Reproduction commands:

- `/home/ww/miniconda3/envs/ovimap-map/bin/python /home/ww/crove/ovimap-module-validation/scripts/evaluation/run_ovimap_module_study.py --spec /home/ww/crove/ovimap-module-validation/docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json --phase all --output-root /mnt/shared/ww/ovimap-module-validation-v1 --resolved-config /mnt/shared/ww/ovimap-module-validation-v1/attempt_003/resolved_config.json`

Next evidence-based action: Connect the independent-scene development driver and provide authorized FIT/CAL/SELECT scenes before scientific selection.
