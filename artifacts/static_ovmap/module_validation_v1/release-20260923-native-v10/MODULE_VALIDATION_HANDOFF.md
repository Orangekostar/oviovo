# OVI-MAP Module Validation Handoff

- Branch: `research/ovimap-module-validation-v1`
- Commit: `b6ab45b8dadf6672d2bcc2a6cfaed9bf41d60a0c`
- Implementation: `COMPLETE`
- Experiment: `MEASURED_APPLICABLE_ROWS`
- Science: `NO_NET_GAIN`
- Confirmation: `NOT_REQUIRED_NO_RETAINED_CANDIDATE`
- Publication: `NOT_CHECKED_BY_REPORT`

Evidence:

- `/mnt/shared/ww/ovimap-module-validation-v1/attempt_011/scientific_evidence.json`
- `/mnt/shared/ww/ovimap-module-validation-v1/attempt_011/method_matrix.json`
- `/mnt/shared/ww/ovimap-module-validation-v1/attempt_011/selection.json`
- `/mnt/shared/ww/ovimap-module-validation-v1/attempt_011/confirmation.json`

Executed scope:

- Runtime and model paths: /home/ww/crove/ovimap-module-validation/configs/evaluation/ovimap_module_scannet_study.json.
- Native extension: /mnt/shared/ww/ovimap-module-validation-v1/tooling/repairs-20260922/native-query-v10/lib/consistent_gsm.cpython-311-x86_64-linux-gnu.so.
- Study root: /mnt/shared/ww/ovimap-module-validation-v1/scannet_study_v1.
- Actual phase blockers: [].

Reproduction commands:

- `/home/ww/miniconda3/envs/ovimap-map/bin/python /home/ww/crove/ovimap-module-validation/scripts/evaluation/run_ovimap_scannet_semantic.py --config /home/ww/crove/ovimap-module-validation/configs/evaluation/ovimap_module_scannet_study.json --phase all`
- `/home/ww/miniconda3/envs/ovimap-map/bin/python /home/ww/crove/ovimap-module-validation/scripts/evaluation/run_ovimap_scannet_geometry.py --config /home/ww/crove/ovimap-module-validation/configs/evaluation/ovimap_module_scannet_study.json --phase all`
- `/home/ww/miniconda3/envs/ovimap-map/bin/python /home/ww/crove/ovimap-module-validation/scripts/evaluation/run_ovimap_scannet_query.py --config /home/ww/crove/ovimap-module-validation/configs/evaluation/ovimap_module_scannet_study.json --phase all`
- `/home/ww/miniconda3/envs/ovimap-map/bin/python /home/ww/crove/ovimap-module-validation/scripts/evaluation/run_ovimap_scannet_selection.py --config /home/ww/crove/ovimap-module-validation/configs/evaluation/ovimap_module_scannet_study.json --phase all`

Next evidence-based action: Interpret retained or negative mechanisms within the frozen two-scene SELECT/CONFIRM scope.
