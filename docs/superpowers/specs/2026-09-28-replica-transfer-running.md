# Replica transfer execution

Frozen contract: `/mnt/shared/ww/ovimap-replica-composition-transfer-v1/attempt_001/transfer.json`.

Execute/resume (completed outputs are content-verified; changed inputs are rejected):

```bash
cd /home/ww/crove/ovimap-complementary-composition
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_replica_transfer.py \
  --config /mnt/shared/ww/ovimap-replica-composition-transfer-v1/attempt_001/transfer.json \
  --phase all
```

Do not launch a duplicate orchestrator while the current one is running. Visual workers share the original GPU 2 lock. `all` chooses the original isolated interpreter for each model. A scene can be selected with `--scene room0`; query leaves additionally require `--method Q_GAIN` or `--method CP_M4_GAIN_S2` and the matching original interpreter.

Fixed methods: N0, Q_GAIN, S_SIGLIP2_AREA, CP_M2_EQUAL_RAW, CP_M2_EQUAL_CAL, CP_M4_GAIN_S2. Eight scenes, 200 scheduled frames each, query B200. No Replica fitting. Official 51 semantic classes and 48 released AP classes; historical Replica exposure is disclosed.

Logs: `attempt_001/native/<scene>/{frontend_job,mapping_job}/runtime.log` and `attempt_001/logs/<scene>/`.

Actual results only: `attempt_001/scenes/<scene>/rows/replica_transfer/<scene>/<method>.json`. Final `report/results.{json,md}` and `report/per_scene.csv` are produced only with all 48 rows and defined primary metrics. Missing rows are not zero-valued measurements. ScanNet reference changes retain the original CAL/regression/confirmation roles.

The first annotation conversion was preserved under each scene's `study/annotations_initial_code_snapshot` before formatting-only source cleanup. The final conversion under `study/annotations` binds the final source hash. Archived preliminary annotations are not experiment inputs.

Validation:

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python -m pytest tests/replica_transfer -q
/home/ww/miniconda3/bin/ruff check src/static_ovmap/replica_transfer scripts/evaluation/run_ovimap_replica_transfer.py tests/replica_transfer
```
