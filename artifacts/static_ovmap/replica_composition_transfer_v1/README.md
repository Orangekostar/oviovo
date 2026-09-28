# Replica frozen composition transfer

Completed: 8 scenes × 6 methods = 48 result rows, 200 frames per scene.
Models, query checkpoint and temperatures were transferred from ScanNet without
Replica fitting. Replica has historical experimental exposure.

- [Full metric table including APall](report_apall/results.md)
- [Per-scene CSV including APall](report_apall/per_scene.csv)
- [Complete results and paired changes](report_apall/results.json)
- [Original five-metric report](report/results.md)
- [Frozen transfer configuration](transfer.json)
- [Multi-GPU execution plan](parallel/execution_plan.json)

APall is the existing uAP field: both map directly to released `all_ap`.
Actual IoU thresholds are 0.50, 0.55, …, 0.90; AP25 is separate.
Results use equal scene means, 48 instance-AP classes and 51 semantic classes.
M2_CAL has the highest mean for all five distinct metrics, but individual
scenes can regress. The source JSON retains original local provenance paths;
those paths are not portable download links.

The repository includes runner, evaluation adapters, tests, result tables and
execution configurations. Raw Replica data, model weights and large intermediate
captures remain external prerequisites, not part of this result package.

Code: `src/static_ovmap/replica_transfer/` and
`scripts/evaluation/run_ovimap_replica_transfer.py`.

Validation:

```bash
python -m pytest tests/replica_transfer -q
```
