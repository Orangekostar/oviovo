Final status: all 34 required conditions COMPLETE. See T1_ATTRIBUTION_RESULTS.md and T1_ATTRIBUTION_AUDIT.md. The checkpoint below is historical; provisional pending decisions are superseded by artifacts/static_ovmap/t1_attribution_v1/decision.json.

# T1 attribution execution checkpoint

This is an active study, not a completion report. Full scope is preserved in
T1_ATTRIBUTION_SPEC.md and T1_ATTRIBUTION_PLAN.md.

- Isolated branch research/ovimap-t1-attribution-v1 starts exactly at reviewed 8cefe6b.
- Both actual saved FP32 predictions are present and receipt hashes verify. Their returned
  coordinate arrays/order equal the historical primary exactly; native projection is reused
  only after that check and exact owner reconstruction from nearest/distances.
- Immutable pools contain 205 (64 OVI + 141 SF) and 207 (64 OVI + 143 SF) candidates.
- Primary reproduction: masks, classes, retained IDs/order, raw area scores, borrowing and
  suppression decisions equal historical T1 exactly. Metadata now says OVI with bound readout
  SHA, not a hard-coded S1a source name.
- All 14 prediction conditions generated on both runs, no GT inputs and zero new inference.
  CPU preparation including pair IoU took 9.631 / 10.548 seconds; total per-run prediction
  preparation/export was 33.662 / 32.286 seconds. Large output root is
  /mnt/shared/ww/ovimap-t1-attribution-v1/predictions.
- 17 targeted tests pass in the original evaluation environment. Tests include original
  fusion regressions, intervention isolation, strict thresholds, unique-owner assignment,
  projected/source commutation, regional confusion including prediction0, and released trace
  AP/PR/FN parity. No unrelated dynamic tests are run.
- The core seven-condition evaluation is running on both cached predictions. Initial primary
  O-only common-domain raw-area AP is .20644305433078583. This is a new paired baseline,
  not historical native .186723 S1a and not a fusion gain.
- Historical S1a has 61 emitted instance masks but 72 positive semantic vertices outside their
  union. The bridge must retain the full historical semantic reference separately from the
  frozen emitted-registry semantic output; silently combining them is not pure transfer.

Executed commands:

```bash
python scripts/evaluation/run_static_t1_attribution.py --config configs/evaluation/ovimap_t1_attribution_v1.json --output /mnt/shared/ww/ovimap-t1-attribution-v1/predictions
/home/ww/miniconda3/envs/ovimap-map/bin/python -m pytest -q tests/evaluation/test_static_t1_attribution.py tests/evaluation/test_static_t1_evaluator_trace.py tests/evaluation/test_static_proposal_fusion.py tests/evaluation/test_static_projected_masks.py tests/evaluation/test_static_projected_instance_metrics.py tests/evaluation/test_static_released_loader.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/diagnose_static_t1_attribution.py --config configs/evaluation/ovimap_t1_attribution_v1.json --predictions /mnt/shared/ww/ovimap-t1-attribution-v1/predictions --output /mnt/shared/ww/ovimap-t1-attribution-v1/evaluation --conditions AT_O_AREA AT_S_RELEASED AT_S_AREA AT_U00 AT_U10 AT_U01 AT_U11
```

First command exit 0, tests exit 0 (17 passed); evaluation remains live at this checkpoint.
Remaining: complete real evaluator parity; both-run core object and region accounting;
geometry/export and native-readout bridges; score/assignment/NMS-order controls; four evidence
reports, repair decision, final requirements review and task-branch push verification.
