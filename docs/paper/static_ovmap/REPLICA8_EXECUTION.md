# Replica8 execution and result-selection rules

The per-scene driver `scripts/evaluation/run_static_replica_scene.py` now executes the
same stages already validated on Room0: measured enrichment, retained-cache readout,
released GT conversion/postprocessing, semantic metrics, exact reference mask export,
released semantic AP, independent strict-5-cm projection audit and full-owner canonical
geometry AP. Each subprocess has an exact argv, log, elapsed time and terminal status.
GT is supplied to evaluation stages only. Frame schedules, native artifacts, source,
model/preprocessor, text cache and vocabulary identities are checked before execution.

The complete driver has passed on office0, with all stages COMPLETE. Baseline semantic
projection, masks and class/score manifest pass exact parity. All eight retained-cache
conditions have mIoU .169461, mAcc .294310, semantic AP .090972, AP50 .150000 and AP25
.312500: no improvement on this scene. The fixed selector has not been retuned.
Evidence and actual command are in `artifacts/static_ovmap/replica8/office0`.
Five manifest/history/loader tests pass; actual scene execution is the integration check.

Example of the executed command (all paths exist):

```bash
python scripts/evaluation/run_static_replica_scene.py \
  --native-manifest /mnt/shared/ww/ovimap-static-20260913/native_replica8/scenes/office0/attempt-20260913T112637.862473Z/native_mapping_manifest.json \
  --text-cache /mnt/shared/ww/ovimap-static-20260913/text/replica51_rebuilt.npz \
  --gpu 2 \
  --output /mnt/shared/ww/ovimap-static-20260913/evaluated_replica8/office0
```

The output directory must be new; this historical command is recorded for reproduction,
not for overwriting the already completed result. Full-history runs additionally supply
`--include-full-history`, with verified full-cache artifacts in their native manifest.
Those conditions use a separate query namespace and retain the original B0 ordering.

## Complete-history coverage

Room0, office0 and office1 were started before the capture hook existed. Their initial
native results are preserved. `rerun_static_native_mapping.py` now reuses each completed
parent's 200 frontend masks and 200 geometric masks, validates unchanged native build
sources, and runs only fresh mapping/image-VLM work with complete-query capture enabled.
It verifies the 400 reused files before and after mapping and records original parent
timings separately from zero incremental frontend/geometry cost. It does not rerun the
frontends, modify parent results, or claim recovery of the original discarded vectors.

The three capture reruns have been launched under
`/mnt/shared/ww/ovimap-static-20260913/native_full_history`. They are still in progress
at this checkpoint. Office2/3/4 and room1/2 were originally launched with capture enabled
under `native_replica8`. The final full-history table uses the first successful captured
run in these declared roots for each scene, selected for record completeness, not for its
GT score. Old Room0/office0 development and version-drift tables remain separately named.
Fresh mapping can differ from its parent; compare every selector against the native
retained cache from that same run. C1 remains the previously frozen retained-pool S1a;
the full-pool conditions are explicit additional comparisons, not a silent C1 change.

The new image calls belong to fresh native reruns and their cost must be reported. The
capture hook and subsequent semantic selection add no image calls. Reused-stage original
timestamps are provenance, not new elapsed work. Concurrent scenes share host CPU/RAM;
these executions do not establish isolated hardware performance or E2E FPS.

## Aggregation boundary

Replica8 semantic AP must be recomputed by the released evaluator over all eight scene
mask manifests; averaging per-scene AP is not equivalent. Vertex mIoU/mAcc likewise use
the pooled confusion matrix. Keep explicitly labelled per-scene macro diagnostics and
the additional seven-scene view separately. Room0 was used for development; the seven
other scenes must not be described as a new independent test set. Canonical geometry AP
remains distinct from released mP/mR and unverified paper AP. Final aggregation and
requirement-by-requirement completion audit are still pending.
