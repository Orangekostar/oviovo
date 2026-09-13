# Room0 rebuilt baseline B1

The fresh 200-frame native run passes ROOM0_PASS and every native frame audit. Independent
strict 5 cm projection reproduces all 954,492 released instance and semantic labels exactly.
The readout B0 file is the internal original-readout control for this run; externally it is
**B1**, not the historical cached B0. Mask arrays, class/score manifests and converted GT
also pass exact parity before evaluating the paired conditions.

| Rebuilt condition | mIoU | mAcc | Semantic AP | AP50 | AP25 |
|---|---:|---:|---:|---:|---:|
| B1 original readout | .333373 | .370803 | .153614 | .332176 | .393366 |
| B1 + S1a / S1c / frozen C1 | .335586 | .373094 | .155800 | .334877 | .397995 |
| B1 + S1b / quality8 | .333373 | .370803 | .153614 | .332176 | .393366 |
| B1 + random8 / all retained views | .308073 | .344627 | .153614 | .332176 | .393366 |

S1a replaces seven observation subsets and changes owner 85 from pillow to cushion.
The gain is much smaller than on the historical Room0 cache: mIoU +.002214 and semantic
AP +.002186, versus the historical +.029318 and +.032407. These two runs do not establish
a confidence interval or an isolated cause. The frozen selector is unchanged; it has not
been retuned on remaining scene GT. All conditions remain retained-top10 offline readout.

The native run has 9,158,364 vertices, 70 cached owners and 481 retained queries, versus
9,282,303 / 77 / 592 historically. Strict projection leaves 124,345 unmatched vertices,
versus 116,029 historically. Twelve cached owners have no final logged surface identity;
released staging records their exclusion, while raw observations remain available.
Raw and logged colors agree for all shared owners. Enrichment covers all 481 observations,
with 58 mesh-backed centroids and no missing RGB-D frames; 61 cached owners meet min-two.

Released class-agnostic values on the native semantic-filtered geometry are mIoU .3922,
mP50 .7170 and mR50 .4130. They are constant across these readout conditions and are not
integrated AP. B1 full-native-owner canonical AP is still pending. Geometry/query/version
differences must not be attributed to the selector or merged into a single B0 baseline.

The unchanged released evaluator initially failed because it uses relative imports while
the adapter loaded it as a single file. `released_loader.py` now loads each evaluator root
under a distinct package namespace; no metric function or external source was changed.
Six loader/projected-metric tests pass. Re-evaluating the historical saved B0 masks with
the new loader in the native environment reproduces AP .1543156292866941 exactly.
The default Python's newer NumPy lacks legacy `np.in1d`; actual released evaluation uses
`/home/ww/miniconda3/envs/ovimap-map/bin/python`, without patching NumPy or metric code.

Small raw results, executed argument vectors, native manifest, stage timings, failure
record and parity evidence are in `artifacts/static_ovmap/room0_b1`. Large outputs are in
`/mnt/shared/ww/ovimap-static-20260913/room0_b1_*`; native outputs remain at the manifest's
local path. Timings derive from recorded process timestamps and include fresh native
mask/geometry/query work under concurrent host load. Isolated VLM/export time and peak GPU
allocation were not recorded and remain null. These are not controlled E2E speed claims.

The other seven Replica scenes are running. This is a complete Room0 B1 released-metric
unit, not the Replica8 main table or final task completion. ScanNet inputs and the final
requirement-by-requirement audit remain unresolved.
