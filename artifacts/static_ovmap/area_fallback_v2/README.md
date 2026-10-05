# Area Fallback V2

An independent, post-hoc accuracy experiment and G1/G3 cold remeasurement on the fixed compact-table cohorts. The frozen v1 release and its original blocked results remain intact.

Protocol: `PROJECTED_FC_EMPTY_AREA_FALLBACK_V2`. Only empty hard-mask support in the fixed projected G1/G3 recovery views invokes area-occupancy-weighted pooling. Original successful vectors, incumbent N/Q/F evidence, archived U2, views, geometry, vocabulary and the frozen projection head are preserved.

Measured accuracy coverage: 26 scenes, 172/172 scene-method scoring records, 14/14 complete released dataset pools. All 74 original empty masks produced valid final vectors; 175 original successful vectors were reused bit-for-bit. That cache-assisted accuracy probe used zero new full-image encodings or GPU inference, and 74 frozen projection-head calls on CPU. The relevant unit/contract checks passed: 89 tests.

Replica A3 vs D2: APall +0.6450pp, mIoU +0.5814pp. CF18 A3 vs D2: APall -0.0770pp, mIoU +0.0247pp. These are exposed-scene descriptive comparisons, not an unseen-generalization or significance claim. Deployment remains unchanged.

The three LaTeX tables contain measured accuracy, recovery counts and latency. Independent G1/G3 v2 cold measurements completed all 16 scene/arm calls on the same A40, one sample per scene/arm. Mean incremental latency is 45.58s for G1 and 45.63s for G3 (sum of all eight scene costs divided by eight). Model loading took 7.97s and is excluded. Fresh projection, decode/preprocessing, image encoding, pooling/head, classification and output/rank export are included; feature, view and result caches are disabled. The OS page cache was not cleared. Archived U2 latency remains the unchanged control.

The cold remeasurement performed 128 GPU image encodings and 159 GPU region-pool/head calls, including 74 area-fallback poolings. All selected views/masks, source labels/availability, full instance support, semantic labels and current-class ranks match the v2 scientific outputs exactly. Maximum FP32 feature difference is 1.1920929e-7, within the inherited atol=rtol=1e-5 tolerance. Accuracy results remain unchanged.

Large evidence, projected vectors, predictions and official scoring traces:
`/home/ww/ovimap-area-fallback-v2/attempt_001`.

Small artifacts in this directory are generated copies of that independent attempt. The full output hashes and input evidence are in `validation_receipt.json`; `timing_pool.json` and `timing_plan.json` retain cold-measurement provenance. The previous unmeasured report and receipts are archived under the attempt's `history/pre_cold_measurement/` directory. Existing observed measurements are reused; the timing command forbids replay of a reserved or failed physical call.

```bash
/home/ww/miniconda3/envs/oviovo-radseg/bin/python scripts/evaluation/run_ovimap_area_fallback.py \
  --parent /mnt/shared/ww/ovimap-cvpr-compact-tables-v1/attempt_001 \
  --output /home/ww/ovimap-area-fallback-v2/attempt_001 --phase all

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/ww/miniconda3/envs/oviovo-radseg/bin/python -m static_ovmap.cvpr_compact.area_fallback_timing \
  --parent /mnt/shared/ww/ovimap-cvpr-compact-tables-v1/attempt_001 \
  --output /home/ww/ovimap-area-fallback-v2/attempt_001

/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_area_fallback.py \
  --parent /mnt/shared/ww/ovimap-cvpr-compact-tables-v1/attempt_001 \
  --output /home/ww/ovimap-area-fallback-v2/attempt_001 --phase report
```
