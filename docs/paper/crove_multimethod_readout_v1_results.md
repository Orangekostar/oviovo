# CROVE multimethod readout — partial DEV evidence

Task revision: V1_REVISED_V2. Evaluated implementation: `aed101c` (execution before
commit; subsequent changes were formatting and equivalent loop unpacking only).
This is an intermediate result, not the final four-family screening report.

## room0 complete-surface initial batch

All rows use the same 9,282,303 native OVI source rows and owners, fixed Replica41
GT/crosswalk and strict 5 cm projection. Predictions were saved before opening GT.

| Method | mIoU | f-mIoU | CA-AP25 | CA-AP50 | Geometry F5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B_SEM_OVI_NATIVE | 0.340208 | 0.605177 | 0.562641 | 0.470711 | 0.935286 |
| MV_NATIVE_CACHED | 0.377473 | 0.598409 | 0.562641 | 0.470711 | 0.935286 |
| MV_SINGLE | 0.328337 | 0.561435 | 0.562641 | 0.470711 | 0.935286 |
| MV_TOPK_MEAN | 0.353902 | 0.605187 | 0.562641 | 0.470711 | 0.935286 |

`B_SEM_OVI_NATIVE` uses the saved native semantic crosswalk. `MV_NATIVE_CACHED`
uses the official retained last-eight visibility-weighted six-crop features,
reclassified against the complete fixed 41-class vocabulary with bare class names.
It changes the text decision space relative to the saved native mapping; its
+0.037265 mIoU is not attributed solely to view selection. SINGLE and TOPK use
normalized per-view six-crop means and respectively one/four most-visible cached
views. The native cache already selected its candidate observations, so this
batch does not test new views outside the retained bank or individual crop scales.

The three new rows cover 99.8523% of surface rows (71 feature owners). Source
observation totals are 497, 71 and 272 respectively, all within the authorized
0:2000:10 input. Uncovered rows retain native predictions. No additional image
encoder forwards were used; one shared SigLIP text pass was run. Prediction and
sidecar-write time was approximately 6.2–6.4 seconds per new row, excluding shared
geometry/model loading and text encoding. Exact view lists and timings remain in
the local run directory. No end-to-end speedup is claimed.

Previously recorded S0≈0.3997 and S2≈0.4479 are still higher than these initial
new rows; those historical controls have not yet been rerun in this batch.
No winner has been frozen and no new deployment recommendation is made.

Executed command:

```bash
python scripts/evaluation/run_crove_multiview_room0.py
```

Outputs: `configs/evaluation/results/crove_multimethod_readout_v1/room0_*.json`.
Full prediction sidecars: `$HOME/oviovo_baseline_runs/20260912_crove_multimethod_readout_v1/dev/room0/native_cached_batch/`.

## Other progress and remaining obligations

| Family | Real current status | Still required |
| --- | --- | --- |
| M1 | Partial room0 whole-map comparison above | diverse/quality/current-aware, same-condition controls, B3/H2, confirmation |
| M2 | Not run | S2 pseudo-unary and real M1 posterior, shared patch-only/geometry/boundary graph controls |
| M3 | Not run | true 2D mask correspondence, shared arbitration, pairwise/consensus/resem whole-map comparisons |
| M4 | Official trained core loaded; three real-mask reference comparison passed | full-map paired mean/learned inference on DEV and confirmation |

M4 uses the official trained checkpoint, not random weights or a local untrained
replacement. Its three-mask numerical test is explicitly not a whole-map result.
Code retains official attribution and Apache-2.0 license.

B3/H2 legacy-to-pointwise equality is verified separately in `bridge_B3.json` and
`bridge_H2.json` (mIoU 0.135886/0.135734). No M1–M4 dynamic result exists yet.
room1 remains the frozen static confirmation scene, with raw inputs present but
derived OVI/S2 inputs pending. Office original assets remain missing.

DEV_SCREENING_STATUS=PARTIAL

STATIC_CONFIRMATION_STATUS=PARTIAL

DYNAMIC_CONFIRMATION_STATUS=NOT_RUN_RAW_MISSING

Code and compact intermediate results are intended for the named research branch;
large geometry/prediction sidecars remain local. Official pretrained weights are
referenced at their original HF source and are not redistributed. No new model
has been trained. Limited combinations, recovery attribution, selection,
confirmation, visualizations and final GitHub handoff remain outstanding.
