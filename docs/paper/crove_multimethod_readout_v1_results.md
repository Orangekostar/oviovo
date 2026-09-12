# CROVE multimethod readout — partial DEV evidence

Task revision: V1_REVISED_V2. Evaluated implementation: `aed101c` (execution before
commit; subsequent changes were formatting and equivalent loop unpacking only).
The paired adapter, semantic controls and initial graph batch use `1efef91`
(executed before commit; subsequent changes were formatting/import ordering and
the explicitly documented forward-count reporting correction).
The diverse/quality batch and real-posterior graph controls use `cbb6543`;
the tie-rule fix was verified by exact full-map prediction recomputation.
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
| MV_DIVERSE_MEAN | 0.394472 | 0.407740 | 0.562641 | 0.470711 | 0.935286 |
| MV_QUALITY | 0.400847 | 0.408872 | 0.562641 | 0.470711 | 0.935286 |
| MV_CURRENT_AWARE (static QUALITY alias) | 0.400847 | 0.408872 | 0.562641 | 0.470711 | 0.935286 |

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

DIVERSE and QUALITY reuse that same native six-crop cache and k=4 budget.
DIVERSE starts with the highest visible-area view and selects angularly diverse
camera directions around each owner's physical centroid. QUALITY uses exactly
those inputs, weighted by depth-consistent unique-pixel fraction, square-root
projected/native mask support and a 0.5 image-boundary truncation factor. The full
formula is frozen in `multiview_quality_room0_registry.json`; no GT is used.
There are 64 actual surface owners and 251 candidate observations (older initial
batch counts also included absent owners; their predictions were unaffected).
Coverage is 99.85233%/99.85201%; two zero-quality owners fall back. New image
forwards remain zero. Shared selection/projection take 9.22/14.87 s; prediction/write
takes 5.50/5.61 s. Higher mIoU comes with substantially lower f-mIoU, not an
across-metric improvement. Static CURRENT_AWARE is an exact QUALITY prediction
alias, not evidence for dynamic adaptation. Per-crop normalized features and
structural-background patch observations still require separate work.

S0 and S2 have now been rerun on this same complete surface (see below).
No winner has been frozen and no new deployment recommendation is made.

Executed command:

```bash
python scripts/evaluation/run_crove_multiview_room0.py
```

Outputs: `configs/evaluation/results/crove_multimethod_readout_v1/room0_*.json`.
Full prediction sidecars: `$HOME/oviovo_baseline_runs/20260912_crove_multimethod_readout_v1/dev/room0/native_cached_batch/`.

## Paired trained adapter and S2 graph DEV results

All rows below preserve the same 9,282,303 source rows and owners. CA-AP50
is 0.470711 and geometry F5 is 0.935286 throughout.

| Method | mIoU | f-mIoU |
| --- | ---: | ---: |
| B_SEM_CROVE_S0 | 0.399675 | 0.683362 |
| B_SEM_CROVE_S2 | 0.447889 | 0.673573 |
| ADAPTER_CLIP_MEAN | 0.364617 | 0.638058 |
| ADAPTER_CLIP_LEARNED | 0.396887 | 0.644932 |
| S2_PATCH_ONLY | 0.443890 | 0.668636 |
| S2_GRAPH_GEOM | 0.443963 | 0.668539 |
| MV_QUALITY_PATCH_ONLY | 0.396774 | 0.409143 |
| MV_QUALITY_GRAPH_GEOM | 0.399117 | 0.409253 |

The adapter pair shares official trained ConvNeXt-L features, class text,
projected source-support masks, and selected views. Learned pooling improves
mIoU by 0.032270 over mean pooling but remains below S2. Depth-consistent source
projection uses a 5 cm tolerance without dilation; these owner-derived masks are
not independent M3 mask evidence. The bank contains 133 selected frames and 251
owner candidates; 130 actual image forwards yield 204 valid feature observations.
Both variants fall back identically on unsupported rows; coverage is 99.76885%.
Shared measured projection time is 7.06 s and frame processing time is 19.61 s,
excluding model loading, cache compression/writes and GT evaluation. Peak allocated
GPU memory is 1,738,725,376 bytes. The initial forward counter of 133 was corrected
to 130 from nonempty feature caches; predictions and scores were unchanged.

M2 welds compatible repeated vertices and uses actual triangle topology, not global
kNN. The fixed 2 cm patches contain 629,549 nodes and 1,291,958 undirected edges,
with 100% source-row backprojection and 1,988,010 welded physical samples.
Patch construction took 22.30 s. S2 pseudo-unaries are confidence-weighted costs,
not recovered class posteriors. Five damped mean-field iterations use lambda 0.2.
Patch-only loses 0.003999 mIoU versus pointwise S2; graph propagation adds only
0.000074 over patch-only. This is a negative result versus S2, not a demonstrated
graph improvement. Boundary graph controls remain pending.

M1-posterior controls reuse exactly the same patch mapping and geometry graph.
Actual full-class owner posteriors are averaged by physical mass before negative-log
costs; unsupported source rows keep M1 fallback. Reliability is not multiplied
again after M1 view weighting. Patch-only changes 43,178 source labels; graph
propagation changes a further 1,707 versus patch-only (43,268 versus pointwise M1).
Graph propagation recovers some patch loss but remains below pointwise QUALITY.
The posterior tie rule was strengthened to preserve the physically dominant
baseline class among cost minimizers; full-map recomputation verified zero
prediction changes for both saved variants. This is a current room0 M1 leader
control, not the final cross-protocol family selection.

Reproduction commands:

```bash
/home/ww/oviovo_baseline_builds/maskadapter/venv/bin/python scripts/evaluation/run_crove_adapter_room0.py
python scripts/evaluation/run_crove_room0_semantic_controls.py
python scripts/evaluation/run_crove_graph_room0.py
python scripts/evaluation/run_crove_multiview_quality_room0.py
python scripts/evaluation/run_crove_graph_room0.py --unary mv_quality
```

## Other progress and remaining obligations

| Family | Real current status | Still required |
| --- | --- | --- |
| M1 | Partial room0 native/top-k/diverse/quality; static current-aware alias | per-crop/structural patch evidence, B3/H2 state adaptation, confirmation |
| M2 | Partial room0 S2 and M1 posterior patch-only/geometry graph | boundary control, B3/H2 and confirmation |
| M3 | Not run | true 2D mask correspondence, shared arbitration, pairwise/consensus/resem whole-map comparisons |
| M4 | Official trained core and room0 full-map mean/learned pair completed | B3/H2 and confirmation |

M4 uses the official trained checkpoint, not random weights or a local untrained
replacement. Its three-mask numerical test is explicitly not a whole-map result.
Code retains official attribution and Apache-2.0 license.

B3/H2 legacy-to-pointwise equality is verified separately in `bridge_B3.json` and
`bridge_H2.json` (mIoU 0.135886/0.135734). No M1–M4 dynamic result exists yet.
room1 remains the frozen static confirmation scene, with raw inputs present but
derived OVI/S2 inputs pending. The native room1 GPU frontend failed with verified
CUDA OOM under existing GPU occupancy. Full-resolution CPU frontend preparation
is running for room1 mapping and room0 independent M3 masks; real first-frame
outputs are verified. These are running jobs, not completed asset receipts.
Office original assets remain missing.

DEV_SCREENING_STATUS=PARTIAL

STATIC_CONFIRMATION_STATUS=PARTIAL

DYNAMIC_CONFIRMATION_STATUS=NOT_RUN_RAW_MISSING

Code and compact intermediate results are intended for the named research branch;
large geometry/prediction sidecars remain local. Official pretrained weights are
referenced at their original HF source and are not redistributed. No new model
has been trained. Limited combinations, recovery attribution, selection,
confirmation, visualizations and final GitHub handoff remain outstanding.
