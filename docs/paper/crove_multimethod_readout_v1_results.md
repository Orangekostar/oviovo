# CROVE multimethod readout — partial DEV evidence

Task revision: V1_REVISED_V2. Evaluated implementation: `aed101c` (execution before
commit; subsequent changes were formatting and equivalent loop unpacking only).
The paired adapter, semantic controls and initial graph batch use `1efef91`
(executed before commit; subsequent changes were formatting/import ordering and
the explicitly documented forward-count reporting correction).
The diverse/quality batch and real-posterior graph controls use `cbb6543`;
the tie-rule fix was verified by exact full-map prediction recomputation.
Independent-mask pipeline diagnostics use `943a96c` (subsequent pre-commit
changes were formatting and additional residual-count reporting only).
Apartment cached-view and adapter pairs use `d4e3d67`; the shared bridge extraction
preserves the executed cached-view computation and is exercised by the adapter pair.
Apartment diverse/quality and posterior graph controls use `43f8aae`;
the extracted evaluator AST is identical to the previously evaluated kernel.
Apartment independent-mask binding and restored-row attribution use `ba94d66`.
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
alias, not evidence for dynamic adaptation. Structural-background patch
observations still require separate work. The original room0 and native-build
`VLModel.encode_image_with_bbox` both normalize each of six crop features before
averaging; M1 subsequently normalizes that per-view mean. The prior statement
that per-crop normalization required re-encoding was incorrect. Individual crop
scale ablations are unavailable from these means, but the prescribed normalized
six-crop hierarchy is already present (also documented in task evidence E15).
The audited source SHA256 values are `607ed3d0ddf8eb39b8989e606d0d7797d8966bc902952fff819ccb020a5169a9`
(room0 checkout) and `b0925c0c78fd46be779fb9fbcde171fcaef58bd89a7f7d99b8fc1b24a1836ba2`
(native build). Native source manifest commit `58a804e2d7c82ba05a489eb071aba3367301fed8`
also contains this operation. These are source audits, not reconstructed raw crops.

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

### Apartment cached-view B3/H2 DEV pair

Every method uses the exact saved B3 (15,622,601 rows) and H2 (15,625,540 rows)
current sets, unchanged owner IDs and the verified legacy evaluator row ordering.
Native features are hash-bound to the lifted entity manifests and color IDs.
Local frame IDs map to original 766–1021/1217–1472 windows, and t1 owners retain
the existing +1,000,000 offset. Text covers the complete public 20 known-class
common-v2 vocabulary, not classes selected from GT. No image encoder is rerun.

| State | Method | current mIoU | Surface precision | Surface F1 | Original gates |
| --- | --- | ---: | ---: | ---: | --- |
| B3 | MV_NATIVE_CACHED | 0.135886 | 0.734413 | 0.431335 | PASS |
| B3 | MV_SINGLE | 0.132403 | 0.669867 | 0.423467 | FAIL |
| B3 | MV_TOPK_MEAN | 0.139546 | 0.343176 | 0.324609 | FAIL |
| H2 | MV_NATIVE_CACHED | 0.135734 | 0.734726 | 0.435710 | PASS |
| H2 | MV_SINGLE | 0.133032 | 0.670218 | 0.427587 | FAIL |
| H2 | MV_TOPK_MEAN | 0.139394 | 0.343541 | 0.327174 | FAIL |
| B3 | MV_DIVERSE_MEAN | 0.135886 | 0.734413 | 0.431335 | PASS |
| B3 | MV_QUALITY | 0.137077 | 0.736586 | 0.434848 | PASS |
| H2 | MV_DIVERSE_MEAN | 0.135734 | 0.734726 | 0.435710 | PASS |
| H2 | MV_QUALITY | 0.137684 | 0.736894 | 0.439203 | PASS |
| B3 | MV_QUALITY_PATCH_ONLY | 0.137077 | 0.736694 | 0.434867 | PASS |
| B3 | MV_QUALITY_GRAPH_GEOM | 0.137093 | 0.736702 | 0.434868 | PASS |
| H2 | MV_QUALITY_PATCH_ONLY | 0.137684 | 0.737002 | 0.439222 | PASS |
| H2 | MV_QUALITY_GRAPH_GEOM | 0.137699 | 0.737010 | 0.439224 | PASS |
| B3 | ADAPTER_CLIP_MEAN | 0.115477 | 0.203297 | 0.244898 | FAIL |
| B3 | ADAPTER_CLIP_LEARNED | 0.135273 | 0.198584 | 0.241745 | FAIL |
| H2 | ADAPTER_CLIP_MEAN | 0.115927 | 0.203402 | 0.245449 | FAIL |
| H2 | ADAPTER_CLIP_LEARNED | 0.136847 | 0.201748 | 0.245421 | FAIL |

For the six M1 rows, Ghost remains 0 and background F1 remains 0.366360. Native cached
aggregation reproduces the original point semantics and roles exactly (zero
changed rows) and all nine legacy metrics in both states. SINGLE changes 179,999
roles; TOPK changes 1,217,362 roles in each state. Their surface metric losses
are role-conditioned evaluation changes, not deleted or moved geometry. TOPK's
mIoU gain does not pass the original minimum surface-precision gate of 0.724413.
No final-current winner is selected from these failed candidates.

The bank contains 125 feature owners, with 853/125/482 selected observations for
native/single/top-k respectively. Coverage is 99.45831% in B3 and 99.45841% in H2;
missing observations preserve the exact old label and role. Prediction/write
cost is recorded per output, excluding shared model, geometry and GT work.
DIVERSE and QUALITY have now completed both states, with identical selected
views and own-state valid-row quality projection. DIVERSE reproduces all nine
native metrics. QUALITY improves current mIoU and surface F1 in both states and
passes the original gates, with Ghost 0 and unchanged background F1. It changes
8,250 roles in each state; geometry remains frozen. Coverage is
99.42440%/99.42379%; 482 candidate owner-views span 301 frames, with zero new image
forwards. Shared selection/projection take 32.40/67.81 s, excluding loading,
writes and evaluation. Dynamic cross-visit CURRENT_AWARE is still pending;
QUALITY is an eligible candidate, not a frozen final-family winner.

```bash
python scripts/evaluation/run_crove_multiview_apartment.py
python scripts/evaluation/run_crove_multiview_quality_apartment.py
```

The paired trained adapter has completed both states through the same bridge. Native
cached poses match the original global TESSE exported trajectories within 5e-7;
no per-visit first-pose normalization is applied. Each state's projected masks
use only its own frozen valid source rows. Both states and pooling heads share
the dense image forward. It used 237 actual forwards for 291 selected frames,
with 262/264 valid independent owner observations from 79/80 owners in B3/H2.
Coverage is 90.19937%/90.20308%. Shared projection/frame processing takes
31.91/55.27 s, excluding loading, cache writes and GT evaluation. Peak allocated
GPU memory is 1,792,770,560 bytes. Mean and learned controls share all inputs.

Learned pooling improves mIoU over mean pooling in both states, but all four rows
have Ghost 1.0 and fail the original Ghost and surface-precision gates. Background
F1 is 0.445280. Neither static head gains nor dynamic mean-pool gains imply an
eligible final-current readout. B3/H2 source sets and owners remain fixed.
An independent role-free geometry audit measures 6,895 confirmed-free conflicts
among 7,783 changed-region source rows in each state. All eighteen readouts have exact
source/owner equality and zero all-geometry conflict-count delta. Thus the change
in official Ghost is role-conditioned; no geometry is added by these readouts.
See `apartment_readout_geometry_audit.json` for counts and role distributions.

```bash
/home/ww/oviovo_baseline_builds/maskadapter/venv/bin/python scripts/evaluation/run_crove_adapter_apartment.py
python scripts/evaluation/audit_crove_apartment_readout_geometry.py
```

| Family | Real current status | Still required |
| --- | --- | --- |
| M1 | Partial room0 and Apartment native/single/top-k/diverse/quality B3/H2 pairs | structural patch evidence, dynamic state adaptation, confirmation |
| M2 | room0 S2/M1 and Apartment QUALITY patch-only/geometry graph pairs | dynamic local S2 control, full-input boundary evaluation and confirmation |
| M3 | room0 13-frame diagnostic; Apartment complete 512-frame/state independent banks, owner-only run started | full pairwise/consensus scores, resem and confirmation |
| M4 | Official trained room0 and Apartment B3/H2 mean/learned pairs completed | static confirmation and limited combinations |

M4 uses the official trained checkpoint, not random weights or a local untrained
replacement. Its three-mask numerical test is explicitly not a whole-map result.
Code retains official attribution and Apache-2.0 license.

B3/H2 legacy-to-pointwise equality is verified separately in `bridge_B3.json` and
`bridge_H2.json` (mIoU 0.135886/0.135734). The M1 dynamic pair above is now evaluated;
M4 also has complete dynamic paired scores; M3 dynamic results remain pending.
Apartment M2 topology preserves all 15,622,601/15,625,540 current source rows,
including isolated rows, and excludes faces containing invalid vertices. B3/H2
have 2,225,953/2,226,715 patch nodes and 2,712,720/2,712,879 undirected edges;
source backprojection is 100%. Different visits cannot weld or exchange graph
messages. Topology construction reads no GT and took 60.48/62.29 s.
Both state graph pairs are now evaluated. Patch-only changes 17,984 labels versus
pointwise QUALITY in each state but leaves current mIoU unchanged. Geometry
propagation changes another 5,685 labels and adds only 0.0000152/0.0000155 mIoU.
All four rows pass the original gates. This is a very small DEV increment,
not broad evidence of graph benefit; real boundary controls remain required.
Pointwise confidence is inherited, not relabeled as calibrated graph confidence.
The old Apartment source semantic cache is owner fallback, not independent local
S2 evidence, and is not presented as such a control.

```bash
python scripts/evaluation/build_crove_apartment_graphs.py
python scripts/evaluation/run_crove_graph_apartment.py
```

### H2 restored-source attribution (completed nine readouts)

The exact H2-minus-B3 set contains 2,939 source rows, representing 666 exact-XYZ
physical samples. No B3 row is removed. A post-prediction nearest-current-GT
diagnostic finds support strictly within 5 cm for 1,207 rows / 322 physical
samples; none of the restored rows is within 5 cm of confirmed-free centers.
This source-to-GT lookup is explanatory and differs from the official
GT-to-prediction mIoU lookup. Unmatched restored rows are not counted as errors
or evidence of correctness.

| H2 readout | Changed restored labels | Correct GT-supported rows | Correct physical samples |
| --- | ---: | ---: | ---: |
| Native / diverse / top-k | 0 | 589 | 145 |
| Single | 112 | 701 | 172 |
| Quality / quality patch / quality graph | 223 | 701 | 172 |
| Adapter mean | 1,812 | 554 | 139 |
| Adapter learned | 538 | 414 | 97 |

QUALITY changes 112 couch rows to chair and 111 ceiling rows to lamp. The
additional supported correct rows come from the first transition; the second
has no current-GT support under this diagnostic. This does not imply that the
whole map improved by the same amount. The artifact
`apartment_H2_recovered_semantic_attribution.json` includes all original-to-new
labels, role transitions and GT confusion counts. Physical transition mass
weights duplicate rows by inverse multiplicity; each table sums to exactly
2,939 rows and, within floating-point tolerance, 666 physical samples.

```bash
python scripts/evaluation/audit_crove_recovered_semantics.py
```

### Apartment independent-mask preparation

Both B3/H2 banks contain all 512 authorized frame observations (256 per visit),
bound by original native-mask hashes and each state's frozen patch hash.
They project actual current-source representatives through global TESSE poses
into original CropFormer PNG interiors, using measured depth within 5 cm.
Only same-visit nodes are observed; different visits never form a static union.
Empty, occluded and boundary observations supply no vote. No new segmentation
inference or GT input was used. The per-visit pairwise/consensus owner-only
runner is now executing with unchanged initial room0 parameters; results are
not yet claimed. Dynamic AP is N/A because this common-v2 protocol has no
corresponding instance GT. Static parameter selection is still pending.

```bash
python scripts/evaluation/build_crove_apartment_mask_bank.py
python scripts/evaluation/run_crove_consensus_apartment.py
```

room1 remains the frozen static confirmation scene, with raw inputs present but
derived OVI/S2 inputs pending. The native room1 GPU frontend failed with verified
CUDA OOM under existing GPU occupancy. Full-resolution CPU frontend preparation
is running for room1 mapping and room0 independent M3 masks; real first-frame
outputs are verified. These are running jobs, not completed asset receipts.
Office original assets remain missing.

The independent mask pipeline has been exercised on frames 0:130:10, not the
complete 200-frame protocol. It binds actual source points nearest the frozen
patch centers to depth-consistent CropFormer interiors; one-pixel region/image
boundaries supply no vote. The boundary path requires at least two joint views
before mask-disagreement attenuation, and uses real RGB means only when supported
by at least two observations. Geometry graph nodes/edges and symmetry are preserved.

The M3 adaptation retains mask visibility/containment, undersegmented-observer
removal and iterative consensus from the pinned MaskClustering mechanism. Unlike
the original implementation it uses fixed patches, frame-local boundary exclusion,
three fixed iterations and per-frame union support. Pairwise and consensus share
the input filter, source readout, 2-view/0.60-score/0.15-margin gates and residual
rules. New IDs require a supported split or merge; simple one-parent renaming is
suppressed. The diagnostic produced 29/32 new IDs, without deleting residuals;
source indices, semantics and confidence were verified exactly unchanged across
all 9,282,303 rows. No GT was opened, no AP was measured and no improvement is
claimed. Exact diagnostic counts are in `room0_mask_pipeline_diagnostic.json`.
Both formal boundary and M3 entry points reject incomplete frame sets.

```bash
python scripts/evaluation/build_crove_room0_mask_bank.py --available-only
python scripts/evaluation/run_crove_consensus_room0.py --diagnostic-available
# After all 200 input frames are ready:
python scripts/evaluation/build_crove_room0_mask_bank.py
python scripts/evaluation/run_crove_graph_room0.py --boundary
python scripts/evaluation/run_crove_graph_room0.py --unary mv_quality --boundary
python scripts/evaluation/run_crove_consensus_room0.py
```

DEV_SCREENING_STATUS=PARTIAL

STATIC_CONFIRMATION_STATUS=PARTIAL

DYNAMIC_CONFIRMATION_STATUS=NOT_RUN_RAW_MISSING

Code and compact intermediate results are intended for the named research branch;
large geometry/prediction sidecars remain local. Official pretrained weights are
referenced at their original HF source and are not redistributed. No new model
has been trained. Limited combinations, recovery attribution, selection,
confirmation, visualizations and final GitHub handoff remain outstanding.
