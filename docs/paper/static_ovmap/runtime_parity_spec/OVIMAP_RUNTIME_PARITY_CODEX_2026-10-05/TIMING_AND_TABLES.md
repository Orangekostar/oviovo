# Runtime measurement and compact paper-table contract

## 1. What the measured latency means

The primary value is **feature/view/result-cache-cold incremental recovery latency,
model and common map/incumbent evidence resident, seconds per scene**. It is not
end-to-end mapping, not per-frame control latency, and not GPU active time. The
user's previous values are references for discussion only: U2 6.61, G1 45.58 and G3
45.63 seconds. Their differences do not identify individual stage costs because
workloads differ. No fixed final speed or speedup is promised. Shape-only warm-up image inputs are
counted separately from scientific/cold scene inputs; they are not free compute.

In particular, shape-only kernel warmup and the allocator are not semantic feature
caches. OS file pages may remain warm; do not call this a disk-cold benchmark. An
entire scene's decoded images, selected views, ray grids/BVH or pooled vectors may
not be computed before the timer or retained from another arm. Preloaded raw xyz,
faces, raw/painted IDs and incumbent N/Q/F are permitted identically for both paths.

The complete return boundary includes output-array/rank construction and the
inherited necessary prediction/file writes. A common new per-call evidence writer
may avoid duplicate legacy records in **both** variants. Disclose this adapter
change, keep its wall time included, and compare reference/selected under that same
boundary. Moving the candidate's writes or camera work out of the measured callable
is not optimization. Record common preflight/model load and post-call parity
separately. None(A1) has zero incremental recovery by definition, not a measured
zero-duration function.

## 2. Repetitions, fairness and coverage

Pilot maximum: 2 scenes x 4 implementations x 2 repeats = 16 G1 cold calls. These
include the R0 profile calls. No appended pilot reruns for a nicer median. Choose
implementation by the equal-scene mean of the two repeats, with the defined 3%
simplicity band, and verify before freezing.

Final with a selected fast candidate: 8 scenes x 4 labels x 2 repeats = 64 calls.
Labels: U2_CONTROL, G1_REFERENCE, G1_SELECTED, G3_SELECTED.
Final if reference retained: 8 x 3 x 2 = 48 calls, U2_CONTROL/G1_REFERENCE/G3_REFERENCE.
Thus normal cold calls are at most 80 for the complete task. Four documented
infrastructure-failure attempts are allowed in addition; a valid slow run cannot
be discarded as an outlier. Final runs are different identities from pilot runs;
old cold reservations are never edited.

Round 1 uses listed scene and arm order; round 2 reverses both. Run one job with one
physical GPU and fixed host thread settings, no overlapping maps/evaluations on the
same allocated resources. Record environment library versions and actual device
UUID, CPU and threads, not just “A40.” If device changes, old/new times are separate
series, not one ratio. Use the same model precision, TF32, determinism and launch
settings; do not tune thread count after seeing the winners.

For arm a and scene s:
    mean_s(a) = (t(s,a,1) + t(s,a,2))/2
    mean_cohort(a) = sum_s mean_s(a)/8
    G1_speedup = sum_{s,r} t(s,REFERENCE,r) / sum_{s,r} t(s,SELECTED,r)
    reduction = 1 - mean_cohort(SELECTED)/mean_cohort(REFERENCE)

Report both individual values, scene means, cohort mean, each round's cohort mean,
and range of within-scene repetition differences. Do not report the minimum per
scene, average ratios instead of aggregate latency ratio, or a variance inflated by
treating every pixel/view as an independent scene. Two repeats do not establish
statistical significance. A >5% reduction is an engineering materiality criterion.
The original scientific A3 v2 has not been reselected by AP.

## 3. Stage attribution without corrupting timing

Place one synchronized wall timer around the whole callable. Track exclusive host
wall spans with a lightweight nested stage timer. Use these categories:
1. `registry_geometry`: candidate preparation, in-call shared-input checks,
   triangle ownership preparation, acceleration structure/bounds construction;
2. `view_search`: depth input I/O, rays, first-hit queries, depth checking, grouped
   counts, Top-3, selected-mask and request materialization;
3. `rgb_preprocess`: selected RGB/mask loading, color conversion, exact resize and
   normalization; keep asynchronous work dependencies accounted for;
4. `recognition`: original FC encoding, normal/fallback pooling, visual projection,
   transfer of region vectors and CPU cosine classification;
5. `export_bookkeeping`: output/rank construction, files, compact required receipts;
6. `other_sync`: measured residual dispatch/synchronization and gaps, not silently
   discarded time.

Never double-count nested spans. Record CUDA event service times for FC encoder and
region/fallback kernels separately; these annotate recognition, they do not add to
wall time. Do not sum host launch time + GPU service time as a total. For the current
sequential execution the recognition phase ends only after vectors are available
for CPU classification; use its actual dependency, not a synchronize after every
mask. Any final CUDA wait belongs to other_sync. Exclude intrusive kernel profiler
runs from the formal series. Both variants receive identical timing instrumentation.

Store detailed subspans for mesh reopen/decompression, RGB decodes, candidate hashes
and serialization when useful. If stages account for <98% of total wall time,
explain the residual; do not normalize the parts to force 100%. Overhead reductions
from bookkeeping versus geometric queries must be shown separately. An optimized
implementation may emit smaller truthful scoped diagnostics, but cannot claim
uncomputed full-image totals or omit prediction outputs.

Work counters: all scheduled/completed frames, skipped-by-safe-bound frames,
full-image pixels, queried rays, BVH builds, raw candidates, admissible views,
materialized/packed masks, unique decoded RGBs, FC image inputs and batch calls,
normal/fallback pools, transferred vector bytes, cache hits, output bytes. Count
logical repeated requests separately from physical encodings. An empty-candidate
scene still contributes its measured overhead to the denominator.

## 4. Main tables remain three

Tables 1–2: use the same bound area-fallback v2 172 scene results and 14 pools. No
new backbone/threshold/scene-selection row. Preserve actual comparison results,
including CF18 recovery losses and FC-only's strong metrics. Numeric output uses
percent once; all derived differences use percentage points. If an imported source
is fractions, convert at render time; do not fit values to the user's rounded table.

Table 3: None, archived U2, independent G1, independent G3. Scientific counts/metrics
are inherited only after exact export parity. Time for each non-None row is the
same **new** final resident-model/cold-call series, not a mixture of reference old
U2 and accelerated new G1. When no candidate survives pre-freeze selection, use the measured
retained reference and state it. When the frozen candidate survives parity but is
not materially faster in the final series, keep its evaluated Table3 rows and
report that outcome; a future reference recommendation does not relabel those rows. No claim of a new scientific method from runtime
repeats. Retain dagger ambiguity in the original matcher trace; no false TP/FP
precision from ambiguous ties. n/N is output coverage, never correct-object recall.

The main caption can be short:
“Recovery quality and incremental latency on Replica-8. n/N is the number of
candidate owners receiving output semantics, not true-positive recall. TP/FP are
definite added matcher entries at IoU 0.50; tied attributions are reported separately.
Latency averages eight scenes and two repeats with resident model/common map and
no recovery feature/view/result cache. Mapping and evaluation are excluded.”
The changed implementation version belongs in the method/implementation note.

## 5. Supplementary S7: reuse, not a new five-model study

Rows: A2 Native incumbents + same G1-FC recovery; A5 FC-only incumbents + same
G1-FC recovery; A3 D2 incumbents + same G1-FC recovery.
Columns: readout, shared recovery, Replica APall/AP50/mIoU, CF18 APall/AP50/mIoU.
All values link to existing Table2 cells and the same recovered set. Label this
**readout comparison**, not controlled backbone-only ablation. It does not separate
backbone, crop/pooling and text-template effects. A5 FC unavailable means unknown,
not hidden Native fallback. All A5 geometry/input prerequisites stay accounted.

A short model note may count actual unique network parameters, encoder batch/image
size and frozen/fitted components. N and Q share weights, so do not double network
parameters. Do not load additional models just to reproduce OVI-MAP Table7. No
claim of universal backbone independence follows from this reuse table.

## 6. Supplementary S8: runtime with same-scope controls

Columns: U2_CONTROL, G1_REFERENCE, G1_SELECTED (or reference retained).
Rows: registry/geometry, view search, RGB/preprocessing, recognition,
export/bookkeeping, other/sync, complete incremental total. Add GPU service times
in a small separate diagnostic block with “non-additive” in its heading. Publish
per-scene/repeat rows as CSV/JSON. G3 selected time may be an additional SI column;
there is no strict G3 speedup claim unless a contemporary reference was actually
measured (not required here).

Do not paste OVI-MAP Table8 numbers measured on RTX3090 into the same ranked timing
columns as current A40 values. Its thread/keyframe protocol is different. This task
measures recovery, not the original whole mapper. No end-to-end FPS or online claim.

## 7. Table provenance and render check

Every accuracy/count cell links to parent result content identity plus output-parity
receipt; every runtime cell links to all 16 constituent measured call IDs. Shared
cells across tables use one data object, not typed copies. Missing leaves => null
and a specific reason, not averages over seven scenes or a predicted time. No
placeholder '--' can be presented as a measured zero.

Generate LaTeX and one small PDF preview with real final data. Check no clipping
and legible type. Keep full running-cost logs in shared storage, but upload compact
scores/counts/timing/parity with the paper tables. Do not add multiple new main
figures or a 4th main table for profiling details.
