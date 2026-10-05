# Final Codex execution instruction — prediction-preserving recovery runtime

## 0. Mandate, scope and success criterion

Work in `Orangekostar/oviovo`. Implement and actually run a bounded optimization of
the measured area-fallback **v2** semantic-recovery pipeline. Produce measured
runtime breakdowns, demonstrate scientific-output parity, reuse the existing
recognition controls, render paper tables, and push the code and compact evidence.
Do not stop at this plan, synthetic tests, a profiler screenshot, or an unexecuted CLI.

The target is the existing **G1/G3 recovery**, not a new classifier, new recovery
acceptance rule, new scene subset, or faster but different reconstruction. Do not
change depth, poses, frame schedule, full-scene occlusion, candidate registry,
thresholds, chosen frames/masks, v2 fallback, vocabulary, text prototypes, FP32
operators, N/Q/F temperatures, D2, positive/unknown semantics, official ranking, or
matching protocol. Preserve the raw/painted support and all real output owners.
No map reconstruction, segmentation inference, Native crop inference, model search,
training, or additional benchmark is authorized. Actual FC inference for cold
measurements is authorized and must be counted. Never claim this task is zero-GPU.

The paper keeps **three main tables**. Reuse A2/A5/A3 for a small supplementary
readout comparison inspired by OVI-MAP Table 7; do **not** call it a strict backbone
ablation. Add a supplementary recovery-runtime breakdown inspired by Table 8.
Do not copy OVI-MAP's parallel/skip-frame online protocol into this offline task.

The scientific method remains `CT_A3_ER`; only its execution implementation is
selected. Deployment remains `N0_UNCHANGED`. A negative speed result is a valid
completed result. Faster execution never establishes a new accuracy gain, online
operation, statistical significance, or cross-mapper generality.

## 1. Resolve and freeze the actual measured source FIRST

Inspected remote commit: `77335848b776d99ffb4ca772e33fa391d9fd8e5e`.
New task branch: `research/ovimap-runtime-parity-v1`.
New worktree: `/mnt/shared/ww/ovimap-runtime-parity-v1/worktree`.
New result root: `/mnt/shared/ww/ovimap-runtime-parity-v1/attempt_001`.
Source hints:
- `/mnt/shared/ww/ovimap-cvpr-compact-tables-v1/worktree`;
- `docs/paper/static_ovmap/AREA_FALLBACK_V2_RESULTS.md`;
- `artifacts/static_ovmap/area_fallback_v2`;
- the compact-table task root and receipt-linked timing/results directories.

The remote inspected source lacks v2. Treat the user's reported 172 outputs, 14
pools, 74 recovered empty pooling masks, 175 unchanged prior vectors, G1=45.58s and
G3=45.63s as **discovery clues**, not values to inject or mandatory target scores.
Resolve exact producer files, effective model/operator identities, manifests,
source pools and latest eight-scene v2 latency receipts using actual records.
The name `v2`, a file timestamp, or a report paragraph alone is not a binding.

Read `git status --short`, HEAD, worktree listing and the narrowly relevant tracked
changes/untracked producers. Follow recorded paths only within the named project
roots, plus dependencies referenced by those receipts. Do not scan or hash all of
`/mnt/shared`. Prefer an explicit supplied `--v2-results-root` when present. If
multiple copies have identical content, treat them as aliases; if distinct copies
exist, bind the one whose producers/receipts match the measured outputs. Do not
select by better metrics or newest modification time.

Create `reference_binding.json` with source HEAD, actual source hashes, effective
fallback operator description from code, source→artifact lineage, model/checkpoint
identities, source path map, full 26-sequence registry, and per-arm U2/G1/G3 semantics.
Record unknown items explicitly. The actual fallback implementation is immutable;
do not infer its formula from the words “area-weighted fallback.” Confirm whether
U2 is the unchanged legacy control or also consumes a v2 path and preserve that
actual choice. A2/A3/A5 must still share identical successful G1 recovery evidence.

Preserve the source worktree, its index, all old results, cold-run reservations,
and external jobs. If v2 is dirty/untracked, make a separate worktree from its
compatible HEAD, copy only the measured producer/import dependency changes and
compact relevant v2 reports into it, and make a dedicated binding commit. Record
the original HEAD + patch + file hashes. Never reset/stash/clean the source tree.
Do not copy scans, weights, big tensors, the entire parent evidence archive, or
unrelated dirty work into Git. Reuse an existing target worktree only with matching
task and binding identity. Create an immutable reference snapshot/commit for the
new task. `reference` must execute these real operators, not an approximate rewrite.

The old `require_frozen_execution` hashes source files, and old timing reservations
forbid rerunning the same leaf. Implement a **new task adapter and new ledger** that
consumes the immutable v2 scientific binding explicitly. Do not edit the old freeze,
masquerade as an old task, monkeypatch away its checks, or overwrite prior timing.
Both reference and fast execution must use the same new bookkeeping boundary.
Retain the actual reference math and the original end-to-end work; any adapter-only
change is disclosed and shared by both. The old 45.58/45.63/6.61 values are context,
not the formal denominator for this task's speedup.

If the measured v2 source or its evidence cannot be resolved, stop dependent GPU
work with `BLOCKED_V2_REFERENCE`, publish the supported code review/partial artifacts,
and list the exact missing paths/identities. Do not silently fall back to v1 or
invent a replacement v2. No clarification is needed when the local files resolve it.

## 2. Inspect the real execution chain and bind every change to a finding

Read the actual counterparts of these pinned anchors (SOURCE_EVIDENCE.md):
- `cvpr_compact/projected_views.py`: camera rays, FullSceneProjector, top-k retention,
  request identities and RGB/mask loading.
- `cvpr_compact/region_worker.py`: FCSession.encode, image grouping, signed pooling,
  U2 plan, v2 override/dispatcher, outcome handling.
- `cvpr_compact/recovery_run.py`: resident inputs, build views, classify, export,
  output/receipt writes and repeated source reads.
- `cvpr_compact/timing.py` / `runtime.py`: exact timer, source freeze, reservations,
  run ordering, CUDA synchronization, post-timer parity.
- `a7_evidence_upgrade/region_adapter.py`: image size/normalization, original pooling,
  FP32 feature generation; actual v2 fallback producer separately.
- `recovery_wave2/binding.py`: existing verification memoization.
- `cvpr_compact/outputs.py`, evaluation and table writers: exact output support,
  current-class ranks, A5 unknowns, matcher traces, units and cell provenance.

Create one concise `source_to_task.md` with finding → actual function → proposed
change → invariant → evidence artifact. Confirm a suspected waste still exists in
v2 before implementing it. Existing image grouping, model residency, inference_mode
and input verification memoization are **already present**, not new contributions.

Implement a narrow new namespace, suggested `src/static_ovmap/runtime_parity/`,
with binding, kernels/loaders, runner/profiling, parity and reporting responsibilities.
Names may be consolidated. Keep the old scientific entrypoints/results unchanged.
Expose:

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python \
  scripts/evaluation/run_ovimap_runtime_parity.py \
  --spec configs/static_ovmap/runtime_parity_v1.json \
  --source-worktree /mnt/shared/ww/ovimap-cvpr-compact-tables-v1/worktree \
  --phase all --resume
```

This is a CLI to IMPLEMENT, not one claimed to exist. Support phases `bind`,
`profile`, `screen`, `verify`, `freeze`, `benchmark`, `tables`, `publish`, `all`;
optional `--v2-results-root`, `--gpu`, `--output-root`, repeated `--path-map`.
Use actual inherited controller/FC interpreters if the displayed default is absent.
Model workers, not the controller's presumed environment, execute FC. Full `all`
means all missing authorized work, not the old 172-row neural pipeline or all tests.

## 3. Implement at most three cumulative execution candidates

R0_REFERENCE: actual v2 operations, no claimed acceleration.

R1_IO_EXACT:
- reuse immutable geometry already present in RecoveryInputs instead of reopening
  the same compressed surface; identical initial resident inputs for R0/R1;
- prepare camera validation/intrinsic inverse once per frame, preserving actual
  ray arithmetic; create any reusable pixel grid inside each measured call;
- count owner areas/bboxes in grouped passes; retain only true Top-3 candidates
  before full mask packing/hashing; preserve the exact tie order;
- parse capture/request manifests once per call; decode/convert one RGB per image
  identity and share it within that call, with per-target masks still validated;
- preserve all fixed selected views and complete image preprocessing; no ROI crop
  is sent to the FC backbone; no new fallback or per-scene special case.

R2_ROI_EXACT: R1 plus conservative query-ray pruning. Move candidate frustum tests
before ray casting; compute a union of conservative projected candidate bounds;
cast required pixels against the **unchanged complete scene**, not target meshes.
Handle near-plane/camera-inside/uncertain cases by full-frame queries. Details and
mandatory parity/fallback behavior are in IMPLEMENTATION_CONTRACTS.md.

R3_RUNTIME_EXACT: R2 plus allocator/transfer/serialization improvements only:
- remove per-image empty_cache in the fast path, retaining no scene feature values
  between calls; allocator storage is not semantic feature caching;
- defer already-computed per-region GPU→CPU copies to a per-frame collection when
  original numerical operations/order and exception outcomes remain unchanged;
- eliminate redundant host/device round trips used only for duplicate identity
  calculations when actual input-tensor identity can be preserved exactly;
- reuse repeated within-call verification/calculation results, not prior run data.
Do not move preprocessing normalization from GPU to CPU unless exact input tensor
parity is proven; leave it unchanged otherwise. Do not batch backbone images or
rewrite region reductions this round. No AMP, FP16/BF16, TF32 toggle, compile,
channels-last, different neural kernel settings, resize reduction or model changes.

All candidates must exist as explicit flags/variants, not manual edits chosen per
scene. When an optimization is already present, a required API is unsupported, or
precision/parity fails, record `NOT_APPLICABLE` / `REJECTED_PARITY` and keep the
previous cumulative candidate. Do not introduce a fourth algorithmic direction.
No cross-scene CPU/GPU pipelining in this task; it would change the latency target.

## 4. Bounded pilot, profiling and implementation selection

Pilot scenes are fixed to **room0 and office1** (normal and small-support coverage),
not selected by timing or accuracy. The pilot target is G1. Each enabled R0/R1/R2/R3
gets two isolated cold calls on each pilot: at most 16 calls total. The baseline
profile calls occupy the first R0 pilot slots; they are not extra runs. First round
uses ascending variants; second uses reverse order. Same-stage lightweight timers
are enabled for all variants. Do not run an expensive whole-scene kernel profiler;
CPU/raycast call counters, exclusive stage wall time and CUDA event annotations
are enough unless one bounded segment is needed to explain a specific discrepancy.

Immediately compare planned selections, masks, encoding successes, unit vectors,
labels and exports against the immutable v2 reference. Retain the actual failed
costs; a mismatch must not be converted into a feature-cache hit or filled with
reference labels. Fix a general defect or reject the offending optimization. Do
not whitelist scenes/objects based on GT, known winners, or reference decisions.

For each passing variant compute mean time per scene over both repeats, then the
mean of those two scene means. Choose the lowest mean including R0. If variants
are within 3% of the fastest, choose the lower variant index (simpler execution).
This is implementation selection only; AP/mIoU cannot be selection criteria.
No repeated runs to obtain a lucky minimum. No claim of significance from two trials.

## 5. Verify the candidate before final timing

Use the fixed 8+18 parent geometries. At most 52 CPU projection passes are allowed
outside pilots/final timing: one optimized pass per scene plus a reference pass
only when required baseline per-frame diagnostics are genuinely missing. No new
maps or new FC inference on CF18 is authorized.

Compare all candidates' per-frame visible areas/bboxes/admissibility and the final
ordered Top-3 request evidence against stored v2 reference data. Full-resolution
selected masks must match exactly. If a reference diagnostic is absent, compute
that reference in the CPU-only parity lane rather than infer it from final labels.
Global image statistics that sparse queries no longer compute must be null/scoped,
not falsely compared to old full-frame diagnostics.

Use exact scientific content keys, not raw producer-bound request IDs. Build an
explicit old↔new mapping; keep new implementation hashes honest. For unchanged
CF18 selections, consume parent vectors only in this **non-timed parity lane**,
regenerate A2/A3/A5 exports, and compare owner/semantic arrays and ranks exactly.
This is cached projection/export verification, not CF18 live optimized-model timing.
Actual fast FC parity is measured on the pilots and all Replica final calls.

If any projection/input/selection parity fails, disable that optimization globally,
choose the next passing pilot candidate and perform its needed verification once
within the same budgets. Never fix mismatches by overwriting candidate output with
parent masks or labels. Unresolved failures mean reference is retained. This task
must not turn into an accuracy-changing method search.

## 6. Freeze and run the final cold benchmark

Before looking at final runtime results, commit `implementation_freeze.json` with
reference source, candidate flags, scientific binding, hardware settings, fixed
scene/repeat/arm order and the new timing definition. Do not mutate frozen source
for a nicer runtime after completion; ordinary crash fixes receive a new attempt
identity with old evidence retained.

If a fast candidate is selected, run on all eight Replica scenes, two rounds each:
`U2_CONTROL`, `G1_REFERENCE`, `G1_SELECTED`, `G3_SELECTED`: **64 real cold calls**.
If reference is retained, run `U2_CONTROL`, `G1_REFERENCE`, `G3_REFERENCE`: **48 calls**.
Round one uses listed scene/arm order, round two reversed scene/arm order. No new
G3-reference timing is needed in the candidate branch; therefore do not claim a
strict contemporary G3 speedup against the historical 45.63s. G3's new latency and
parity are still measured. Main speedup is new G1_REFERENCE versus G1_SELECTED.

Model, frozen text, raw common map, incumbent N/Q/F, and evaluation projection may
be resident for every call. Recovery views, recovery vectors, selected-frame lists,
BVH, candidate ROI caches, recovery RGBs and prior outputs must not survive a call.
All G1/G3 view search/BVH construction starts inside its own timer. Same-call input
and dense-frame reuse is allowed. The allocator may retain unused storage but not
live scene-derived tensors. Run one identical shape-only warm-up per model process,
with no scene pixels, outside the timer. Do not flush the OS page cache or require
administrator access; disclose its uncontrolled state. Preserve original TF32 and
determinism settings, one GPU, projector threads=4, encoder batch=1, and one timed
job. Report actual CPU/GPU models, versions, thread controls and memory peaks.

Record each measured total with CUDA synchronization only at total start/end;
lightweight stage attribution is defined in TIMING_AND_TABLES.md. Do not insert
per-op synchronizations or move expensive work out of the candidate's interval.
Compare to v2 reference AFTER stopping the timer, never by reading its output in
the prediction call. All output writes within the inherited return boundary stay
inside; one shared new receipt format is used for reference and fast executions.
Legacy special audit work is disclosed separately, not silently counted only in R0.

Exactly two finite measurements per scene/arm are required for an aggregate.
Use every valid measured value, never the minimum. Performance outliers remain.
A demonstrated external crash/contention failure may retry once with a new attempt
record; at most four extra environment-failure attempts over the task. Do not retry
for slow speed, weak gain, or changed predictions. `--resume` reuses completed
measurements by source+variant+scene+round identity; it must not invoke old reserved
measurements. A live process is not a failed attempt and must not be killed.

## 7. Accuracy and cost acceptance

Exact equality is required for physical mask support, selected frames/masks/bboxes,
per-request available/failed/fallback state, recovered set, class labels, source
order, output owner/semantic arrays, area rank strings and matcher/export context.
FP32 features/scores may use atol=rtol=1e-5, **only with exact downstream labels**.
Store maximum errors, including normal and fallback regions separately. A near-tie
flip is a parity failure even when AP happens to stay identical.

The previous 74/175 checks are scoped historical findings, not expected counts for
new timed repeats. Check the actual v2 cohort inventory; do not conflate duplicated
G1/G3 timing requests with unique scientific inputs or with recovered objects.
Prediction-identical outputs can inherit locked v2 metrics and matcher evidence
using explicit identity maps; they do not need new GPU or 172 evaluator reruns.
Import and verify all 172 scene rows and 14 pools once; retain their original science
identity. Source/timing identity may differ. Never enter rounded user-report values
as measured cells. If exact parity fails, do not claim metric reuse is valid.

For speed, report T_ref/T_fast and (T_ref−T_fast)/T_ref from the paired complete new
measurements. A ≥5% mean reduction with all parity checks is the predeclared material
runtime target, not a statistical test. Publish smaller/no gains honestly. Do not
switch implementations per scene, remove slow scenes, or replace the frozen nominee
based on the final eight-scene results. If the final gain is not material, retain
reference as the recommended execution and report the candidate as evaluated.

Track ray counts, considered/admissible views, packed masks, RGB decodes, mesh loads,
model/image batch calls, pooling/fallback calls, unique frame content and cache hits.
No old feature/view/result hits are permitted in a valid cold call. Lower physical
ray/IO work is allowed; successful neural request semantics are not reduced.

## 8. Paper tables and publication

Keep main Tables 1–2 scientific values unchanged, use the bound complete v2 rows.
Update main Table 3 with a single consistent current timing series for U2, selected
G1 and selected G3 and the same v2 accuracy/counts. None remains zero by definition.
Do not combine an old fast U2 time with newly redefined candidate measurements.
Display TP/FP attribution ambiguities and the actual valid candidate denominator.
A reference-retained run is not labeled an optimized success. If the frozen fast
implementation passes parity but its final speed gain is small/negative, keep that
evaluated implementation in the new Table3 and show the reference comparison in S8;
do not swap individual Table3 rows to faster final results. Recommending the reference
for future use is a separate statement. A final parity failure invalidates that
implementation timing/metric reuse; report incomplete timing rather than borrow a
G3 reference measurement that was never made in this task.

Generate supplementary readout table S7 from **A2/A5/A3** on both cohorts. They share
G1 recovery, but differ in incumbent Native / FC-only / D2 treatment. Label it
“Readout comparison with shared recovered instances,” not “pure backbone ablation.”
N/Q share network weights; parameters are optional actual counts, never copied from
OVI's model-name table or doubled for Q. A5 still has Native-painted geometry and
input prerequisites; it is not a wholly Native-free pipeline. No new backbones.

Generate supplementary runtime table S8 with U2 control, G1 reference and G1 selected
columns; full pipeline stages/time and separate GPU service diagnostics as specified.
Also publish per-scene timing and each repetition, selection/pairing, memory and work
counters. G3 may appear as an extra SI column but is not a new main table. Render
LaTeX to PDF, inspect for clipping/readability, do not fabricate empty cells or shrink
to unreadable fonts. Source every numerical cell from machine results.

Required repository outputs:
- `docs/paper/static_ovmap/RUNTIME_PARITY_RESULTS.md`;
- `RUNTIME_PARITY_HANDOFF.md`, `RUNTIME_PARITY_SELECTION.md`, `RUNTIME_PARITY_CLAIMS.md`;
- real new code/CLI/config, limited production tests, narrow parent bridges;
- `artifacts/static_ovmap/runtime_parity_v1/` containing binding, code provenance,
  freeze, pilot/final time records, parity maps, stage/cost counters, imported compact
  metric references, Table3/S7/S8 LaTeX/PDF and per-cell provenance.
Keep heavy weights, scans, full matrices/TSDF/dense tensors on shared storage with
exact reproducing commands. Target ≤25 MiB new compact artifacts. Preserve licenses,
no credential or font-file uploads. Do not copy the old 35 MiB archive merely to
show publication completeness.

Run focused tests once after implementation and rerun only failures/changed paths.
Use real production projector and v2 fallback parity, not just this package's toy
kernels. One final completion review checks the declared workload, no full-CROVE
suite, no serial repeated whole-history hashing or expanding test-count quota.
Separate statuses: source binding, implementation, parity, timing coverage, metric
reuse, measured speed result, and publication. Keep incomplete leaf totals explicit.

Commit all actual deliverables, then ordinary `git push -u origin
research/ovimap-runtime-parity-v1`. Verify full local HEAD equals `git ls-remote`
for that branch. Only then write external `publication/final.json` with
`PUSH_VERIFIED`. Receipt must remain outside its own commit to avoid an infinite
verification-commit loop. Authentication failure means PUSH_FAILED with an exact
retry command; never claim success. No force push, no automatic deployment change.

Finish by linking the four GitHub reports, giving the full SHA and actual source
v2 binding, showing main Table3 and S8, listing speedup/parity/cold-call counts and
actual new image/pooling costs. State explicitly that unchanged accuracy is expected
and that runtime optimization does not repair CF18's scientific recovery tradeoff.
