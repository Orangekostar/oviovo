# Final Codex execution instruction — compact CVPR experiment tables

## 0. Mandate and fixed result scope

Implement, execute, analyze, render and publish the three manuscript tables defined
here. Do not merely produce another plan, mock table or smoke-test report. Use this
instruction together with IMPLEMENTATION_CONTRACTS.md, TABLE_CONTRACTS.md and
PROTOCOL_SPEC.json. These documents are a single specification; their numeric
choices agree. Resolve a genuine contradiction by documenting it before dependent
measurements, not by choosing the version that scores better.

Repository: `Orangekostar/oviovo`.
Base: `1a1c4513ecf824c55274a456c40657271d631883`.
New branch: `research/ovimap-cvpr-compact-tables-v1`.
Task root: `/mnt/shared/ww/ovimap-cvpr-compact-tables-v1/attempt_001`.
Parent recovery: `/mnt/shared/ww/ovimap-recovery-wave2-v1/attempt_001`.
Parent backbone: `/mnt/shared/ww/ovimap-backbone-wave1-v1/attempt_001`.
Upstream OVI source pin: `f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424`.

Use a new worktree; preserve existing branches, assets, binaries and completed
receipts. Resume an existing task worktree only when its task/base/spec identity
matches. Normal fixes are permitted; no destructive checkout, forced push or
silent replacement of historical artifacts.

The scope is two cohorts, eight internal configurations and THREE main tables.
Main output count is 172 scene-method records; shared geometry consists of 26
native maps, not 172 reconstructions. A3 is the preregistered full method, not an
alias for whichever row later wins. G3 stays a diagnostic. No new neural training,
Q-head fitting, temperature fitting, weight search, SAM replacement, association
variant, renderer parameter sweep or extra benchmark is authorized.

Negative and mixed results are valid scientific outputs. Completion is not a
promise of a positive result. Do not hide a stronger simple control or substitute
U2 results for unexecuted G1. Deployment remains `N0_UNCHANGED`.

Copy PROTOCOL_SPEC.json verbatim to
`configs/static_ovmap/cvpr_compact_tables_v1.json` in the new worktree. The script
must accept `--spec`, `--phase`, `--resume`, `--gpu`, and repeatable `--path-map`
for relocating old referenced roots without changing file identities. Use the
inherited main/native/FC interpreter paths; default controller is
`/home/ww/miniconda3/envs/ovimap-map/bin/python`. Select the GPU from the inherited
working job configuration unless an explicit `--gpu` is supplied; use the same
physical device for the Table3 timing series and report it.

Required final run, from the new worktree:
```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python \
  scripts/evaluation/run_ovimap_cvpr_compact.py \
  --spec configs/static_ovmap/cvpr_compact_tables_v1.json --phase all --resume
```
The command must exist and execute the workload by the end of development; it is
not a command claimed to be available on the unmodified parent commit.

## 1. Repository reading and binding, before development decisions

Inspect the exact base files listed in SOURCE_EVIDENCE.md. Bind actual effective
producer sources and model assets from the parent receipts; a Git HEAD alone does
not establish which Python wrapper produced a cached feature. Reuse archived
producer sources when required by their receipts. Do not weaken old identity
checks to make a mismatched cache appear usable.

Create `src/static_ovmap/cvpr_compact/` and an entry script
`scripts/evaluation/run_ovimap_cvpr_compact.py`. Reuse low-level verified utilities;
do NOT repurpose the old task IDs, developer-only scene selectors, recovery arms,
confirmation-lock APIs, or task-local publication assumptions.

The existing `backbone_wave1.readouts.build_sources` shows the N/Q/F data path,
`recovery_wave2` supplies fixed-support exports and expanded-registry evaluation,
and `module_validation` supplies raw preparation and native capture. Their high-
level runners are restricted to old scenes and are not a CF18 driver.

Write `resolved_inputs.json` once for the task, containing:
- exact cohort order and per-capture physical family / prior exposure;
- original RGB-D, poses, aligned captured RGB/depth, camera intrinsics and 200-slot
  schedule; input units and image color convention;
- reusable map/TSDF/surface/capture/native-deferred metadata and source receipts;
- actual native binary, mapping options, upstream patch stack and producer hashes;
- N/Q/FC assets, category/text order, Q scaler/checkpoint and final temperatures;
- read-only parent caches and the new writable content cache;
- per-leaf source/resource availability, without claiming that local CF18 assets
  exist until they have actually been inspected.

Use configured parent roots and linked manifests first. Search plausible dataset
roots once, to a bounded depth. Hash each consumed artifact once and reuse an
unchanged filesystem-stamp memo. Do not recursively re-audit every historical
map or repeatedly hash all model files for each object.

## 2. Benchmarks, frames and dataset preparation

Replica order is `replica8.txt` (office0–4, room0–2).
ScanNet order is `scannet_cf18.txt`, all eighteen explicitly listed captures.
These are seven physical ScanNet scene families, not eighteen independent rooms.
The old four development scenes do not count as CF18 and cannot fill any CF18 cell.

Use `assets.native_schedule(N)`: step=N//200; the first 200 elements of
range(0,N,step), with the original indices. Keep matching inherited schedules
exactly. Do not substitute linspace sampling, refill invalid slots, or borrow
extra frames for G1. Read source N from the sensor/export record. Preserve native
handling of invalid poses, no-segment frames and other genuinely skipped slots.
Store both scheduled and actually completed frames.

Prepare only missing inputs for these eighteen ScanNet captures. Reuse the user's
already authorized official ScanNet downloader and existing consent/access. Download
only the required .sens/.txt/mesh/segments/aggregation/label-map files from the
verified original layout; do not accept new terms or obtain alternative credentials.
Use `export_sensor` with original JPEG bytes and native depth conversion. Do not
use `prepare_plan` (which selects unrelated random families) or `export_development`
(which silently excludes other roles). A new fixed-list bridge is required.

Obtain the exact ScanNet200 and Replica valid IDs, names, original text matrices
and annotation conversion from the pinned implementation and parent receipts.
The mapper's legacy task flag can differ from final Scannet200 evaluation; do not
change geometry because of a scoring-vocabulary name. Evaluation geometry/labels
must never become the input to the recovery visibility projector.

If access is genuinely unavailable, produce a named blocked leaf and continue all
independent work. A partial CF18 pool is NOT a CF18 score. Never drop a troublesome
capture. Do not initiate another dataset or external-system retraining to compensate.

## 3. Common native maps and exact E baseline

The only common geometry recipe is BB00_NATIVE: original CropFormer, native depth
fusion and mode-4 instance association, actual inherited remaining options, and
0.01 m voxels. Reuse compatible original maps; generate a missing native map once
for a sequence. Do not apply SYNC, RAW, RATIO, BIDIR, A/S recovery-wave mapping changes.
No algorithmic C++ changes are needed. If installation requires an isolated rebuild
of the existing capture stack, document it and validate one short disabled-hook
path; never rebuild every dependency or alter the original binary in place.

For a new capture, adapt the actual BB00 command and its environment from the
parent receipt. Preserve the already verified CropFormer model/config and output
PNG conversion. Reuse identical precomputed masks when present. Explicitly bind
scene/path/schedule substitutions rather than treating an old development receipt
as a new main-benchmark receipt.

Generate/recover Native and N/Q/F exactly once per map:
- N: original successful retained observations, original top-10 storage / actual
  native readout limit, canonical-relative Native labels and separate cosine N
  scores for D2. Do not confuse the two score representations.
- Q: frozen Q_GAIN policy and scaler, own causal acquisition history, maximum 200
  attempted queries. A genuinely exhausted candidate stream stops with its actual
  count; do not fabricate/repeat acquisitions to satisfy the old exact-200 assert.
  Use the already implemented exhaustion behavior as the template.
- F: original FC_FROZEN dense region operators, existing-object static requests,
  original maximum-three-view selection and area aggregation. Preserve its genuine
  technical failures, cap exclusions and availability.
- E: the original D2 grouped probability fusion, using the exact published final
  N/Q/F temperature vector used by the archived Replica D2, shared across the new
  main cohorts. Bind its three actual values and source receipt; do not substitute
  a leave-one-CAL-scene-out vector or fit CF18-specific scalars. Verify the eight
  archived Replica anchors use this final vector. Actual source availability is
  respected. No fourth source and no new gamma.

A0 and A1 must have identical incumbent owner partitions; only classes differ.
Use the original triangle-painted incumbent support, not the full numerical raw
owner raster. Match the eight historic Replica A0/A1 anchors by payload and scorer
identity when reuse is possible. Do not force exact agreement with historical
rounded metrics after an actual remapping; report the paired new baseline instead.

## 4. Freeze the experiment before new main-cohort evaluation

All visual models, prompts, temperatures, recovery thresholds and the eight method
recipes are fixed by this package. There is no learned-model selection or new
scalar fitting. A3/G1 is the named full method; G3 is NOT eligible to replace it.

Develop/test on the existing `scene0056_00` capture and synthetic fixtures first.
At most eight smoke outputs on that one existing development map are allowed; they
are not part of the 172 main outputs. Correct technical defects, do not tune new
threshold grids based on these scores. A technical projected-mask montage may be
viewed without GT labels to verify camera alignment, not to choose successful objects.

Commit the implementation and `freeze/experiment.json` before any new main-cohort
metric or result-driven choice. The freeze records all 26 sequences, method IDs,
models, numerical constants, preselected full method and data-exposure ledger.
Main input preparation and label-free capture may precede this commit. New main
predictions used to assess performance and all main scoring follow the commit.

Fixes discovered later remain permitted, but freeze an amendment, state whether
main outcomes were seen, invalidate and rerun only affected descendants, and
never call that corrected run an untouched confirmation. Do not rename A3 to A5
or pick different methods for different tables after seeing main scores.

## 5. Independent geometry-driven views: implement G1/G3

Follow C1–C5 in IMPLEMENTATION_CONTRACTS.md exactly. This is the only new scientific
input-generation module. It MUST NOT depend on whether an owner had a captured
native request. A geometrically visible eligible owner with zero archived requests
must still be able to receive a G1 request.

Construct the same omitted-owner registry as recovery_wave2 on the frozen raw
surface and native-painted incumbent support: absent positive owner ID, at least
100 residual source rows, maximum 128 candidates with the existing spatial ordering.
This is the shared candidate denominator for U2, G1 and G3. Do not use GT vertex
counts or GT labels to select or rank requests.

Raycast the FULL predicted mesh, including non-candidate and unknown-owner triangles,
from each original completed RGB-D camera. Use aligned captured RGB uint8 and
`depth_m` float32 with its exact K and pose. Do not repeat depth conversion, color
warping or world-axis alignment. Assign a face owner only when all three vertex
raw owners agree; other triangles remain occluders with owner zero.

Use integer pixel rays with camera z component 1, without normalizing the ray;
first-hit ray parameter is then optical-axis depth. Accept only finite positive
measured depth and the fixed tolerance max(0.02 m, 0.02*measured depth). Require at
least 100 valid target pixels and bbox extent at least 2 in each direction. Do not
dilate, fill holes, use target-only occlusion, substitute GT geometry, or recover
with a different method when there is no admissible view.

Rank each object's valid views by decreasing mask pixels, then original frame ID,
then mask digest. G1 takes the first; G3 the first three distinct frames. G1 is an
exact prefix of G3. Selection is finalized before any class score is read. Failure
of the selected view does not authorize looking for a better-classified view.

Create a new projected-request schema with final geometry identity, raw owner,
frame/K/pose/RGB/depth/mask identities, score-independent selection rank and both
canonical half-open and legacy native crop boxes. Do not fake a native paid request,
segment lineage or historical map state. Read-back of G1 manifests must not call
`_load_request` on nonexistent old capture arrays.

## 6. Encode requests once and construct the eight outputs

Use the actual pinned N and F inference operators and text matrices. Shared full-
frame FC tensors are reusable across targets and methods only under the effective
model/preprocessing/tensor identity; new masks need their own region pooling.
No cached probabilities may be used to reconstruct missing visual features.
Batch by frame, pool all required regions, then release dense tensors as needed.
New output must live in the new cache. Preserve parent caches read-only.

For G1_FC, a successful unit region vector is matched to the original FC text
prototypes with cosine similarity. G3 aggregates successful PRESELECTED views by
original valid-mask pixel area and normalizes once. Do not borrow unselected views
or change prompts. Zero successful selected views means unavailable, not Native
fallback. For G1_NATIVE, feed the same RGB frame and full target mask (also used as
its union mask) through original six crops and original single-native classification.
Keep the native bbox quirk explicit as in C4; this is a same-view comparison, not a
same-backbone/pooling-only ablation.

| ID | Existing owners | Omitted owners | Required cohort |
|---|---|---|---|
| CT_A0_NATIVE | original Native | none | 8 + 18 |
| CT_A1_E | fixed D2 | none | 8 + 18 |
| CT_A2_R | original Native | G1_FC | 8 + 18 |
| CT_A3_ER | fixed D2 | same G1_FC | 8 + 18 |
| CT_A4_NATIVE_RECOVERY | fixed D2 | same G1 view, native encoder | 8 + 18 |
| CT_A5_FC_ONLY | original F evidence only | same G1_FC | 8 + 18 |
| CT_H_U2 | fixed D2 | exact archived U2 rule | Replica only |
| CT_G3 | fixed D2 | G3_FC | Replica only |

A2/A3/A5 use exactly the same recovered FC owners, masks and labels. A4 may have
different technical availability; disclose the common-success comparison using
saved scores, without creating another main benchmark row. A5 existing owners use
exactly the original A1 F observations, not fewer, and do not obtain extra G1 views.
If F is unavailable, retain its geometric owner support and assign semantic class
0 (unknown). No hidden Native fallback. Describe A5 as "FC-only, matched evidence",
not an exhaustive optimal FC pipeline on all possible observations.

Original incumbent masks are immutable in all eight arms. Recovery appends only
(raw_owner==i) & (native_painted_owner==0) for an absent owner. Recognition uses the
full visible owner mask, not a differently clipped residual mask. New labels may
change official instance ranks; recompute them. Only A0/A1/A2/A3/A4/U2/G3 guarantee
positive valid incumbent labels. A5 needs a new export wrapper allowing class 0;
do not weaken the old `expanded_prediction` restrictions globally.

## 7. Released evaluation and actual recovery outcomes

Use the pinned released evaluator and the current-class official area ranks, NOT
frozen N0 ranking or fused semantic probability as confidence. The metric APall
uses nine thresholds .50 through .90. Record the actual loaded float values;
validate their mathematical values with tolerance 1e-12, not Python-list exact
equality against decimal literals. Never replace the loaded evaluator vector. AP25 and mAcc are calculated and retained in
SI even though not displayed in Tables 1–2. Serialize scores/metrics with their
units: fractions in machine files, percent once at presentation, deltas in pp.

Build an evaluator context for each ACTUAL owner partition. In particular, the
N0-source mask registry must include new recovered owners as genuinely unavailable
N rows; otherwise the evaluator silently cannot see the recovered masks. A5's
unknown class is excluded by the released positive-class instance export but
remains an error/unknown in the semantic confusion matrix. No GT subsetting to
only the exported or correctly classified objects.

Pool every method over the complete ordered cohort using the released evaluate
call; sum confusion matrices before computing mIoU/mAcc. Never average scene AP.
A missing CF18 sequence yields an incomplete pool and NA in the main table, with
per-scene work preserved. Equal score values are not evidence of equal predictions.
Exact aliases require owner, semantics, official rank, geometry, projection, GT
context and ordered pooling identity agreement.

For Table 3, trace added TP50/FP50 SCORE ENTRIES using the actual released matcher,
with terminal parity to its computed AP. Retain duplicate/tie/ambiguous identities
and old TP losses in the evidence file. Do not silently replace these by unique
GT counts. Aggregate the same definition across all eight Replica captures.
Compute n/N from the common prelocked candidate registry and actual successful
recovered outputs. Also store target-min100 eligible counts; source-min100 and
scoring-min100 are different spaces.

## 8. Measure Table 3 time; no warm-cache "zero cost" claim

Perform exactly one measured recovery-only replay for each of U2/G1/G3 on each
Replica scene: at most 24 timing replays, no new method variants or map builds.
This supplements the cached scientific runs; it does NOT increase the 172 result
count. Use isolated empty feature caches per scene/arm and forbid reading parent
or other-arm feature tensors during these timing runs. Reuse weights and text
prototypes already resident in memory, and reuse per-frame encoding within that
one arm only.

The main column is **feature-cache-cold incremental recovery time, model resident,
s/scene**. Include own candidate lookup, archived request recovery OR independent
full-mesh visibility/request construction, image decode/preprocessing, all distinct
necessary image encodings, region pooling/projection, classification, and output/
rank export. Exclude mapping, benchmark scoring, model loading and downloading;
record model load time separately. Synchronize CUDA at timed boundaries. Do not
subtract overlapping worker wall times or estimate cold time by summing unrelated
warm-cache hits. Build and run the standalone recovery callable under the timer.

Serialize these timing runs against this task's other mapper/evaluation jobs;
record concurrent external load if apparent, but do not kill unrelated processes.
Store cold-replay feature/prediction parity with scientific outputs. Do not use a
newly changed cold replay's score as a different silent experiment. No-op scenes
still incur measured eligibility-check overhead; only the "none" reference has
zero incremental recovery cost. Report the arithmetic mean over all eight scenes
(not only recovered scenes); also retain all stage times and hardware in SI.

No repeated bootstraps of GPU timings, power profiling, new budget curve or online
benchmark. One valid timed sample per scene/arm is enough for this scope. Missing
real timings are NA with a reason, never zero or an invented speedup.

## 9. External comparison rows: attribution, not a new model zoo

The three external rows remain Mask3D+OpenMask3D, Segment3D+OpenMask3D and OVO-SLAM.
Use the authored OVI Table 3 values in a SEPARATE "reported in OVI-MAP" block with
citation, source version/table location, their input/supervision setting and the
flag `AUTHOR_PROTOCOL_AS_REPORTED`. Read the source values; do not regenerate them
from our baselines, infer missing cells or interchange columns.

The shared dataset description does not prove byte-exact scorer/threshold/rank
compatibility of external predictions. Do not merge these values into the paired
reproduction block or claim their raw protocol has been independently verified.
No global-best bolding, computed speedup, significance or SOTA claim across that
boundary. This is a compact literature context, not three newly reproduced systems.
If exact local released predictions with verified protocol are already present,
they may be evaluated once and flagged REEVALUATED instead; keep a single main row
and retain the original literature value in provenance. Do not install/train all
three systems, retrieve new checkpoints or launch up to 78 external runs by default.
An unavailable/mismatched source cell stays NA; the internal experiment is not
replaced by a fabricated complete table. Publication distinguishes internal-
measurement completion from external protocol-comparability status.

## 10. Generate the actual three tables and analysis

Use one typed result store and the exact row mapping in TABLE_CONTRACTS.md and TABLE_BINDINGS.json.
Export JSON, CSV and three LaTeX files; no hand-entered numeric copies. Every cell
must point to the corresponding pooled receipt or attributed external source.
A0/A1/A3 in Table 1, Table 2 and Table 3 refer to the SAME predictions and values.
Produce the actual rendered table preview using the available LaTeX environment,
inspect overflow/readability, and fix formatting without deleting unfavorable rows.

Main tables: 6, 6 and 4 rows. Main table font at least 8.5 pt; prefer 9 pt. Use
booktabs and grouped dataset headings, no vertical rules, tiny text or forced
negative spacing. If the conference template imposes a larger minimum, follow it.
Placeholders are allowed only in the execution package; completed experiment cells
must contain results or explicit NA status. Do not write paper-ready claims for
incomplete G1 or CF18 results.

Analyze compactly:
- Table 1: A3–A0 total effect and A3–A1 recovery increment in each dataset.
- Table 2: the 2x2, A3 vs A4 and A3 vs A5, using full maps; preserve negative results.
- Table 3: whether independent requests enlarge usable coverage, produce TP rather
  than FP, and whether three views justify cost. Do not infer gain from n alone.
- Report raw pairwise deltas and Pareto tradeoffs, plus flags for APall>0 and
  AP50/mIoU drops no greater than .10 pp versus A1. These flags are descriptive
  engineering preferences, not significance tests or a reason to suppress rows.
A3 stays named full method whether or not it wins. State when a simple control
makes part of the proposed framework unnecessary.

## 11. Bounded execution, validation and fault handling

Default: two native CPU map jobs with eight threads each, three evaluation jobs
with four BLAS threads each, one shared GPU worker per designated device. Reuse the
existing lock; do not kill other users' processes or claim remote GPUs are available.
Allow actual new CropFormer/native/FC inference only for required missing anchors
and changed views. Count image inputs separately from batch calls and region ops.

Use one focused test invocation for new functions, one real development end-to-end
path and an unchanged-baseline parity check. Checks must cover camera z, occluders,
no-request owners, fixed G1/G3 prefix, unknown A5 labels, append-only support,
expanded evaluation and table reuse/units. Test production call sites, not only
a reference kernel. Do not run the whole CROVE/dynamic/repository test suite, hash
all historic files, inspect every old map, or build unrelated models. No test-count
quota; do not grow tests merely to report a larger number.

Bound identical-leaf retries to two automatic retries. Preserve failed commands
and costs. Fix the underlying error before rerunning its dependency descendants.
An incomplete TSDF capture is restarted only for that capture, not all finished
maps. Do not change precision, frame list, resolution or candidate criteria to
silently avoid OOM. Chunking/scheduling changes preserving exact inputs are allowed.
A semantic-inference all-failed condition is a technical block, not a valid zero
recovery result. Genuine no-visible-view outcomes are valid scientific outcomes.

Expose phases `bind`, `prepare`, `anchors`, `views`, `encode`, `predict`,
`evaluate`, `time`, `tables`, `publish`, and `all`. Make `all --resume` execute all
missing authorized work and preserve exact complete work. No automatic expansion
into W040+U2 or a new parameter sweep. Do not deploy the selected candidate.

## 12. Publication and final acceptance

Create in the repository:
- `docs/paper/static_ovmap/COMPACT_TABLES_RESULTS.md`;
- `COMPACT_TABLES_HANDOFF.md`, `COMPACT_TABLES_SELECTION.md`, `COMPACT_TABLES_CLAIMS.md`
  in the same directory;
- actual new module/script/tests, task configuration and any narrowly needed
  compatibility bridge (with inherited component attribution);
- `artifacts/static_ovmap/cvpr_compact_tables_v1/` with experiment matrix, input/
  source manifests, per-scene and pooled metrics, compact sources/decisions,
  coverage/matching evidence, cold timings, cost/failure ledger, freeze record,
  three LaTeX tables, table preview, cell provenance, and validation summary.

Target a compact artifact set below 50 MiB. Keep large scans, tensors, models,
TSDFs and full-size masks in shared storage with a manifest and exact reconstruction
commands. Include real compact scores/decisions/metrics on GitHub, not only absolute
paths. Never publish ScanNet access secrets, model credentials or copyrighted fonts.

Before publishing, run one goal-completion review linking each required output to
actual files. Implementation, scientific coverage, external comparison provenance,
timing coverage, performance outcome and publication are SEPARATE status fields.
Full internal scientific completion means 172 complete fixed outputs (identity-
proven aliases allowed), 14 complete internal pools and all 24 timing leaves.
Resource blocks cannot be relabeled COMPLETE. If a parent-scope incompatibility
prevents reuse, fix the new bridge, not the old frozen parent.

Commit actual changes; normal push to the named branch. Read local full HEAD and
`git ls-remote origin refs/heads/research/ovimap-cvpr-compact-tables-v1`; only when
both full 40-character SHAs agree write `publication/final.json` in shared storage
with `PUSH_VERIFIED`. Do not create an infinite verification-commit chain; the final
receipt lives outside its own commit. No force push or unrelated branch mutation.
An authentication failure produces the exact retry command and PUSH_FAILED status,
not a request for credentials in the report or a claim that upload succeeded.

Final response: link the four published MD reports, give the full verified SHA,
report 26-sequence/internal-pool/timing coverage, the three concise tables, A3
paired deltas and the strongest simple control, actual compute, and remaining
blocks. Do not claim new unseen-scene generalization solely because files have a
new date, nor that synthetic checks prove the method improves accuracy.
