# Execution checkpoint (not completion)

Workspace: /home/ww/crove/ovimap-m2-reviewer-evidence
Branch: research/ovimap-m2-reviewer-evidence-v1
Base: 8fee8294c1a3e83feeae28782f6ef4700f08d35c
Output: /mnt/shared/ww/ovimap-m2-reviewer-evidence-v1/attempt_001

Implemented study-local source binding, genuine score reconstruction, subset
fusion, shared scalar objective, separate ranking views and released pooled call,
and per-scene core execution. All legacy production modules remain unchanged.
The source package is preserved byte-for-byte and its supplied hashes passed.

Actual evidence:

- source_binding.json: 10 scenes, 70 initially bound consumed files.
- source_audits and legacy_parity: both CAL scenes, five controls' labels,
  owners and ranks exact. Native cosine reconstruction has no top1 disagreement.
- calibration/new_folds.json: real opposite-scene shared and cosine-N0 fits.
  Shared T: 0.010000492837221748 in both folds. Cosine N0 held-out0056:
  0.011285594474064128; held-out0534: 0.010000492837221748.
- scene0534_00: all 26 core ranking rows measured; five historical frozen-rank
  controls match all five metrics within1e-10.
- All ten scenes now have 26 core ranking rows each (260 total).
- CAL nomination frozen before Replica: RV_A2_NS2 and S_SIGLIP2_AREA.
  Final shared T=0.010000492837221748; cosine-native T=0.010357429721631367.
- Both CAL and Replica have 26 released-evaluator pools. Replica pooling
  completed successfully in session11742; original evaluate versus merged
  released matches gives exact AP, PR and FN parity. Receipts live under
  validation/ScanNet_pool.json and validation/Replica_pool.json.
- Real official export parity passed on scene0534_00 (78 instances).
- Query replay fixtures now cover deterministic random prefixes, irreversible
  split evidence, quota-zero COMBINE coverage, and receipt tamper rejection.
  The coverage fixture was corrected to satisfy the native valid-depth gate;
  production COMBINE behavior was not changed. Five query/selection tests pass.
- fresh_inventory.json: 14 authorized local scenes, all in explicit historical
  excluded families, zero eligible fresh families. Fresh success implementation
  is still required; the data limitation does not block other stages.
- 12 focused tests currently pass (fusion, calibration, rank/confusion/export).

Repair: initial evaluation adapter used absolute mask paths, which the released
parser warns against. Changed to relative paths and added evaluator-code cache
identity. One initial scene0056 N0 row and cache are preserved under repairs/
relative_manifest_v1. Recomputed descendants use the fixed adapter; old published
experiments and all prediction decisions are untouched.

Outstanding requirements remain substantial: B200 causal controls/curve gate, object/probability
and cost diagnostics, text robustness, real confirmation success path, complete
runner phases/resume, three reports, scoped publication and full-SHA audit.
Do not mark the study complete from this checkpoint or the unit-test count.

Query-control continuation:

- Added query_jobs.py: study access, frozen GAIN reuse, attested historical
  cache imports, causal new-policy jobs, immutable receipts and CUDA peaks.
- Added budget_controls.py: replacement-source-only CAL crossfits/final fits,
  dual-rank standalone/RAW/CAL rows, CAL-only curve gate and process-isolated
  full acquisition/evaluation orchestration. These fits and gate have NOT run yet.
- scene0056_00 Q_COMBINE_B200: 200 successes, all cache hits, zero new forwards.
- scene0056_00 RV_Q_RANDOM_s17_B200: 200 successes, 131 new forwards, 69 cache
  hits, 786 physical crops, peak allocated 2802071552 bytes, inference26.415s.
- Initial cache import failed because JSON lists were compared directly with
  request tuples; canonical digest comparison fixes the serialization mismatch.
  Initial GAIN reuse receipt is preserved in repairs/query_json_roundtrip/.
  No completed query result was deleted. Failed-attempt total physical cost
  was not separately recorded and must not be reported as measured zero.
- Initial in-process loop session64073 terminated after seed17 because CUDA
  allocator residue tripped the idle-GPU guard before seed23. Production runner
  now starts each visual leaf in a separate process, retaining the guard.
- Full query-controls orchestration is LIVE in exec session10450 on GPU1.
  It resumed the three completed scene0056 jobs; inspect this handle/process
  before any restart. No Replica query evaluation has been opened yet.
- Eight query/selection fixtures and changed-file Ruff checks pass.

Diagnostics continuation:

- All ten scenes have objects.json.gz, summary.json, used_operations.json.gz,
  mechanisms.json, released_attribution.json.gz and attribution_AP25_AP50.json.
  Objects retain full distributions, source requests, strict unique geometry
  matches, stable examples, ablation margins and missing-source strata.
- Exact released traces preserve added/lost GT matches plus separate duplicate,
  ignored and unmatched-FP events at every original overlap. Trace parity passed.
- Replica_macro_bootstrap.json contains 2000 scene draws with seed17 for each
  core method/rank/metric delta; these are macro intervals, never pooled CIs.
- Probability fixtures cover clip-only NLL, unnormalized multiclass Brier sum,
  15-bin ECE including confidence1, actual hard-vote tie labels, and singleton
  softmax. Two diagnostic tests pass; all changed files pass Ruff.
- office1 frozen-rank M2_CAL AP50=.19117647058823528 versus N0=.25:
  added blanket owner80; lost blanket owner79 and desk owner8. At both harmed
  owners Q and S2 agree on a wrong label (cloth and tv-screen respectively).
  Blanket class AP is unchanged; desk AP falls1→0. Details are actual trace data.
- Both CAL scenes now have all five B200 acquisition receipts (200 successes
  each, including reused GAIN). Replacement-source crossfits/final fits completed:
  COMBINE .011326161171616115; random17 .011759627198134013;
  random23 .010765086550345544; random41 .013764027970635112.
- Orchestrator session10450 / PID819399 is still live, evaluating CAL controls
  before gate freeze and Replica control acquisition. Do not duplicate it.
- Remaining: query stage completion/pools/cost report, robust text tests, full
  fresh success path and exposure audit, complete phase/resume implementation,
  scientific reports and artifact publication/full requirement audit.

Robustness continuation:

- CAL curve_gate.json is now FROZEN and NOT_TRIGGERED before new Replica
  query evaluation: GAIN uAP=.03665012765302552, mIoU=.23444966273475054;
  COMBINE .03241838889551922/.23865477941591023; random seed mean
  .03906695475461139/.24129766047893245. No B100/B400 should run.
- Recovered all ten scenes' exact frozen aggregates; native readout is unchanged,
  Q/S2 original full-score reconstruction error <6e-17 with exact top1 parity.
  Recovery session13627 exited0. No image inference or controller replay.
- CPU text encoding completed in original isolated environments for Replica:
  native session70994 and S2 session19239 both exited0, 118 new texts each.
  Original vectors and canonical references are reused exactly; extension
  preserves original columns and encodes only the fixed distractors.
- All eight Replica scenes completed 6 methods ×4 vocabularies, paired+union
  object evaluations, raw scores/probabilities and text/model identities.
  Robustness evaluation session95330 exited0. Results under robustness/results/.
- On the same157 paired identifiable objects, M2_CAL accuracy: original50.318%,
  photo55.414%, closeup46.497%, expanded50.318%; expanded distractor win0%.
  These are descriptive object-weighted figures, not official AP or a prompt
  selection. No method, temperature, prompt or comparator was changed.
- Added callable robustness phase; three robustness/diagnostic fixtures pass,
  including distractor winners counted as errors, and changed-file Ruff passes.
- Query controller session10450 remains live in Replica office0; inspect before
  resuming. Fresh success path/exposure audit, cost reporting, full runner/resume,
  final reports, publication and exact requirement audit remain outstanding.

Fresh continuation:

- Added fresh.py metadata-only audit across known repository/runtime roots:
  522 historical references/directories; 14 locally authorized raw scenes,
  all explicitly excluded physical families; zero eligible.31 oversized metadata
  files are recorded as uninspected; unresolved families cannot become eligible.
  Sensor headers confirm the original native200-slot schedule without RGB export.
- Real fresh phase and resume return FRESH_BLOCKED_INSUFFICIENT_LOCAL_UNEXPOSED_FAMILIES;
  authoritative receipt is fresh/status.json, linked to immutable fresh/audits/.
  The early fresh_inventory.json is retained as historical preliminary evidence.
- Added fresh_execution.py with a new own-plan authorization contract, local raw
  byte verification, real sensor export/native capture, S2 semantic worker, causal
  frozen GAIN worker, source formation, all-four-scene prediction lock, both
  rank evaluations and released pools. No old authorization guard was changed.
  Visual leaves use fresh processes; S2/Q allocator peaks and sampled locked
  native/frontend GPU-device peaks have explicit distinct measurement scopes.
- Fresh execution success path has NOT been run on real fresh data (none exists).
  Two fixtures verify literal-NUL family selection, duplicate-family exclusion,
  unknown/incomplete exclusion, four-scene contract and audit-tamper rejection.
  CLI import/help, real blocked execution/resume and changed-file Ruff pass.
- Query orchestrator session10450 remains live; latest observed sceneoffice1
  random17 complete, all completed jobs200successes. Inspect before any restart.
- Remaining major work: query completion/pools/cost accounting, complete core
  setup/all/report/publish orchestration and dependency-resume audit, tablesA–F,
  three final documents, artifact bundle, branch push/full remote SHA, full-spec
  completion audit. Fresh missing data must remain clearly not-tested.

Cost/core orchestration continuation:

- Added core_pipeline.py: callable original-example-preserving cosine/shared
  CAL folds, CAL-first prediction/evaluation, nomination, final fits, Replica
  core, official export and both datasets' released pools. The existing22 cosine
  example records match reconstructed resume inputs exactly (session40396 exit0).
  Original ad-hoc four-fold-fit timing remains missing; it is not invented as0.
- Strengthened read_scene checks against source_binding config/source hashes
  and the native feature hash attested by the N0 source. Real office0 parity
  passes; existing output/audit schemas and prediction semantics are unchanged.
- Added cost_ledger.py: shared historical frontend/mapping, historical Q/S2,
  per-source logical counts, exact required operation unions including N0 map,
  actual study physical jobs, evaluator jobs, CPU text jobs and scalar-fit costs.
  Snapshots have content-keyed directories under costs/. Partial job coverage is
  explicit; failed initial work and missing historical memory/timings remain
  declared missing, not free. These are not online end-to-end latency claims.
- Added budget_summary.py. Real CAL summary completed (session29991 exit0):
  60 scene rows,30 released pools,12 seed-spread summaries. No random ensemble.
- GPU1 controller session10450 / PID819399 continues office2 onward.
  Added independent GPU2 acquisition driver session30900 for room0,room1,room2.
  CLI per-scene/policy/budget locks now prevent duplicate leaves if controllers
  meet; identical completed jobs resume regardless of the requested GPU.
  The ongoing original room0 leaf started before this CLI lock was added; it
  should finish before GPU1 reaches room0. Inspect actual processes before reuse.
- Eight changed query/selection fixtures pass; Ruff and diff checks pass.
- Still outstanding: full Replica query completion and pools, final cost
  snapshot, report/publish/all phases and full dependency-resume audit, tablesA–F,
  final docs/bundle/remote SHA and requirement-by-requirement completion review.

Report continuation:

- Added reporting.py and claims.py. Evidence-derived tablesA–F, scene/pool/macro
  CSVs, calibration CSV, cost CSV, merged object ledger, paired comparisons,
  three MD document drafts and11 separate contribution decisions are generated.
  Report readiness rejects missing/smoke stages; fresh insufficient-local-data
  is an explicitly allowed prerequisite outcome, never a performance failure.
- Real report CLI completed with PARTIAL, not final. Latest draft snapshot:
  reports/eaa388e9f100c945eb001ad7a16433cc78bcbd49ef8eb99d85ec2bb5f16a38b1/.
  Earlier draft8f5f3e... is preserved. Reports remain external until final review.
- Official-current-class Replica pooled M2_CAL has uAP8.910%, AP5020.175%,
  mIoU26.792%; N0 has8.685%,21.477%,27.261%. This is a metric tradeoff.
  Fixed157-object probability pool: RAW→CAL NLL3.32554→1.74688,
  Brier.94642→.66166; constant.01 NLL1.74291/Brier.66077. No need for
  source-specific fitting is established by these controls. Claims remain
  provisional while Replica matched-query results are incomplete.
- Added finish_query_stage() to collect both dataset query pools/seed spreads
  and cost completion. The report phase invokes it once scene_stage.json exists.
  This is needed because currently live controller10450 imported an earlier
  runner and will finish scene rows without invoking the later pool-finalizer.
- GPU1 session10450 remains live on office3. GPU2 session30900 completed all
  room0 acquisitions and is on room1. New CLI leaves take per-job locks. Avoid
  restarting either controller or changing query_jobs.py/calibration inputs.
- One report-state fixture and two probability/bootstrap fixtures pass; changed
  paths Ruff and diff checks pass. Still required: remaining queries, final
  summaries, publish/all phases, dependency-resume and final artifact audit,
  final docs/bundle, normal push and full remote SHA match.

Measured-matrix audit continuation:

- Session10450 is now confirmed terminal (exit1): the office3 seed41 leaf
  rejected GPU1 at2037MiB/0% with GPU_BUSY. Subsequent nvidia-smi showed GPU1
  empty; resumed the existing content-bound controller as session71407,
  PID852526. Completed receipts are reused. No query source/code identities
  were changed and no busy-device guard was weakened. The historical owner of
  that temporary allocation was not captured, so it is not attributed here.
- GPU2 session30900 remains live, completed room1 and started room2. Latest
  observed B200 receipts:40/50 (all CAL, office0–2, room0–1; office3 four,
  room2 GAIN, office4 pending). Both controllers were verified by live ps.
- Added audit.validate_matrix and integrated it into report collection:
  exact scene/dataset/method/rank keys, duplicate/missing/unexpected rejection,
  finite bounded metrics, exact APall=uAP, and pool scene order plus every
  ordered evaluation identity. Query expected methods depend on the frozen
  curve gate, not merely on whatever rows happen to exist.
- Real checks passed:260 core scene rows/52 pools and60 CAL query rows/30 pools.
  The compact reporting fixture rejects duplicate rows, swapped pool inputs,
  altered APall and missing scenes. Both reporting tests pass; changed-path
  Ruff and git diff checks pass.
- Real partial report CLI succeeded with the new audit, snapshot
  reports/7cd6ee9cbbee546f1127acd85322fcc29e822e03273af47319eeb194ad927929/.
  This is still PARTIAL. Publication/all, remaining query results and final
  requirement-by-requirement review remain outstanding; no completion claim.

Publication/workflow implementation continuation:

- Added callable publish/all CLI phases. publication.py packages this study's
  report tables, byte-exact principal metrics/nomination, scalar fits, configs,
  locked predictions, source audits, compressed diagnostics/query receipts and
  external consumed-file identities. It requires a final report and exact
  matrix audit, checks the100MiB ceiling, scopes staged paths, performs a normal
  task-branch push and writes an external full-SHA equality receipt.
- workflow.py runs real phase subprocesses with dependency statuses. A failed
  query stage blocks report/publication but not diagnostics, robustness or
  freshness. Status is persisted after every stage. This workflow was not
  launched against the live experiment to avoid duplicating its controller.
- Four focused reporting/publication/dependency tests pass; changed-path Ruff
  and diff checks pass. Actual CLI --help lists all nine phases. Actual publish
  invocation correctly stopped at missing final_report.json before copying,
  committing or pushing. Full real bundle construction remains unverified until
  final query evidence exists; it is not claimed published.
- Sessions71407 and30900 remain live. GPU1 resumed the office3 seed41 leaf;
  GPU2 completed room2 COMBINE and random17 and continues the remaining seeds.
- Still required: completed query matrix and pools, final report/claim review,
  actual bundle size/dependency validation, full specification audit, final
  docs/commit/push and remote SHA verification.

Dependency/review continuation:

- Publication now includes consumed evaluation receipts (with exact projection,
  annotation and evaluator identities), robustness aggregate receipts, and
  recursively collected explicit absolute file identities. It does not crawl
  historical releases or hash large weights again. Real recorded-dependency
  merge succeeded:21,396 unique identities from342 distinct evaluation receipts
  plus current query/text/aggregate receipts, with no conflicting identities.
- Restricted the learned-query B200 claim lookup to B200 even if a future
  permitted curve exists; B100/B400 cannot overwrite the matched-budget row.
- Added REQUIREMENT_AUDIT.md mapping prompt sections0–17 to concrete evidence
  and explicit remaining work. This is an incomplete audit, not a completion
  certificate. Fresh success remains not tested on real fresh data.
- All29 tests under tests/m2_reviewer_study passed in0.66s (the prescribed
  compact fixture groups only). Changed-path lint and diff checks pass.
- CPU-only evaluation session94272 runs room0 then room1 without image inference.
  room0 completed30 query rows; room1 in progress. Main71407 remains live on
  office4 COMBINE; parallel30900 on room2 random41. Latest confirmed acquisitions
  at least45/50. Do not duplicate these jobs; main controller will reuse room
  evaluation caches when it reaches them.
- Final bundle construction/size and actual publication remain pending final
  query completion, summary and root review. No push has occurred in this step.
