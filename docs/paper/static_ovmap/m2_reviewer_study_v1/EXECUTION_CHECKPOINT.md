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
