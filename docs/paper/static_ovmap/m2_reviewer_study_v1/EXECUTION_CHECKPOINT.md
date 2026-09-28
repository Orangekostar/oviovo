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
