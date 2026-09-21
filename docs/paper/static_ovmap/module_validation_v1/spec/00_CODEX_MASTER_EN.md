# OVI-MAP module validation — final Codex execution contract

Version: 2026-09-20-v1. This is an execution instruction, not another proposal.

## 1. Objective and authority

Work in `Orangekostar/oviovo`, starting from `b6455520c758a3413988e0827b3e1f34667bdfd1`. Implement and experimentally evaluate three isolated hypotheses:

1. **S — selective semantic correction:** preserve OVI geometry, obtain alternative region evidence, and learn whether replacing the incumbent label is beneficial.
2. **G — complete-partition selection:** preserve TSDF geometry, construct a bounded set of complete local instance partitions from native segment evidence, and compare learned versus old agreement-based selection.
3. **Q — pre-query utility:** decide whether to pay for a new native visual query from information available before that query, under a common causal budget.

The deliverable is working scoped code, actual experiments where prerequisites are present, a deterministic model/module selection decision, and a verified GitHub push. Do not stop after writing another plan. Do not claim that a method is useful because code, tests, or experiment rows completed.

The delivered `CODEX_FINAL_EXECUTION_EN.md` is the exact concatenation of this file and files `01`–`05`. Read that single file, or read this file and files `01`–`05` in order; they contain the same instruction. They jointly define one contract. `PROTOCOL_SPEC.json` encodes its constants; its `specification_only` marker means you must resolve local paths and versions into a runnable configuration, not feed unresolved metadata into the old runners. `CODE_REVIEW_AND_AUDIT_ZH.md` and `SOURCES.md` explain provenance; they do not add executable experiments. The earlier `THREE_INNOVATIONS_REVIEW_ZH.md` is background only. If an old task says zero training, zero new image inference, or no mapping replay, those restrictions are superseded **only for the scoped operations explicitly authorized here**. Preserve old source, configurations and result directories.

## 2. Authorization and limits

Authorized: frozen native SigLIP; frozen SigLIP2-L/16-384; one official WOW-Seg region recognizer and its fixed name mapper; bounded native OVI replay with instrumentation; CropFormer inference only to create missing front-end caches for the preselected legally available development/confirmation scenes; fitting the small predictors specified here; main experiments, listed ablations, selection and one frozen confirmation; commit and ordinary push to the task branch.

Not authorized: retraining segmentation/VLM backbones; launching new SF, SAM, Open3DIS, LEGO, OVRCOAT, GeoGuide, APPLE or NBV-Gym pipelines; changing the benchmark trajectory, evaluator, class subset or projection threshold; a broad parameter/prompt/model search; using test GT in prediction, adoption, hypothesis proposal or acquisition; silently filling missing assets with ground truth; repeating the old 34/8/6-condition studies; whole-repository dynamic-CROVE regression or repeated safety audits. The referenced papers motivate mechanisms, not extra workloads.

Do not download restricted datasets or accept new restricted-data terms on the user's behalf. Use authorized local datasets and existing permitted credentials. Official public model downloads are authorized. A missing dependency is a specific status, not permission to invent a substitute.

## 3. Workspace and startup

Create/resume `research/ovimap-module-validation-v1` from the reviewed commit. Prefer a new worktree beside the existing project. Do not reset, stash, delete, or force-push unrelated user work. If the task branch already exists, verify its ancestry and resume only its own receipts. If it is unrelated, create `research/ovimap-module-validation-v1-<first8-of-start-HEAD>` and record why.

Default output root: `/mnt/shared/ww/ovimap-module-validation-v1`; use a new `attempt_001`, `attempt_002`, ... child when an existing root has incompatible configuration. An exact compatible interrupted run resumes atomically at its first incomplete artifact. Never overwrite completed old studies.

Copy this task bundle into `docs/paper/static_ovmap/module_validation_v1/spec/`. Implement a single public orchestration entry point:

```bash
python scripts/evaluation/run_ovimap_module_study.py \
  --spec docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json \
  --phase all \
  --output-root /mnt/shared/ww/ovimap-module-validation-v1
```

Supported phases, with explicit receipts: `bind`, `capture`, `semantic`, `geometry`, `query`, `select`, `confirm`, `report`, `all`. `all` executes missing prerequisites in this order and finishes reporting/release preparation even when a scientifically dependent phase cannot be run. It must not substitute zero arrays or toy measurements for a blocked phase. Each leaf job also accepts `--resolved-config` and can be resumed independently.

## 4. Required implementation surface

Create a small package `src/static_ovmap/module_validation/` rather than changing historical algorithm behavior:

- `contracts.py`, `assets.py`: typed records, provenance, path resolution and scene splits.
- `native_capture.py`: immutable native image/segment/state capture and replay access.
- `region_evidence.py`, `semantic_selector.py`: S.
- `entity_hypotheses.py`, `partition_quality.py`: G.
- `query_state.py`, `query_gain_policy.py`: Q.
- `evaluation.py`, `selection.py`, `reporting.py`: common metrics, model choice and reports.

Keep adapters for incompatible model environments separate. Use compact NPZ/JSONL/PNG interfaces. Do not build a service, dashboard, generic workflow engine or new experiment database.

Store official OVI patches in `third_party_patches/ovimap/module_validation_v1/`, with a script that applies them to a **separate clean checkout** at `f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424`. Never modify the original baseline installation in place. The checked-out C++ extension, loaded `.so`, patch digest and compiler/runtime environment belong in receipts. Patching a file without loading its rebuilt extension is not integration.

## 5. Data and experimental precedence

There are three distinct evidence levels; keep them distinct everywhere:

- **HISTORICAL:** the Room0 `AT_O_AREA` common-point-domain study. Its approximately 20.644% uAP is not the universal native OVI baseline.
- **DEVELOPMENT:** physically disjoint FIT/CAL/SELECT scenes chosen before model outcomes; FIT trains, CAL chooses teacher/head checkpoint/adoption settings, SELECT decides which modules are retained.
- **CONFIRMATION:** untouched scenes for this study, run once after `selection.json` is frozen. Known Replica8 and old benchmark scene results are exposed, not blind validation.

All three modules are coded. Their empirical prerequisites are different. A blocked training-data dependency cannot be replaced with a random crop split of Room0. Continue independent runnable work and publish exact incompleteness; never report overall empirical completion when an indispensable branch was not measured. This requirement is not a request for endless preflight checking.

## 6. Execute in stages, not a Cartesian product

1. Resolve assets and splits once; capture a faithful native baseline and input/state records.
2. Run the five direct region readouts in file 02, plus the unchanged native reference. Select the **alternative** teacher on CAL, not Room0 or confirmation; the native model remains the incumbent.
3. Fit and evaluate the paired semantic selectors if the independent FIT labels contain identifiable beneficial and harmful replacement events. An inferior teacher overall can still supply useful selective corrections; do not require a direct net win as a training gate.
4. Construct native G partitions, run the old and learned scorers on exactly the same hypotheses, and evaluate final non-overlapping partitions.
5. Collect bounded randomized **training-only** acquisition logs, fit Q, then evaluate causal equal-budget policies; no offline final-map state in the causal result.
6. Select modules using file 05. Test only the eligible combination path, not all eight combinations of three modules.
7. Freeze selection, confirm once, produce final reports and push.

## 7. Tests: necessary, small, and linked to failure modes

No test-count target. Write a compact suite for: identity/geometry invariants; crop-feature compatibility; real region-mask consumption; partition completeness and no parent/child duplication; split isolation; pre-query information exclusion; budget debit and pending-result handling; evaluator trace parity on one payload; resume invalidation. Prefer a few parametrized tests. Run one real-input smoke per genuinely distinct integration boundary (visual region adapter, C++ snapshot, causal acquisition), not repeated full-scene smoke studies.

When a check fails, fix the observed fault and rerun the affected check/downstream artifacts. Do not rerun all historical research. Do not turn negative scientific results into implementation bugs without evidence.

## 8. Completion and communication

Keep a compact `progress.md`: implemented/running/completed/blocked and the concrete next action. Finish with actual method rows, decisions, measured costs and exact file paths. Separate:

- implementation: COMPLETE / PARTIAL;
- experiments per branch: MEASURED / BLOCKED_<reason> / NOT_REQUIRED_BY_FROZEN_GATE;
- scientific result: NET_GAIN / TRADEOFF / NO_NET_GAIN / INCONCLUSIVE;
- confirmation: CONFIRMED_ON_STUDY_HOLDOUT / NOT_CONFIRMED / NOT_RUN_<reason>;
- publication: PUSH_VERIFIED / PUSH_FAILED_<actual_error>.

The final release rules in file 05 are mandatory even for a negative result. No background-work promises. No claim of end-to-end real-time speed from cached processing. No claim that these engineering adaptations reproduce entire 2026 methods or establish novelty merely by combining them.

All implementation constants below are frozen v1 engineering choices for a bounded test, not literature-derived optimal values. Fix a verified implementation error when found, but do not change scientific constants after SELECT/CONFIRM outcomes to obtain a win. Document this scope as a minimum module-validation study; it does not establish the entire proposed online research system.
