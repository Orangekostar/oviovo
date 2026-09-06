# OVI-ReScene Real-Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce real Apartment and asset-backed 3RScan evidence for supported-domain ReScene identity, controlled G/F/R comparisons, and downstream OVI ownership/completion.

**Architecture:** Compact only legally supported D/A/M rows into a hash-bound V3 inference view, reuse the pinned native ReScene forward, and keep full denominators in sidecars. Build GT and metric adapters outside the method path, then run one frozen matrix and let measured attribution select any limited adaptation.

**Tech Stack:** Python 3.10, NumPy, SciPy, PyTorch 2.6/CUDA 12.6, Hydra, Concerto/Sonata, ReScene4D, stmetrics 0.1.0, Pillow, trimesh/plyfile, pytest.

**Spec:** `docs/superpowers/specs/2026-09-06-ovi-rescene-real-evidence-design.md`

## Global Constraints

- Start from commit `31130314f62a227a8f8d1b37f0784532e8c235fb` on branch `research/ovi-rescene-real-evidence`.
- Never modify prior external run artifacts or frozen OVI maps.
- Use only same-visit RGB, source normals, and supported rows as model input.
- Keep evaluator GT, cross-time identity, and object transforms out of every method input.
- Store large outputs outside Git and commit only small hash-bound summaries.
- Do not report estimates, renamed custom metrics, or missing values as zeros.
- Use test-first RED/GREEN cycles for every production change.

---

### Task 1: Supported D/A/M View

**Files:**
- Create: `src/oviv2/rescene_supported_view.py`
- Create: `tests/oviv2/test_rescene_supported_view.py`

**Interfaces:**
- Consumes: `StaticPreparedInput`, `RecoveredModelSupport`.
- Produces: `SupportedInferenceView`, `EntityCoverage`, and `build_supported_inference_view(prepared, support)`.

- [ ] **Step 1: Write failing tests** for stable `S/h/A_keep/g_compact`, two-visit offsets, nine-channel features, compact CSR mappings, full denominators, unsupported entities, no `-1` dereference, and source immutability.
- [ ] **Step 2: Verify RED** with `pytest -q tests/oviv2/test_rescene_supported_view.py`; failures must be missing module/symbol behavior.
- [ ] **Step 3: Implement immutable contracts and vectorized compaction** using `flatnonzero`, prefix sums, and stable old-domain order; reject empty visits and inconsistent representatives.
- [ ] **Step 4: Verify GREEN and regressions** with `pytest -q tests/oviv2/test_rescene_supported_view.py tests/oviv2/test_rescene_input_bridge.py`.
- [ ] **Step 5: Commit** with `git add src/oviv2/rescene_supported_view.py tests/oviv2/test_rescene_supported_view.py && git commit -m "feat: add supported ReScene inference view"`.

### Task 2: V3 Artifact and Audit

**Files:**
- Create: `scripts/evaluation/prepare_ovi_rescene_supported_v3.py`
- Create: `tests/evaluation/test_prepare_ovi_rescene_supported_v3.py`
- Create: `configs/evaluation/ovi_rescene_supported_v3.json`

**Interfaces:**
- Consumes: Task 1 view plus prior V2 static, calibration, and recovered-support artifacts.
- Produces: `input_contract_v3.json`, `model_input/`, `adapter_pair/`, `mappings.npz`, and `entity_coverage.json`.

- [ ] **Step 1: Write failing tests** for atomic publish/no-clobber, strict schemas, all file hashes, parent V2 bindings, mapping round trips, count conservation, tamper rejection, and real-config path validation.
- [ ] **Step 2: Verify RED** with `pytest -q tests/evaluation/test_prepare_ovi_rescene_supported_v3.py`.
- [ ] **Step 3: Implement loader, publisher, and auditor** without changing `prepare_ovi_rescene_input_v2.py` or its receipt.
- [ ] **Step 4: Verify GREEN** with `pytest -q tests/evaluation/test_prepare_ovi_rescene_supported_v3.py tests/evaluation/test_prepare_ovi_rescene_input_v2.py`.
- [ ] **Step 5: Build the real Apartment artifact** with `/home/ww/miniconda3/envs/persist4d/bin/python scripts/evaluation/prepare_ovi_rescene_supported_v3.py build --config configs/evaluation/ovi_rescene_supported_v3.json`, then run the same command with `audit` and record exact D/A/M counts.
- [ ] **Step 6: Commit** code, tests, config, and the small audit summary.

### Task 3: Native Apartment Forward and Resolver

**Files:**
- Modify: `scripts/evaluation/rescene_pair_executor.py`
- Modify: `tests/evaluation/test_rescene_pair_executor.py`
- Create: `src/oviv2/query_entity_resolver.py`
- Create: `tests/oviv2/test_query_entity_resolver.py`
- Create: `scripts/evaluation/evaluate_ovi_rescene_apartment_v3.py`
- Create: `tests/evaluation/test_evaluate_ovi_rescene_apartment_v3.py`

**Interfaces:**
- Consumes: audited V3 artifact, pinned ReScene/Concerto checkpoints, compact pair, and raw M-space masks/logits.
- Produces: raw forward receipt, legacy and supported relations, diagnostics, B4/B5 delta, and representative visuals.

- [ ] **Step 1: Write executor RED tests** proving V3 acceptance, complete hash/mapping validation, V2 compatibility, and tamper rejection.
- [ ] **Step 2: Verify executor RED**, implement a V3-specific loader selected by artifact ID, then verify `pytest -q tests/evaluation/test_rescene_pair_executor.py`.
- [ ] **Step 3: Write resolver RED tests** for entity aggregation, conditional/full coverage, mutual dominance, deterministic one-to-one conflict resolution, duplicates, collisions, and unsupported denominators.
- [ ] **Step 4: Verify resolver RED**, implement chunked aggregation without materializing QxD, then verify `pytest -q tests/oviv2/test_query_entity_resolver.py`.
- [ ] **Step 5: Write and implement the report driver** under RED/GREEN tests; it must serialize B4 once, both B5 readouts, raw/unique counts, coverage, and visual selection provenance.
- [ ] **Step 6: Run one real GPU forward** with `RESCENE_DEVICE=cuda:0`, the pinned Python, frozen checkpoint, source checkout, and V3 artifact; save raw tensors outside Git and exact runtime/memory in the receipt.
- [ ] **Step 7: Verify focused integration** with the five baseline test files plus all Task 1-3 tests, inspect generated PNGs, and commit small summaries/visuals.

### Task 4: RSCAN_T2_DEV_V1 and GT Binder

**Files:**
- Create: `src/evaluation/rscan_t2_dev.py`
- Create: `tests/evaluation/test_rscan_t2_dev.py`
- Create: `scripts/evaluation/build_3rscan_t2_dev.py`
- Create: `src/evaluation/rscan_gt_instances.py`
- Create: `tests/evaluation/test_rscan_gt_instances.py`
- Create: `configs/evaluation/manifests/3rscan_t2_dev_v1.json`

**Interfaces:**
- Consumes: official metadata/validation list, available complete assets, Persist4D sequence database, annotated PLY/semantic JSON, and declared global transforms.
- Produces: deterministic two-session pairs, split-isolation receipt, GT instance grids, and primary/sensitivity assignments.

- [ ] **Step 1: Write selection RED tests** for validation-only, complete assets, exactly T=2, at least three environments when available, UUID ordering, and train-environment exclusion.
- [ ] **Step 2: Verify selection RED**, implement selection/build CLI, run tests, then freeze the real manifest and re-audit all bindings.
- [ ] **Step 3: Write GT RED tests** for static/rigid matrix direction, 5 cm voxelization, Hungarian maximum-IoU assignment, IoU 0.50/0.25 thresholds, unmatched/duplicate/fragment/merge, symmetry, and ambiguity.
- [ ] **Step 4: Verify GT RED**, implement the binder/evaluator-only adapter, then run both new test files and existing `test_rscan_dataset.py`/`test_rscan_temporal.py`.
- [ ] **Step 5: Build real GT sidecars** outside Git, commit only the manifest and aggregate counts, then commit Task 4.

### Task 5: Controlled G/F/R and D0/D1/D2 Matrix

**Files:**
- Create: `src/evaluation/rscan_method_views.py`
- Create: `tests/evaluation/test_rscan_method_views.py`
- Create: `scripts/evaluation/run_3rscan_t2_matrix.py`
- Create: `tests/evaluation/test_run_3rscan_t2_matrix.py`
- Create: `configs/evaluation/ovi_rescene_3rscan_t2_matrix_v1.json`

**Interfaces:**
- Consumes: Task 4 pairs/GT, frozen checkpoint, common alignment, support views, and official stmetrics dataset spec.
- Produces: G_full, G_supported, F, R_legacy, R_supported across available D0/D1/D2 rows with separate custom and official metrics.

- [ ] **Step 1: Write RED tests** for identical method inputs, F visit independence, R joint input, domain nesting, GT isolation, raw-forward reuse, and missing-D2 coverage semantics.
- [ ] **Step 2: Verify RED**, implement method adapters and fail-closed config validation, then verify both new test files.
- [ ] **Step 3: Add stmetrics parity tests** using native prediction/target structures and assert t-AP/t-REC/per-stage AP keys remain separate from custom association metrics.
- [ ] **Step 4: Run G/F/R jobs** on available GPUs with one pair per process, never sharing an output root; rerun failed jobs only after preserving their logs.
- [ ] **Step 5: Aggregate macro and per-pair values**, coverage, runtime, memory, and error taxonomy into small JSON/CSV results and commit Task 5.

### Task 6: Ownership and Completion Experiment

**Files:**
- Create: `src/evaluation/ovi_ownership_completion.py`
- Create: `tests/evaluation/test_ovi_ownership_completion.py`
- Create: `scripts/evaluation/evaluate_ovi_ownership_completion.py`
- Create: `tests/evaluation/test_evaluate_ovi_ownership_completion.py`
- Create: `configs/evaluation/ovi_rescene_ownership_completion_v1.json`

**Interfaces:**
- Consumes: fixed OVI geometry/visibility, Task 5 G/R identities, estimated registration, and evaluator-only GT/oracle transforms.
- Produces: A0/A1/A2 AP/PQ/error metrics plus C0/C1/C2/O1/O2 opportunity, recoverable-surface, precision, and F-score.

- [ ] **Step 1: Write RED tests** proving identity does not relocate geometry, all methods share XYZ/visibility, O1/O2 are oracle-only, and completion opportunity uses actual OVI shape.
- [ ] **Step 2: Verify RED**, implement pure metrics and the provenance-bound driver, then verify both new test files.
- [ ] **Step 3: Run at least one real pair**, record all defined rows and any undefined denominators explicitly, then commit code/config/small results.

### Task 7: Evidence-Gated Adaptation

**Files:**
- Create: `scripts/evaluation/decide_ovi_rescene_adaptation.py`
- Create: `tests/evaluation/test_decide_ovi_rescene_adaptation.py`
- Create when selected by the decision: `configs/evaluation/ovi_rescene_local_adaptation_v1.json`

**Interfaces:**
- Consumes: raw/conditional association, D0/D1/D2, registration-oracle, and completion-opportunity summaries.
- Produces exactly one action: `REPORT_MODEL_LIMIT`, `ADAPT_DECODER_MASK_HEAD`, `TUNE_RESOLVER`, `FIX_REGISTRATION`, `CHANGE_PAIR_BUDGET`, or `RETAIN_IDENTITY_ONLY`.

- [ ] **Step 1: Write RED tests** covering the six ordered decision rules in the design and rejection of missing/non-real metrics.
- [ ] **Step 2: Verify RED**, implement the deterministic decision CLI, and verify GREEN.
- [ ] **Step 3: Execute the selected action**: decoder adaptation freezes the encoder and uses disjoint train/validation environments; resolver tuning uses only development GT; registration repair leaves identity fixed; pair-budget changes rerun every compared method.
- [ ] **Step 4: Rerun the affected matrix rows** and commit the decision, exact config, and measured before/after summaries.

### Task 8: Integration, Handoff, and Push

**Files:**
- Create: `configs/evaluation/results/ovi_rescene_real_evidence/experiments.csv`
- Create: `docs/paper/ovi_rescene_real_evidence_handoff.md`
- Modify: `docs/paper/benchmark_tables_baselines.md` only for rows backed by final real result artifacts.

**Interfaces:**
- Consumes: all completed run receipts and result summaries.
- Produces: claim-evidence table, reproducibility ledger, paper-facing values, limitations, and pushed branch.

- [ ] **Step 1: Validate every CSV row and table value** against a hash-bound real artifact; reject estimates and protocol-incompatible baselines.
- [ ] **Step 2: Run focused unit/integration suites**, `python -m py_compile` on changed Python files, and `git diff --check`; inspect all committed visuals.
- [ ] **Step 3: Review the complete diff personally** for GT leakage, frozen-source changes, output collisions, missing coverage, and metric naming.
- [ ] **Step 4: Commit final docs/results**, push `research/ovi-rescene-real-evidence`, fetch the remote ref, and verify its SHA equals local HEAD.

