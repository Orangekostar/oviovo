# OVI-MAP x ReScene Persist4D B5 Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a source-bound, fail-closed answer to whether the recovered Persist4D ReScene checkpoint adds Apartment cross-visit OVI relation evidence beyond frozen B4.

**Architecture:** Add two small CPU audit CLIs before any learned executor: one binds and audits the checkpoint/topology, and one compares the native T=2 preprocessing witness with the frozen OVI pair contract. Only if both input features and token conservation pass may the existing ReScene backend call a minimal executor and the existing projection feed a B4/B5 comparator; otherwise publish a complete blocked receipt and stop before GPU.

**Tech Stack:** Python 3.13 for repository audits and tests, the pinned Persist4D Python 3.10 environment for source-bound ReScene construction, PyTorch, NumPy, pytest, JSON/NPZ artifacts, Git.

**Spec:** `docs/superpowers/specs/2026-09-05-ovi-rescene-b5-identity-design.md`

## Global Constraints

- Base is exactly `2136865993033e35c44dac12444363a4788e8452` on `research/ovi-rescene-persist4d-b5-identity`.
- Never rerun OVI, B0--B7 maps, Office, learned B7, dense recovery, or 3RScan ranking.
- Do not train, fine-tune, sweep thresholds, change ProjectionConfig, or use evaluator targets as method input.
- Checkpoint, pair arrays, native samples, masks, and large logs remain outside Git.
- No GPU command is legal until checkpoint, topology, feature, temporal/batch, and token-conservation gates all pass.
- Keep Office `HELD_OUT` with `attempt_count=0`; keep learned B7 `NOT_RUN_BY_SCOPE`.

---

### Task 1: Machine-Readable Checkpoint And Topology Audit

**Files:**
- Create: `configs/external/persist4d_rescene_checkpoint_v1.json`
- Create: `scripts/evaluation/audit_persist4d_rescene_checkpoint.py`
- Create: `tests/evaluation/test_audit_persist4d_rescene_checkpoint.py`
- Create: `configs/evaluation/results/ovi_rescene_b5_identity/checkpoint_audit_summary.json`

**Interfaces:**
- Consumes: checkpoint path/hash, Persist4D evidence checkout/commit, upstream checkout/commit, Concerto path/revision/hash.
- Produces: `audit_checkpoint(config_path: Path, output_path: Path) -> dict[str, object]` and a canonical JSON receipt containing C0 and C1 status.

- [x] **Step 1: Write failing source/checkpoint tests.** Cover a non-symlink regular checkpoint, wrong checkpoint hash, dirty or wrong source commit, malformed state dict, a missing model tensor, an unexpected model tensor, a shape mismatch, and exact tensor consumption.
- [x] **Step 2: Verify RED.** Run `pytest -q tests/evaluation/test_audit_persist4d_rescene_checkpoint.py`; require failures because the audit module does not exist.
- [x] **Step 3: Implement minimal audit.** Use stable file reads, `torch.load(..., map_location="cpu", weights_only=False)`, compact dtype/shape/prefix inventories, exact Git object checks, and strict model-key comparison. Never print tensor values.
- [x] **Step 4: Verify GREEN.** Rerun the same module, then run `python -m compileall -q scripts/evaluation/audit_persist4d_rescene_checkpoint.py` and `git diff --check`.
- [x] **Step 5: Run the real C0/C1 audit.** Use `/home/ww/miniconda3/envs/persist4d/bin/python`, the exact checkpoint and source paths, and write only compact JSON under the configured external root plus the checked-in summary.

### Task 2: Native And OVI Input Contract Audit

**Files:**
- Create: `scripts/evaluation/audit_ovi_rescene_b5_input_contract.py`
- Create: `tests/evaluation/test_audit_ovi_rescene_b5_input_contract.py`
- Create: `configs/evaluation/results/ovi_rescene_b5_identity/input_contract_summary.json`
- Create: `configs/evaluation/results/ovi_rescene_b5_identity/apartment_pair_receipt.json`

**Interfaces:**
- Consumes: one native preprocessed witness JSON, frozen B0/B2 snapshots and entity records, native OVI manifests/PLY headers/color logs, and the frozen 0.02 m contract.
- Produces: `audit_input_contract(...) -> dict[str, object]` with feature, color, normal, centering, batch/time, token counts, collision voxels, merge/drop/duplication counts, and a single gate status.

- [x] **Step 1: Write failing tests.** Synthetic fixtures must prove 9-channel native detection, palette RGB rejection, missing camera RGB rejection, exact two-visit temporal values, collision-free permutation PASS, cross-entity merge BLOCK, and no mutation of input files.
- [x] **Step 2: Verify RED.** Run `pytest -q tests/evaluation/test_audit_ovi_rescene_b5_input_contract.py`; require failures because the input-audit module does not exist.
- [x] **Step 3: Implement the audit.** Parse only bounded headers/metadata and memory-map or stream point arrays; use shared-pair centering and exact 2 cm floor quantization; compute per-entity and global token inventories without generating neural outputs.
- [x] **Step 4: Verify GREEN.** Rerun the audit tests plus `pytest -q tests/oviv2/test_ovi_rescene_adapter.py` and `git diff --check`.
- [x] **Step 5: Materialize the real witnesses once.** Run one existing native T=2 dataset/collator sample on CPU, audit the frozen Apartment pair, and bind the external witness files by absolute path, bytes, SHA-256, producer command, Git commit, and input hashes.
- [x] **Step 6: Apply the gate.** If source camera RGB is absent or any unversioned token merge/drop exists, write `BLOCKED_INPUT_FEATURE_CONTRACT` or `BLOCKED_RESCENE_TOKEN_CONSERVATION`, skip Tasks 3 and 4, and continue directly to Task 5. No GPU command may run.

### Task 3: Conditional Source-Bound Pair Executor

Status: skipped by the frozen Task 2 C2 gate; no GPU command was authorized.

**Files:**
- Create only after Task 2 PASS: `configs/evaluation/rescene_two_visit_backend_persist4d_repro.json`
- Create only after Task 2 PASS: `scripts/evaluation/rescene_pair_executor.py`
- Create only after Task 2 PASS: `tests/evaluation/test_rescene_pair_executor.py`

**Interfaces:**
- Consumes: existing backend subprocess arguments and one hash-bound `NeuralSampleMap` artifact.
- Produces: the existing exact output manifest keys and `token_indices`, `query_masks`, `token_scores`, `query_scores` NPZ arrays.

- [ ] **Step 1: Write failing executor tests.** Cover checkpoint mismatch, malformed pair, invalid/nonfinite temporal or feature tensors, one missing visit, permutation restoration, merge/drop rejection, malformed output, and mask/query dimension mismatch using a CPU stub model.
- [ ] **Step 2: Verify RED.** Run `pytest -q tests/evaluation/test_rescene_pair_executor.py`; require failures because the executor does not exist.
- [ ] **Step 3: Implement the minimal executor.** Construct the pinned upstream model, strict-load every checkpoint model tensor, apply the frozen preprocessing and query semantics, and write the existing schema atomically.
- [ ] **Step 4: Verify GREEN.** Run the executor tests and `pytest -q tests/oviv2/test_rescene_backend_contract.py tests/oviv2/test_rescene_backend_integration.py`.
- [ ] **Step 5: Freeze config and source in a design commit.** Record exact config/source/checkpoint hashes before any result-bearing inference.
- [ ] **Step 6: Run C3 once.** Run the frozen Apartment pair on one GPU through `run_rescene_pair_backend.py`; require PASS, exact checkpoint hash, finite scores, both visits covered, and restored mask width equal to adapter token count. Reuse this output as the final learned evidence.

### Task 4: Conditional B4/B5 Relation Comparator

Status: skipped by the frozen Task 2 C2 gate; B4 was not reconstructed.

**Files:**
- Create only after C3 PASS: `scripts/evaluation/compare_ovi_rescene_b5_b4_identity.py`
- Create only after C3 PASS: `tests/evaluation/test_compare_ovi_rescene_b5_b4_identity.py`
- Create only after C3 PASS: compact B4, B5, and diff JSON files under `configs/evaluation/results/ovi_rescene_b5_identity/`.

**Interfaces:**
- Consumes: one frozen B4 relation reconstruction, one C3 `TemporalQueryEvidence`, existing `project_queries_to_instances()`, and frozen `ProjectionConfig(0.25, 0.25, 0.10)`.
- Produces: normalized relation summaries and B5-only one-to-one evidence keyed by `(state, sorted t0 IDs, sorted t1 IDs)`.

- [ ] **Step 1: Write failing comparator tests.** Cover topology normalization, state totals, exact intersection/B4-only/B5-only sets, one-to-one subsets, deterministic first-20 evidence, and rejection of mismatched pair/config/source hashes.
- [ ] **Step 2: Verify RED.** Run `pytest -q tests/evaluation/test_compare_ovi_rescene_b5_b4_identity.py`; require failures because the comparator does not exist.
- [ ] **Step 3: Implement the comparator.** Reuse the existing projection and geometric reasoner; do not add a matcher, change B4 tokenization, or compose a current map.
- [ ] **Step 4: Verify GREEN.** Run comparator, query projection, and geometric reasoner tests.
- [ ] **Step 5: Rebuild B4 once and compare.** Serialize B4 once, project cached C3 evidence once, optionally run frozen registration only on B5-only one-to-one persistent relations, and emit the legal scientific verdict without tuning.

### Task 5: Blocked Or Completed Result Package, Review, And Push

**Files:**
- Create: `configs/evaluation/results/ovi_rescene_b5_identity/decision.json`
- Create: `docs/superpowers/reports/2026-09-05-ovi-rescene-b5-identity-results.md`
- Create: `docs/superpowers/reports/2026-09-05-ovi-rescene-b5-identity-handoff.md`
- Update: `docs/superpowers/reports/2026-09-05-ovi-rescene-persist4d-checkpoint-audit.md` only with verified post-audit facts.

**Interfaces:**
- Consumes: all completed audit receipts and, only when unlocked, cached C3/B4/B5/diff receipts.
- Produces: one of the three legal verdicts, all 37 handoff answers, external artifact manifest, verified branch push.

- [x] **Step 1: Audit cheap held-out/data state.** Verify Office attempt count is zero and recount the 44 selected 3RScan visits without downloading assets or running ranking.
- [x] **Step 2: Write compact result JSON.** Every unavailable value must be `null` with an explicit blocked/not-run reason; never substitute zero or a predicted number.
- [x] **Step 3: Write results and handoff.** State local-reproduction provenance, paper gap, gate statuses, permitted claims, forbidden claims, external artifact paths/hashes, and the single highest-value next action.
- [x] **Step 4: Run focused verification.** Run all changed test modules, relevant adapter/backend/projection tests, compile changed Python paths, and `git diff --check`.
- [x] **Step 5: Perform primary-agent review.** Inspect the full diff, source boundaries, schemas, failure ordering, result-to-claim consistency, Office/3RScan leakage, and Git scope.
- [x] **Step 6: Run the full suite once.** On failure, diagnose with targeted tests and permit at most one second full-suite attempt.
- [ ] **Step 7: Commit and push.** Commit logical groups, push `research/ovi-rescene-persist4d-b5-identity`, and require `git rev-parse HEAD` to equal `git ls-remote origin refs/heads/research/ovi-rescene-persist4d-b5-identity` before reporting `UPLOAD_STATUS=VERIFIED`.
