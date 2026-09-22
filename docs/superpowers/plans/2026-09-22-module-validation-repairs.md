# Module-validation engineering repairs

**Goal:** Repair unsafe reuse, incomplete executable boundaries, and misleading delivery statuses found in the previous release.

**Architecture:** Preserve the S/G/Q algorithms and historical attempt. Bind execution to content identities, validate phase dependencies and outputs before reuse, and expose reproducible native/smoke jobs. Reports derive status and cost from actual receipts.

**Spec:** `docs/paper/static_ovmap/module_validation_v1/spec/CODEX_FINAL_EXECUTION_EN.md`.

**Execution:** Primary agent implements and reviews in the existing isolated worktree. Follow systematic-debugging, TDD, and verification-before-completion. No new research choices or independent-scene substitutions.

- [x] Harden `evaluation.py`: immutable prediction buffers, revalidation at lock, complete G support, evaluation-context cache identity, unmatched projections. Regressions in `test_evaluation.py`.
- [x] Honor metric tolerance in CAL/SELECT tie-breaks and return INCONCLUSIVE on undefined final metrics. Regressions in selection/head tests.
- [x] Harden `StudyRunner`: hash outputs and dependencies, invalidate stale downstream receipts, record failures and timings, preserve older code/config attempts, isolate report output until explicit export. Regressions in `test_orchestrator.py`.
- [x] Add native build/replay and real capture/S/G/Q smoke execution entrypoints with exact commands and input hashes. Successful native replay: `tooling/repairs-20260922/native-v7`; complete public invocation: `attempt_003`.
- [x] Replace hard-coded report success/cost/blockers with receipt-derived values; export all finalized receipts and correct the old completion claim. Reconcile manifests and release receipt.
- [x] Run focused module tests (95 passed), lint, actual boundary smokes, artifact checks; commit and push the authorized task branch; compare remote code SHA (`2cad816d98ea07270a032d22f47fe76005c278c0`).

The repair checklist is complete for these scoped engineering fixes. Overall study implementation remains PARTIAL: the complete independent-scene scientific driver and required ScanNet scenes are outstanding, explicitly recorded in the corrected delivery.

Verification uses `/home/ww/miniconda3/envs/ovimap-map/bin/python -m pytest -q tests/module_validation` and `ruff check src/static_ovmap/module_validation scripts/evaluation/run_ovimap_module_study.py tests/module_validation`.
