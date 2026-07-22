# OVIV2 TESSE-CD T2 Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate the strict provenance JSON consumed by `finalize_tesse_t2.py` from the frozen OVIV2 manifest and the four completed, repeated TESSE-CD runs.

**Architecture:** A dependency-light CLI treats the `FROZEN` manifest as the authority for commit, config, dataset, model, and mapping-root identity. It revalidates mapping, bridge, evaluator, and finalizer inputs; derives only observed command/environment/hardware fields; and atomically publishes a deterministic JSON file after all identities agree.

**Tech Stack:** Python 3.12, strict JSON, SHA-256, pytest, existing TESSE-CD validators.

---

### Task 1: Bind clean mapping provenance

**Files:**
- Modify: `scripts/evaluation/run_oviv2_tesse_cd.py`
- Test: `tests/evaluation/test_run_oviv2_tesse_cd.py`

- [ ] Add a failing test that requires production provenance to contain the SHA-256 digest of `git status --porcelain=v1 -z`.
- [ ] Run `pytest -q tests/evaluation/test_run_oviv2_tesse_cd.py -k production_provenance` and confirm the missing field fails.
- [ ] Add the repository status capture beside `repository_commit`.
- [ ] Re-run the focused test and require it to pass.

### Task 2: Generate strict finalizer provenance

**Files:**
- Create: `scripts/evaluation/build_oviv2_tesse_t2_provenance.py`
- Create: `tests/evaluation/test_build_oviv2_tesse_t2_provenance.py`

- [ ] Add failing fixture tests for a valid four-run package and for rejected PREPARED/dirty/mismatched/missing-weight/existing-output inputs.
- [ ] Run `pytest -q tests/evaluation/test_build_oviv2_tesse_t2_provenance.py` and confirm import failure.
- [ ] Implement `build_provenance(...)` and a CLI accepting one freeze manifest, four official run roots, exact `ROLE=PATH` weight bindings, and a fresh output path.
- [ ] Require exact frozen repository/config/model/output-root bindings, clean mapping provenance, validated official run/evaluator identities, byte-identical metric repeats, and a full expected weight-role set.
- [ ] Emit canonical absolute file records for dataset/config/weight/raw output fields, preserve observed commands/environment/hardware, and publish with no replacement.
- [ ] Re-run the focused test and require it to pass.

### Task 3: Verify integration and review

**Files:**
- Verify: `scripts/evaluation/build_oviv2_tesse_t2_provenance.py`
- Verify: `scripts/evaluation/run_oviv2_tesse_cd.py`
- Verify: `tests/evaluation/test_build_oviv2_tesse_t2_provenance.py`
- Verify: `tests/evaluation/test_run_oviv2_tesse_cd.py`

- [ ] Run the focused tests plus freeze, official evaluator, and T2 finalizer suites.
- [ ] Run `python -m py_compile` and `git diff --check`.
- [ ] Commit the implementation and request an independent review against the frozen-run contract.
- [ ] Fix every critical or important finding, re-run the complete verification command, and report the final commit.
