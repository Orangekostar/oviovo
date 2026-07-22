# TESSE-CD T2 Publication Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind T2 metric/status pairs to one frozen run and publish finalizer files and temporal directories without overwrite or partial output.

**Architecture:** Add one strict run-identity parser/pair validator and one atomic JSON publisher to the T2 finalizer. Build temporal artifacts in a sibling staging directory, fsync the tree, then publish through an exclusive empty-directory reservation.

**Tech Stack:** Python standard library (`hashlib`, `json`, `os`, `re`, `shutil`, `tempfile`), pytest.

---

### Task 1: Run identity and F1 contracts

**Files:**
- Modify: `tests/evaluation/test_finalize_tesse_t2.py`
- Modify: `scripts/evaluation/finalize_tesse_t2.py`

- [ ] **Step 1: Write failing tests**

Add fixture identities and tests equivalent to:

```python
identity = {"run_id": "run-a", "config_sha256": "a" * 64}
metrics["run_identity"] = identity
status["run_identity"] = {**identity, "run_id": "run-b"}
with pytest.raises(ValueError, match="run identity"):
    build_scene_evidence(metrics_path, status_path, method_key="DUALMAP", mode="native")
```

Also remove `run_identity` from either side and require rejection, assert a valid pair is preserved, require full-result identities to match provenance/config hashes, and parameterize F1 values with `True`, `"0.5"`, and a mapping.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/evaluation/test_finalize_tesse_t2.py -k 'run_identity or numeric_type'`.

Expected: identity mismatch/missing and bool/string acceptance tests fail because pairing and strict type checks do not exist.

- [ ] **Step 3: Implement strict validation**

Add a parser based on:

```python
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

def _run_identity(payload, *, label):
    raw = payload.get("run_identity")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} run identity is required")
    run_id = raw.get("run_id")
    digest = raw.get("config_sha256")
    if type(run_id) is not str or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(f"{label} run identity run_id is not canonical")
    if type(digest) is not str or not _is_lower_sha256(digest):
        raise ValueError(f"{label} run identity config_sha256 is invalid")
    return {"run_id": run_id, "config_sha256": digest}
```

Require metrics/status identities to compare exactly in scene evidence and full results. In full results, require both run IDs to equal provenance `run_id` and each digest to appear in freshly hashed provenance configs. Store identities by scene. Before `float(value)`, require `type(value) in {int, float}`.

- [ ] **Step 4: Verify GREEN**

Run `pytest -q tests/evaluation/test_finalize_tesse_t2.py` and require all tests pass.

### Task 2: Atomic finalizer JSON publication

**Files:**
- Modify: `tests/evaluation/test_finalize_tesse_t2.py`
- Modify: `scripts/evaluation/finalize_tesse_t2.py`

- [ ] **Step 1: Write failing no-clobber and fault tests**

Run both scene and full CLI commands against a target containing `b"sentinel"`; require unchanged bytes. Inject an `os.link` failure into the publication helper and require no target or sibling temporary file remains.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/evaluation/test_finalize_tesse_t2.py -k 'no_clobber or publication_failure'`.

Expected: full CLI overwrites the target and the atomic helper is absent.

- [ ] **Step 3: Implement sibling-temp publisher**

Implement `_atomic_json_no_replace(path, payload)` using `tempfile.mkstemp(dir=path.parent)`, `json.dump`, file flush/fsync, `os.link(temp, path)`, parent-directory fsync, and unconditional temporary unlink. Route both CLI output branches through it.

- [ ] **Step 4: Verify GREEN**

Run the full T2 finalizer module tests and require all pass.

### Task 3: Transactional temporal artifact publication

**Files:**
- Modify: `tests/evaluation/test_export_tesse_temporal_artifact.py`
- Modify: `scripts/evaluation/export_tesse_temporal_artifact.py`

- [ ] **Step 1: Write failing staging tests**

Monkeypatch `shutil.copyfile` to fail during checkpoint copying. Require `output` not to exist, no sibling staging directory to remain, then restore copying and retry successfully. Pre-create `output/marker`, require `FileExistsError`, and assert marker bytes remain unchanged.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/evaluation/test_export_tesse_temporal_artifact.py -k 'copy_failure or no_clobber'`.

Expected: copy failure leaves the directly-created output directory and existing output raises the old error type.

- [ ] **Step 3: Implement staging and reserved publication**

Create staging with `tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)`. Generate all paths relative to staging, fsync every file and directory, then reserve with `output.mkdir()`, rename staging over the empty reservation, and fsync the parent. Before reservation, remove staging on every failure. After reservation, wrap failures as `RuntimeError("temporal artifact publication-uncertain")` without deleting the final path.

- [ ] **Step 4: Verify GREEN**

Run the complete temporal exporter tests and require all pass.

### Task 4: Protocol verification and implementation commit

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run focused modules**

Run `pytest -q tests/evaluation/test_finalize_tesse_t2.py tests/evaluation/test_export_tesse_temporal_artifact.py`.

- [ ] **Step 2: Run protocol gate**

Run the existing 14-file TESSE-CD protocol focused command and require no failures.

- [ ] **Step 3: Check code and diff**

Run `python -m py_compile scripts/evaluation/finalize_tesse_t2.py scripts/evaluation/export_tesse_temporal_artifact.py`, `git diff --check`, and `git status --short`.

- [ ] **Step 4: Commit implementation**

Stage the two production files, two test files, and this plan; commit with `fix: harden TESSE-CD evidence publication`.
