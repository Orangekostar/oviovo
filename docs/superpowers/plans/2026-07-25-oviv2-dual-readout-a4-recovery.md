# OVIV2 Dual-Readout A4 Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the externally killed A4 search candidate into a new, provenance-bound PASS search root without mutating or rerunning valid A0-A3 evidence.

**Architecture:** Add a strict standalone merger that snapshots the original and A4-only retry statuses, validates their file-backed configs/logs/run identities against the production manifest, and transactionally publishes a composite search status. Continue using the existing search runner for the A4-only retry and the existing packager/tuner for downstream metrics.

**Tech Stack:** Python standard library (`hashlib`, `json`, `os`, `pathlib`, `stat`, `tempfile`), pytest, existing OVIV2 search/config validators.

---

### Task 1: Exact recovery input contract

**Files:**
- Create: `tests/evaluation/test_recover_oviv2_tesse_dual_readout_search.py`
- Create: `scripts/evaluation/recover_oviv2_tesse_dual_readout_search.py`

- [ ] **Step 1: Write fixture builders and failing contract tests**

Build original and retry roots with real config/log/run-manifest files. The happy path must call:

```python
result = recover_search(
    manifest_path=MANIFEST,
    original_status=make_original_status(
        pass_ids=("a0", "a1", "a2", "a3"), failed_id="a4", exit_code=-9
    ),
    retry_status=make_retry_status(pass_ids=("a4",)),
    output_root=tmp_path / "recovered",
)
assert json.loads(result.read_text())["status"] == "PASS"
```

Parameterize invalid original A4 values for `exit_code`, nonempty stdout/stderr, `failure_reason`, `input_hashes`, and `run_identity`. Require exact original candidate order A0-A4, exact retry candidate list `[a4]`, and empty `unscheduled_candidate_ids`.

- [ ] **Step 2: Run tests to verify RED**

Run `pytest -q tests/evaluation/test_recover_oviv2_tesse_dual_readout_search.py`.

Expected: collection fails because the recovery module does not exist.

- [ ] **Step 3: Implement stable source snapshots**

Define a witness that retains path, bytes, parsed payload, and `(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)`. Its `revalidate()` must `os.stat(..., follow_symlinks=False)` and reject a changed identity or symlink. Read through `O_NOFOLLOW`; reject duplicate JSON keys, non-finite constants, non-regular files, and JSON larger than 16 MiB.

- [ ] **Step 4: Implement exact status and SIGKILL gates**

Use production `ValueError` checks equivalent to:

```python
if original["status"] != "FAIL":
    raise ValueError("original search status must be FAIL")
if [r["candidate_id"] for r in original["candidates"]] != list(CANDIDATE_IDS):
    raise ValueError("original candidates must be exact A0-A4 order")
if retry["status"] != "PASS" or [r["candidate_id"] for r in retry["candidates"]] != ["a4"]:
    raise ValueError("retry must contain exactly one PASS A4")
```

Accept original A4 only when scene is Apartment, status is FAIL, exit code is `-9`, stdout/stderr are exact empty-file records, semantic failure reason/input hashes/run identity are null, and both unscheduled lists are empty.

- [ ] **Step 5: Verify GREEN**

Run the Task 1 test module and require all input-contract cases to pass.

### Task 2: Cross-run identity and artifact validation

**Files:**
- Modify: `tests/evaluation/test_recover_oviv2_tesse_dual_readout_search.py`
- Modify: `scripts/evaluation/recover_oviv2_tesse_dual_readout_search.py`

- [ ] **Step 1: Write failing mismatch tests**

Add one mutation test for each binding: manifest SHA, Apartment base config, Office binding, A4 config bytes, declared temporal config, non-temporal hash, input-binding-values hash, run algorithm/input hash, code commit, and source bindings. Each must raise a message naming the mismatch.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/evaluation/test_recover_oviv2_tesse_dual_readout_search.py -k 'mismatch or differs'`.

Expected: at least one mutated binding is accepted.

- [ ] **Step 3: Validate file-backed records**

Compare each record exactly with freshly snapshotted bytes:

```python
expected = {
    "path": str(snapshot.path),
    "sha256": hashlib.sha256(snapshot.data).hexdigest(),
    "byte_count": len(snapshot.data),
}
if dict(record) != expected:
    raise ValueError(f"{label} record mismatch")
```

For configs, additionally compare the canonical config hash, canonical algorithm hash, manifest declaration, non-temporal hash, and input-binding-values hash. Original and retry A4 config bytes must be identical.

- [ ] **Step 4: Validate every PASS run identity**

Snapshot `<output_root>/run_manifest.json` for A0-A3 and retry A4. Require each record's `run_identity` to equal the run manifest's `algorithm_hash`, `input_sha256`, `code_commit`, and `source_bindings`; require `input_hashes == source_bindings`; require the run config record to match the candidate config. All PASS records must share the same code commit and source bindings.

- [ ] **Step 5: Verify GREEN**

Run the complete recovery test module and require every mismatch to fail closed.

### Task 3: Transactional composite publication

**Files:**
- Modify: `tests/evaluation/test_recover_oviv2_tesse_dual_readout_search.py`
- Modify: `scripts/evaluation/recover_oviv2_tesse_dual_readout_search.py`

- [ ] **Step 1: Write failing provenance and publication tests**

Require A0-A3 original records plus retry A4, five candidate Apartment directories, and:

```python
recovery = status["recovery"]
assert recovery["strategy"] == "immutable_single_candidate_retry_v1"
assert recovery["replaced_candidate_id"] == "a4"
assert recovery["original_status"]["sha256"] == sha256(original_bytes)
assert recovery["retry_status"]["sha256"] == sha256(retry_bytes)
```

Pre-create an output marker and require unchanged bytes. Mutate a source witness before publication and require no output. Inject rename failure and require no staging directory remains.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/evaluation/test_recover_oviv2_tesse_dual_readout_search.py -k 'publication or provenance or no_clobber'`.

Expected: provenance and transactional publication behavior are absent.

- [ ] **Step 3: Build and publish the composite tree**

Build a PASS payload from original A0-A3 plus retry A4. Add source status `{path, sha256, byte_count}` records, the strategy, replaced candidate, and original empty-log `-9` signature. In a sibling staging directory, write/fsync canonical JSON and create/fsync five candidate directories. Revalidate every source, reserve the destination with exclusive `mkdir`, rename staging over the empty reservation, and fsync the parent. Remove staging on pre-publication failure; never overwrite an existing destination.

- [ ] **Step 4: Add CLI wiring**

Expose required `--manifest`, `--original-status`, `--retry-status`, and `--output` arguments. Print canonical JSON containing the absolute recovered `status_path`; return zero only after successful publication.

- [ ] **Step 5: Verify GREEN and static checks**

Run:

```bash
pytest -q tests/evaluation/test_recover_oviv2_tesse_dual_readout_search.py
python -m py_compile scripts/evaluation/recover_oviv2_tesse_dual_readout_search.py
git diff --check
```

Expected: all tests pass, compilation succeeds, and the diff check is silent.

### Task 4: Execute A4-only retry and merge evidence

**Files:**
- Create at runtime: one immutable A4-only search root.
- Create at runtime: one composite A0-A4 search root.

- [ ] **Step 1: Verify source commit and protected files**

Require HEAD `8034e79d9cb853166222610981a7e6893f6cca70`. Require zero diff for `src/oviv2/runtime.py`, `src/oviv2/runner_config.py`, `scripts/run_oviv2_replica.py`, `scripts/evaluation/run_oviv2_tesse_cd.py`, and both v1 scene configs.

- [ ] **Step 2: Run only A4 in a fresh root**

Invoke:

```python
run_search(
    manifest_path=manifest,
    apartment_base_config=apartment_config,
    office_base_config=office_config,
    output_root=retry_root,
    gpu_ids=("0",),
    max_parallel=1,
    candidate_ids=("a4",),
)
```

Expected: PASS with exactly A4, exit code zero, and no Office execution.

- [ ] **Step 3: Merge into a new root**

Run the recovery CLI with the production manifest, original status, retry status, and a new composite output. Require PASS, exact A0-A4 order, and immutable recovery provenance.

- [ ] **Step 4: Package and tune without protocol changes**

Package results into `$COMPOSITE/candidates/<id>/apartment/result.json` using validated external sources. Run the existing tuner with `--results-root "$COMPOSITE"` and require a PASS selection covering exactly A0-A4.

### Task 5: Reviews and regression verification

**Files:**
- Verify all production, test, spec, and plan changes.

- [ ] **Step 1: Request specification review**

Require explicit PASS for immutable inputs, exact SIGKILL signature, cross-run equivalence, no-clobber publication, and downstream compatibility.

- [ ] **Step 2: Request independent quality review**

Resolve every Critical or Important issue with a failing test before changing implementation.

- [ ] **Step 3: Run focused and broad tests**

Run recovery, search runner, packager, tuner, temporal exporter/bridge, and occlusion modules. Then run the repository suite and require zero failures.

- [ ] **Step 4: Verify protected-file invariance and final diff**

Require the six protected files to have zero diff from `8034e79`, run `git diff --check`, inspect `git status --short`, and record exact test counts and output roots before commit.
