# OVIV2 Temporal Staging Recovery Proxy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a no-clobber, nonformal recovery proxy from a preserved OVIV2 temporal staging tree and prove that the existing strict temporal exporter accepts the recovered geometry artifacts.

**Architecture:** A standalone recovery command snapshots and validates the preserved files, copies only schedule/capture/temporal streams and official neutral checkpoints into an independent staging tree, writes a legacy base `source_index.json`, and atomically publishes the proxy with an explicit non-submission receipt. Existing runner and evaluator schemas remain unchanged.

**Tech Stack:** Python 3 standard library, NumPy fixture data, existing OVIV2 map snapshot writer/reader, pytest, existing temporal exporter.

---

### Task 1: Minimal valid proxy contract

**Files:**
- Create: `tests/evaluation/test_recover_oviv2_temporal_staging_proxy.py`
- Create: `scripts/evaluation/recover_oviv2_temporal_staging_proxy.py`

- [ ] **Step 1: Write the failing end-to-end test**

Create `_build_preserved_staging(root)` with one `OVIV2` Apartment checkpoint at
frame 0/timestamp 100. Write `inputs/schedule.json`, the three JSONL streams,
`checkpoints/00000000-100/checkpoint_status.json`, and a real
`neutral_current/{snapshot.npz,entities.json}` via `write_map_snapshot()`. Write
`capture_status.json` using exact relative SHA-256 records. The test API is:

```python
result = recover_temporal_staging_proxy(
    source_root=source,
    output_root=tmp_path / "proxy",
)
assert result == tmp_path / "proxy/source_index.json"
index = json.loads(result.read_text(encoding="utf-8"))
assert set(index) == {
    "schema_version", "dataset", "mode", "method", "scene", "schedule",
    "capture_status", "trajectories", "frame_coverage",
    "lifecycle_transitions", "checkpoints",
}
assert export_temporal_artifact(result, tmp_path / "temporal").is_file()
```

Also assert the source tree inventory and every source file digest are unchanged,
and assert the receipt status is exactly
`NONFORMAL_RECOVERY_PROXY_NOT_SUBMISSION_EVIDENCE` with
`submission_eligible is False`. Require receipt keys
`{schema_version,status,submission_eligible,source_root,source_files,source_index,recovery_tool}`;
`source_files` is the canonical path-sorted list of absolute source witnesses and
`recovery_tool` binds this script's SHA-256 and byte count. Assert the source index
contains none of `runtime_diagnostics`, `run_manifest`, `run_execution`,
`frozen_run_identity`, or `nonformal_authorization`.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
pytest -q tests/evaluation/test_recover_oviv2_temporal_staging_proxy.py::test_recovers_exporter_readable_proxy_without_mutating_source
```

Expected: collection fails with `ModuleNotFoundError` for the recovery module.

- [ ] **Step 3: Implement the minimal public API and source records**

Define:

```python
def recover_temporal_staging_proxy(*, source_root: Path, output_root: Path) -> Path:
    validated = _validate_preserved_staging(source_root.resolve())
    staging = _create_sibling_staging(output_root.resolve())
    try:
        source_index = _materialize_proxy(validated, staging)
        validated.revalidate()
        _publish_no_clobber(staging, output_root.resolve())
    except BaseException:
        _remove_owned_staging(staging)
        raise
    return output_root.resolve() / source_index.relative_to(staging)
```

Use `loads_strict()` for JSON. Define a frozen `_FileWitness` containing resolved
path, bytes, SHA-256, byte count, device, inode, mode, size, mtime, and ctime. Its
constructor rejects paths outside the source root, symlinks, and non-regular files;
`revalidate()` rejects any identity or digest change. Source records must contain
exactly `path`, `sha256`, and `byte_count` with canonical relative POSIX paths.

Validate the schedule and capture identities:

```python
schedule["dataset"] == "TESSE-CD"
capture["schema_version"] == 1
capture["status"] == "PASS"
capture["mode"] == "causal_checkpoints"
capture["scene"] in schedule["scenes"]
capture["scheduled_frame_indices"] == capture["captured_frame_indices"]
```

Require schedule entries to produce the exact captured frame sequence. Verify the
capture records for schedule, trajectories, frame coverage, and lifecycle against
fresh witnesses.

- [ ] **Step 4: Copy one official checkpoint and publish**

Read each declared checkpoint-status record, require exact frame/timestamp and
`consumed_through_frame_exclusive == frame + 1`, and locate sibling
`neutral_current/snapshot.npz` and `entities.json`. Copy every verified source to
the same relative path under a sibling temporary directory. Write a base source
index with `method="OVIV2"`, fresh destination-relative records, then write the
nonformal receipt. Revalidate all source witnesses, fsync the staged tree, reserve
the destination with exclusive `mkdir`, rename the staging tree over the owned
empty reservation, and fsync the parent.

- [ ] **Step 5: Run the end-to-end test and verify GREEN**

Run the exact Task 1 test command. Expected: `1 passed` and the downstream exporter
creates `temporal_manifest.json`.

- [ ] **Step 6: Commit Task 1**

```bash
git add scripts/evaluation/recover_oviv2_temporal_staging_proxy.py \
  tests/evaluation/test_recover_oviv2_temporal_staging_proxy.py
git commit -m "Add temporal staging recovery proxy"
```

### Task 2: Fail-closed source and checkpoint validation

**Files:**
- Modify: `tests/evaluation/test_recover_oviv2_temporal_staging_proxy.py`
- Modify: `scripts/evaluation/recover_oviv2_temporal_staging_proxy.py`

- [ ] **Step 1: Write failing capture/schedule tests**

Parameterize mutations for captured frame mismatch, schedule timestamp mismatch,
capture scene mismatch, and incorrect schedule/stream digest. Each call must raise
`ValueError` containing the failing contract name and must leave the destination
absent.

- [ ] **Step 2: Verify capture/schedule tests are RED**

Run:

```bash
pytest -q tests/evaluation/test_recover_oviv2_temporal_staging_proxy.py -k 'capture or schedule or stream'
```

Expected: at least one malformed fixture is accepted by the Task 1 implementation.

- [ ] **Step 3: Add exact schema and record validation**

Require the schedule, capture, source records, schedule entries, and checkpoint
status dictionaries to have the exact field sets accepted by the existing temporal
exporter. Compare each declared record with the corresponding fresh witness,
including canonical relative path. Reject absolute paths, `.`/`..`, empty parts,
duplicate records, and path aliases.

- [ ] **Step 4: Write failing checkpoint tests**

Add independent tests for a missing neutral file, checkpoint-status symlink,
duplicate declared status record, frame mismatch, timestamp mismatch, incorrect
exclusive boundary, and an undeclared second status with the same official
frame/timestamp. Each must fail before publication.

- [ ] **Step 5: Verify checkpoint tests are RED**

Run:

```bash
pytest -q tests/evaluation/test_recover_oviv2_temporal_staging_proxy.py -k 'checkpoint or symlink or boundary'
```

Expected: at least one invalid checkpoint is accepted.

- [ ] **Step 6: Enforce unique official checkpoint resolution**

Scan only direct `checkpoints/*/checkpoint_status.json` regular files. Parse all
statuses strictly, index them by `(checkpoint_frame, timestamp_ns)`, and require
each official schedule identity to resolve exactly once and to equal the
corresponding capture `checkpoint_statuses` record. Reject any symlink in the
selected checkpoint directory or neutral subtree.

- [ ] **Step 7: Verify Task 2 GREEN and commit**

```bash
pytest -q tests/evaluation/test_recover_oviv2_temporal_staging_proxy.py
git add scripts/evaluation/recover_oviv2_temporal_staging_proxy.py \
  tests/evaluation/test_recover_oviv2_temporal_staging_proxy.py
git commit -m "Harden temporal proxy source validation"
```

Expected: all recovery tests pass.

### Task 3: Transactional publication and CLI

**Files:**
- Modify: `tests/evaluation/test_recover_oviv2_temporal_staging_proxy.py`
- Modify: `scripts/evaluation/recover_oviv2_temporal_staging_proxy.py`

- [ ] **Step 1: Write failing publication tests**

Pre-create the destination with `marker.txt` and require unchanged bytes. Wrap and
monkeypatch the existing `_ValidatedStaging.revalidate()` method so the wrapper
mutates a source immediately before invoking the real method; require publication
failure with no output. Patch `os.rename` to fail and require the temporary staging
directory to be removed while the source stays unchanged.

- [ ] **Step 2: Verify publication tests are RED**

Run:

```bash
pytest -q tests/evaluation/test_recover_oviv2_temporal_staging_proxy.py -k 'publication or no_clobber or mutation'
```

Expected: at least one atomicity invariant fails.

- [ ] **Step 3: Complete no-clobber failure handling**

Record the reservation inode after exclusive `mkdir`. On pre-rename errors, remove
only the owned empty reservation and sibling staging directory. If reservation
identity changes or rename state is uncertain, preserve both paths and raise
`RuntimeError("temporal recovery proxy publication uncertain")`. Never recursively
delete by an unverified destination name.

- [ ] **Step 4: Add the CLI**

Expose required `--source-root` and `--output-root` arguments. On success print one
canonical JSON line:

```json
{"source_index":"/absolute/proxy/source_index.json","status":"NONFORMAL_RECOVERY_PROXY_NOT_SUBMISSION_EVIDENCE"}
```

Return zero only after publication and post-publication source-index verification.

- [ ] **Step 5: Verify CLI and commit**

```bash
pytest -q tests/evaluation/test_recover_oviv2_temporal_staging_proxy.py
python -m py_compile scripts/evaluation/recover_oviv2_temporal_staging_proxy.py
git diff --check
git add scripts/evaluation/recover_oviv2_temporal_staging_proxy.py \
  tests/evaluation/test_recover_oviv2_temporal_staging_proxy.py
git commit -m "Finish temporal recovery proxy publication"
```

Expected: recovery tests pass, compilation succeeds, and diff check is silent.

### Task 4: Regression verification and A4 diagnostic recovery

**Files:**
- Verify: `scripts/evaluation/export_tesse_temporal_artifact.py`
- Runtime output only: `/home/pluto/crove_runs_20260816_final/a4_apartment_485666f_recovery_proxy`
- Runtime output only: `/home/pluto/crove_runs_20260816_final/a4_apartment_485666f_temporal_recovered`

- [ ] **Step 1: Run focused and exporter regressions**

```bash
pytest -q tests/evaluation/test_recover_oviv2_temporal_staging_proxy.py \
  tests/evaluation/test_export_tesse_temporal_artifact.py
python -m py_compile scripts/evaluation/recover_oviv2_temporal_staging_proxy.py
git diff --check
```

Expected: zero failures.

- [ ] **Step 2: Deploy the exact recovery script to node103**

Copy the verified script to `/tmp/recover_oviv2_temporal_staging_proxy.py` and run
it with source
`/home/pluto/crove_runs_20260816_final/a4_apartment_485666f_compact_preserved_final`
and the nonexistent proxy output above. Record the printed source-index path and
receipt digest.

- [ ] **Step 3: Export the recovered temporal artifact**

Run the existing `export_tesse_temporal_artifact.py` against the proxy
`source_index.json` and the nonexistent recovered temporal output. Require a valid
`temporal_manifest.json` and preserve the proxy receipt beside the evaluation
outputs.

- [ ] **Step 4: Run common-v2 and official Khronos diagnostics**

Evaluate the recovered temporal manifest with the frozen common-v2 target, aliases,
and Apartment label space. Prepare the Khronos bridge and run its official
evaluator using the preserved normalized A4 config. Label every produced summary
`NONFORMAL_RECOVERY_PROXY_NOT_SUBMISSION_EVIDENCE`; do not copy values into the
paper result tables.

- [ ] **Step 5: Diagnose the runner counter mismatch separately**

Finish the exact A3 frame-by-frame parity loop and record the first divergent frame,
counter, runtime value, accumulated-record value, and preceding state transition.
If it diverges, add a failing unit test before changing runner/runtime code. If it
does not diverge, compare the diagnostic loop's accumulation and finalization path
with the formal runner before proposing a fix.

- [ ] **Step 6: Final review**

Inspect the full diff, confirm no frozen T1 file changed, run the formal source
manifest verifier, and report exact test counts, runtime paths, metric outputs, and
the nonformal evidence limitation.
