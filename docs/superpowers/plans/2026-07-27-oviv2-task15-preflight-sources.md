# OVIV2 Task15 Preflight Sources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a source-backed, causally audited Apartment preflight bundle from the exact A0-A4 development transaction without requiring the final T1/T4 freeze.

**Architecture:** First make the fixed exact-profile transaction use a strict, explicitly unfrozen Apartment execution context while leaving every Office/final-freeze path unchanged. Then add an independent bundle producer that reopens the verified exact executions, audits causal inputs and checkpoints, runs the real occlusion evaluator, copies only consumer-required source files byte-for-byte, and atomically publishes evidence consumable by the existing search preflight validator.

**Tech Stack:** Python 3.10, pytest, descriptor-relative POSIX I/O, canonical JSON, SHA-256, existing OVIV2 TESSE runners/evaluators.

---

## Formal Test Command Prefix

Use this prefix for every test command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python \
  -m pytest -p no:cacheprovider
```

Do not touch the five user-owned untracked files listed by the root task.

### Task 1: Remove The Final-Freeze Cycle From Apartment Development Evidence

**Files:**
- Modify: `scripts/evaluation/run_oviv2_t1_reference.py`
- Modify: `scripts/evaluation/verify_oviv2_dual_readout_development_gates.py`
- Create: `tests/evaluation/test_run_oviv2_t1_reference.py`
- Modify: `tests/evaluation/test_verify_oviv2_dual_readout_development_gates.py`

- [ ] **Step 1: Write failing tests for the new exact spec and argv**

Add tests requiring exact specs with only
`profile,config,output_root,source_manifest`, and requiring child argv:

```python
assert argv == [
    python, runner, "--config", config, "--output", output,
    "--receipt", f"{output}/t1_exact_receipt.json",
    "--source-manifest", source,
]
```

for `reference`, and:

```python
assert argv == [python, runner, "--config", config, "--output", output]
```

for A0-A4. Assert that `--freeze-manifest` and `--run-slot` are rejected in
the development transaction and that an Office config is rejected before a
process starts.

- [ ] **Step 2: Verify RED**

Run:

```bash
<formal-prefix> -q \
  tests/evaluation/test_run_oviv2_t1_reference.py \
  tests/evaluation/test_verify_oviv2_dual_readout_development_gates.py
```

Expected: failures show the current required freeze fields/flags.

- [ ] **Step 3: Implement the explicit development reference entry point**

Keep the ordinary `run_reference()` frozen contract intact. Add
`run_development_reference` with keyword-only `config`, `output`, `receipt`,
and `argv`; defaulted `repo=REPO_ROOT`,
`source_manifest=DEFAULT_SOURCE_MANIFEST`, and `runner=run_v1`; and return type
`dict[str, Any]`.

It must accept only Apartment config, validate the canonical no-freeze argv,
call `runner(config, output, freeze_manifest=None, run_slot=None)`, and emit the
same exact artifact audit plus the execution mode
`apartment_development_unfrozen`. Make the CLI expose this mode without adding
an Office bypass.

- [ ] **Step 4: Replace freeze binding with exact execution-context binding**

In the development verifier, require:

```python
PROFILE_CONFIG_BINDING_FIELDS = {
    "mode", "config_sha256", "algorithm_hash", "profile_sha256"
}
```

with `mode == "apartment_development_unfrozen"`. Derive common fingerprints
from the verified config fields:

```python
{
    "source_manifest_sha256": source_record["sha256"],
    "input_manifest_sha256": config["input_manifest_sha256"],
    "schedule_sha256": config["schedule_manifest_sha256"],
    "ground_truth_sha256": config["occlusion_target_manifest_sha256"],
    "source_bindings_sha256": canonical_sha256(run_manifest["source_bindings"]),
}
```

The observation receipt must bind the config, source manifest, run manifest,
production receipt, root inode, PID, argv, and execution mode. It must not
contain a freeze-manifest record.

- [ ] **Step 5: Add fail-closed regression tests**

Cover an Office config, an injected freeze flag, missing source hashes,
changed config after execution, mismatched mode, duplicate root/PID, and a
source-binding mismatch. Keep the fixed nine-process sequence and exact
cumulative artifact equality tests.

- [ ] **Step 6: Verify GREEN and commit**

Run the two files above, then:

```bash
<formal-prefix> -q \
  tests/evaluation/test_compare_oviv2_cumulative_artifacts.py \
  tests/evaluation/test_run_oviv2_tesse_cd_v2.py
git diff --check
```

Commit only the four Task 1 files:

```bash
git add scripts/evaluation/run_oviv2_t1_reference.py \
  scripts/evaluation/verify_oviv2_dual_readout_development_gates.py \
  tests/evaluation/test_run_oviv2_t1_reference.py \
  tests/evaluation/test_verify_oviv2_dual_readout_development_gates.py
git commit -m "fix: decouple Apartment development evidence from freeze"
```

### Task 2: Build And Audit Independent Preflight Source Bundles

**Files:**
- Create: `scripts/evaluation/build_oviv2_tesse_search_preflight_sources.py`
- Create: `tests/evaluation/test_build_oviv2_tesse_search_preflight_sources.py`

- [ ] **Step 1: Write the fixture and fixed-position RED test**

Build a real-shaped development evidence fixture with the nine execution
records. The public `build_preflight_sources` API has keyword-only
`development_evidence`, `search_manifest`, `apartment_base_config`, `targets`,
`dataset_root`, and `output` parameters accepting `str | Path`, and returns
`dict[str, Any]`.

Assert it selects positions `{a0:1,a1:2,a2:4,a3:6,a4:8}` and rejects a
changed sequence, profile, observation record, config hash, code commit, or
source-manifest hash.

- [ ] **Step 2: Verify RED**

Run:

```bash
<formal-prefix> -q tests/evaluation/test_build_oviv2_tesse_search_preflight_sources.py
```

Expected: import failure because the producer does not exist.

- [ ] **Step 3: Implement strict readers and exact-run selection**

Add duplicate-key/nonfinite rejecting JSON readers, bounded descriptor reads,
symlink rejection, stable `(dev,ino,size,mtime_ns,ctime_ns)` witnesses, and
strict development-evidence schema validation. Reuse
`verify_exact_profile_runs()` before selecting any execution. Do not trust a
cached `status: PASS` field.

- [ ] **Step 4: Write RED tests for causal leakage**

Add one test per causal invariant from this exact inventory:

```python
("future_cache_frame", "coverage_gap", "coverage_timestamp_drift",
 "checkpoint_consumed_boundary", "compact_frame", "compact_revision",
 "compact_timestamp", "checkpoint_source_binding")
```

Each case must assert no final output exists and that the reported violation
identifies the candidate, frame, source role, observed value, and causal bound.

- [ ] **Step 5: Implement `_audit_future_leakage`**

Implement `_audit_future_leakage` with keyword-only `candidate_id: str`,
`run_root: Path`, `run: Mapping[str, Any]`,
`source_index: Mapping[str, Any]`, and
`checkpoint_index: Mapping[str, Any]`, returning `dict[str, Any]`.

Return the exact evidence schema:

```python
{
    "schema_version": 1,
    "manifest_id": "oviv2_tesse_future_leakage_evidence_v1",
    "scene": "apartment",
    "candidate_id": candidate_id,
    "source_index": run["source_index"],
    "records": [],
}
```

Only return an empty list after validating every scheduled frame, source-frame
mapping, checkpoint status, compact metadata record, timestamp, and exclusive
consumption boundary. A nonempty violation list must raise before publication.

- [ ] **Step 6: Write RED tests for copy and bundle binding**

Require byte-identical copies of `run_manifest`, `source_index`, trajectories,
lifecycle transitions, frame coverage, and runtime diagnostics at their
original relative paths. Reject absolute/escaping paths, aliased inodes,
duplicate roles, mutation during copy, and any copied digest mismatch.

- [ ] **Step 7: Implement verified bundle staging**

Copy via already-open descriptors into a new random sibling staging directory.
For each candidate, materialize `future_leakage.json`, run the existing
`evaluate_temporal_occlusion_package()` against the verified original
checkpoint index, and write `temporal_occlusion_result.json`. Reject unless the
recomputed gate has exactly 66 eligible anchors and at least 53 unique mappings.

Build `candidate_sources.json` with exact schema and canonical A0-A4 order.
Call existing `build_preflight()` using the staged sources and write staged
`preflight.json`. Finally call the search runner's
`_validate_preflight_gate_evidence()` against A0-A4 before publication.

- [ ] **Step 8: Add publication RED tests**

Cover existing output, post-mkdir wrapper failure, EEXIST/no-create,
source replacement before rename, parent replacement, partial write, fsync
failure, rename success followed by parent swap, and fd counts. The required
exception is:

```python
class PreflightPublicationUncertain(RuntimeError):
    preserved: tuple[PreservedArtifact, ...]
```

Every preserved record contains logical path/name, parent and artifact
device/inode, full `st_mode`, and `owned|unknown|unbound`. No failure path may
recursively delete a mutable name.

- [ ] **Step 9: Implement atomic no-clobber publication**

Use `renameat2(RENAME_NOREPLACE)` through the already-open parent descriptor.
Revalidate all input witnesses immediately before rename, fsync staged files
and directories, rename once, then fsync the parent. On uncertainty, preserve
all artifacts and chain the original cause. Propagate the original exception
only when creation provably did not occur.

- [ ] **Step 10: Verify GREEN and commit**

Run:

```bash
<formal-prefix> -q \
  tests/evaluation/test_build_oviv2_tesse_search_preflight_sources.py \
  tests/evaluation/test_build_oviv2_tesse_search_preflight.py \
  tests/evaluation/test_run_oviv2_tesse_dual_readout_search.py \
  tests/evaluation/test_evaluate_oviv2_tesse_temporal_occlusion.py
git diff --check
```

Commit only the new producer and its test:

```bash
git add scripts/evaluation/build_oviv2_tesse_search_preflight_sources.py \
  tests/evaluation/test_build_oviv2_tesse_search_preflight_sources.py
git commit -m "feat: build source-backed Apartment preflight bundles"
```

### Task 3: Cross-Module Verification And Review

**Files:**
- Modify only files required by verified review findings.

- [ ] **Step 1: Run focused Task15 verification**

```bash
<formal-prefix> -q \
  tests/evaluation/test_run_oviv2_t1_reference.py \
  tests/evaluation/test_verify_oviv2_dual_readout_development_gates.py \
  tests/evaluation/test_build_oviv2_tesse_search_preflight_sources.py \
  tests/evaluation/test_build_oviv2_tesse_search_preflight.py \
  tests/evaluation/test_run_oviv2_tesse_dual_readout_search.py \
  tests/evaluation/test_evaluate_oviv2_tesse_temporal_occlusion.py
```

- [ ] **Step 2: Run repository gates**

```bash
<formal-prefix> -q tests/oviv2 tests/evaluation
git diff --check
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python \
  scripts/evaluation/verify_oviv2_dual_readout_development_gates.py \
  --repo . --base-commit 8034e79d9cb853166222610981a7e6893f6cca70 \
  --verify-source-manifest \
  configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json
```

- [ ] **Step 3: Complete two-stage review**

Dispatch a spec reviewer, fix every Critical/Important finding with a failing
test, and obtain PASS. Then dispatch a fresh code-quality reviewer, fix every
Critical/Important finding with a failing test, and obtain PASS. Re-run the
focused and repository gates after the final commit.

- [ ] **Step 4: Prepare the real Task15 commands**

Generate materialized A0-A4 configs and an exact-profile specs JSON under the
ignored experiment root. Do not start the nine real runs until all Task 3
verification and review gates pass. The final preflight command must use the
new producer and the result must be independently accepted by the existing
search consumer before the three-GPU search starts.
