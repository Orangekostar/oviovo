# OVI-MAP Paper-Parity Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the current non-paper-equivalent OVI-MAP run from populating Table 1, add an executable CVPR 2026 protocol audit, and publish new OVI-MAP metrics only after the Replica-8 artifacts pass the paper protocol.

**Architecture:** Keep the existing neutral evaluator and the OVI-MAP paper evaluator as separate named contracts. A paper-parity audit reads the result metadata plus the eight native semantic-feature files, records coverage diagnostics, and emits a machine-readable PASS/FAIL report. PASS additionally requires hash-bound native-mapping, released-semantic, and class-agnostic-AP manifests; the released precision/recall evaluator alone cannot satisfy the AP requirement. The benchmark importer re-runs the audit against the exact result hash before accepting OVI-MAP Table 1 bindings; failed historical runs remain immutable evidence and their registry rows are quarantined.

**Tech Stack:** Python 3.10, NumPy, pickle, JSON, pytest, existing benchmark result importer, released OVI-MAP evaluator, Replica-8.

---

## File Structure

- Create `src/evaluation/baselines/ovimap_paper_audit.py`: paper protocol constants, native artifact summaries, and strict parity checks.
- Create `scripts/evaluation/audit_ovimap_paper_parity.py`: CLI that hashes inputs and writes the audit JSON.
- Create `tests/evaluation/test_ovimap_paper_audit.py`: unit coverage for protocol and artifact failures.
- Modify `tools/import_benchmark_results.py`: reject OVI-MAP T1 bindings without a passing audit whose embedded result SHA-256 matches the imported JSON.
- Modify `tests/evaluation/test_import_benchmark_results.py`: importer regression coverage.
- Create `docs/paper/results/baselines/ovimap/replica/20260719-s10-200f-full-instances-v2/paper_parity_audit.json`: immutable failure evidence for the current run.
- Modify `docs/paper/benchmark_tokens.tsv`: quarantine current OVI-MAP T1 bindings.
- Regenerate `docs/paper/benchmark_tables_baselines.md` and `docs/paper/benchmark_tables_baselines.tex`: show OVI-MAP as unfilled until parity passes.
- Modify `docs/paper/BASELINE_RUN_REPORT.md`: record the exact paper values, reproduced values, protocol mismatch, and rerun gate.
- Create `scripts/evaluation/run_ovimap_paper_protocol.py`: run released Replica-51 post-processing and evaluation without relabeling its metrics.
- Create `tests/evaluation/test_run_ovimap_paper_protocol.py`: command and layout validation.
- Modify `tests/test_benchmark_table_package.py`: assert all OVI-MAP T1 tokens remain unfilled until parity passes.

### Task 1: Encode the OVI-MAP Paper Protocol and Artifact Audit

**Files:**
- Create: `tests/evaluation/test_ovimap_paper_audit.py`
- Create: `src/evaluation/baselines/ovimap_paper_audit.py`
- Create: `scripts/evaluation/audit_ovimap_paper_parity.py`

- [ ] **Step 1: Write failing protocol tests**

Add fixtures with a paper-equivalent result and with the current 41-class result. Assert that the required contract is:

```python
PAPER_PROTOCOL = {
    "name": "ovimap_cvpr2026_replica",
    "scene_ids": ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"],
    "frame_count_per_scene": 200,
    "frame_step": 10,
    "semantic_vocabulary": "Replica-51",
    "semantic_metrics": "per_vertex_miou_macc_and_class_aware_mask_ap",
    "instance_metrics": "class_agnostic_mask_miou_ap25_ap50_ap75",
    "vlm": "siglip-large-patch16-384",
}
```

The current result must fail for missing protocol metadata, Replica-41 semantics, and an unverified instance-AP contract.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_ovimap_paper_audit.py -q
```

Expected: FAIL because `src.evaluation.baselines.ovimap_paper_audit` does not exist.

- [ ] **Step 3: Implement protocol validation and feature summaries**

Implement:

```python
def summarize_feature_file(path: str | Path, *, image_width: int = 1200, image_height: int = 680) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        instances = pickle.load(handle)
    records = list(instances.values())
    query_count = sum(len(record.get("frame_id", ())) for record in records)
    eligible_count = sum(len(record.get("frame_id", ())) >= 2 for record in records)
    boxes = [box for record in records for box in record.get("box_2d", ())]
    full_frame = sum(
        int(box[0]) == 0 and int(box[1]) == 0
        and int(box[2]) >= image_width - 1 and int(box[3]) >= image_height - 1
        for box in boxes
    )
    return {
        "feature_instance_count": len(records),
        "eligible_feature_instance_count": eligible_count,
        "query_count": query_count,
        "average_queries_per_feature_instance": query_count / len(records) if records else 0.0,
        "full_frame_bbox_count": full_frame,
        "full_frame_bbox_ratio": full_frame / len(boxes) if boxes else 0.0,
    }
```

Implement `audit_paper_parity(result, scene_summaries)` so PASS requires exact protocol fields, all eight scenes, 200 frames per scene, and raw native summaries for all scenes. Coverage statistics are diagnostic and must not be compared to paper values as an acceptance threshold.

- [ ] **Step 4: Implement the audit CLI**

Accept `--result`, repeated `--scene-feature SCENE=PATH`, and `--output`. Hash the result and all feature files, call `audit_paper_parity`, and atomically write sorted JSON.

- [ ] **Step 5: Run focused tests and verify GREEN**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_ovimap_paper_audit.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/evaluation/baselines/ovimap_paper_audit.py \
  scripts/evaluation/audit_ovimap_paper_parity.py \
  tests/evaluation/test_ovimap_paper_audit.py
git commit -m "feat: audit OVI-MAP paper protocol parity"
```

### Task 2: Block Non-Equivalent OVI-MAP Results at Import

**Files:**
- Modify: `tests/evaluation/test_import_benchmark_results.py`
- Modify: `tools/import_benchmark_results.py`

- [ ] **Step 1: Write the failing importer test**

Create an `OVIMAP` result binding `T1_OVIMAP_REPLICA8_MIOU` without `protocol_audit` and assert:

```python
with pytest.raises(ImportFailure, match="paper-parity audit"):
    import_results(...)
```

Add a passing case whose result audit record contains `status="PASS"`, `protocol_name="ovimap_cvpr2026_replica"`, and a real audit path. The audit file itself must report PASS without failures and its `result_source.sha256` must match the imported result JSON.

- [ ] **Step 2: Run the test and verify RED**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_import_benchmark_results.py -q
```

Expected: FAIL because the importer currently accepts any `VERIFIED` OVI-MAP result.

- [ ] **Step 3: Add the strict OVI-MAP gate**

Before processing OVI-MAP T1 bindings, require:

```python
audit = result.get("protocol_audit", {})
if audit.get("status") != "PASS" or audit.get("protocol_name") != "ovimap_cvpr2026_replica":
    raise ImportFailure("OVI-MAP T1 result requires a passing paper-parity audit")
audit_path = Path(str(audit.get("path", "")))
_require_file(audit_path)
audit_document = json.loads(audit_path.read_text(encoding="utf-8"))
if audit_document.get("status") != "PASS" or audit_document.get("failures") != []:
    raise ImportFailure("OVI-MAP paper-parity audit file must PASS without failures")
if audit_document.get("result_source", {}).get("sha256") != _sha256(result_path):
    raise ImportFailure("OVI-MAP paper-parity audit result hash mismatch")
recomputed = audit_paper_parity(result, audit_document.get("scene_artifacts", {}))
if recomputed["status"] != "PASS":
    raise ImportFailure("OVI-MAP result does not satisfy the paper protocol audit")
```

Keep T4 hardware-only OVI-MAP results outside this metric gate.

- [ ] **Step 4: Run importer tests and verify GREEN**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_import_benchmark_results.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add tools/import_benchmark_results.py tests/evaluation/test_import_benchmark_results.py
git commit -m "fix: gate OVI-MAP metrics on paper parity"
```

### Task 3: Audit and Quarantine the Current Replica-8 Result

**Files:**
- Create: `docs/paper/results/baselines/ovimap/replica/20260719-s10-200f-full-instances-v2/paper_parity_audit.json`
- Modify: `docs/paper/benchmark_tokens.tsv`
- Modify: `docs/paper/benchmark_tables_baselines.md`
- Modify: `docs/paper/benchmark_tables_baselines.tex`
- Modify: `docs/paper/BASELINE_RUN_REPORT.md`

- [ ] **Step 1: Run the audit on all eight feature files**

Use the existing repaired outputs, with `room1` from `formal_replica8/room_1`, and write the audit beside the historical result.

Expected audit status: `FAIL`, including missing paper protocol metadata and coverage diagnostics showing only 40 instances with at least two views across Replica-8 and a full-frame query-box ratio of 1.0.

- [ ] **Step 2: Verify the historical result is rejected**

Run the benchmark importer with the current OVI-MAP result.

Expected: failure containing `paper-parity audit`.

- [ ] **Step 3: Quarantine OVI-MAP T1 registry rows**

For every `T1_OVIMAP_*` row, clear `source_json` and `json_pointer`, set `status=UNFILLED`, and set the note to:

```text
Quarantined: 20260719 native run fails OVI-MAP CVPR 2026 paper-protocol parity audit.
```

Do not modify OVI-MAP T4 rows.

- [ ] **Step 4: Regenerate derived tables without the quarantined result**

Run `tools/import_benchmark_results.py` with all currently valid non-OVI-MAP T1 results plus the OVI-MAP T4 result. Confirm OVI-MAP T1 cells remain tokens rather than numbers.

- [ ] **Step 5: Update the run report**

Record:

```text
Paper Table 2 Replica: instance mIoU/AP25/AP50/AP75 = 36.3/76.7/50.8/22.0 (%).
Paper Table 3 Replica: semantic mIoU/mAcc/AP25/AP50/APall = 26.5/32.2/34.5/21.2/8.5 (%).
Current native run: neutral semantic mIoU/mAcc = 3.10/4.97 (%); neutral AP25/AP50 = 32.59/18.37 (%).
Released Replica-51 room0 evaluator on the same artifacts: semantic mIoU/mAcc = 2.75/3.28 (%).
```

State that the AP columns were previously described as class-agnostic but numerically compared to the semantic-instance table; they are not accepted as a paper reproduction.

- [ ] **Step 6: Run table-package tests**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_import_benchmark_results.py \
  tests/test_benchmark_table_package.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

```bash
git add docs/paper/results/baselines/ovimap/replica/20260719-s10-200f-full-instances-v2/paper_parity_audit.json \
  docs/paper/benchmark_tokens.tsv docs/paper/benchmark_tables_baselines.md \
  docs/paper/benchmark_tables_baselines.tex docs/paper/BASELINE_RUN_REPORT.md
git commit -m "fix: quarantine non-equivalent OVI-MAP metrics"
```

### Task 4: Add an Executable Released Paper-Protocol Runner

**Files:**
- Create: `tests/evaluation/test_run_ovimap_paper_protocol.py`
- Create: `scripts/evaluation/run_ovimap_paper_protocol.py`

- [ ] **Step 1: Write failing command-construction tests**

Given a synthetic Replica-8 source layout, assert that the runner constructs, in order, the released `preprocess_gt_mesh`, `mesh_postprocess_utils`, and `eval_sem_seg` commands, uses Replica-51, and never converts released precision/recall diagnostics into AP.

- [ ] **Step 2: Run the test and verify RED**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_run_ovimap_paper_protocol.py -q
```

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement preflight and command execution**

The runner must verify all eight `instance_mesh_200.ply`, `inst_sem_siglip-l-16-384_200_incre_combine.pkl`, and backend color-log files, the Replica GT meshes, the local SigLIP model, and the released evaluator checkout. Before released post-processing it deterministically replaces malformed pkl colors from authoritative `LogInstanceColor` records and records both hashes plus dropped stale IDs. It writes per-command stdout, stderr, exit status, source hash, and elapsed time. Completion additionally requires 51 expected output files, eight finite instance-diagnostic rows, finite semantic mIoU/mAcc, and a parseable finite `results_replica.json`; empty successful scripts must fail output validation. Any missing input, nonzero command, or invalid output marks the run `BLOCKED` or `FAILED`, never `VERIFIED`.

- [ ] **Step 4: Run tests and verify GREEN**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_run_ovimap_paper_protocol.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add scripts/evaluation/run_ovimap_paper_protocol.py \
  tests/evaluation/test_run_ovimap_paper_protocol.py
git commit -m "feat: run released OVI-MAP paper evaluation"
```

### Task 5: Produce a New Native Artifact Set Before Restoring Table 1

**Files:**
- External output: `/home/ww/oviovo_baseline_runs/20260719_ovimap_paper_parity/`
- Create after success: `docs/paper/results/baselines/ovimap/replica/<new-run-id>/result.json`
- Modify after success: `docs/paper/benchmark_tokens.tsv`
- Regenerate after success: `docs/paper/benchmark_tables_baselines.md`
- Regenerate after success: `docs/paper/benchmark_tables_baselines.tex`

- [ ] **Step 1: Resolve the native-input blocker**

Obtain executable access to `baseline-ovimap-58a804e` or an equivalent host Python environment for the built ROS/C++ modules. Record the exact Replica RGB-D trajectory source and the exact CropFormer instance-ID export used by the paper. Do not patch the OVI-MAP algorithm to chase paper values.

- [ ] **Step 2: Run room0 as the parity gate**

Run 200 evenly sampled frames with the paper configuration. Then run the released Replica-51 post-processing and both paper metric families. Preserve all raw outputs.

- [ ] **Step 3: Compare artifact invariants**

The new run must explain the current anomalies before expansion: only 4 room0 feature instances had at least two views, average queries per feature instance was 3.2 rather than the paper's reported 8.6 aggregate, and every saved semantic query box covered the full image. Treat these as diagnostics, not tunable target thresholds.

- [ ] **Step 4: Expand to Replica-8 only after room0 is valid**

Run all eight scenes, execute Task 4's evaluator, and generate a new parity audit. Existing result directories remain unchanged.

- [ ] **Step 5: Restore registry bindings only on PASS**

Finalize a new result that points to a `protocol_audit.status=PASS` sidecar. Generate the audit after adding that pointer so `result_source.sha256` binds the exact imported JSON. PASS requires a native manifest with exact frame provenance and Replica-51/SigLIP/GT hashes, a validated released semantic manifest, and a separate evaluator manifest with the paper's class-agnostic mIoU/AP25/AP50/AP75 contract. Import it through the guarded importer, regenerate tables, and run the full benchmark package tests. If parity remains unresolved, keep OVI-MAP T1 cells unfilled and report the exact failing invariant.

## Self-Review

- Spec coverage: separates paper Table 2 and Table 3, blocks current mixed metrics, binds PASS to evaluator manifests and outputs, preserves failed evidence, and defines the native rerun gate.
- Placeholder scan: no implementation `TODO`/`TBD` placeholders are present; unresolved native execution is an explicit external-access gate in Task 5.
- Type consistency: the result audit record uses `status`, `protocol_name`, and `path`; the audit sidecar uses `result_source.sha256` to bind the exact result JSON.
- Safety: no paper number is copied into a measured token, no historical result is overwritten, and no reference checkout is modified.
