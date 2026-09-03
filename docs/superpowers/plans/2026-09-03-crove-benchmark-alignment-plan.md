# CROVE Benchmark Alignment Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a source-bound B0-B9 audit that separates Khronos protocol, aggregation, frontend, temporal-slice, identity, and current-maintenance effects, then freezes and uploads the resulting benchmark decision.

**Architecture:** Pure Python modules own schema validation and numeric diagnostics; thin CLIs bind source files and atomically publish JSON. Khronos remains external and is changed only through reviewable patches. Dataset adapters expose immutable inventories and causal sessions; experiment reports consume frozen artifacts rather than runtime state.

**Tech Stack:** Python 3.13, dataclasses, NumPy, CSV/JSON/YAML, pytest, Git source manifests, Khronos C++ patch files.

**Spec:** `docs/superpowers/specs/2026-09-03-crove-benchmark-alignment-design.md`

## Global Constraints

- Base SHA is exactly `668aefc49034ef97d090b811b1c3f291ecafbd66`.
- Stage order is B0, B1, B2, B3, B4, B5, B6, B7, B8, B9.
- Historical A6/P5/P6 artifacts and evaluator outputs are immutable.
- Ranking-eligible runtime inputs never use future observations or GT semantics, instances, transforms, identities, changes, or evaluator outcomes.
- GT conditions are labeled `ORACLE_DIAGNOSTIC` and `NOT_RANKING_ELIGIBLE`.
- Paper-like Khronos absolute F1 tolerance is `0.02`, frozen before execution.
- Khronos numeric parity uses `rtol=0` and `atol=1e-12`; row identity is exact.
- `FRONTEND_DOMINATED` requires at least 50% gap closure on a majority of available headline metrics.
- The 3RScan pilot uses the first 10 sorted validation environments satisfying the frozen change rule.
- Official/post-release Khronos and `TESSE_CURRENT_DIAGONAL` have separate protocol IDs and output roots.
- Large RGB-D, meshes, maps, checkpoints, and third-party trees remain outside Git.
- Every missing external dataset produces a `BLOCKED_DATASET_ACCESS` report, never an estimated result.
- The frozen T1 source verifier and full repository test suite must pass before push.

---

### Task 1: Freeze B0/B1 Evidence and Pre-Result Decisions

**Files:**
- Create: `configs/evaluation/external_benchmark_sources.json`
- Create: `configs/evaluation/benchmark_suitability_pre_result.json`
- Create: `docs/superpowers/reports/2026-09-03-dynamic-mapping-literature-benchmark-audit.md`
- Create: `docs/superpowers/experiments/2026-09-03-benchmark-alignment-decision-ledger.md`
- Test: `tests/evaluation/test_benchmark_alignment_preregistration.py`

**Interfaces:**
- Consumes: official repository commits, papers, dataset documentation, the master prompt, frozen P6 reports, and local asset inventories.
- Produces: source IDs referenced by every scorecard cell and immutable B0/B1 ledger entries.

- [ ] **Step 1: Write the failing schema test**

```python
def test_suitability_scorecard_is_complete_and_source_bound():
    payload = json.loads(SCORECARD.read_text())
    assert payload["status"] == "FROZEN_PRE_RESULT"
    assert set(payload["benchmarks"]) == {
        "tesse_official", "tesse_current_diagonal", "3rscan", "panoptic_flat"
    }
    for row in payload["benchmarks"].values():
        assert set(row["scores"]) == EXPECTED_DIMENSIONS
        assert all(item["score"] in {0, 1, 2} for item in row["scores"].values())
        assert all(item["source_ids"] for item in row["scores"].values())
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/evaluation/test_benchmark_alignment_preregistration.py`

Expected: FAIL because the frozen source registry and scorecard do not exist.

- [ ] **Step 3: Publish exact source and scorecard documents**

The source registry uses one object per external repository with keys
`repo`, `commit`, `license`, `role`, `paper`, and `retrieval_date`. The
scorecard contains the 12 exact dimensions from the spec, integer scores, a
short rationale, and source IDs. Record all local repository/data SHA-256 and
the observed Khronos release/latest/used-artifact identity boundary in B0.

- [ ] **Step 4: Run GREEN and integrity checks**

Run: `pytest -q tests/evaluation/test_benchmark_alignment_preregistration.py && git diff --check`

Expected: PASS with every score source resolving to the literature report or source registry.

- [ ] **Step 5: Commit before reading CROVE pilot scores**

```bash
git add configs/evaluation/external_benchmark_sources.json \
  configs/evaluation/benchmark_suitability_pre_result.json \
  docs/superpowers/reports/2026-09-03-dynamic-mapping-literature-benchmark-audit.md \
  docs/superpowers/experiments/2026-09-03-benchmark-alignment-decision-ledger.md \
  tests/evaluation/test_benchmark_alignment_preregistration.py
git commit -m "docs: freeze benchmark suitability before pilots"
```

### Task 2: Add Source-Bound Khronos Aggregation Parity

**Files:**
- Create: `src/evaluation/khronos_metric_parity.py`
- Create: `scripts/evaluation/audit_khronos_metric_parity.py`
- Create: `tests/evaluation/test_khronos_metric_parity.py`
- Create: `docs/superpowers/reports/2026-09-03-khronos-metric-parity.md`
- Modify: `docs/superpowers/experiments/2026-09-03-benchmark-alignment-decision-ledger.md`

**Interfaces:**
- Consumes: result directory containing `static_objects.csv`, `dynamic_objects.csv`, and `background_mesh.csv`; frozen upstream Khronos checkout.
- Produces: `KhronosParityResult` and atomic JSON audit with exact source records.

- [ ] **Step 1: Write RED tests for upstream row semantics**

```python
def test_parity_compares_full_grid_duplicates_slices_and_weighting(tmp_path):
    fixture = write_triangular_khronos_fixture(tmp_path)
    result = audit_metric_parity(fixture, upstream_utils=FROZEN_UTILS)
    assert result.row_grid_equal
    assert result.duplicate_policy_equal
    assert result.slice_values_equal
    assert result.metric_values_equal
    assert result.max_abs_delta <= 1e-12

def test_conflicting_duplicate_is_reported_as_non_parity(tmp_path):
    fixture = write_conflicting_duplicate_fixture(tmp_path)
    with pytest.raises(KhronosParityError, match="conflicting duplicate"):
        audit_metric_parity(fixture, upstream_utils=FROZEN_UTILS)
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/evaluation/test_khronos_metric_parity.py`

Expected: FAIL because the parity module does not exist.

- [ ] **Step 3: Implement the independent parity model**

```python
@dataclass(frozen=True, slots=True)
class KhronosParityResult:
    row_grid_equal: bool
    duplicate_policy_equal: bool
    slice_values_equal: bool
    metric_values_equal: bool
    local_metrics: Mapping[str, float]
    upstream_metrics: Mapping[str, float]
    max_abs_delta: float
```

Load upstream `plotting/utils.py` from the declared checkout, compare raw
`(Name, Query)` maps and all 4D/Robot/Query/Online slices, then compare local
and upstream Object/Dynamic/Change/Background F1. Reject missing, non-finite,
out-of-domain, or source-changing inputs. The CLI writes sorted finite JSON
through an atomic rename.

- [ ] **Step 4: Run GREEN on synthetic and frozen A6/P5 fixtures**

Run: `pytest -q tests/evaluation/test_khronos_metric_parity.py tests/evaluation/test_tesse_cd_metrics.py`

Expected: PASS and the report states either parity or the exact mismatch without rewriting historical metrics.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/khronos_metric_parity.py \
  scripts/evaluation/audit_khronos_metric_parity.py \
  tests/evaluation/test_khronos_metric_parity.py \
  docs/superpowers/reports/2026-09-03-khronos-metric-parity.md \
  docs/superpowers/experiments/2026-09-03-benchmark-alignment-decision-ledger.md
git commit -m "eval: add Khronos aggregation parity audit"
```

### Task 3: Close the Khronos Paper-Protocol Identity Gate

**Files:**
- Create: `configs/external/khronos_source_manifest.json`
- Create: `docs/superpowers/reports/2026-09-03-khronos-protocol-reproduction.md`
- Modify: `docs/superpowers/experiments/2026-09-03-benchmark-alignment-decision-ledger.md`
- Test: `tests/evaluation/test_khronos_source_manifest.py`

**Interfaces:**
- Consumes: release `742227a88de8b2ac23ac54d719b321c3af88dc75`, latest official main, current TESSE run manifests, RSS paper/config/data evidence.
- Produces: one of `PAPER_PROTOCOL_EXACT`, `PAPER_PROTOCOL_APPROX_PUBLIC`, `BLOCKED_PAPER_ASSET`, or `BENCHMARK_REPRODUCTION_NO_GO`.

- [ ] **Step 1: Write RED manifest tests**

```python
def test_source_manifest_distinguishes_release_latest_and_used_artifact():
    payload = json.loads(MANIFEST.read_text())
    assert payload["rss_release"]["commit"] == RSS_RELEASE
    assert payload["latest_public"]["commit"]
    assert payload["artifact_evaluator_identity"]["status"] in {
        "PROVEN", "UNPROVEN_ARTIFACT_EVALUATOR_IDENTITY"
    }
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/evaluation/test_khronos_source_manifest.py`

Expected: FAIL because the source manifest does not exist.

- [ ] **Step 3: Audit and run only reproducible paper-like inputs**

Verify the paper's 8 cm, 5 m, GT-semantics, GT-pose condition from primary
sources. If exact paper assets/config/evaluator are absent, record
`BLOCKED_PAPER_ASSET` or `PAPER_PROTOCOL_APPROX_PUBLIC`; do not relabel a
post-release run as exact. Bind every command and output source.

- [ ] **Step 4: Run GREEN**

Run: `pytest -q tests/evaluation/test_khronos_source_manifest.py && git diff --check`

Expected: PASS with the report carrying the frozen `0.02` tolerance and all four reference metrics.

- [ ] **Step 5: Commit**

```bash
git add configs/external/khronos_source_manifest.json \
  docs/superpowers/reports/2026-09-03-khronos-protocol-reproduction.md \
  docs/superpowers/experiments/2026-09-03-benchmark-alignment-decision-ledger.md \
  tests/evaluation/test_khronos_source_manifest.py
git commit -m "docs: close Khronos protocol identity audit"
```

### Task 4: Add Non-Interfering Khronos Exact Attribution

**Files:**
- Create: `external_patches/khronos_eval_exact_attribution.patch`
- Create: `src/evaluation/khronos_attribution.py`
- Create: `scripts/evaluation/audit_khronos_exact_attribution.py`
- Create: `tests/evaluation/test_khronos_exact_attribution.py`
- Modify: `configs/external/khronos_source_manifest.json`
- Modify: `docs/superpowers/experiments/2026-09-03-benchmark-alignment-decision-ledger.md`

**Interfaces:**
- Consumes: source-bound current public Khronos tree and one frozen `.4dmap` evaluation input.
- Produces: dynamic/object JSONL association sidecars and a non-interference receipt.

- [ ] **Step 1: Write RED tests for sidecar schema and byte identity**

```python
def test_exact_sidecar_conserves_metric_mass_and_has_node_identity(tmp_path):
    rows = read_exact_associations(write_sidecar_fixture(tmp_path))
    assert {row.status for row in rows} == {"TP", "FP", "FN"}
    assert all(row.map_name and row.query_time_ns >= row.trajectory_timestamp_ns for row in rows)
    assert summarize_status(rows) == {"TP": 3, "FP": 2, "FN": 1}

def test_noninterference_requires_byte_identical_official_csvs(tmp_path):
    assert_official_outputs_identical(UNPATCHED, PATCHED)
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/evaluation/test_khronos_exact_attribution.py`

Expected: FAIL because no parser or reviewed patch exists.

- [ ] **Step 3: Implement parser, validator, and C++ patch**

The patch serializes map name, query time, trajectory timestamp, predicted and
GT node IDs, distance, status, and metric type before aggregation. Default
official CSV code paths remain byte-identical. The Python validator rejects
duplicate event identities, impossible time order, mass mismatch, source
tampering, and official CSV differences.

- [ ] **Step 4: Apply/build/run patch and prove non-interference**

Run: `pytest -q tests/evaluation/test_khronos_exact_attribution.py tests/evaluation/test_crove_dyn_provenance.py`

Expected: PASS. A real replay receipt must show byte-identical unpatched and patched official CSV SHA-256 values before B5 is marked successful.

- [ ] **Step 5: Commit**

```bash
git add external_patches/khronos_eval_exact_attribution.patch \
  src/evaluation/khronos_attribution.py \
  scripts/evaluation/audit_khronos_exact_attribution.py \
  tests/evaluation/test_khronos_exact_attribution.py \
  configs/external/khronos_source_manifest.json \
  docs/superpowers/experiments/2026-09-03-benchmark-alignment-decision-ledger.md
git commit -m "eval: add Khronos exact attribution sidecar"
```

### Task 5: Add TESSE Current-Diagonal and Fairness Diagnostics

**Files:**
- Create: `src/evaluation/benchmark_alignment.py`
- Create: `src/evaluation/tesse_current_slice.py`
- Create: `src/evaluation/tesse_oracle_inputs.py`
- Create: `scripts/evaluation/evaluate_tesse_current_slice.py`
- Create: `scripts/evaluation/audit_tesse_input_fairness.py`
- Create: `scripts/evaluation/build_tesse_oracle_inputs.py`
- Create: `tests/evaluation/test_benchmark_alignment.py`
- Create: `tests/evaluation/test_tesse_current_slice.py`
- Create: `tests/evaluation/test_tesse_oracle_inputs.py`
- Create: `docs/superpowers/reports/2026-09-03-tesse-input-fairness.md`
- Create: `docs/superpowers/reports/2026-09-03-tesse-current-vs-full4d.md`
- Modify: `docs/superpowers/experiments/2026-09-03-benchmark-alignment-decision-ledger.md`

**Interfaces:**
- Consumes: exact attribution rows, map timestamps, frozen A6/static-anchor/P5/P6-C results, K0/K1/C0/C1 condition manifests.
- Produces: `TESSE_CURRENT_DIAGONAL`, frontend gap-closure classification, and rank comparison.

- [ ] **Step 1: Write RED tests for time domains and gap closure**

```python
def test_current_slice_requires_equal_robot_and_belief_time():
    result = evaluate_current_diagonal(rows=ROWS, robot_times=ROBOT_TIMES)
    assert all(row.robot_time_ns == row.query_time_ns for row in result.rows)

def test_frontend_classification_uses_majority_at_fifty_percent():
    assert classify_frontend_gap(K0, C0, C1).status == "FRONTEND_DOMINATED"

def test_single_favorable_cell_is_not_material_mismatch():
    assert not classify_rank_mismatch(OFFICIAL, CURRENT).material

def test_oracle_inputs_are_diagnostic_and_source_bound(tmp_path):
    manifest = build_oracle_input_manifest(ORACLE_FIXTURE, output=tmp_path / "oracle.json")
    assert manifest["condition"] == "ORACLE_DIAGNOSTIC"
    assert manifest["ranking_eligible"] is False
    assert manifest["source_bindings"]
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/evaluation/test_benchmark_alignment.py tests/evaluation/test_tesse_current_slice.py tests/evaluation/test_tesse_oracle_inputs.py`

Expected: FAIL because the diagnostic modules do not exist.

- [ ] **Step 3: Implement immutable diagnostic models**

Use explicit `robot_time_ns`, `belief_time_ns`, and `trajectory_timestamp_ns`.
Reject missing map-time bindings and future timestamps. Aggregate TP/FP/FN by
the same formulas as the paired evaluator. Fairness computes positive finite
gaps only and requires majority closure at `>=0.5`. Rank mismatch implements
the three preregistered signals without a weighted score. The oracle-input
builder validates simulator semantic/instance streams before constructing C1,
marks every artifact non-ranking, and fails closed when the source DB or
per-frame GT stream cannot prove RGB-D/timestamp alignment.

- [ ] **Step 4: Run frozen variants and publish reports**

Run: `pytest -q tests/evaluation/test_benchmark_alignment.py tests/evaluation/test_tesse_current_slice.py tests/evaluation/test_tesse_oracle_inputs.py tests/evaluation/test_evaluate_crove_ovimap_static_anchor.py`

Expected: PASS. Missing oracle inputs remain `INCONCLUSIVE_MISSING_CONDITION`, not zero.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/benchmark_alignment.py src/evaluation/tesse_current_slice.py \
  src/evaluation/tesse_oracle_inputs.py \
  scripts/evaluation/evaluate_tesse_current_slice.py \
  scripts/evaluation/audit_tesse_input_fairness.py \
  scripts/evaluation/build_tesse_oracle_inputs.py \
  tests/evaluation/test_benchmark_alignment.py \
  tests/evaluation/test_tesse_current_slice.py \
  tests/evaluation/test_tesse_oracle_inputs.py \
  docs/superpowers/reports/2026-09-03-tesse-input-fairness.md \
  docs/superpowers/reports/2026-09-03-tesse-current-vs-full4d.md \
  docs/superpowers/experiments/2026-09-03-benchmark-alignment-decision-ledger.md
git commit -m "eval: add TESSE current-state alignment diagnostics"
```

### Task 6: Add the Causal 3RScan Pilot Adapter and Exact-ID Metrics

**Files:**
- Create: `src/evaluation/datasets/rscan.py`
- Create: `src/evaluation/rscan_temporal.py`
- Create: `scripts/evaluation/evaluate_3rscan_temporal.py`
- Create: `configs/evaluation/manifests/3rscan_causal_pilot_v1.json`
- Create: `tests/evaluation/test_rscan_dataset.py`
- Create: `tests/evaluation/test_rscan_temporal.py`
- Create: `docs/superpowers/reports/2026-09-03-3rscan-pilot.md`
- Modify: `docs/superpowers/experiments/2026-09-03-benchmark-alignment-decision-ledger.md`

**Interfaces:**
- Consumes: official `3RScan.json`, validation scan list, per-scan assets, and session-prefix predictions.
- Produces: immutable environment/session/change inventory, frozen 10-environment manifest, community metric inputs, and exact-ID diagnostics.

- [ ] **Step 1: Write RED adapter and metric tests**

```python
def test_selection_is_result_independent_and_sorted(tmp_path):
    selected = select_pilot(metadata_fixture(tmp_path), validation_fixture(tmp_path), count=10)
    assert len(selected) == 10
    assert [item.reference_id for item in selected] == sorted(item.reference_id for item in selected)

def test_session_prefix_rejects_future_predictions():
    with pytest.raises(RScanProtocolError, match="future session"):
        evaluate_temporal_identity(GT, predictions_for_sessions=(0, 1, 2), evaluation_session=1)

def test_exact_identity_metrics_count_switch_reid_and_change_types():
    metrics = evaluate_temporal_identity(GT_FIXTURE, PREDICTION_FIXTURE, evaluation_session=2)
    assert metrics.id_switches == 1
    assert metrics.false_reid == 1
    assert metrics.rigid_recall == pytest.approx(0.5)
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/evaluation/test_rscan_dataset.py tests/evaluation/test_rscan_temporal.py`

Expected: FAIL because the 3RScan modules do not exist.

- [ ] **Step 3: Implement validated immutable records and metrics**

Parse fixed IDs and rigid/nonrigid/removed metadata, derive added IDs from
session-vs-reference annotations only in the evaluator, hash every required
asset, preserve session order, and expose no GT transform through method input.
Keep t-AP/t-REC compatibility outputs distinct from custom exact-ID fields.

- [ ] **Step 4: Freeze selection, run available pilot stages, report missing assets**

Run: `pytest -q tests/evaluation/test_rscan_dataset.py tests/evaluation/test_rscan_temporal.py`

Expected: PASS. The real report contains measured values only for completed source-bound runs and `BLOCKED_DATASET_ACCESS` for unavailable RGB-D/runtime stages.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/datasets/rscan.py src/evaluation/rscan_temporal.py \
  scripts/evaluation/evaluate_3rscan_temporal.py \
  configs/evaluation/manifests/3rscan_causal_pilot_v1.json \
  tests/evaluation/test_rscan_dataset.py tests/evaluation/test_rscan_temporal.py \
  docs/superpowers/reports/2026-09-03-3rscan-pilot.md \
  docs/superpowers/experiments/2026-09-03-benchmark-alignment-decision-ledger.md
git commit -m "data: add causal 3RScan pilot evaluation"
```

### Task 7: Add the Panoptic Mapping Flat Pilot Adapter and Metrics

**Files:**
- Create: `src/evaluation/datasets/panoptic_flat.py`
- Create: `src/evaluation/panoptic_flat_current.py`
- Create: `scripts/evaluation/evaluate_panoptic_flat_current.py`
- Create: `tests/evaluation/test_panoptic_flat_dataset.py`
- Create: `tests/evaluation/test_panoptic_flat_current.py`
- Create: `docs/superpowers/reports/2026-09-03-panoptic-flat-pilot.md`
- Modify: `docs/superpowers/experiments/2026-09-03-benchmark-alignment-decision-ledger.md`

**Interfaces:**
- Consumes: run1/run2 RGB, depth, GT/predicted panoptic labels, poses, timestamps, change log, structural GT, and causal predictions.
- Produces: separate GT-panoptic oracle and predicted-panoptic non-oracle summaries.

- [ ] **Step 1: Write RED adapter and metric tests**

```python
def test_flat_adapter_requires_run1_before_run2(tmp_path):
    dataset = load_flat_dataset(flat_fixture(tmp_path))
    assert [frame.run_id for frame in dataset.frames] == ["run1", "run1", "run2", "run2"]

def test_flat_conditions_cannot_be_aggregated_together():
    with pytest.raises(FlatProtocolError, match="input condition"):
        aggregate_flat_results([GT_RESULT, PREDICTED_RESULT])

def test_current_metrics_count_stale_and_recovery():
    result = evaluate_flat_current(GT_FIXTURE, PREDICTION_FIXTURE)
    assert result.stale_geometry_fp == 2
    assert result.recovery_latency_frames == 1
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/evaluation/test_panoptic_flat_dataset.py tests/evaluation/test_panoptic_flat_current.py`

Expected: FAIL because the Flat modules do not exist.

- [ ] **Step 3: Implement strict adapter and current-state metrics**

Validate image pairing, depth type, labels, finite 4x4 poses, strictly ordered
timestamps, change types, and source hashes. Keep `Flat-GT-Panoptic` marked
oracle and prevent cross-condition aggregation. Evaluate only causal prefixes.

- [ ] **Step 4: Run available pilot stages and publish report**

Run: `pytest -q tests/evaluation/test_panoptic_flat_dataset.py tests/evaluation/test_panoptic_flat_current.py`

Expected: PASS. If official Flat assets are absent, the real-data status is `BLOCKED_DATASET_ACCESS` while synthetic protocol validation remains PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/datasets/panoptic_flat.py \
  src/evaluation/panoptic_flat_current.py \
  scripts/evaluation/evaluate_panoptic_flat_current.py \
  tests/evaluation/test_panoptic_flat_dataset.py \
  tests/evaluation/test_panoptic_flat_current.py \
  docs/superpowers/reports/2026-09-03-panoptic-flat-pilot.md \
  docs/superpowers/experiments/2026-09-03-benchmark-alignment-decision-ledger.md
git commit -m "data: add Panoptic Flat current-state pilot"
```

### Task 8: Freeze B9 Decision, Handoff, and Paper Benchmark Layout

**Files:**
- Create: `docs/superpowers/reports/2026-09-03-benchmark-decision.md`
- Create: `docs/superpowers/reports/2026-09-03-benchmark-alignment-handoff.md`
- Modify: `docs/superpowers/experiments/2026-09-03-benchmark-alignment-decision-ledger.md`
- Test: `tests/evaluation/test_benchmark_alignment_reports.py`

**Interfaces:**
- Consumes: frozen B0-B8 reports, manifests, scorecard, and measured artifacts.
- Produces: exactly one allowed decision, a transparent Pareto report, all 15 handoff answers, and a paper table recommendation.

- [ ] **Step 1: Write RED report-completeness tests**

```python
def test_handoff_answers_all_fifteen_questions_and_one_decision():
    text = HANDOFF.read_text()
    assert all(f"{index}." in text for index in range(1, 16))
    chosen = [value for value in ALLOWED_DECISIONS if value in DECISION.read_text()]
    assert len(chosen) == 1
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/evaluation/test_benchmark_alignment_reports.py`

Expected: FAIL because B9 reports do not exist.

- [ ] **Step 3: Apply the preregistered decision tree**

Do not alter B1 suitability scores. Report each metric and unavailable cell,
classify root causes, separate diagnostic from paper-eligible claims, list
Office/held-out work, and recommend the main/supplement table hierarchy only
when supported by the frozen evidence.

- [ ] **Step 4: Run all required regression gates**

```bash
pytest -q tests/evaluation/test_crove_dyn_provenance.py \
  tests/evaluation/test_crove_failure_attribution.py \
  tests/evaluation/test_crove_runtime_attribution.py \
  tests/evaluation/test_prepare_temporal_khronos_bridge.py \
  tests/evaluation/test_export_tesse_temporal_artifact.py \
  tests/evaluation/test_run_crove_ovimap_static_anchor.py
python scripts/evaluation/verify_oviv2_dual_readout_development_gates.py \
  --verify-source-manifest configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json
git diff --check
python -m compileall -q src scripts tests
pytest -q
```

Expected: all applicable checks pass; external-data tests skip only through explicit missing-asset guards.

- [ ] **Step 5: Commit and push**

```bash
git add docs/superpowers/reports/2026-09-03-benchmark-decision.md \
  docs/superpowers/reports/2026-09-03-benchmark-alignment-handoff.md \
  docs/superpowers/experiments/2026-09-03-benchmark-alignment-decision-ledger.md \
  tests/evaluation/test_benchmark_alignment_reports.py
git commit -m "docs: close CROVE benchmark alignment audit"
git push -u origin research/crove-benchmark-alignment-audit
LOCAL_SHA=$(git rev-parse HEAD)
REMOTE_SHA=$(git ls-remote --heads origin research/crove-benchmark-alignment-audit | awk '{print $1}')
test "$LOCAL_SHA" = "$REMOTE_SHA"
```

Expected: `UPLOAD_STATUS = VERIFIED` only after local and remote SHA values are identical.
