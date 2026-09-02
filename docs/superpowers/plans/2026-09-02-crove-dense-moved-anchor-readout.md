# CROVE Dense Moved-Anchor Readout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit P4A readout mode that translates immutable dense OVI-MAP object templates to CROVE's causal current centroids, then measure it separately from the frozen compact baseline.

**Architecture:** Keep `compose_anchor_checkpoint()` as the single overlay authority and add a default-preserving moved-geometry selector. The composition runner validates and records whether an output is a formal compact baseline, visualization shadow, or evaluation candidate. A separate read-only audit compares compact and dense moved geometry without modifying either run.

**Tech Stack:** Python 3.13, NumPy, SciPy `cKDTree`, pytest, existing `MapSnapshot` and OVI-MAP anchor composition APIs.

---

### Task 1: Dense moved-template geometry in the composition core

**Files:**
- Modify: `tests/oviv2/test_ovimap_static_anchor.py`
- Modify: `src/oviv2/ovimap_static_anchor.py`

- [ ] **Step 1: Write the failing translation and default-preservation tests**

Add focused tests that call `compose_anchor_checkpoint()` with a moved bound
entity. The candidate assertion must be equivalent to:

```python
dense, _, diagnostics = compose_anchor_checkpoint(
    anchor=anchor,
    temporal=temporal,
    exports=(moved_batch,),
    state=state,
    config=config,
    anchor_manifest_sha256="a" * 64,
    class_names=("Chair",),
    moved_geometry_mode="anchor_centroid_translation",
)
entity = dense.entities[0]
delta = np.asarray(moved_batch.samples[0].centroid_xyz) - anchor.entities[0].points_xyz.mean(0)
np.testing.assert_allclose(entity.points_xyz, anchor.entities[0].points_xyz + delta)
assert entity.semantic_label == temporal.entities[0].semantic_label
assert entity.lifecycle_state == temporal.entities[0].lifecycle.lifecycle.value
assert entity.metadata["geometry_authority"] == "ovimap_anchor_template"
assert entity.metadata["state_authority"] == "crove_temporal"
assert entity.metadata["template_anchor_id"] == anchor.entities[0].entity_id
assert entity.metadata["transform_source"] == "current_export_centroid_translation"
assert entity.metadata["readout_resolution_m"] is None
assert diagnostics.moved_anchor_ids == (anchor.entities[0].entity_id,)
```

Also compare a call with no new argument against an explicit
`moved_geometry_mode="temporal_compact"` call and require identical positions,
fields, metadata, state, and diagnostics. Add one test where no interval sample
exists and the candidate uses the temporal prediction centroid, plus invalid
mode and non-finite centroid rejection tests.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest -q tests/oviv2/test_ovimap_static_anchor.py \
  -k 'dense_moved or moved_geometry_mode'
```

Expected: fail because `compose_anchor_checkpoint()` does not accept
`moved_geometry_mode` and does not emit dense template metadata.

- [ ] **Step 3: Implement the minimal readout selector**

Add these exact modes and validation:

```python
_MOVED_GEOMETRY_MODES = frozenset(
    {"temporal_compact", "anchor_centroid_translation"}
)

def _moved_geometry_mode(value: object) -> str:
    if not isinstance(value, str) or value not in _MOVED_GEOMETRY_MODES:
        raise ValueError("moved_geometry_mode is invalid")
    return value
```

Extend `_copy_prediction()` with an optional `points_xyz` override. Add a
private helper that computes a finite float64 anchor centroid, chooses the
latest export centroid or temporal geometry centroid, applies one translation
to all anchor points, converts once to float32, and returns the temporal
prediction with the required provenance metadata. Add the optional keyword to
`compose_anchor_checkpoint()`:

```python
moved_geometry_mode: str = "temporal_compact"
```

Validate it before processing. In the existing moved branch, keep the compact
path byte-for-byte equivalent; invoke the helper only for
`anchor_centroid_translation`.

- [ ] **Step 4: Run focused and module tests and verify GREEN**

Run:

```bash
python -m pytest -q tests/oviv2/test_ovimap_static_anchor.py
python -m ruff check src/oviv2/ovimap_static_anchor.py \
  tests/oviv2/test_ovimap_static_anchor.py
git diff --check
```

Expected: all pass with no lint or whitespace errors.

- [ ] **Step 5: Commit the core behavior**

```bash
git add src/oviv2/ovimap_static_anchor.py \
  tests/oviv2/test_ovimap_static_anchor.py
git commit -m "feat: preserve dense anchors in moved readout"
```

### Task 2: Explicit runner role and candidate publication

**Files:**
- Modify: `tests/evaluation/test_run_crove_ovimap_static_anchor.py`
- Modify: `scripts/evaluation/run_crove_ovimap_static_anchor.py`

- [ ] **Step 1: Write failing runner-contract tests**

Extend the existing hermetic package fixture so the anchor has more points than
the temporal moved entity. Add tests equivalent to:

```python
result = compose_run(
    source_run_manifest=source,
    anchor_manifest=anchor,
    output_root=output,
    moved_geometry_mode="anchor_centroid_translation",
    readout_role="visualization_shadow",
)
manifest = json.loads(result.read_text())
assert manifest["readout_contract"] == {
    "moved_geometry_mode": "anchor_centroid_translation",
    "readout_role": "visualization_shadow",
}
```

Read the final checkpoint through `read_map_snapshot()` and verify its moved
entity point count equals the anchor point count and its semantic/lifecycle
fields equal the compact source. Add parameterized rejection tests for:

```text
temporal_compact + visualization_shadow
temporal_compact + evaluation_candidate
anchor_centroid_translation + formal_baseline
unknown mode or role
```

Each rejected call must leave the requested output path absent. Preserve the
existing default fixture test and assert it records
`temporal_compact + formal_baseline`.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest -q tests/evaluation/test_run_crove_ovimap_static_anchor.py \
  -k 'readout or dense or publishes_hash_bound'
```

Expected: fail because `compose_run()` and the CLI do not accept the new
contract and the manifest lacks `readout_contract`.

- [ ] **Step 3: Implement mode/role validation and propagation**

Add:

```python
_READOUT_ROLES = frozenset(
    {"formal_baseline", "visualization_shadow", "evaluation_candidate"}
)

def _readout_contract(mode: object, role: object) -> dict[str, str]:
    allowed = {
        ("temporal_compact", "formal_baseline"),
        ("anchor_centroid_translation", "visualization_shadow"),
        ("anchor_centroid_translation", "evaluation_candidate"),
    }
    if not isinstance(mode, str) or not isinstance(role, str) or (mode, role) not in allowed:
        raise ValueError("moved geometry mode and readout role are incompatible")
    return {"moved_geometry_mode": mode, "readout_role": role}
```

Extend `compose_run()` with default arguments, validate before any output
directory is created, pass the mode to every checkpoint composition, and add
the returned mapping to the manifest. Add CLI choices:

```text
--moved-geometry-mode temporal_compact|anchor_centroid_translation
--readout-role formal_baseline|visualization_shadow|evaluation_candidate
```

- [ ] **Step 4: Run runner and protected tests and verify GREEN**

Run:

```bash
python -m pytest -q tests/evaluation/test_run_crove_ovimap_static_anchor.py \
  tests/oviv2/test_ovimap_static_anchor.py
python -m pytest -q tests/oviv2/test_t1_exactness.py \
  tests/oviv2/test_t1_noninterference.py
python -m ruff check scripts/evaluation/run_crove_ovimap_static_anchor.py \
  tests/evaluation/test_run_crove_ovimap_static_anchor.py
git diff --check
```

Expected: all pass; T1 protected behavior remains unchanged.

- [ ] **Step 5: Commit the runner contract**

```bash
git add scripts/evaluation/run_crove_ovimap_static_anchor.py \
  tests/evaluation/test_run_crove_ovimap_static_anchor.py
git commit -m "feat: publish dense moved-readout candidates"
```

### Task 3: Read-only compact-versus-dense audit

**Files:**
- Create: `src/evaluation/crove_dense_readout_audit.py`
- Create: `scripts/evaluation/audit_crove_dense_readout.py`
- Create: `tests/evaluation/test_crove_dense_readout_audit.py`

- [ ] **Step 1: Write failing geometry-audit tests**

Create compact, dense, and anchor `MapSnapshot` fixtures with one moved entity.
The public function contract is:

```python
audit_dense_moved_entities(
    anchor: MapSnapshot,
    compact: MapSnapshot,
    dense: MapSnapshot,
    moved_anchor_ids: tuple[str, ...],
    *,
    distance_threshold_m: float = 0.05,
) -> dict[str, object]
```

Assert per entity:

```text
anchor_point_count
compact_point_count
dense_point_count
dense_to_compact_nn_median_m
dense_to_compact_nn_p90_m
dense_to_compact_coverage_at_threshold
anchor_bbox_min_xyz / anchor_bbox_max_xyz
dense_bbox_min_xyz / dense_bbox_max_xyz
translation_xyz
rigid_translation_residual_max_m
state_fields_equal
```

Require an exact constant translation residual no greater than `1e-6`, dense
point count equal to anchor point count, matching entity IDs, and matching
semantic/lifecycle state between compact and dense. Add rejection tests for a
non-translation deformation, missing moved ID, changed semantic state, wrong
metadata authority, and nonpositive threshold.

- [ ] **Step 2: Run the audit tests and verify RED**

Run:

```bash
python -m pytest -q tests/evaluation/test_crove_dense_readout_audit.py
```

Expected: collection fails because the audit module does not exist.

- [ ] **Step 3: Implement the minimal audit**

Use sorted entity-ID mappings, NumPy vector operations for translation and
bounds, and `scipy.spatial.cKDTree(compact_points).query(dense_points, workers=1)`
for deterministic nearest-neighbor distances. Return canonical Python values;
do not mutate or publish snapshots in this module.

The CLI reads bound anchor/compact/dense checkpoints selected by frame, invokes
the function, adds source file SHA-256/byte-count records and serialized
snapshot sizes, and atomically writes canonical JSON plus Markdown. It rejects
Office and refuses overwrite.

- [ ] **Step 4: Run focused validation and verify GREEN**

Run:

```bash
python -m pytest -q tests/evaluation/test_crove_dense_readout_audit.py
python -m ruff check src/evaluation/crove_dense_readout_audit.py \
  scripts/evaluation/audit_crove_dense_readout.py \
  tests/evaluation/test_crove_dense_readout_audit.py
python -m py_compile src/evaluation/crove_dense_readout_audit.py \
  scripts/evaluation/audit_crove_dense_readout.py
git diff --check
```

Expected: all pass.

- [ ] **Step 5: Commit the audit tooling**

```bash
git add src/evaluation/crove_dense_readout_audit.py \
  scripts/evaluation/audit_crove_dense_readout.py \
  tests/evaluation/test_crove_dense_readout_audit.py
git commit -m "feat: audit dense moved-anchor readout"
```

### Task 4: Apartment shadow experiment and phase decision

**Files:**
- Create: `docs/superpowers/reports/P4A_DENSE_MOVED_READOUT_REPORT.md`
- Modify: `docs/superpowers/experiments/2026-09-02-crove-dynamic-recovery-decision-ledger.md`

- [ ] **Step 1: Verify the runner CLI before the real run**

```bash
python scripts/evaluation/run_crove_ovimap_static_anchor.py --help
python scripts/evaluation/audit_crove_dense_readout.py --help
```

Expected: both commands exit 0 and show the explicit mode/role or audit inputs.

- [ ] **Step 2: Publish the Apartment visualization shadow**

```bash
python scripts/evaluation/run_crove_ovimap_static_anchor.py \
  --source-run-manifest /home/ww/oviovo_baseline_runs/20260901_crove_ovimap_anchor/crove_a6_c069510_source/run_manifest.json \
  --anchor-manifest /home/ww/oviovo_baseline_runs/20260901_crove_ovimap_anchor/anchor/apartment/anchor_manifest.json \
  --output-root /home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p4a/apartment_shadow \
  --moved-geometry-mode anchor_centroid_translation \
  --readout-role visualization_shadow
```

Expected: PASS manifest; Office is not read; the existing compact run is not
modified.

- [ ] **Step 3: Audit final and representative moved checkpoints**

Run the audit at frame 1472 and at the first checkpoint where each of
`ovimap:2`, `ovimap:6`, `ovimap:85`, `ovimap:131`, and `ovimap:179` is emitted
as moved. Write results below
`/home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p4a/audit/`.

Expected: dense point count equals anchor point count, translation residual is
at most `1e-6`, all state fields are equal, and no compact artifact changes.

- [ ] **Step 4: Write the P4A report and decision ledger entry**

Record exact source hashes, per-entity point counts, nearest-neighbor median and
p90, 5 cm coverage, bounding boxes, and serialized sizes. Mark the output
`visualization-only`; do not claim official metric improvement. Authorize a
separate evaluation candidate only if all shadow invariants pass.

- [ ] **Step 5: Run the phase protection suite**

```bash
python -m pytest -q tests/oviv2/test_temporal_association.py \
  tests/oviv2/test_temporal_identity.py \
  tests/oviv2/test_temporal_runtime.py \
  tests/oviv2/test_ovimap_static_anchor.py \
  tests/evaluation/test_run_crove_ovimap_static_anchor.py \
  tests/oviv2/test_t1_exactness.py \
  tests/oviv2/test_t1_noninterference.py
git diff --check
```

Expected: all pass.

- [ ] **Step 6: Commit the phase evidence**

```bash
git add docs/superpowers/reports/P4A_DENSE_MOVED_READOUT_REPORT.md \
  docs/superpowers/experiments/2026-09-02-crove-dynamic-recovery-decision-ledger.md
git commit -m "docs: record dense moved-readout shadow gate"
```
