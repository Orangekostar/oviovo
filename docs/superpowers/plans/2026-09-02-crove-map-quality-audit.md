# CROVE Map-Quality Audit Implementation Plan

> **Execution:** implement task by task with test-driven development. This plan
> covers P0 only; later phases require the P0 Decision Ledger result.

**Goal:** Produce a deterministic, source-bound audit proving the exact native
PLY producer and quantifying dense-anchor to sparse-temporal geometry authority
transitions without changing benchmark behavior.

**Architecture:** One standalone evaluation script reads the native, anchor,
and composed manifests; validates their declared file records; streams the
native ASCII PLY vertex section to classify instance-palette colors; binds the
producer source files from the recorded native command; compares the final
composed checkpoint entities with their anchor counterparts; and atomically
writes canonical JSON and Markdown. The tool imports no runtime mapper and
never writes inside input artifact roots.

**Tech Stack:** Python 3.10+, standard library, NumPy only for existing NPZ
contracts if needed, pytest.

---

## File Map

- Create `scripts/evaluation/audit_crove_map_quality.py`: P0 validators,
  streaming PLY audit, geometry-authority comparison, CLI, JSON/Markdown output.
- Create `tests/evaluation/test_audit_crove_map_quality.py`: hermetic contract,
  tamper, palette, authority, CLI, and determinism tests.
- Create `docs/superpowers/reports/2026-09-02-crove-map-quality-audit.md`: only
  after the real Apartment audit passes.
- Create `docs/superpowers/experiments/2026-09-02-crove-dynamic-recovery-decision-ledger.md`:
  phase decision with exact artifact hashes and GO/NO-GO.

Do not modify runtime code, configs, evaluators, existing reports, or T1
protected files in P0.

### Task 1: Freeze input and file-record contracts

**Files:**
- Create: `tests/evaluation/test_audit_crove_map_quality.py`
- Create: `scripts/evaluation/audit_crove_map_quality.py`

- [ ] Write failing tests for `_verified_file_record` and manifest loading.
  Require a mapping with exact `path`, lowercase 64-character `sha256`, and
  non-negative integer `byte_count`. Resolve only explicitly relative records
  against their owning manifest directory. Reject symlinks, missing/non-regular
  files, path escape, byte-count mismatch, hash mismatch, malformed JSON, and
  non-PASS native/anchor/composed manifests.
- [ ] Run:

```bash
python -m pytest -q tests/evaluation/test_audit_crove_map_quality.py
```

  Expected RED: module import fails.
- [ ] Implement canonical JSON loading, streaming SHA-256, file-record
  validation, and atomic output helpers. Public functions must accept `Path`
  objects and must not mutate caller mappings.
- [ ] Re-run the focused tests and `python -m py_compile`.

### Task 2: Stream and classify the native PLY

**Files:**
- Modify: `tests/evaluation/test_audit_crove_map_quality.py`
- Modify: `scripts/evaluation/audit_crove_map_quality.py`

- [ ] Add small ASCII PLY fixtures with scalar vertex properties
  `x y z red green blue`, an optional face element, and an authoritative
  instance-color log. Test:
  - exact header format, vertex/face count, and property capture;
  - palette color to instance-ID binding;
  - unknown/background color counts;
  - duplicate instance colors and conflicting log entries fail closed;
  - RGB values outside `[0,255]`, truncated vertices, or unsupported list
    properties on the vertex element fail closed;
  - reordering records does not change canonical summary order.
- [ ] Verify RED for the missing parser.
- [ ] Implement a bounded-memory ASCII PLY vertex reader. It must read exactly
  the declared vertex records, parse properties by header index, count RGB
  triples, and avoid loading faces or all vertices. Report
  `visualization_mode="instance_palette"` only when registered palette colors
  are unique and present. Otherwise report a precise failure, never infer RGB
  texture from appearance.
- [ ] Run focused tests and a synthetic CLI smoke test.

### Task 3: Bind the exact producer chain

**Files:**
- Modify: `tests/evaluation/test_audit_crove_map_quality.py`
- Modify: `scripts/evaluation/audit_crove_map_quality.py`

- [ ] Construct a native manifest fixture whose mapping command points to a
  synthetic OVI-MAP root. Add exact producer source fixtures containing the
  required calls and enum markers.
- [ ] Test that the audit records path, SHA-256, and byte count for:
  - `scripts/panoptic_mapping_.py`;
  - `consistent_gsm/src/global_segment_map_py.cpp`;
  - `global_segment_map/src/meshing/label_tsdf_mesh_integrator.cc`.
- [ ] Test rejection when the command entrypoint lies outside its recorded
  mapping cwd, a source is missing, or the required
  `generateMesh(...False, False, True)`, `kInstance`, or
  `instance_color_map_` marker is absent.
- [ ] Implement producer-root derivation from `commands.mapping.cwd` and bind
  all three files. Record the native manifest's pinned OVI-MAP commit separately
  from the current file hashes because the native environment may contain
  declared compatibility patches.

### Task 4: Quantify geometry authority transitions

**Files:**
- Modify: `tests/evaluation/test_audit_crove_map_quality.py`
- Modify: `scripts/evaluation/audit_crove_map_quality.py`

- [ ] Add anchor and composed fixtures with one unchanged anchor, two moved
  anchors, one removed anchor, and two new temporal entities. The composed
  manifest must bind the final checkpoint entity JSONL and diagnostics.
- [ ] Test exact aggregates:
  - anchor/current point counts by authority and overlay state;
  - moved-anchor before/after point counts and retained-point ratios;
  - new temporal count, total, minimum, median, and maximum;
  - missing moved replacement, duplicate entity ID, inconsistent authority,
    mismatched anchor ID, and diagnostics/metadata disagreement fail closed.
- [ ] Implement strict JSONL parsing and deterministic comparison. Use entity
  IDs, `metadata.authority`, `metadata.overlay_state`, and
  `metadata.anchor_entity_id` as separate fields; do not infer authority from
  semantic labels or colors.
- [ ] Label the dense-to-sparse conclusion `MEASURED_EVIDENCE` only when all
  source bindings pass.

### Task 5: Add CLI and deterministic reports

**Files:**
- Modify: `tests/evaluation/test_audit_crove_map_quality.py`
- Modify: `scripts/evaluation/audit_crove_map_quality.py`

- [ ] Test exact CLI arguments:

```text
--native-manifest PATH
--anchor-manifest PATH
--composed-manifest PATH
--output-json PATH
--output-markdown PATH
```

- [ ] Require output paths outside all input artifact roots. Write JSON with
  `sort_keys=True`, compact separators, `allow_nan=False`, and a final newline.
  Write a concise Markdown table from the same in-memory result. No timestamps
  are permitted, so identical inputs produce byte-identical outputs.
- [ ] Include schema version, status, evidence labels, all input/output source
  records, producer chain, PLY classification, point aggregates, per-moved-ID
  comparison, and conclusion. Never include evaluator metrics in P0.
- [ ] Run:

```bash
python -m pytest -q tests/evaluation/test_audit_crove_map_quality.py
python -m py_compile scripts/evaluation/audit_crove_map_quality.py
git diff --check
```

### Task 6: Run the real Apartment audit and gate P0

**Files:**
- Create: `docs/superpowers/reports/2026-09-02-crove-map-quality-audit.md`
- Create: `docs/superpowers/experiments/2026-09-02-crove-dynamic-recovery-decision-ledger.md`

- [ ] Run the CLI on:

```bash
python scripts/evaluation/audit_crove_map_quality.py \
  --native-manifest /home/ww/oviovo_baseline_runs/20260901_crove_ovimap_anchor/apartment_prefix_000262/native_mapping_manifest.json \
  --anchor-manifest /home/ww/oviovo_baseline_runs/20260901_crove_ovimap_anchor/anchor/apartment/anchor_manifest.json \
  --composed-manifest /home/ww/oviovo_baseline_runs/20260901_crove_ovimap_anchor/composed_selective/apartment/run_manifest.json \
  --output-json /home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p0/map_quality_audit.json \
  --output-markdown /home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p0/map_quality_audit.md
```

- [ ] Re-run to a second output pair and compare SHA-256 hashes for deterministic
  equality.
- [ ] Record measured values and exact source/output hashes in the repository
  report. Update the Decision Ledger:
  - GO if all bindings pass, the PLY producer is exact, colors are proven to be
    instance palette IDs, and moved/new sparsity is quantified;
  - NO-GO otherwise, with later phases blocked.
- [ ] Run the frozen T1/core regression set and the protected-source verifier.
- [ ] Inspect the full diff and commit only P0 code, tests, design/plan, report,
  and ledger. Do not add run outputs or large PLY/NPZ artifacts.

### Task 7: Review checkpoint

- [ ] Review the complete P0 diff for path handling, hash validation, memory
  bounds, determinism, causality language, and evidence-label correctness.
- [ ] Confirm `git status --short` contains no generated assets.
- [ ] If P0 is GO, write a separate P1/P2 plan based on its measured report.
  P3-P6 remain blocked until P2 identifies a dominant failure contributor.

