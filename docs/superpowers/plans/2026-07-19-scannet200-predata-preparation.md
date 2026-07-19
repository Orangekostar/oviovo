# ScanNet200 Pre-Data Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify all ScanNet200 data-validation, manifest-freezing, asset-preflight, and runner-spec infrastructure that is legal and useful before publisher-approved data arrives.

**Architecture:** A pure data validator inspects an official ScanNet scene root, enforces containment and required files, and produces immutable scene inventory records. A manifest freezer consumes only complete validated inventories. Runner specifications reference the frozen manifest and method assets but cannot execute or emit results while the dataset status is blocked.

**Tech Stack:** Python 3.10 standard library, dataclasses, pathlib, hashlib, JSON, pytest, existing manifest/result conventions.

---

### Task 1: Define ScanNet200 Scene Validation Contracts

**Files:**
- Create: `src/evaluation/datasets/__init__.py`
- Create: `src/evaluation/datasets/scannet200.py`
- Create: `tests/evaluation/test_scannet200_dataset.py`

- [ ] **Step 1: Write failing valid-scene tests**

Create a synthetic scene named `scene0011_00` with:

```text
scene0011_00.sens
scene0011_00.txt
scene0011_00_vh_clean_2.ply
scene0011_00_vh_clean_2.labels.ply
scene0011_00_vh_clean_2.0.010000.segs.json
scene0011_00.aggregation.json
```

Desired API:

```python
inventory = validate_scannet200_scene(root, "scene0011_00")
assert inventory.scene_id == "scene0011_00"
assert set(inventory.files) == {
    "sensor",
    "metadata",
    "mesh",
    "labels_mesh",
    "segments",
    "aggregation",
}
assert all(record.sha256 for record in inventory.files.values())
```

- [ ] **Step 2: Write failure tests**

Cover missing file, empty file, invalid scene ID, a symlink escaping `dataset_root`, a directory in place of a file, and duplicate resolved paths.

- [ ] **Step 3: Run and verify RED**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_scannet200_dataset.py -q
```

Expected: module import failure.

- [ ] **Step 4: Implement immutable inventory types**

```python
@dataclass(frozen=True)
class ScanNetFileRecord:
    role: str
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ScanNetSceneInventory:
    scene_id: str
    dataset_root: str
    files: Mapping[str, ScanNetFileRecord]
```

Normalize the mapping to a read-only proxy or tuple-backed deterministic representation before returning it.

- [ ] **Step 5: Implement strict validation**

Scene IDs must match `^scene[0-9]{4}_[0-9]{2}$`. Resolve every path and require `resolved_path.is_relative_to(resolved_root)`. Hash files in 1 MiB chunks. Reject missing and zero-byte files.

- [ ] **Step 6: Run tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_scannet200_dataset.py -q
git add src/evaluation/datasets/__init__.py \
  src/evaluation/datasets/scannet200.py \
  tests/evaluation/test_scannet200_dataset.py
git commit -m "feat: validate official ScanNet200 scenes"
```

---

### Task 2: Validate the Frozen Five-Scene Candidate Set

**Files:**
- Modify: `src/evaluation/datasets/scannet200.py`
- Modify: `tests/evaluation/test_scannet200_dataset.py`

- [ ] **Step 1: Add failing set-level tests**

Desired API:

```python
SCANNET200_5_SCENES = (
    "scene0011_00",
    "scene0050_00",
    "scene0231_00",
    "scene0378_00",
    "scene0518_00",
)

inventories = validate_scannet200_split(root, SCANNET200_5_SCENES)
assert tuple(item.scene_id for item in inventories) == SCANNET200_5_SCENES
```

Reject duplicate scenes, missing scenes, reordering by filesystem traversal, and any scene outside the declared set.

- [ ] **Step 2: Verify RED**

Expected: missing constant/function.

- [ ] **Step 3: Implement deterministic split validation**

Iterate in caller order and return a tuple. Aggregate failures into one `ScanNetValidationError` whose message lists every failed scene and role, so one preflight identifies all missing data.

- [ ] **Step 4: Run tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_scannet200_dataset.py -q
git add src/evaluation/datasets/scannet200.py tests/evaluation/test_scannet200_dataset.py
git commit -m "feat: validate the ScanNet200 five-scene split"
```

---

### Task 3: Freeze a Manifest Only From Validated Data

**Files:**
- Create: `scripts/evaluation/freeze_scannet200_manifest.py`
- Create: `tests/evaluation/test_freeze_scannet200_manifest.py`

- [ ] **Step 1: Write a failing deterministic-output test**

Run the desired CLI twice against the synthetic five-scene fixture and assert byte-identical JSON containing:

```json
{
  "schema_version": 1,
  "manifest_id": "scannet200_5_static_v1",
  "dataset": "ScanNet200",
  "split": "scannet200_5",
  "vocabulary": {
    "name": "scannet200",
    "source_path": "...",
    "source_sha256": "..."
  },
  "scenes": []
}
```

Each scene contains the inventory records, pose source, depth source, split role, and file hashes.

- [ ] **Step 2: Write refusal tests**

Assert the CLI leaves no output for incomplete scenes, missing vocabulary metadata, wrong source hash, or an existing output without `--replace`.

- [ ] **Step 3: Run and verify RED**

Expected: CLI missing.

- [ ] **Step 4: Implement atomic manifest writing**

The CLI arguments are:

```text
--dataset-root
--vocabulary
--vocabulary-sha256
--output
--replace
```

Validate all five scenes first. Write sorted indented JSON to a sibling temporary file, fsync, then `os.replace`. `frozen_at` is supplied through `SOURCE_DATE_EPOCH` or an explicit `--frozen-at` argument so repeat tests remain deterministic.

- [ ] **Step 5: Run tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_freeze_scannet200_manifest.py -q
git add scripts/evaluation/freeze_scannet200_manifest.py \
  tests/evaluation/test_freeze_scannet200_manifest.py
git commit -m "feat: freeze validated ScanNet200 manifests"
```

Do not create `configs/evaluation/manifests/scannet200_5.json` during this task because real data is still absent.

---

### Task 4: Add Method Asset and Capability Preflight

**Files:**
- Create: `configs/evaluation/baselines/scannet200_methods.json`
- Create: `scripts/evaluation/preflight_scannet200_methods.py`
- Create: `tests/evaluation/test_preflight_scannet200_methods.py`

- [ ] **Step 1: Write failing schema tests**

The method config must contain exactly `OPENFUSION`, `OVIMAP`, `CONCEPTGRAPHS`, and `DUALMAP`. Each record declares upstream commit, mode, runner kind, required weights/configs, supported metrics, and unsupported metrics with reasons.

OpenFusion instance AP remains unsupported and must be marked `N/A`, not omitted or zero.

- [ ] **Step 2: Write asset failure tests**

Synthetic preflight must report all missing files in one structured JSON and return non-zero. Hash mismatches are failures even when files exist.

- [ ] **Step 3: Verify RED**

Expected: config/CLI missing.

- [ ] **Step 4: Implement preflight**

CLI:

```text
--method-config
--manifest
--output
```

When the formal manifest is absent, write status `BLOCKED`, reason `SCANNET_DATA_NOT_VALIDATED`, list assets that are independently ready, and do not attempt a run.

- [ ] **Step 5: Run tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_preflight_scannet200_methods.py -q
git add configs/evaluation/baselines/scannet200_methods.json \
  scripts/evaluation/preflight_scannet200_methods.py \
  tests/evaluation/test_preflight_scannet200_methods.py
git commit -m "feat: preflight ScanNet200 baseline assets"
```

---

### Task 5: Generate Non-Executing Runner Specifications

**Files:**
- Create: `scripts/evaluation/generate_scannet200_run_specs.py`
- Create: `tests/evaluation/test_generate_scannet200_run_specs.py`

- [ ] **Step 1: Write failing generation tests**

Given a valid synthetic manifest and method preflight, require one JSON run specification per method. Every spec includes:

- method, mode, upstream commit;
- immutable manifest path/hash;
- scene order;
- environment/container identity;
- input/output roots;
- command argv as a list, not a shell string;
- required metric contract;
- GPU allocation hint;
- status `READY`.

If the manifest or assets are blocked, no `READY` spec may be written.

- [ ] **Step 2: Run and verify RED**

Expected: generator missing.

- [ ] **Step 3: Implement deterministic generation**

Write into a temporary directory and atomically rename it. Do not execute any command. Reject output directories that already contain a status/result unless `--replace-empty` is used and the directory is proven empty.

- [ ] **Step 4: Run tests and commit**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest \
  tests/evaluation/test_generate_scannet200_run_specs.py -q
git add scripts/evaluation/generate_scannet200_run_specs.py \
  tests/evaluation/test_generate_scannet200_run_specs.py
git commit -m "feat: generate ScanNet200 baseline run specs"
```

---

### Task 6: Refresh the Real Blocker With Executable Next Steps

**Files:**
- Update externally: `/home/ww/oviovo_baseline_runs/20260714_non_oviovo_baselines/scannet200/blocked_status_20260719.json`
- Modify: `docs/paper/BASELINE_RUN_REPORT.md`

- [ ] **Step 1: Run real validation against the expected asset root**

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python \
  scripts/evaluation/freeze_scannet200_manifest.py \
  --dataset-root /home/ww/oviovo_benchmark_assets/scannet200 \
  --vocabulary /home/ww/oviovo_aaai_workspace/02_repos/main/ConceptGraphs/conceptgraph/scannet200_classes.txt \
  --vocabulary-sha256 4e4c4ef1e39dad24538b8feeba5be43abca39b21c5e5492ca6413f2908ec49ad \
  --output /home/ww/oviovo_baseline_runs/20260714_non_oviovo_baselines/scannet200/scannet200_5.json
```

Expected while data is absent: non-zero exit with all five missing scene directories and no manifest output.

- [ ] **Step 2: Run method preflight in blocked mode**

Record which source checkouts, environments, configs, and weights are ready independently of data.

- [ ] **Step 3: Write a structured blocker**

Include commands, exit codes, stdout/stderr paths, official access URL, required user action, candidate scene IDs, and the exact next validation command. Do not include credentials or agreement contents.

- [ ] **Step 4: Update and commit the report only if evidence paths changed**

Do not change registry ScanNet tokens from `UNFILLED`.

---

## Completion Gate

```bash
/home/ww/miniconda3/envs/oviovo-conceptgraphs/bin/python -m pytest -q -p no:cacheprovider \
  tests/evaluation/test_scannet200_dataset.py \
  tests/evaluation/test_freeze_scannet200_manifest.py \
  tests/evaluation/test_preflight_scannet200_methods.py \
  tests/evaluation/test_generate_scannet200_run_specs.py
```

Then assert:

- no formal `configs/evaluation/manifests/scannet200_5.json` exists without real validation;
- all ScanNet registry tokens remain `UNFILLED`;
- the blocker contains real commands and exit statuses;
- no reference checkout is dirty;
- the workspace verifier passes.
