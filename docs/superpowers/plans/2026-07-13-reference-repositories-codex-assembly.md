# OVIOVO Reference Repositories and Codex Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an isolated, reproducible collection of nine AAAI reference repositories and two documents that tell Codex how to use them without wholesale code merging.

**Architecture:** Reference repositories live outside the OVIOVO worktree under `/home/ww/oviovo_references` and are treated as read-only. Baseline implementations and evaluators are grouped separately, while `MANIFEST.md` freezes provenance and `CODEX_ASSEMBLY_PROMPT.md` defines the staged integration contract for the existing Python OVIOVO worktree.

**Tech Stack:** Git, Markdown, Bash verification commands, GitHub-hosted upstream repositories.

---

## File Structure

- Create: `/home/ww/oviovo_references/baselines/*` - seven shallow baseline clones.
- Create: `/home/ww/oviovo_references/evaluation/*` - two shallow evaluation clones.
- Create: `/home/ww/oviovo_references/MANIFEST.md` - repository provenance, license status, role, and exact commit.
- Create: `/home/ww/oviovo_references/CODEX_ASSEMBLY_PROMPT.md` - executable instructions for a later Codex integration session.
- Preserve: `/home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates` - implementation target, unchanged by this acquisition plan.

### Task 1: Acquire Baseline Repositories

**Files:**
- Create: `/home/ww/oviovo_references/baselines/OVI-MAP`
- Create: `/home/ww/oviovo_references/baselines/OpenVox`
- Create: `/home/ww/oviovo_references/baselines/OpenFusion`
- Create: `/home/ww/oviovo_references/baselines/ConceptGraphs`
- Create: `/home/ww/oviovo_references/baselines/DualMap`
- Create: `/home/ww/oviovo_references/baselines/Khronos`
- Create: `/home/ww/oviovo_references/baselines/panoptic_mapping`

- [ ] **Step 1: Create the isolated directory structure**

Run:

```bash
mkdir -p /home/ww/oviovo_references/baselines /home/ww/oviovo_references/evaluation
```

Expected: both directories exist and the OVIOVO worktree remains outside this tree.

- [ ] **Step 2: Shallow-clone the seven baseline repositories**

Run each command independently so one failure does not prevent later clones:

```bash
git clone --depth 1 https://github.com/OVI-MAP/OVI-MAP.git /home/ww/oviovo_references/baselines/OVI-MAP
git clone --depth 1 https://github.com/BIT-DYN/OpenVox.git /home/ww/oviovo_references/baselines/OpenVox
git clone --depth 1 https://github.com/UARK-AICV/OpenFusion.git /home/ww/oviovo_references/baselines/OpenFusion
git clone --depth 1 https://github.com/concept-graphs/concept-graphs.git /home/ww/oviovo_references/baselines/ConceptGraphs
git clone --depth 1 https://github.com/Eku127/DualMap.git /home/ww/oviovo_references/baselines/DualMap
git clone --depth 1 https://github.com/MIT-SPARK/Khronos.git /home/ww/oviovo_references/baselines/Khronos
git clone --depth 1 https://github.com/ethz-asl/panoptic_mapping.git /home/ww/oviovo_references/baselines/panoptic_mapping
```

Expected: each successful target contains `.git`; no submodule initialization, dataset download, or environment creation occurs.

- [ ] **Step 3: Verify baseline origins and commits**

Run:

```bash
for repo in /home/ww/oviovo_references/baselines/*; do
  git -C "$repo" remote get-url origin
  git -C "$repo" rev-parse HEAD
done
```

Expected: seven upstream URLs and seven 40-character commit SHAs. Any failed clone is recorded for Task 3 rather than replaced with a different fork.

### Task 2: Acquire Evaluation Repositories

**Files:**
- Create: `/home/ww/oviovo_references/evaluation/rescene4d`
- Create: `/home/ww/oviovo_references/evaluation/stmetrics`

- [ ] **Step 1: Shallow-clone the evaluation repositories**

Run:

```bash
git clone --depth 1 https://github.com/GradientSpaces/rescene4d.git /home/ww/oviovo_references/evaluation/rescene4d
git clone --depth 1 https://github.com/GradientSpaces/stmetrics.git /home/ww/oviovo_references/evaluation/stmetrics
```

Expected: both target directories contain `.git`; dependencies and datasets are not installed.

- [ ] **Step 2: Verify evaluator provenance**

Run:

```bash
git -C /home/ww/oviovo_references/evaluation/rescene4d remote get-url origin
git -C /home/ww/oviovo_references/evaluation/rescene4d rev-parse HEAD
git -C /home/ww/oviovo_references/evaluation/stmetrics remote get-url origin
git -C /home/ww/oviovo_references/evaluation/stmetrics rev-parse HEAD
```

Expected: official GradientSpaces origins and exact commit SHAs.

### Task 3: Create the Repository Manifest

**Files:**
- Create: `/home/ww/oviovo_references/MANIFEST.md`

- [ ] **Step 1: Collect checked-out metadata**

Run:

```bash
for repo in /home/ww/oviovo_references/baselines/* /home/ww/oviovo_references/evaluation/*; do
  test -d "$repo/.git" || continue
  printf '%s\t%s\t%s\t%s\n' \
    "${repo#/home/ww/oviovo_references/}" \
    "$(git -C "$repo" remote get-url origin)" \
    "$(git -C "$repo" branch --show-current)" \
    "$(git -C "$repo" rev-parse HEAD)"
done
```

Expected: one tab-separated record for every successful clone.

- [ ] **Step 2: Write the manifest using the collected exact values**

The manifest must contain these sections and fixed policy statements:

```markdown
# OVIOVO Reference Repository Manifest

Acquired: 2026-07-13
Target project: /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates

## Repositories

| Path | Upstream | Branch | Commit | Role | License | Status |
| --- | --- | --- | --- | --- | --- | --- |

## Acquisition Policy

- Repositories are shallow, read-only references.
- Submodules, datasets, weights, build products, and environments were not acquired.
- OpenVox and OpenFusion have no declared standard repository license and are reference-only.

## Citation-Only Projects

OVO-SLAM, OpenFusion++, DovSG, RADIO-ViPE, ThinkGraphs, VLMaps, OpenScene, and OpenMask3D are not cloned because they are not required as runnable baselines in the approved benchmark plan.
```

Populate each repository row with the exact output from Step 1. A failed clone receives `Status = failed` plus its original clone command below the table.

- [ ] **Step 3: Verify manifest completeness**

Run:

```bash
rg -n 'OVI-MAP|OpenVox|OpenFusion|ConceptGraphs|DualMap|Khronos|panoptic_mapping|rescene4d|stmetrics' /home/ww/oviovo_references/MANIFEST.md
```

Expected: all nine approved repository names occur in the manifest.

### Task 4: Create the Codex Assembly Prompt

**Files:**
- Create: `/home/ww/oviovo_references/CODEX_ASSEMBLY_PROMPT.md`

- [ ] **Step 1: Write a self-contained Codex prompt**

The prompt must contain the following enforceable sections:

```markdown
# Mission
Extend the existing OVIOVO project into an ownership-aware open-vocabulary dynamic 3D mapping system. Reference repositories are evidence and baseline sources, not merge targets.

# Authoritative Paths
- OVIOVO implementation target: /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates
- Reference root: /home/ww/oviovo_references
- Provenance manifest: /home/ww/oviovo_references/MANIFEST.md

# Hard Constraints
- Treat every repository under the reference root as read-only.
- Never copy code from OpenVox or OpenFusion because their repositories do not declare a compatible standard license.
- Do not merge ROS1, ROS2, C++, and Python build systems.
- Do not use GT semantic labels during inference.
- Do not claim dynamic robustness or real-time performance without benchmark evidence.
- Preserve unrelated user changes in the OVIOVO worktree.

# Required Workflow
1. Inspect OVIOVO interfaces and MANIFEST.md.
2. Produce a capability matrix mapping each reference to reusable contracts, evaluator behavior, and baseline outputs.
3. Fix true runtime open-vocabulary semantic evaluation before algorithm work.
4. Add benchmark adapters for Replica/ScanNet, Khronos, and 3RScan/stmetrics.
5. Implement original modules in this order: visibility negative evidence, reversible ownership removal, background reclaim, dormant-object re-identification, probabilistic lifecycle state.
6. Add unit tests before each behavior and protocol-level integration tests after each adapter.
7. Run controlled-backend and end-to-end baseline comparisons without changing baseline algorithms.

# Reference Roles
- OVI-MAP: evaluation protocol and closest static instance-semantic baseline.
- OpenVox: conceptual reference for probabilistic voxel association; no code copying.
- OpenFusion: dense TSDF baseline; no code copying.
- ConceptGraphs: object-centric map baseline.
- DualMap: object status and dynamic navigation baseline.
- Khronos: continuous dynamic dataset, evaluator, and metric-semantic baseline.
- panoptic_mapping: classical long-term consistent mapping baseline.
- rescene4d and stmetrics: sparse cross-session t-AP and t-Recall evaluation.

# Required Deliverables
- A written compatibility and license audit before edits.
- True runtime-semantic and GT-oracle-instance evaluators with distinct names and outputs.
- Reproducible dataset adapters and baseline runner commands.
- Original lifecycle modules with unit and integration tests.
- Static, continuous-dynamic, sparse-temporal, ablation, and efficiency result tables.

# Acceptance Criteria
- No reference repository is modified.
- GT labels never influence runtime semantic predictions.
- Removed or moved objects can release stale voxel ownership.
- Dormant objects can reactivate with their original identity after relocation.
- Benchmark outputs include semantic mIoU, instance AP, change F1, ghost rate, t-mAP, ID switches, latency, and memory.
- Every paper claim maps to a reproduced table or is weakened or removed.
```

Add an initial instruction requiring Codex to stop after its compatibility matrix and implementation plan so the user can review architecture decisions before code changes.

- [ ] **Step 2: Verify prompt constraints and deliverables**

Run:

```bash
rg -n 'read-only|OpenVox|OpenFusion|GT semantic|negative evidence|reversible ownership|background reclaim|re-identification|Acceptance Criteria' /home/ww/oviovo_references/CODEX_ASSEMBLY_PROMPT.md
```

Expected: every safety boundary and original contribution appears explicitly.

### Task 5: Final Verification

**Files:**
- Verify: `/home/ww/oviovo_references`
- Verify: `/home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates`

- [ ] **Step 1: Count successful clones**

Run:

```bash
find /home/ww/oviovo_references/baselines /home/ww/oviovo_references/evaluation -mindepth 2 -maxdepth 2 -type d -name .git | wc -l
```

Expected: `9`.

- [ ] **Step 2: Confirm no heavyweight artifacts were created by acquisition**

Run:

```bash
find /home/ww/oviovo_references -type d \( -name outputs -o -name weights -o -name checkpoints -o -name .venv -o -name build \) -prune -print
```

Expected: only directories already versioned by upstream may appear; no downloaded data, generated environment, or local build directory has a modification time from the acquisition workflow.

- [ ] **Step 3: Confirm OVIOVO implementation files were not changed**

Run:

```bash
git -C /home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates status --short
```

Expected: no new implementation changes caused by acquisition; pre-existing untracked user files remain untouched.

- [ ] **Step 4: Report disk use and document paths**

Run:

```bash
du -sh /home/ww/oviovo_references
test -f /home/ww/oviovo_references/MANIFEST.md
test -f /home/ww/oviovo_references/CODEX_ASSEMBLY_PROMPT.md
```

Expected: disk usage is reported and both documents exist.
