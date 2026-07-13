# OVIOVO Reference Repositories and Codex Assembly Design

Date: 2026-07-13

## Goal

Create a reproducible, isolated collection of runnable reference repositories for AAAI experiments and generate a Codex prompt that uses those repositories as read-only design, baseline, and evaluation references. The existing OVIOVO worktree remains the only implementation target.

## Scope

The reference collection will be created at `/home/ww/oviovo_references` with nine repositories grouped by purpose:

```text
oviovo_references/
├── baselines/
│   ├── OVI-MAP/
│   ├── OpenVox/
│   ├── OpenFusion/
│   ├── ConceptGraphs/
│   ├── DualMap/
│   ├── Khronos/
│   └── panoptic_mapping/
├── evaluation/
│   ├── rescene4d/
│   └── stmetrics/
├── MANIFEST.md
└── CODEX_ASSEMBLY_PROMPT.md
```

Repositories:

| Local path | Upstream | Role | License status |
| --- | --- | --- | --- |
| `baselines/OVI-MAP` | `https://github.com/OVI-MAP/OVI-MAP.git` | Closest online instance-semantic mapping baseline and evaluation protocol | MIT |
| `baselines/OpenVox` | `https://github.com/BIT-DYN/OpenVox.git` | Probabilistic instance voxel reference | No declared standard license; reference only |
| `baselines/OpenFusion` | `https://github.com/UARK-AICV/OpenFusion.git` | Dense online open-vocabulary TSDF baseline | No declared standard license; reference only |
| `baselines/ConceptGraphs` | `https://github.com/concept-graphs/concept-graphs.git` | Object-centric open-vocabulary mapping baseline | MIT |
| `baselines/DualMap` | `https://github.com/Eku127/DualMap.git` | Dynamic object status and navigation baseline | Apache-2.0 |
| `baselines/Khronos` | `https://github.com/MIT-SPARK/Khronos.git` | Short- and long-term dynamic mapping baseline and dataset evaluator | BSD-3-Clause |
| `baselines/panoptic_mapping` | `https://github.com/ethz-asl/panoptic_mapping.git` | Classical long-term consistent panoptic mapping baseline | BSD-3-Clause |
| `evaluation/rescene4d` | `https://github.com/GradientSpaces/rescene4d.git` | Temporally consistent 4D instance benchmark reference | MIT |
| `evaluation/stmetrics` | `https://github.com/GradientSpaces/stmetrics.git` | Temporal AP and recall evaluator | MIT |

OVO-SLAM, OpenFusion++, DovSG, RADIO-ViPE, ThinkGraphs, VLMaps, OpenScene, and OpenMask3D remain citation-only references unless a later experiment plan demonstrates that their official code and output protocol are necessary.

## Repository Acquisition Policy

Each repository will be cloned with `--depth 1` from its default branch. Submodules, datasets, model weights, build products, and package environments will not be downloaded in this step. `MANIFEST.md` will record the upstream URL, checked-out branch, exact commit SHA, acquisition date, role, license status, and clone result.

If a clone fails, the remaining repositories continue. The failure and command needed to retry are recorded in the manifest. Existing non-empty repository directories are never overwritten; their current remote and commit are inspected and recorded instead.

## Codex Assembly Prompt

`CODEX_ASSEMBLY_PROMPT.md` will instruct Codex to work in phases:

1. Inspect OVIOVO and the manifest before proposing changes.
2. Treat every reference repository as read-only.
3. Build a capability matrix that maps reference components to OVIOVO interfaces.
4. Reuse evaluation contracts and dataset adapters before implementing algorithms.
5. Preserve OVIOVO's Python module boundaries instead of importing whole ROS/C++ systems.
6. Implement the paper's original core only in OVIOVO: visibility-based negative evidence, reversible voxel ownership, background reclaim, dormant-object re-identification, and probabilistic lifecycle state.
7. Separate true runtime open-vocabulary semantics from GT-oracle instance evaluation.
8. Add adapters and tests incrementally, with one source project and one behavior per change.
9. Respect repository licenses and avoid copying code from repositories without an explicit compatible license.
10. Produce benchmark-compatible outputs for static semantic mapping, continuous dynamics, and sparse cross-session identity consistency.

The prompt will explicitly prohibit wholesale code merging, dependency-environment unification, and claims of real-time or dynamic performance without measured evidence.

## Data Flow

```text
Reference repositories (read-only)
  -> capability and interface inventory
  -> dataset/evaluator adapters
  -> controlled baseline runners
  -> original OVIOVO dynamic modules
  -> unified benchmark outputs
```

No reference repository writes into another repository. Shared datasets and model weights remain external and are referenced through configuration paths.

## Verification

Completion requires:

1. All nine target directories exist as Git repositories or have a recorded clone failure.
2. Every successful clone has a reachable upstream URL and a recorded commit SHA.
3. `MANIFEST.md` matches the checked-out repositories.
4. `CODEX_ASSEMBLY_PROMPT.md` names the OVIOVO target, all reference roles, license restrictions, implementation phases, and acceptance criteria.
5. No files in the OVIOVO implementation are modified by repository acquisition.
6. No datasets, model weights, build trees, or Python/Conda environments are created.

## Non-Goals

- Running or installing every baseline in the acquisition step.
- Combining ROS1, ROS2, C++, and Python environments into one environment.
- Copying complete mapping backends into OVIOVO.
- Downloading benchmark datasets or model checkpoints.
- Claiming that repository availability implies protocol-compatible experimental results.
