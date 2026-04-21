# Task: Add a switchable CropFormer proposal backend alongside SAM2 for comparison

You are modifying an existing research codebase for online open-vocabulary dynamic mapping.

The codebase already has:

* a modular proposal stage
* a working SAM2 proposal backend
* Replica dataset loading
* visualization for proposal overlays
* a pipeline where proposal backend is intended to be swappable

Now extend the system so that the proposal stage can use **either SAM2 or CropFormer**, selected by config / CLI.

This is for **controlled comparison only**.
Do not redesign the entire pipeline.

---

## Goal

Make the proposal stage support two interchangeable backends:

* `sam2`
* `cropformer`

Both should produce the same unified internal output type, e.g. `List[Proposal2D]`.

The rest of the pipeline should remain unchanged.

I want to be able to run:

* SAM2 on Replica room0
* CropFormer on Replica room0

and compare:

* proposal count
* overlay quality
* fragmentation behavior
* downstream compatibility

---

## Important design constraints

### 1. Keep the existing proposal abstraction

Do not break the current swappable backend design.

There should be a clean interface such as:

* `ProposalBackend`
* `SAM2ProposalBackend`
* `CropFormerProposalBackend`

The proposal module should choose backend by config.

---

### 2. Unify output format

No matter which backend is used, the output of proposal stage must be normalized into the same `Proposal2D` structure.

Each proposal should contain at least:

* mask
* bbox
* score if available
* backend name
* optional metadata

If CropFormer outputs a different raw structure, adapt it cleanly.

---

### 3. Do not couple the pipeline tightly to CropFormer internals

Keep CropFormer wrapped behind a backend class.

The rest of the code should only depend on the common interface.

---

## Reference repositories

You may inspect these local repositories for reusable patterns:

* `/home/phl/vv/paper2/OVO`
* `/home/phl/vv/paper2/DualMap`

You may also inspect whether there is already any local CropFormer-related code or environment setup available in the current machine or project tree.

Use existing code if helpful, but keep the final integration clean and independent.

---

## What to implement

### A. Add a CropFormer proposal backend

Implement something like:

* `src/models/cropformer_proposal_backend.py`

Responsibilities:

* load CropFormer model/config/checkpoint if available
* run inference on RGB input
* optionally use depth only for alignment or metadata, but CropFormer itself can remain RGB-driven
* convert raw output into unified `Proposal2D`

If the exact local CropFormer installation/checkpoint path is not obvious:

* first inspect and summarize
* then implement a configurable path-based loader
* do not hardcode hidden assumptions without documenting them

---

### B. Extend proposal module

Modify the proposal module so backend can be selected by config:

* `sam2`
* `cropformer`

The default should remain current behavior unless config says otherwise.

---

### C. Add config entries

Update config files so proposal backend can be selected cleanly.

Example fields:

* `proposal.backend: sam2 | cropformer`
* `proposal.sam2.*`
* `proposal.cropformer.config_path`
* `proposal.cropformer.checkpoint_path`
* `proposal.cropformer.device`

Do not guess exact field names rigidly if current config style differs; stay consistent with current project style.

---

### D. Add a comparison runner

Add a script for easy comparison, for example:

* `run_replica20f_compare_proposals.py`

This script should:

1. run proposal extraction on the same Replica sequence using `sam2`
2. run proposal extraction on the same sequence using `cropformer`
3. save outputs into separate folders
4. generate overlays and contact sheets
5. summarize:

   * number of proposals per frame
   * average proposal count
   * optional mask area stats

Suggested output structure:

* `outputs/replica20f_compare/sam2/...`
* `outputs/replica20f_compare/cropformer/...`

---

### E. Add visualization compatibility

Ensure existing visualization tools work for both backends.

At minimum, for both backends produce:

* per-frame overlay
* contact sheet
* manifest.json

If needed, add backend label into overlay title or manifest.

---

### F. Add tests

Add lightweight tests for:

1. backend selection by config
2. output normalization into `Proposal2D`
3. comparison script wiring
4. visualization compatibility for both backend output formats

Do not require heavy model inference in unit tests.
Use mocks or fake outputs where appropriate.

---

## Important behavior expectations

### 1. Do not change downstream module contracts

Depth refinement, lifting, split, association, etc. should not need to know whether proposals came from SAM2 or CropFormer.

### 2. Preserve high-level comparison fairness

The same frames, same dataset loader, same visualization logic, same downstream interfaces should be used.

Only the proposal backend should change.

### 3. Make backend provenance explicit

Each proposal and each manifest entry should record backend source:

* `sam2`
* `cropformer`

So later debugging is easy.

---

## Implementation steps

Before coding:

1. inspect current proposal backend abstraction
2. inspect current SAM2 backend implementation
3. inspect current config and runner structure
4. inspect whether CropFormer code/checkpoints are already available locally
5. summarize what is reusable
6. then implement the cleanest integration

If CropFormer assets are missing, do not fake success.
Instead:

* implement the backend wrapper
* implement config/path plumbing
* implement graceful error messages
* document exactly what is missing

---

## Deliverables

I want you to produce:

1. CropFormer backend wrapper
2. proposal module updated for backend switching
3. config updates
4. comparison runner script
5. visualization outputs for both backends
6. brief README/interface note describing how to switch between SAM2 and CropFormer
7. tests for backend switching and normalized output

---

## Final instruction

This task is not about improving the whole method yet.

It is only about creating a **clean A/B comparison setup** for proposal backends.

Keep the integration minimal, typed, modular, and easy to inspect.

Do not over-engineer.
Do not rewrite the pipeline.
Only add a clean switchable CropFormer backend next to SAM2.
