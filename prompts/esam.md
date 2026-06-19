# Task: Add a switchable SAM2 / E-SAM frontend proposal model

You are modifying an existing research codebase for online open-vocabulary dynamic mapping.

The codebase already has:

* a modular proposal stage
* a working SAM2 proposal frontend
* Replica dataset loading
* visualization for proposal overlays
* a runtime pipeline where proposal frontend is intended to be swappable

Now extend the system so that the proposal stage can use **either SAM2 or E-SAM**, selected by config / CLI.

Important local reference:

* E-SAM source code is already available at
  `/home/phl/vv/E-SAM`

This task is only about **frontend proposal model switching**.
Do not redesign the rest of the pipeline.

---

## Goal

Make the proposal stage support two interchangeable frontend models:

* `sam2`
* `esam`

Both must produce the same unified internal output type, e.g. `List[Proposal2D]`.

The rest of the pipeline should remain unchanged.

I want to be able to run:

* SAM2 on Replica room0
* E-SAM on Replica room0

and compare:

* proposal count
* overlay quality
* fragmentation behavior
* downstream compatibility

---

## Important design constraints

### 1. Keep the existing proposal abstraction

Do not break the current swappable proposal design.

There should be a clean interface such as:

* `ProposalBackend`
* `SAM2ProposalBackend`
* `ESAMProposalBackend`

The proposal module should choose the frontend model by config.

### 2. Unify output format

No matter which frontend is used, proposal output must be normalized into the same `Proposal2D` structure.

Each proposal should contain at least:

* mask
* bbox
* score if available
* backend/frontend name
* optional metadata

If E-SAM produces a different raw structure, adapt it cleanly.

### 3. Do not tightly couple pipeline code to E-SAM internals

Keep E-SAM wrapped behind a backend/frontend adapter class.

The rest of the code should depend only on the common proposal interface.

---

## Reference codebases

You may inspect and reuse patterns from:

* current project proposal abstraction
* existing SAM2 frontend implementation
* local E-SAM source at `/home/phl/vv/E-SAM`
* `/home/phl/vv/paper2/OVO`
* `/home/phl/vv/paper2/DualMap`

Use them as references for:

* model loading patterns
* config style
* output normalization
* visualization integration

Do not blindly copy code.
Keep the final integration clean and independent.

---

## What to implement

### A. Add an E-SAM proposal frontend adapter

Implement something like:

* `src/models/esam_proposal_backend.py`

Responsibilities:

* load E-SAM model/config/checkpoint if available
* run inference on RGB input
* produce entity-level masks
* convert raw E-SAM output into unified `Proposal2D`

If exact local E-SAM entrypoints/checkpoints/configs are not obvious:

* inspect `/home/phl/vv/E-SAM`
* summarize the reusable inference path
* implement a configurable wrapper
* do not hardcode undocumented assumptions

If required assets are missing, do not fake success.
Implement:

* path/config plumbing
* clear error messages
* documentation of missing requirements

---

### B. Extend proposal module

Modify the current proposal module so frontend selection can be done by config:

* `sam2`
* `esam`

Default behavior should remain current behavior unless config explicitly changes it.

---

### C. Add config support

Update config files so proposal frontend can be selected cleanly.

Example idea:

* `proposal.backend: sam2 | esam`
* `proposal.sam2.*`
* `proposal.esam.repo_root`
* `proposal.esam.config_path`
* `proposal.esam.checkpoint_path`
* `proposal.esam.device`

Stay consistent with current project config style.

---

### D. Add a comparison runner

Add a script for direct comparison, for example:

* `run_replica20f_compare_proposals.py`

This script should:

1. run proposal extraction on the same Replica sequence using `sam2`
2. run proposal extraction on the same Replica sequence using `esam`
3. save outputs into separate folders
4. generate overlays and contact sheets
5. summarize:

   * number of proposals per frame
   * average proposal count
   * optional mask area statistics

Suggested output structure:

* `outputs/replica20f_compare/sam2/...`
* `outputs/replica20f_compare/esam/...`

---

### E. Keep visualization compatible

Ensure existing visualization tools work for both proposal frontends.

At minimum, for both frontends produce:

* per-frame overlay
* contact sheet
* manifest.json

If useful, annotate output with frontend name:

* `sam2`
* `esam`

---

### F. Add tests

Add lightweight tests for:

1. backend/frontend selection by config
2. output normalization into `Proposal2D`
3. comparison runner wiring
4. visualization compatibility for both output formats

Do not require heavy model inference in unit tests.
Use mocks or fake outputs where appropriate.

---

## Important behavior expectations

### 1. Do not change downstream contracts

Depth refinement, runtime_vis, lifting, split, association, etc. should not need to know whether proposals came from SAM2 or E-SAM.

### 2. Preserve fair comparison

Use the same:

* dataset loader
* frame sampling
* visualization logic
* output structure
* downstream interfaces

Only the proposal frontend should change.

### 3. Make provenance explicit

Each proposal and each manifest entry should record proposal source:

* `sam2`
* `esam`

So later debugging and comparison are easy.

---

## Required implementation steps

Before coding:

1. inspect the current proposal abstraction and SAM2 implementation
2. inspect `/home/phl/vv/E-SAM`
3. identify:

   * inference entrypoint
   * config usage
   * checkpoint loading path
   * output mask format
4. summarize what is reusable
5. then implement the cleanest adapter

If E-SAM integration needs environment-specific imports or path manipulation, do it carefully and document it clearly.

---

## Deliverables

I want you to produce:

1. an E-SAM proposal adapter
2. proposal module updated for SAM2 / E-SAM switching
3. config updates
4. comparison runner script
5. visualization outputs for both frontends
6. brief README/interface note describing how to switch between SAM2 and E-SAM
7. tests for switching and normalized output

---

## Engineering requirements

* keep the code typed and modular
* do not rewrite the whole pipeline
* preserve current project structure
* add TODOs where future cleanup or optimization is needed
* keep the integration minimal and inspectable
* do not silently swallow import/model loading failures

---

## Final instruction

This task is not about improving the whole method.

It is only about creating a **clean A/B comparison setup** for two proposal frontends:

* SAM2
* E-SAM

Keep the integration minimal, typed, modular, and easy to inspect.

Do not over-engineer.
Do not rewrite downstream modules.
Only add a clean switchable E-SAM frontend next to SAM2.

---

## Execution note

Before writing code:

1. inspect the current project proposal module
2. inspect the current SAM2 adapter
3. inspect `/home/phl/vv/E-SAM`
4. explain briefly:

   * how E-SAM should be wrapped
   * what outputs need normalization
   * where config paths should live
5. then implement

If there are blockers in E-SAM local setup, report them clearly instead of guessing.
