# Task: Add switchable SAM2 / EntitySAM proposal frontend

Modify the current project so the proposal stage supports **two selectable frontends**:

* `sam2`
* `entitysam`

Local EntitySAM repo:

* `/home/phl/vv/paper2/entitysam`

The rest of the pipeline must stay unchanged.

---

## Goal

Create a clean A/B comparison setup for proposal frontends on Replica room0.

Both frontends must output the same normalized internal type:

* `List[Proposal2D]`

So downstream modules do not need to know whether proposals came from SAM2 or EntitySAM.

---

## What to do

### 1. Inspect feasibility first

Before coding, inspect:

* current proposal abstraction
* current SAM2 adapter
* `/home/phl/vv/paper2/entitysam`

Summarize briefly:

* whether EntitySAM has runnable inference code
* where config/checkpoint/inference entrypoint are
* what output format it produces
* whether it can be wrapped for per-frame proposal generation

If the repo is incomplete, do **not** fake a fully working integration.
Instead:

* build the adapter skeleton
* add config/path plumbing
* add clear runtime error messages
* document missing assets

---

### 2. Add EntitySAM adapter

Create something like:

* `src/models/entitysam_proposal_backend.py`

Responsibilities:

* load EntitySAM model/config/checkpoint if available
* run inference on RGB input
* convert raw output into unified `Proposal2D`

Keep it wrapped cleanly behind the proposal interface.

---

### 3. Extend proposal module

Update proposal selection so config can choose:

* `sam2`
* `entitysam`

Keep default behavior unchanged unless config says otherwise.

---

### 4. Add config fields

Update config to support:

* `proposal.backend: sam2 | entitysam`
* `proposal.entitysam.repo_root`
* `proposal.entitysam.config_path`
* `proposal.entitysam.checkpoint_path`
* `proposal.entitysam.device`

Stay consistent with existing config style.

---

### 5. Add comparison runner

Create or extend a runner like:

* `run_replica20f_compare_proposals.py`

It should:

1. run SAM2 on the same Replica sequence
2. run EntitySAM on the same Replica sequence
3. save overlays/contact sheets separately
4. save manifest/stat summary

Suggested output folders:

* `outputs/replica20f_compare/sam2/`
* `outputs/replica20f_compare/entitysam/`

---

### 6. Keep visualization compatible

Existing visualization should work for both frontends.

For both, output:

* per-frame overlay
* contact sheet
* manifest.json

Include source field:

* `sam2`
* `entitysam`

---

### 7. Add tests

Add lightweight tests for:

* config-based frontend switching
* normalized `Proposal2D` output
* comparison runner wiring
* visualization compatibility

Use mocks/fake outputs if needed.
Do not require heavy inference in unit tests.

---

## Requirements

* do not rewrite downstream pipeline
* keep code typed and modular
* keep proposal frontend swappable
* do not tightly couple pipeline to EntitySAM internals
* do not silently hide missing model/config/checkpoint failures
* keep provenance explicit in proposal metadata

---

## Deliverables

I want:

1. EntitySAM proposal adapter
2. proposal module updated for SAM2 / EntitySAM switching
3. config updates
4. comparison runner
5. visualization outputs for both frontends
6. short README/interface note
7. tests

---

## Final instruction

This task is only about **frontend proposal switching**.

Do not redesign the whole method.
Do not rewrite downstream modules.

Implement the cleanest minimal integration for:

* `sam2`
* `entitysam`

and make the comparison setup easy to inspect.
