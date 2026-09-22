# ScanNet experiment data preparation

> **For agentic workers:** Execute inline with superpowers:executing-plans. Project AGENTS.md reserves data semantics, implementation and review for the primary agent.

**Goal:** Use the supplied downloader's official release layout to prepare the frozen 12 development + 2 confirmation captures, with resumable transfers and native-compatible frame exports.

**Architecture:** A scoped preparation CLI reads only literal release constants from `/home/ww/getscannet.py`; it never executes the script's full-release branch or SSL override. Public split files are pinned to ScanNet commit `3830fce7f8b2e48ef047ef7fd76ea5f62903f51c`. Restricted transfers require the user's explicit authorization. Candidate order is fixed before any image-model outputs; availability checks produce a separate final acquisition lock.

**Tech Stack:** Python standard library, curl, NumPy and OpenCV in the existing mapping environment.

**Spec:** `docs/paper/static_ovmap/module_validation_v1/spec/00_CODEX_MASTER_EN.md` and `01_ASSETS_NATIVE_INTERFACES_EN.md`.

## Constraints

- Exactly 8 FIT, 2 CAL, 2 SELECT, 2 CONFIRM physical families; exclude actual ScanNet exposure, not 3RScan scene aliases.
- Native start 0, end from frame count, negative step `floor((end-start)/200)`; retain first 200 slots with explicit positive step and replay end.
- Download only six required file types plus label map, official split files and metadata needed for availability checks.
- Preserve prior attempts, original downloader and baseline installation. Shared-disk output only.
- Confirmation raw availability may be prepared now; mapping and visual processing wait for frozen selection.
- Data preparation completion is not completion of the still-missing independent-scene S/G/Q experiment driver.

## Task 1 — Scoped download preparation

Files: `src/static_ovmap/module_validation/scannet_download.py`, `scripts/evaluation/prepare_ovimap_scannet.py`, `tests/module_validation/test_scannet_preparation.py`.

- [x] Add failing tests for literal-only script reading, v1 `.sens` routing, exact physical-family ordering, rejection of overlap, authorization gating, and verified transfer resume.
- [x] Implement `DownloaderLayout.from_script(path)`, `native_schedule(frame_count)`, `prepare_plan(...)`, and `download_selected(...)`. Plan records script/split/exclusion hashes, provisional scene roles and exact URLs; download locks availability before transferring large files.
- [x] Verify corrupt or partial files cannot be treated as completed downloads; transient server failures do not silently change scene selection.
- [x] Run public-metadata preparation and inspect the actual provisional 14-scene list.

## Task 2 — Native input export and discovery

Files: `src/static_ovmap/module_validation/scannet_frames.py`, the preparation CLI, `assets.py`, and the preparation tests.

- [x] Test a small binary sensor fixture against explicit JPEG/depth/pose/calibration values; verify selected indices, truncated-stream rejection and non-overwrite behavior.
- [x] Implement a streaming ScanNet v4 reader based on the pinned official SensorData format; export only frozen scheduled frames and retain original millimeter depth, camera matrices, frame IDs and JPEG payloads.
- [x] Bind explicit ScanNet roots by official split identity, including arbitrary directory names; preserve ambiguity rejection.
- [x] Run the scoped module tests and lint. Native ScannetLoader also passed a synthetic 200-frame input check; real-data validation depends on authorized data.

## Task 3 — Authorized execution and handoff

- [x] With explicit data authorization, perform availability checks, lock 14 captures, download missing assets and export only development input frames.
- [x] Without that authorization, finish and test the preparation code and public plan; report the precise authorization blocker without accepting terms.
- [x] Record actual data status and remaining experiment integration in the project progress document. Never label provisional candidates as complete or measured scenes.

Authorized execution completed: 14 raw captures downloaded, 12 development scenes exported (2,400 slots), public bind passes. Real-input faults fixed: curl retry offsets and v1/v2 frame-count differences that leave scheduled IDs unchanged. Validation: 116 scoped tests, Ruff, annotation alignment and native-loader checks pass. Capture is wired into the public entry point but execution is blocked by the occupied GPU. Independent-scene S/G/Q scientific drivers remain pending; no new scientific measurement has been produced.
