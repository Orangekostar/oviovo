# Replica composition transfer implementation plan

> Execute inline with executing-plans. Primary agent owns semantics, implementation, experiments and review; only a deterministic capture-recipe copy is delegated.

**Goal:** Obtain actual eight-scene, six-method frozen-transfer measurements.
**Architecture:** New `src/static_ovmap/replica_transfer/` wraps unchanged native-v10 capture and composition primitives with Replica vocabulary, input binding and released evaluation.
**Tech stack:** Existing isolated Python environments, FP32 Torch, NumPy, native-v10, released evaluator. No new dependencies.
**Spec:** `docs/superpowers/specs/2026-09-28-replica-composition-transfer-design.md`.

## Tasks

- [x] Protocol and binding (`protocol.py`, `tests/replica_transfer/test_protocol.py`): reject scene/method subsets outside the fixed set, preserve exact transferred temperatures/checkpoint, load 51 prediction classes and 48 released AP classes. Test failed access and vocabulary behavior first; then implement and run scoped tests.
- [x] Replica capture (`capture.py`): mechanically copy the bound capture recipe, changing only input receipt, Replica loader and filename conventions; test and inspect exact diff. Bind all scheduled RGB-D, trajectory and camera bytes before capture. Start the first real capture while implementing independent downstream work.
- [x] Scene preparation (`jobs.py`, `text.py`, `ground_truth.py`): generate model-specific FP32 text embeddings for official raw Replica names, run unchanged GT conversion on the existing mesh/label files, freeze projection/N0 and native export parity, static S2 request/readout/source tables. GT never enters prediction or selection.
- [x] Real jobs (`jobs.py`, public runner): reuse native replay and forced reread kernels, immutable receipts and full score fusion. Add transfer-specific method authorization without changing ScanNet guards; resume exact outputs only. Tests cover changed inputs and transferred-calibration enforcement.
- [x] Evaluation/report (`evaluation.py`, `reporting.py`): initialize released evaluator with Replica, preserve cache content identities and per-scene immutable geometry/ranks, trace/match references, full row coverage, means/deltas/costs. Verify with a real completed scene before all-scene aggregation.
- [ ] Execute all eight scenes and six fixed methods, diagnose actual failures without scientific retuning, verify physical/logical counts and Q_GAIN/M4 sequence parity. Report exact data/metric boundaries and preserve all outcomes.
- [ ] Final review against the design, compact artifacts/external hashes, reproducibility commands and concise Chinese measured-result delivery. Local branch isolates this work from the completed ScanNet release.

Commands use `/home/ww/miniconda3/envs/ovimap-map/bin/python`, semantic workers use the bound repaired semantic environment, and scoped lint uses `/home/ww/miniconda3/bin/ruff`.

## Execution checkpoint

Implementation snapshot: `8bddf183`. Ten transfer tests and 29 relevant original-kernel tests passed; scoped Ruff passed. Query and evaluation kernel copies were compared function-by-function against their frozen originals. A real released-evaluator synthetic test confirms that wall classification affects 51-class semantics but not 48-class AP.

Both frozen FP32 text spaces and all eight annotation conversions are complete. The converted semantic/instance arrays exactly match the released mapping rules. room0 capture completed 200/200 frames without gaps; all 9,158,364 surface owner labels match the native mesh. Native projected masks and serialized ranks match the original exporter. Fixed 5 cm projection matches 830,147 of 954,492 whole-scene GT vertices (86.9727%). No method metric results are claimed at this checkpoint.

At the user's request, GPT-5.6 Luna performs read-only progress monitoring. The primary retains debugging, scientific decisions, result validation and final completion authority. The initial capture-only launcher will hand off after room1's verified completion to the existing full experiment orchestrator, so subsequent scenes run through a single scene-by-scene six-method pipeline.

First full-scene integration verification: room0 has six COMPLETE rows, with verified evaluator receipts, trace parity, identical native geometry/owners/ranks, and unchanged transferred temperatures. Q_GAIN and M4 each attempted 200 requests with zero failures; their 200 paid requests match exactly. Q_GAIN executed 200 model forwards; M4 executed 177 forwards and used 23 exact static-cache hits. The final eight-scene conclusion remains pending.
