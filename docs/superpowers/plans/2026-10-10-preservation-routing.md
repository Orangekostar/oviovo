# Preservation routing implementation plan

> Execute inline with executing-plans; primary owns numerical implementation, experiments and review. User authorized autonomous fixed-protocol execution.

**Goal:** implement and execute the unchanged 1010 protocol and publish actual applicable results.
**Architecture:** task-owned binding/registry/control flow; reuse read-only LR input loaders and output/scoring kernels. Shared production R/G arithmetic is used by training, inference and focused tests. Fixed DEV state machine limits activated scientific work.
**Tech Stack:** existing Python environments, FP32 PyTorch, one A40, NumPy, released scorer, normal Git.
**Spec:** docs/paper/static_ovmap/preservation_routing_v1/CODEX_FINAL_EXECUTION_EN.md and its three contracts.

## Constraints
Exact base6767eb90, parent read-only, split unchanged, 622 base-positive originals only, batch16/micro1, 2000 updates/arm, seeds17/29, no hyperparameter or timing gate, at most30000 updates. H2 names locked before training; annotations only after nominations. Full CLI observed before ordinary push. Production checks <=12 groups.

## Execution
- [x] P0: binding.py,parent_audit.py: verify parent stores/receipts286/22 and16000 updates; copy exact split/class/features manifests with separate writable root. Preserve loss availability; compare TRAIN/DEV FC, warmup, selected heads; fixed-key cancellation intervention.
- [x] P1: data.py,teachers.py: expose FC vectors in target-free inputs; clean/corrupt prefixes2/4/8 teachers; TRAIN-only correctness sidecars; deterministic H2 names and real-proposal frame plan. Engineer two real CUDA objects <=20 discarded updates; numerical tests first.
- [x] P2: models.py,losses.py,training.py,recognition.py: shared direct/residual equations; exact state subtraction and conditional item mean; common initialization/draws; six DEV checkpoints with A/M/C/CE and harm; save reference, selected,last and full scalar curves/RNG. 8000 main updates then fixed architecture2000 repeat only if qualified.
- [x] P3 conditional: identical frozen-base local architecture and five controls; count-normalized null routing; 5x2000 per seed; direct-pair and gradient tests.
- [x] P4: mechanisms.py, prediction.py,evaluation.py: lock nominations; exposed oldH all selected; conditional real SAM2 proposals/H2/map outputs; preserve G1 partition and protected labels; ranks/scorer/pools use actual registry. Two-scene production integration, no extra benchmark selection.
- [x] P5: reporting.py and CLI: all phase switches/resume, explicit negative vs technical exit states; three tables, four reports, loss curves/decisions, inference weights/dependencies. Strict export/sample roundtrip and actual allCLI. Original16-file final review, normal push/local-remote fullSHA receipt.

## Acceptance commands
`PYTHONPATH=.:src:<parent>/tooling/test_dependencies <FC_python> -m pytest -q tests/test_preservation_routing.py`
`<controller_python> -u scripts/evaluation/run_ovimap_preservation_routing.py --spec configs/static_ovmap/preservation_routing_v1.json --parent-root /mnt/shared/ww/ovimap-learned-object-readout-v1/attempt_001 --output-root /mnt/shared/ww/ovimap-preservation-routing-v1/attempt_001 --gpu 0 --phase all --resume`
`git diff --check` (CRLF-aware for CSV and unchanged input pack), strict selected-head sample roundtrip, `git ls-remote origin refs/heads/research/ovimap-preservation-routing-v1` equality.

## Observed implementation corrections
Before scientific updates, the two-scene checks exposed inherited SU baseline-key lookup and removed NumPy `in1d` in the existing FC environment. The task-owned adapter supplies exact baseline aliases and a scorer-local `isin` equivalent; upstream code and environments remain unchanged. Actual released scoring on office0/scene0011_00 reproduced all five parent G1 metrics. Failed CLI attempts and their measured clocks are retained externally under history/.

## Actual completion
All four seed17 R arms completed2,000 updates with matching initialization/draws. No arm passed the fixed foundation gate; conditional scientific G/seed29/H2/SAM2/new-map work is NOT_TRIGGERED. Post-lock TRAIN785/DEV87/oldH101 recognition, selected small weights plus frozen initialization, strict sample roundtrip, all table families and reports completed. Final focused tests:12 passed; actual final full all/resume exit0. The report-envelope duplicate source_binding failure is fixed and archived. Final upload authority is external publication/final.json after normal push/fullSHA verification.
