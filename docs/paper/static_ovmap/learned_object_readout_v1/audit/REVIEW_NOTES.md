# Delivery review notes

This is a review of a research execution specification, not completed GPU science.

## Material issues corrected before delivery

1. The existing 26-scene cohort cannot become the new supervised training set. A bounded authorized 24/4/4 family inventory and explicit missing-data state replace that assumption.
2. The held-out H set cannot be preprocessed from annotations before model selection. Its annotation parsing/proposals/features are now deferred until both seed nominations.
3. A frozen visual head must still propagate gradients to learned pooling. No-gradient feature extraction and differentiable frozen-head use are separated.
4. The actual Mask-Adapter `_2d` dense projection helper was read, and the contract now uses its norm/drop/visual-head path rather than assuming an arbitrary pointwise approximation.
5. Raw-value pooling and nonlinear FC projection cannot be swapped. The architecture and reference test preserve this distinction.
6. VIEW/SURFACE models share layer shapes and raw tokens; nonlinear group keys make the grouping test nontrivial.
7. Four static corruption conditions share a supervised per-object view bank; no branch-specific successful-view selection. They are proposal diagnostics, not a hidden new map benchmark.
8. Checkpoint/architecture selection uses only DEV base labels. Benchmark best-candidate ranking is separately marked exploratory; no best-of-seed or per-dataset switching.
9. Optional gate training is explicitly outside this phase because independent old/new D2/G1 training pairs are not bound. No simulated GT “old predictions” are substituted.
10. Runtime is recorded but never an accuracy selection gate; large cold timing and exhaustive safety test work are excluded.

## Remaining empirical prerequisites

- Server availability and authorized access to the additional ScanNet families.
- Actual raw calibration and sufficient base/heldout object counts.
- Real upstream head parity, valid CPU/GPU gradient path and supervised convergence.
- Transfer from GT-derived perturbed training proposals to real predicted map masks.
- Improvement over learned 2D and view-grouped controls and five-metric regression gates.

The included 12 tests use small synthetic CPU tensors. They do not validate the actual FC/MaskAdapter weights, ScanNet data or any task metric.
