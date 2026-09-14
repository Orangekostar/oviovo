# Local ownership implementation plan

Use executing-plans inline. User authorizes autonomous execution and task-branch push; primary owns architecture, algorithms, scientific decisions and final review.

Goal: exactly six new cached Room0 cells from exact U00, with no model inference and one frozen configuration. Spec: [LOCAL_OWNERSHIP_SPEC.md](LOCAL_OWNERSHIP_SPEC.md).
Reviewed and actual start HEAD: 98cafc4a96bf906878116ae3284da28bc974ad4d, clean. Worktree /home/ww/crove/ovimap-local-ownership, branch research/ovimap-local-ownership-v1. Baseline: 19 directly relevant tests pass.

Architecture: ownership_evidence.py implements frame selection, shared z-buffer and leave-atom-out sparse contingencies; local_ownership.py implements exact signature/connectivity atoms, feasible unaries and bounded spatial solver. New runner loads verified U00 and source observations only. Thin evaluation wrapper reuses old released exports/traces/semantic/object/region helpers, dispatches global versus local validation and does not require old AT closing checks. A reporting script assembles the four tables and handoff.

- [x] WP0 read actual source and receipts, establish clean isolated task branch, baseline regression.
- [x] WP0 bind consumed files once, selected original frames and calibration; verify T0/pool/coords/metadata identities and observation registration. Write frozen config before GT evaluation.
- [x] WP1–3 write deterministic failing fixtures for projection/ties/abstention, leave-atom-out, neutral atom feasibility, LOCAL/SPATIAL same unaries, energy descent, global-vs-local validation and cache identity. Implement corresponding new functions and run focused fixtures.
- [x] WP1 produce same-U00 OVI_FILL on both runs. Preserve masks/classes/IDs/ranks exactly; unknown method rejects.
- [x] WP2 one real input smoke, then full 32-view projection and sparse evidence on both complete clouds. Atomize 2cm grid/exact signature/1.5cm connectivity, no parent lock. Record all abstentions, counts and sparse contributing observations.
- [x] WP3 construct same-frame histogram-supported radius-capped graph (6/3cm) and normalized weights. Solve LOCAL then sequential SPATIAL (lambda .10, <=5 sweeps), energy strictly decreases per accepted move. Persist source maps, evidence/unary/graph identities and times.
- [x] WP4 adapter validates source support/coverage and fixed projection; legacy global check remains separate. Candidate manifest canonical parity then reuse overlapping AP; evaluate six unique maps and semantics. Fixed U00 regional accounting and GT-object retention/extent/fragmentation diagnostics, no GT in predictions.
- [x] WP5 four full-precision tables + displayed results, commands/interpreter/status, costs/peak RSS actually measured, source hashes, large assets manifest, limits and one next experiment. One task-specific final review, scoped commit/push and exact remote SHA.

Fixed parameters are the spec values, not a sweep. Camera integers use np.rint ties-to-even (same old source convention), world/camera-z metres. Atom point order is geometry-derived with deterministic atom-ID tie-breaking; sparse atom-candidate incidence keeps candidate index+1 owners. Pixel contingencies count only each shared z-buffer winner with positive measured-depth-consistent instance evidence. Leave-atom-out subtracts the atom histogram from both candidate and observation operands. Absent remainder abstains. Missing fallback scores use .5; candidates need >=2 views to challenge. LOCAL and SPATIAL share exactly the persisted unaries/feasibility, no score/model blending.

Validation command: /home/ww/miniconda3/envs/ovimap-map/bin/python -m pytest -q tests/evaluation/test_static_local_ownership.py tests/evaluation/test_static_t1_attribution.py tests/evaluation/test_static_proposal_fusion.py tests/evaluation/test_static_projected_masks.py tests/evaluation/test_static_projected_instance_metrics.py . Prediction/evaluation CLI will have explicit --config/--output, recorded actual argv+interpreter; CPU threads=8, chunks=65536, one run at a time. No installations or previous 34-cell rerun.
