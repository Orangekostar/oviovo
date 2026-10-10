# Learned multi-view object readout — Codex execution pack

Date: 2026-10-09. Repository: Orangekostar/oviovo. Baseline source: `30f59c1fa3d783054dc53fc6693e1fd68e8c2a6d`.

Start with **CODEX_FINAL_EXECUTION_EN.md**. Read DATA_AND_SPLITS.md, MODEL_AND_TRAINING_CONTRACT.md, and EVALUATION_AND_RELEASE.md before implementing. PROTOCOL_SPEC.json is the numerical source of truth; the Markdown defines algorithms and exceptions. If an actual contradiction is found, fix the task-owned specification before scientific training and preserve a concise amendment. Do not silently choose a favorable interpretation after results.

This pack is a plan and small CPU reference checks, NOT a trained model or a completed server experiment. New independent training data have not been confirmed available. Inventory and authorized materialization are part of the job. Do not substitute the exposed Replica8/CF18 cohorts for training data.

## Scope

Train four matched region/object heads on new ScanNet families while freezing FC visual and text backbones. Evaluate 2/4/8-view ordinary FC, retrained Mask-Adapter architecture, view-grouped and physically grouped multi-view aggregation, and targeted auxiliary supervision. Keep the current TSDF, G1 instance partition, N/Q evidence, and recovered labels fixed. Accuracy and mechanism take priority; runtime is logged, not a selection gate.

A learned update-risk gate, new foundation models, online deployment, new OVI map construction and an independent whole-map confirmation benchmark are **not** implemented in this first learning phase. The held-out new families support proposal-level recognition evidence only. Do not describe that as independent whole-map AP.

## Files

- CODEX_FINAL_EXECUTION_EN.md — actual staged execution, work packages, commands and completion.
- CODE_REVIEW_AND_PLAN_ZH.md — Chinese evidence/decision/task mapping.
- DATA_AND_SPLITS.md — authorized data discovery, family and category separation, camera registration, masks and observations.
- MODEL_AND_TRAINING_CONTRACT.md — tensors, exact baselines, proposed grouped model, losses and training.
- EVALUATION_AND_RELEASE.md — prediction interface, selections, main tables, repeat run and GitHub delivery.
- SOURCE_EVIDENCE.md — inspected sources and boundaries of claims.
- PROTOCOL_SPEC.json — fixed numeric settings and method registry.
- reference/ — small synthetic reference kernels/tests, not the production implementation.
- audit/ — pack consistency/test results produced during this delivery.

No artifact in this pack supplies synthetic scores as scientific results. Scientific fields remain unfilled until real execution.
