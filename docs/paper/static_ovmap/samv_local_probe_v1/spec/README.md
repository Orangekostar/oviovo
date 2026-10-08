# SAM-V local-map exploration — Codex execution package

Entry point: **CODEX_FINAL_EXECUTION_EN.md**.

This is an execution specification, not a delivered SAM-V integration or new experiment result.
It authorizes a small four-scene study only: six main arms, plus a D2 reference.
Do not automatically expand to Replica-8/CF18, train a network, or replace production.

Read in order:
1. CODEX_FINAL_EXECUTION_EN.md — end-to-end responsibilities and exact CLI contract.
2. IMPLEMENTATION_CONTRACTS.md — data, prompting, inference, map repair and semantic rules.
3. EVALUATION_AND_SELECTION.md — scoring, diagnoses, promotion and measurement.
4. PROTOCOL_SPEC.json — machine-readable constants. These documents must agree.
5. SOURCE_EVIDENCE.md — inspected source anchors and unresolved prerequisites.

CODE_REVIEW_AND_PLAN_ZH.md summarizes the rationale in Chinese. The reference/
code contains small synthetic tests of the proposed rules, not SAM-V/FC/GPU tests.
AUDIT_REPORT.json records only checks actually executed while preparing this package.

Two repositories are pinned:
- Orangekostar/oviovo @ a95c24d990cea57b14bb537e9acca95b95659bda
- gong208/SAM-V @ 33fab24c1d0d21ac56f4e471abf30c6de3e7018b

Checkpoint locations are author-declared; actual files, hashes, licenses and loads
must be bound on the execution server before the implementation freeze. No weights,
font files, scans or previous private maps are included in this package.
