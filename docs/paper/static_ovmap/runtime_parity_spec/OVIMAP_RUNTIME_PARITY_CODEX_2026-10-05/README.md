# OVI-MAP semantic recovery: same-output runtime task

**Entry point:** `CODEX_FINAL_EXECUTION_EN.md`.

This is a Codex execution specification, not completed production development or a
new benchmark report. The goal is to obtain the **same measured v2 predictions
faster**, measure the complete incremental recovery call honestly, and close the
paper's runtime/readout evidence without expanding its scientific experiment matrix.

Read the final instruction, `IMPLEMENTATION_CONTRACTS.md`, `TIMING_AND_TABLES.md`,
then `PROTOCOL_SPEC.json`. The JSON is the machine-readable workload definition;
the Markdown defines semantics. A contradiction must be recorded and resolved
before the affected measurement, never by choosing the better result.

## Deliverables in this package

- `CODEX_FINAL_EXECUTION_EN.md`: production development, binding, selection,
  execution, table generation and GitHub publication instructions.
- `IMPLEMENTATION_CONTRACTS.md`: conservative projection, exact top-k, I/O,
  precision, identity, and reference/optimized parity contracts.
- `TIMING_AND_TABLES.md`: timing boundary, repeated-run schedule, stage attribution,
  and compact main/supplementary table rules.
- `CODE_REVIEW_AND_PLAN_ZH.md`: Chinese explanation of the source-grounded decisions.
- `SOURCE_EVIDENCE.md` / `SOURCE_EVIDENCE.json`: inspected source paths and unresolved v2 boundary.
- `TABLE_TEMPLATES.tex`: optional compact templates; em dashes are placeholders.
- `reference/`: small executable reference invariants, **not** replacement production code.
- `AUDIT_REPORT.json`: actual package-level checks; not CUDA or dataset validation.
- `tables/LAYOUT_PREVIEW.pdf`: one-page placeholder preview, not measured tables.

The inspected remote revision is `77335848b776d99ffb4ca772e33fa391d9fd8e5e`.
It does **not** contain the measured area-fallback v2 at preparation time. Codex
must bind the real server-side v2 producers/results first. The supplied old source
paths are evidence anchors, not permission to recreate the fallback from prose.

No production code was changed, no GPU inference was run, and no remote GitHub
write was performed in preparing this package.
