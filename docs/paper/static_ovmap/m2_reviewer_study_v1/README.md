# OVI-MAP M2_CAL reviewer-evidence execution pack

**Start here:** give Codex the entire directory and require execution of `CODEX_FINAL_EXECUTION_EN.md` using `PROTOCOL_SPEC.json`.

This pack specifies code to implement and experiments to run. It contains no new benchmark results and does not modify a remote repository by itself.

Files:

- `CODEX_FINAL_EXECUTION_EN.md`: self-contained English implementation, evaluation, selection, and publication directive.
- `PROTOCOL_SPEC.json`: machine-readable method matrix, budgets, data permissions and outputs.
- `CODE_REVIEW_AND_AUDIT_ZH.md`: Chinese explanation of evidence, code binding, tradeoffs and audit corrections.
- `SOURCES.md`, `SOURCE_INVENTORY.json`: pinned implementation/primary-literature references; no claim of target-server asset access.
- `PACKAGE_AUDIT.json`: package checks, not model tests.
- `MANIFEST.sha256`: checksums of the package's files.

Target repository: `Orangekostar/oviovo` at `8fee8294c1a3e83feeae28782f6ef4700f08d35c`.
New branch: `research/ovimap-m2-reviewer-evidence-v1`.

Suggested Codex start message:

```text
Read CODEX_FINAL_EXECUTION_EN.md and PROTOCOL_SPEC.json. Implement and execute
the M2 reviewer-evidence study, not another planning-only or smoke-only task.
Preserve legacy source maps and results. Complete the cached ablations, both
ranking/aggregation conventions, B200 query controls, attribution/calibration
analysis and text-only robustness tests. Execute the exactly defined optional
curve and fresh-scene stages when their conditions hold. Publish actual code,
small results and the three reports, then verify the full remote commit SHA.
Do not tune on Replica, replace missing evidence with invented scores, or run
broad safety/regression checks unrelated to this task. Let the evidence support
simplification when simpler controls match or outperform M2_CAL.
```
