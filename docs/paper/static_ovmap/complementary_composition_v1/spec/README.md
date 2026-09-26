# OVI-MAP complementary composition — Codex task package

**Reviewed base:** `Orangekostar/oviovo@498f2c5a2fa511c919a83ff9c27946e37cb46a0f`  
**New task branch:** `research/ovimap-complementary-composition-v1`

This package instructs Codex to implement and run object-level fusion, fixed-query-trajectory SigLIP2 rereads, and a shared-budget mixed query policy. It does not contain new benchmark results or already implemented composition code.

## Files

- `CODEX_FINAL_EXECUTION_EN.md` — authoritative, self-contained execution contract.
- `PROTOCOL_SPEC.json` — authoritative constants and experiment specification; paths must be bound into a runtime config.
- `CODE_REVIEW_AND_AUDIT_ZH.md` — Chinese source-to-task review and design audit.
- `SOURCES.md` — immutable repository references and two external primary sources.
- `audit_package.py` / `PACKAGE_AUDIT.json` — task-package checks only; not method or benchmark tests.

## Start Codex with

```text
Read CODEX_FINAL_EXECUTION_EN.md and PROTOCOL_SPEC.json completely. Execute this
new composition study in Orangekostar/oviovo from the pinned revision.

Implement the six mandatory output configurations and conditional M6, run actual
CAL/regression evaluations, freeze one nominee, and execute the declared bounded
confirmation under the new study contract. Reuse the old native-v10 captures,
weights and verified caches; do not rewrite the previous frozen study.

Do not stop at a proposal, smoke-only implementation, all-null report, or another
large test/release audit. Preserve negative and no-intervention results honestly.
Commit and push the scoped code, configurations, new small results and both
Markdown reports to the new task branch. Verify the complete remote SHA.
```

The existing two confirmation scenes may be processed only after the new selection lock and current exposure check. No new restricted-data download is authorized. The original eight FIT scenes are not rerun, and no Q or semantic selector head is retrained. New inference uses only native SigLIP and the previously bound SigLIP2; scalar temperature fitting is the only added fitting step.

## Audit this task package

```bash
python audit_package.py
```

This command validates package content and internal constants and writes a digest manifest. It neither connects to the repository nor executes models.
