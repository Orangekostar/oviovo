# CVPR compact-table execution package

Entry point: **CODEX_FINAL_EXECUTION_EN.md**.

This package specifies development and measurement, not completed experiments.
The primary method is fixed as A3 (D2 + independent single-view FC recovery).
The G1/G3 methods have not been run by the author of this package.

Read in order:
1. CODEX_FINAL_EXECUTION_EN.md: complete execution and release mandate.
2. IMPLEMENTATION_CONTRACTS.md: numerical, interface, export and cost definitions.
3. PROTOCOL_SPEC.json: the same choices in machine-readable form.
4. SOURCE_EVIDENCE.md: verified code locations and public protocol references.
5. CODE_REVIEW_AND_PLAN_ZH.md: Chinese reasoning, differences from the outline.
6. TABLE_CONTRACTS.md and tables/: exact manuscript tables and provenance rules.

The reference_checks folder checks specification mathematics and bookkeeping only.
It is not the production projector, a server integration test, or a benchmark result.
Run `python reference_checks/check_contracts.py` and `python audit_package.py` from
this folder to reproduce package-level checks.

No weights, font files, training data, licensed scans, server caches, or repository
credentials are distributed. External comparison numbers are attributed literature
values in a separate block, not experiments performed by Codex.
