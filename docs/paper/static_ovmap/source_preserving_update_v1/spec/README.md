# Source-preserving semantic update: Codex execution package

**Start with `CODEX_FINAL_EXECUTION_EN.md`.** This package specifies a new, bounded research execution. It is not a completed experiment or an implementation already installed in the repository.

目标：验证保留N/Q和旧F、只调整新增区域证据的接入方式，能否保留CF18已有纠错而减少Replica误改。

Base: `Orangekostar/oviovo@b000355eb8f91491e002df1499bd0170d91c35ef`.
Target branch: `research/ovimap-source-preserving-update-v1`.

## Read order

1. `CODEX_FINAL_EXECUTION_EN.md`: mandatory phases, executable interface to implement, completion and publication.
2. `IMPLEMENTATION_CONTRACTS.md`: exact evidence alignment, nine methods, output and failure semantics.
3. `EVALUATION_AND_SELECTION.md`: pooling, paired interpretation, research selection, cost accounting.
4. `PROTOCOL_SPEC.json`: authoritative numeric constants and identifiers.
5. `SOURCE_EVIDENCE.md`: verified code paths, symbols and primary references.

`CODE_REVIEW_AND_PLAN_ZH.md` and `EXPERIMENT_MATRIX.md` summarize the design. `reference/` contains only small mathematical reference functions/tests, not a substitute for production integration.

The task is the evidence-injection stage, **not** the later learned update-risk model, adaptive acquisition, structural repair, or independent confirmation study. Those need a separate protocol informed by this experiment. All nine conditions must be attempted and evaluated on the existing full cohorts; there is no post-hoc owner, class, or dataset routing.

The package audit checks specification consistency and reference mathematics. It does not attest that the server data are currently available, the production code is bug-free, or a performance gain exists.
