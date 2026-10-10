# FC-preserving readout and absolute-support routing — Codex execution pack

**Date:** 2026-10-10. **Repository base:** `6767eb90c2fd6999621b356b87269c012821713a`.

Start with **[CODEX_FINAL_EXECUTION_EN.md](CODEX_FINAL_EXECUTION_EN.md)**. Then read [MODEL_CONTRACT.md](MODEL_CONTRACT.md), [DATA_AND_EXECUTION.md](DATA_AND_EXECUTION.md), and [EVALUATION_AND_RELEASE.md](EVALUATION_AND_RELEASE.md). The Chinese explanation is [CODE_REVIEW_AND_PLAN_ZH.md](CODE_REVIEW_AND_PLAN_ZH.md).

This is a **new supervised research experiment**, not a patch to the completed LR experiment. It separates two questions: preserving the original FC readout while learning, and allowing an entire unreliable local-support group to contribute no update. Existing G1 geometry and recovery labels remain fixed.

`PROTOCOL_SPEC.json` is the machine-readable registry. Numerical definitions in the contracts complete it; there are no hidden tuning grids. `EXPERIMENT_MATRIX.csv` gives the paired comparisons. `SOURCE_EVIDENCE.md` separates inspected repository facts, literature context, and new experimental choices.

`reference/` contains small executable arithmetic/gradient examples, **not an implemented production model**. `audit/` records checks run while authoring this pack. These tests do not establish FC/SAM2 loading, raw-data availability, actual training correctness, speed, or accuracy. Those require the finite production checks in the directive.

Expected bounded paths: 8,000 updates if the first 2D screen fails; 10,000 if its repeat fails; at most 30,000 if both pass and all five geometry arms are trained in both seeds. No latency gate. Normal-path complete-map coverage is at most 390 logical rows/30 ordered pools, with historical baselines reused only on exact scoring identity.

Do not push this instruction pack alone and report the task as complete. Codex must implement, run the applicable scientific path, document all outcomes, and publish real code/results/selected heads. A negative completed path is valid; a technical block is not a successful experiment.
