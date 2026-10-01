# OVI-MAP recovery wave 2 — Codex execution package

This is a development-and-experiment specification, not a completed algorithm or measured gain.

Start with **CODEX_FINAL_EXECUTION_EN.md**. Numeric settings and method identifiers are in
**PROTOCOL_SPEC.json**. Detailed contracts are in **IMPLEMENTATION_CONTRACTS.md**.
**CODE_REVIEW_AND_PLAN_ZH.md** explains why these experiments are prioritized.
**SOURCE_EVIDENCE.md** identifies the actual inspected code and measured records.
**reference_kernels.py** and **test_reference_kernels.py** are small standalone examples for
checking the proposed semantics; they are not a replacement for the real mapper or evaluator.

Base: Orangekostar/oviovo@c21297413954ecb7d706d1050a8e07822938ea6f.
Task branch: research/ovimap-recovery-wave2-v1.

Objective: improve the complete method relative to **BB00_NATIVE + D2**, not merely recover
from the deliberately damaged BIDIR arm. Preserve official metrics and publish negative,
redundant and resource-screened outcomes. No guarantee of improvement is made.

A new arm must demonstrate an actual prediction intervention or be explicitly equivalent.
No full-map neural reruns are needed for the fixed-map weight controls. Recovery may require
new FC pooling or a capped number of missing frame encodings. New association/front-end maps
need their own semantic evidence; frozen model rules do not mean reusable owner-level scores.

No training, new checkpoints, new raw datasets, SAM3, or new SAM2/CropFormer forwards are in scope.
The maximum is 28 newly reconstructed full map-scene configurations, not 64; a geometry
screen prevents paying for semantic inference on catastrophic fragmentation.

## Recheck the local specification bundle

```bash
python -m unittest -v test_reference_kernels.py > REFERENCE_TEST_LOG.txt 2>&1
python audit_package.py
```

This rechecks the synthetic examples and rebuilds the manifest/ZIP. It does not run the actual project.
