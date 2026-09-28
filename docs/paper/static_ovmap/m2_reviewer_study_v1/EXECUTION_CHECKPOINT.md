# Execution checkpoint (not completion)

Workspace: /home/ww/crove/ovimap-m2-reviewer-evidence
Branch: research/ovimap-m2-reviewer-evidence-v1
Base: 8fee8294c1a3e83feeae28782f6ef4700f08d35c
Output: /mnt/shared/ww/ovimap-m2-reviewer-evidence-v1/attempt_001

Implemented study-local source binding, genuine score reconstruction, subset
fusion, shared scalar objective, separate ranking views and released pooled call,
and per-scene core execution. All legacy production modules remain unchanged.
The source package is preserved byte-for-byte and its supplied hashes passed.

Actual evidence:

- source_binding.json: 10 scenes, 70 initially bound consumed files.
- source_audits and legacy_parity: both CAL scenes, five controls' labels,
  owners and ranks exact. Native cosine reconstruction has no top1 disagreement.
- calibration/new_folds.json: real opposite-scene shared and cosine-N0 fits.
  Shared T: 0.010000492837221748 in both folds. Cosine N0 held-out0056:
  0.011285594474064128; held-out0534: 0.010000492837221748.
- scene0534_00: all 26 core ranking rows measured; five historical frozen-rank
  controls match all five metrics within1e-10.
- scene0056_00: fixed-adapter run started; inspect process/session and actual
  rows before resuming, never duplicate a live job.
- fresh_inventory.json: 14 authorized local scenes, all in explicit historical
  excluded families, zero eligible fresh families. Fresh success implementation
  is still required; the data limitation does not block other stages.
- 12 focused tests currently pass (fusion, calibration, rank/confusion/export).

Repair: initial evaluation adapter used absolute mask paths, which the released
parser warns against. Changed to relative paths and added evaluator-code cache
identity. One initial scene0056 N0 row and cache are preserved under repairs/
relative_manifest_v1. Recomputed descendants use the fixed adapter; old published
experiments and all prediction decisions are untouched.

Outstanding requirements remain substantial: real official-export/pool parity,
CAL nomination with exact operation-cost tiebreak, final scalars, all eight
Replica core scenes and pools, B200 causal controls/curve gate, object/probability
and cost diagnostics, text robustness, real confirmation success path, complete
runner phases/resume, three reports, scoped publication and full-SHA audit.
Do not mark the study complete from this checkpoint or the unit-test count.
