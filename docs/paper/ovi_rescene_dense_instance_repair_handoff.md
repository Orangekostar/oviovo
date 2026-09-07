# OVI-MAP Backbone x ReScene Dense Instance Repair Handoff

Date: 2026-09-07

Branch: `research/ovi-rescene-dense-instance-repair`

Baseline commit: `dab81331017c7359cf2775f82351c97d2fe1c8bd`

## Outcome

The implementation preserves each visit's native OVI-MAP XYZ and source row
order, projects ReScene point masks back through the explicit D-to-A-to-M-to-D
index chain, and emits exclusive query-owned instances plus lossless residual,
background, and unknown ownership. P0, P1, and P2 therefore differ only in
instance ownership, not geometry.

The scientific result is negative at the primary threshold. Across three pairs
and 164 visit-level GT instances, P0, P1, and P2 all have zero TP at IoU 0.50.
At IoU 0.25, pair-macro F1 is 0.1367 / 0.1566 / 0.0929 for P0 / P1 / P2.
P2 genuinely splits and merges OVI entities, but raw mask quality and fixed
surface coverage are too weak, and the extra fragments increase FP. The result
supports an auditable OVI-backbone integration and a precise bottleneck
diagnosis, not an accuracy or SOTA claim.

## Executed Scope

| Pair | Role | OVI source | Dense evaluation | Fixed-P2 association |
| --- | --- | --- | --- | --- |
| `scene0109_00-scene0109_01` | D2_DEV | reused prior independent per-visit OVI maps | PASS | PASS |
| `scene0359_00-scene0359_01` | frozen D2_EVAL | two new independent per-visit OVI maps | PASS | PASS |
| `scene0459_00-scene0459_01` | frozen D2_EVAL | two new independent per-visit OVI maps | PASS | PASS |

The two D2_EVAL environments were selected before their dense metrics were
opened and were not replaced or retuned. Both are from the checkpoint validation
split, so they are resolver-held-out but not checkpoint-held-out. Office and all
other 3RScan pairs were not run and are outside the claim.

The scene0359 mapper audited 100/125 and 94/106 frames, recording 25 and 12
mapper-skipped frames. Scene0459 audited 529/592 and 524/582 frames, recording
63 and 58 skipped frames. Skipped frames remain explicit in the native receipts;
they were not silently treated as successful mappings.

## Evidence Interpretation

Mean dense OVI coverage of complete GT ranges from 0.1531 to 0.3787 across the
six visits. Union-oracle IoU is above best-atomic IoU, showing OVI fragmentation,
but point-level split-oracle IoU is higher again, showing contamination or
undersegmentation on the fixed surface. Raw-mask IoU is only 0.0383-0.0958,
well below the 0.1531-0.3787 split oracle. All endpoint failures therefore
remain `mixed`; sensor-visible GT was not derived, so missing observation cannot
be separated from reconstruction loss.

P2 records 4-55 split parents and 1-19 merged candidates per visit while
preserving all 16,136,910 dense points across the three pairs. Despite that
structural freedom, it has no valid two-endpoint identity at IoU 0.50. At IoU
0.25, geometry on the fixed P2 pool is strongest (G supported: 5 TP / 134 FP,
5/6 conditional recall), while R records 1 TP / 12 FP. This pool is itself
joint-query-derived and is disclosed as such.

No observation support or evaluator definition was changed. Exact-floor 5 cm
IoU 0.50 remains primary and 0.25 remains sensitivity. System P0/P1/P2 identity
uses each method's own frozen endpoint pool; the G/F/R table separately uses one
fixed P2 pool. The old D1 result remains a cached projection, not a new network
forward. No D1 forward, adapter training, checkpoint update, or threshold sweep
was run.

Historical recovery was not eligible: every system and fixed-P2 association has
zero TP and zero rigid recall at IoU 0.50, and no sensor-visible recoverable
surface was established. `C0`, `C_G`, `C_R`, `O_ID`, and `O_POSE` are recorded
as `NOT_RUN_INELIGIBLE`; all recovery metric fields are null, not zero.

## Claim Boundary

Supported claims are limited to lossless dense point reassignment, real split
and merge behavior, two preregistered D2_EVAL executions, and the measured
coverage/raw-mask bottleneck. Unsupported claims include primary-threshold or
semantic improvement, geometry improvement, successful recovery, adaptation
benefit, checkpoint-held-out generalization, Office performance, and SOTA.

The single highest-value next action is an actual camera/depth-defined D1
re-forward beside D0 and D2, evaluated with the same raw-mask endpoint metrics.
That experiment can separate observation support from OVI-domain mask failure
before any decoder adaptation is authorized.

## Artifact Locations

Tracked compact evidence is under:

```text
configs/evaluation/results/ovi_rescene_dense_instance_repair/
```

Large PLY, NPZ, RGB-D, and third-party checkpoint artifacts remain local under:

```text
/home/ww/oviovo_baseline_runs/20260907_ovi_rescene_dense_instance_repair/
```

Their paths, sizes, and SHA-256 values are recorded in
`compact_artifact_index.json`; they are not claimed as uploaded Git content.

Primary implementation entry points are:

```text
src/oviv2/rescene_dense_instance_readout.py
src/evaluation/ovi_endpoint_diagnosis.py
src/evaluation/dense_instance_repair_metrics.py
src/evaluation/dense_instance_repair_artifacts.py
scripts/evaluation/run_ovi_rescene_dense_instance_repair.py
scripts/evaluation/run_ovi_rescene_dense_p2_association.py
scripts/evaluation/aggregate_ovi_rescene_dense_instance_repair.py
```

## Reproduction

Run from the repository root with the `persist4d` environment. Producers publish
exclusively, so exact reruns require a fresh output root.

The exact configs publish exclusively. Copy a config and change both output
roots before rerunning. The compact inputs and outputs are source-bound; native
mapping commands and asset hashes are retained in the two `d2_ovimap_receipt`
files and their local manifests.

```bash
cd /home/ww/.config/superpowers/worktrees/ovi-rescene-dense-instance-repair

/home/ww/miniconda3/envs/persist4d/bin/python \
  scripts/evaluation/run_ovi_rescene_dense_instance_repair.py \
  --config <fresh-dense-config.json> --evaluated-commit "$(git rev-parse HEAD)"

/home/ww/miniconda3/envs/persist4d/bin/python \
  scripts/evaluation/run_ovi_rescene_dense_p2_association.py \
  --config <fresh-dense-config.json> --evaluated-commit "$(git rev-parse HEAD)"

/home/ww/miniconda3/envs/persist4d/bin/python \
  scripts/evaluation/aggregate_ovi_rescene_dense_instance_repair.py \
  --source scene0109_00-scene0109_01=<result-root> \
  --source scene0359_00-scene0359_01=<result-root> \
  --source scene0459_00-scene0459_01=<result-root> \
  --output-root <fresh-aggregate-root>
```

Dense result evaluated commits are `e19958627cd5f9ed4e0d8740d0698cf08902db6f`
(DEV), `579105196f36f74b17b8dc645d6737b0a7a74294` (0359), and
`f9089393b8932beb1417aaab4c2b763c7b920c64` (0459). Fixed-P2 association
evaluated commits are `7ddcf0e79eddc83df05fbad0fccd2427c7382913`,
`5035222f0ef5b62679fc1dca02c34bfd4879d54d`, and
`a617a171d2a80f2d232268d1d6fbd9ebf7be5f52`, respectively.

## Verification

Fresh validation on the final implementation and evidence tree:

- changed modules and direct dependencies: `114 passed`;
- all 21 branch-changed Python files: `py_compile` PASS;
- the same 21 Python files: Ruff PASS;
- aggregate source hashes, recovery bindings, and Markdown numeric cells:
  `NUMERIC_AND_BINDING_AUDIT=PASS`;
- `git diff --check`: PASS.

The full repository suite was not rerun. The branch adds isolated readout,
evaluation, artifact, and orchestration modules; the three touched shared
surfaces are covered by their direct `run_ovimap_native`, object-transfer, pair
view, artifact, association, and temporal-grouping tests included in the 114.

## Delivery

The delivery commit and remote SHA are reported after the non-force push. The
committed handoff intentionally does not contain its own SHA.
