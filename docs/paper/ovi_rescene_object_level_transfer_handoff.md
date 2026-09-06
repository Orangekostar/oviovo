# OVI-MAP Backbone x ReScene Object-Level Transfer Handoff

Date: 2026-09-06

Branch: `research/ovi-rescene-object-level-transfer`

Baseline commit: `48117bd6f502d7d289eafe81392ef55748bd6c4a`

## Outcome

The complete engineering path is implemented: real 3RScan RGB-D visits are
materialized and independently mapped by native OVI-MAP; a source-bound D2 pair
feeds shared-candidate G/F/R association, dense object grouping, D0-D1-D2
diagnosis, and C0/C1/C2/O1/O2 historical recovery. The measured scientific
result is bounded and mostly negative: grouping yields only a weak IoU-0.25
gain, C1 yields a 0.000268 surface-F1 delta on one pair, and C2/O1/O2 do not
recover any evaluable opportunity voxel.

The recommended architecture remains OVI-MAP as the geometry and open-vocabulary
semantic backbone, with ReScene restricted to temporal identity evidence. The
next experiment should improve OVI proposal/endpoint coverage before decoder
training or recovery-rule tuning.

## Acceptance Status

| Requirement | Status | Evidence |
| --- | --- | --- |
| Fixed 0.3 resolver outside DEV6 | Complete | Two resolver-held-out development environments in `resolver_fixed_threshold_transfer.csv` |
| Real RGB-D to OVI D2 artifact | Complete for one D2_DEV pair | `d2_ovimap_smoke_v1.json`, `d2_pair_view_v1.json` |
| Shared-candidate G/F/R with GT | Complete for the same pair | `shared_candidate_gfr.csv` |
| Actual instance map | Complete | U3 map manifest and tracked previews in `instance_readout_v1.json` |
| Learned recovery plus oracle diagnosis | Complete with negative/null-preserving result | `completion_oracles.csv`, `completion_oracles_v1.json` |
| Held-out D2_EVAL and Office | Not run | No claim is made |
| Adaptation training | Not run by frozen gate | `adaptation_decision_v2.json` selected `NO_ADAPTATION` |

## Key Result

At D2 IoU 0.25, G full records 1 TP / 20 FP / 13 GT, F records
0 / 17 / 13, and R records 0 / 1 / 13 on the same 29 x 21 OVI candidate
pool. Only one persistent GT identity has both predicted endpoints. U3 reduces
the sensitivity FP count from 18 to 17 without changing XYZ, while all variants
remain at zero TP for the primary IoU 0.50 metric.

In historical recovery, C1 changes surface F1 from 0.442020 to 0.442289; C2 is
unchanged. O2 has the official object pose but produces 16 new unevaluable voxels
and zero new correct voxels. This makes proposal/endpoint and evaluable-surface
coverage the next target; it does not justify tuning visibility gates to the
single development pair.

## Artifact Locations

Tracked results are under:

```text
configs/evaluation/results/ovi_rescene_object_level_transfer/
```

The compact inventory, sizes, hashes, and transitive local manifests are in:

```text
configs/evaluation/results/ovi_rescene_object_level_transfer/compact_artifact_index.json
```

Large PLY/NPZ outputs remain local under:

```text
/home/ww/oviovo_baseline_runs/20260906_ovi_rescene_object_level_transfer/
```

They are marked `LARGE_ARTIFACTS_LOCAL_ONLY`; raw RGB-D, third-party weights,
and full meshes are not committed to Git.

## Reproduction

Use the `persist4d` environment and repository root. Result writers are
exclusive: exact reruns require a fresh output root or a copied config with new
output paths.

```bash
cd /home/ww/.config/superpowers/worktrees/ovi-rescene-real-evidence
PYTHONPATH=. /home/ww/miniconda3/envs/persist4d/bin/python \
  scripts/evaluation/run_ovi_rescene_transfer_diagnosis.py \
  --config configs/evaluation/ovi_rescene_transfer_diagnosis_v1.json \
  --evaluated-commit d295568fea0d5a2abee2775a577964fae12589bf

PYTHONPATH=. /home/ww/miniconda3/envs/persist4d/bin/python \
  scripts/evaluation/run_ovi_rescene_object_recovery.py \
  --config configs/evaluation/ovi_rescene_object_recovery_v1.json \
  --evaluated-commit 048d2edabaceb8482b9cff65d381c7eb23ef623c
```

Primary verification:

```bash
/home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q \
  tests/evaluation/test_run_ovi_rescene_object_recovery.py \
  tests/oviv2/test_two_visit_registration.py \
  tests/oviv2/test_two_visit_dense_recovery.py \
  tests/oviv2/test_two_visit_current_map.py \
  tests/evaluation/test_ovi_ownership_completion.py
```

Recorded validation on the final evidence chain:

- Branch-changed tests plus direct dependencies and the frozen two-visit matrix:
  `256 passed`.
- Artifact binding audit: 25 tracked and 27 local-only records passed.
- All branch-changed Python files passed `py_compile`; `git diff --check` passed.
- Full repository: `6246 passed, 10 skipped, 2 failed`. Both failures are in
  files unchanged from the baseline: NumPy 2.2 internally calls the private
  header reader that its test forbids, and `transformers` is absent from the
  `persist4d` environment. Ruff is also not installed in that environment.

## Delivery

The final upload check is intentionally external to this committed file to avoid
a self-referential commit SHA. Report `UPLOAD_STATUS=VERIFIED` only after the
local HEAD exactly equals the remote branch SHA. No force push is permitted.
