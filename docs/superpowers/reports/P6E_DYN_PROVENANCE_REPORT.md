# P6-E Official Dyn Provenance Report

## Scope And Contract

- `CODE_EVIDENCE`: the sidecar consumes the existing official
  `dynamic_objects.csv`, official visualization tables, CROVE bridge manifest,
  and composition manifest. It does not replace the evaluator or modify any
  official metric.
- `CODE_EVIDENCE`: candidate runtime entities are joined to the bridge node
  symbols at each of the 43 causal checkpoints, but official dynamic errors
  remain aggregate trajectory mass without a per-error prediction identity.
- Scene: Apartment only. Office was not accessed.

## Coverage And Mass Conservation

| Provenance status | Official Dyn mass | Fraction |
| --- | ---: | ---: |
| Exact | 0 | 0.000000 |
| Ambiguous | 42,175 | 1.000000 |
| Unavailable | 0 | 0.000000 |

`MEASURED_EVIDENCE`: all 43 official query events are represented. Official
mass is conserved exactly as 188 detected + 19,556 missed + 22,431
hallucinated = 42,175. The sidecar exposes 2,877 checkpoint-local candidate
predictions, including 687 marked dynamic, 907 bound, and 1,970 unbound, but no
official error unit identifies which candidate generated it.

Consequently, detected, missed, and hallucinated mass cannot be assigned to a
proven proposal, association, motion, geometry, or identity mechanism bucket.
Candidate-set membership is not treated as exact error provenance.

## Instrumentation Boundary

`CODE_EVIDENCE`: runtime instrumentation cannot recover the missing evaluator
identity because the official dynamic table exposes only aggregate mass.
Joining an instrumented runtime record to that mass by checkpoint or proximity
would be a heuristic and is forbidden by the exact/ambiguous contract.

`MEASURED_EVIDENCE`: an existing instrumentation-only Apartment run was also
checked for formal equivalence before reuse. At 42/43 emitted checkpoints, 294
of 336 checkpoint files differed from the A6 formal source; the first
checkpoint contained 61 instead of 65 temporal entities. This incomplete,
non-equivalent run is excluded from P6-E evidence. No runtime attribution is
therefore represented as formal evidence in the sidecar.

## Decision

`MEASURED_EVIDENCE`: `NO_GO / ATTRIBUTION_AMBIGUOUS`. Exact provenance is 0%,
so the current evidence cannot justify proposal, association, motion, geometry,
or identity-specific optimization.

`MEASURED_EVIDENCE`: ReScene remains `NO_GO / NOT_NEEDED`. There is no exact
evidence that temporal identity representation is the dominant official Dyn
failure source. Future mechanism work remains blocked until the official
evaluation export provides per-error prediction/GT identity or an equivalent
auditable association table.

## Bound Artifacts

| Artifact | SHA-256 | Bytes |
| --- | --- | ---: |
| Dyn provenance JSON | `78784d10bd7a247fa164fdb2aeb8ca3067196fa41b4ef6fc978ff3cb7517a8cd` | 1,381,648 |
| Dyn provenance CSV | `40dcbb5d8b5d1357b4915daf2acd86ed361f120e751897670df80538d75cee84` | 5,068 |
| Official dynamic objects | `61ffb22a2d95f702517abaa4621f0eadd512e50c1cc7c9a7952058c457ac19ff` | 32,763 |
| Official visualization associations | `f8ae36a20300b5958aa7d05638d6e6616c342a90d45239ad4a30da60934c6d13` | 3,968,436 |
| Official visualization objects | `b97ffa10126715f39e40015eec7970cef44e9277f20ed8c31cab9f4d1a0b601e` | 13,217,778 |
| CROVE bridge manifest | `c6e693e169f1adfdec111be0c949b61446ee2f7ed842e8a066324a98eb9668cf` | 3,789,189 |
| P5 composition manifest | `3cfa389183a930ac5fb10aa34093cf63e64e1c8206f1f4e9ccffef1490369fc3` | 95,950 |
| Source trajectories | `ab05c9ec752e77137e24bc747d1692337906f9263ced93208f4f4980877e4a8f` | 4,791,913 |

Published artifacts are under
`/home/ww/oviovo_baseline_runs/20260903_crove_localized_current_ownership/p6e/p5_actual`.

## Reproduction

```bash
python scripts/evaluation/build_crove_dyn_provenance.py \
  --composition-manifest /home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p5/apartment_candidate/run_manifest.json \
  --bridge-manifest /home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p5/bridge_apartment_candidate/bridge_manifest.json \
  --official-dynamic-csv /home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p5/khronos_apartment_candidate/map/results/dynamic_objects.csv \
  --visualization-objects-csv /home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p5/khronos_apartment_candidate/map/eval_visualization/objects.csv \
  --visualization-associations-csv /home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p5/khronos_apartment_candidate/map/eval_visualization/associations.csv \
  --output-root /home/ww/oviovo_baseline_runs/20260903_crove_localized_current_ownership/p6e/p5_actual
```
