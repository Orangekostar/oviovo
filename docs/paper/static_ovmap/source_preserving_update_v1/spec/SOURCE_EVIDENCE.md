# Source evidence and design provenance

Checked on 2026-10-08. Repository anchor: `Orangekostar/oviovo@b000355eb8f91491e002df1499bd0170d91c35ef`; the branch `research/ovimap-minimal-instance-repair-v1` resolved to this commit during this review. All repository links below are commit-pinned. Path/symbol anchors are used instead of invented GitHub line numbers.

## Reviewed repository interfaces

| ID | Source and symbols | Observed fact | Consequence for this task |
|---|---|---|---|
| S01 | [MINIMAL_REPAIR_RESULTS.md](https://github.com/Orangekostar/oviovo/blob/b000355eb8f91491e002df1499bd0170d91c35ef/docs/paper/static_ovmap/MINIMAL_REPAIR_RESULTS.md) | Complete 234/18 study. IR06 improves CF18 versus D2 but loses Replica; structural additions have little metric effect | Concentrate on incumbent evidence injection rather than another structural search |
| S02 | [binding.py](https://github.com/Orangekostar/oviovo/blob/b000355eb8f91491e002df1499bd0170d91c35ef/src/static_ovmap/minimal_instance_repair/binding.py), `load_scene`, `validate_baseline` | Loads N/Q/F from context, checks valid-ID order, reconstructs D2, loads actual G1/D2 and target projection | Reuse verified distributions and actual payloads; do not enter all old setup phases |
| S03 | [reread.py](https://github.com/Orangekostar/oviovo/blob/b000355eb8f91491e002df1499bd0170d91c35ef/src/static_ovmap/minimal_instance_repair/reread.py), `select_incumbents`, `reread_decision` | Selects lowest D2 margins, averages FULL cosine arrays, hard replaces against old class; IR07 adds view/region conditions | Reconstruct historical decisions; new arms retain full old-source distributions |
| S04 | [recognition_plan.py](https://github.com/Orangekostar/oviovo/blob/b000355eb8f91491e002df1499bd0170d91c35ef/src/static_ovmap/minimal_instance_repair/recognition_plan.py), `prepare_scene`, `region_identity`, `unpack_region` | Stores original support, FULL/core/intersection masks, frame/bank, resize/pad and image identity | Lock original selection and exactly pair FULL observations; no new projection |
| S05 | [recognition_worker.py](https://github.com/Orangekostar/oviovo/blob/b000355eb8f91491e002df1499bd0170d91c35ef/src/static_ovmap/minimal_instance_repair/recognition_worker.py), `acquire_frame`, `assemble_decisions` | Successful frame receipts retain scores, dense receipt/arrays, projected vectors and input tensor identities; incumbent regions use AnyUp while UNION_FC uses coarse pooling | Read A from actual parent outputs, add only missing same-view C; never assume C already exists |
| S06 | [backbone readouts.py](https://github.com/Orangekostar/oviovo/blob/b000355eb8f91491e002df1499bd0170d91c35ef/src/static_ovmap/backbone_wave1/readouts.py), `fuse_readout` | Sources converted using final source temperatures; D2 handled by `simple_dependence`; all-unavailable fallback retains native class | Reconstruct the real baseline rather than implement an approximate name-only D2 |
| S07 | [residuals.py](https://github.com/Orangekostar/oviovo/blob/b000355eb8f91491e002df1499bd0170d91c35ef/src/static_ovmap/paired_evidence_study/residuals.py), `pool`, `simple_dependence`, `paired_residual`, `ratio_update`, `blend` | N/Q group then F group; existing probability-space mixing and paired residual operators | Reuse these operators as controls; not new mathematical inventions |
| S08 | [area_fallback.py](https://github.com/Orangekostar/oviovo/blob/b000355eb8f91491e002df1499bd0170d91c35ef/src/static_ovmap/cvpr_compact/area_fallback.py), `pool_region`, `region_vector` | Original hard support or area fallback, then original FC visual head and normalization | Exact same-view coarse control; avoid an invented pooling/head path |
| S09 | [outputs.py](https://github.com/Orangekostar/oviovo/blob/b000355eb8f91491e002df1499bd0170d91c35ef/src/static_ovmap/minimal_instance_repair/outputs.py), `construct_partition` | Copies G1, can relabel incumbents, cancels uniform class change touching raw-zero rows, recomputes official ranks | Preserve cancellation uniformly; do not import old headline IR06 numbers into a narrowed replay |
| S10 | [evaluation.py](https://github.com/Orangekostar/oviovo/blob/b000355eb8f91491e002df1499bd0170d91c35ef/src/static_ovmap/minimal_instance_repair/evaluation.py), `partition_evaluator`, `_evaluate_scene`, `evaluate_study` | Registry derived from actual positive owners, exact-context aliases, whole semantic confusion, ordered pools | Narrow new evaluator must preserve full G1 recovery owners and actual changed ranks |
| S11 | [reviewer evaluator](https://github.com/Orangekostar/oviovo/blob/b000355eb8f91491e002df1499bd0170d91c35ef/src/static_ovmap/m2_reviewer_study/evaluation.py), `official_view`, `SceneEvaluator` | Masks bound during initialization; current predicted-class area ranks | A probability change alone is not AP improvement; correct mask registry and ranks are essential |
| S12 | [parent config](https://github.com/Orangekostar/oviovo/blob/b000355eb8f91491e002df1499bd0170d91c35ef/configs/static_ovmap/minimal_instance_repair_v1.json) | Exact 8+18 cohort order, 16 incumbents/two FULL banks, original AnyUp asset identity, five-metric gate | Inherit cohorts/models; create a separate task protocol rather than editing the parent |
| S13 | [office1 saved decisions](https://github.com/Orangekostar/oviovo/blob/b000355eb8f91491e002df1499bd0170d91c35ef/artifacts/static_ovmap/minimal_instance_repair_v1/scenes/office1/semantic_decisions.json) | Inspected the beginning: actual aggregate scores, FULL view records and per-region identities exist | Supports cache-first design, but does not certify every one of 26 local files is present |

## Primary external sources, narrowly used

- [AnyUp author project](https://wimmerth.github.io/anyup/): identifies the encoder-agnostic feature-upsample method and ICLR 2026 Oral status. This task reuses its already-computed ordinary model outputs. The new experiment does not claim to invent AnyUp or verify its general claims across backbones.
- [SciPy softmax](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html): normalized exponentiation and stable shifting.
- [SciPy log_softmax](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_softmax.html): stable log-probabilities, including saturated inputs. Use the installed compatible version; these docs are not authorization to upgrade the server environment.

No external paper's reported AP is inserted into the new paired experiment. This package does not claim that weighted probability pooling, log-score differences or an exact no-op case are new science.

## Known facts versus proposed choices

**Observed:** source grouping, saved cosine aggregation, current output rules, actual previous outcomes and scoring interface.

**Prespecified new choices:** common paired-FULL domain; matching hard/stable replays; alpha=.5, eta=.5; beta=alpha*wF; applying residual only inside F; no additional acceptance threshold; no cold inference campaign in this task.

These new choices were selected for interpretation and controlled cost, not because their gains are known. They must be reported as tested configurations, not values derived from a proven optimum.

**Not verified here:** all parent server files and dense caches; full production reconstruction; new model/scene outcomes; overall deployment runtime; new GitHub publication. Codex must verify the relevant actual inputs and execute the study.
