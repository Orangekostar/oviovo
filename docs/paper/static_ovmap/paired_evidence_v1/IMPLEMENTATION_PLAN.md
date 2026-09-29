# Paired evidence implementation plan

**Goal:** Execute the supplied paired-evidence protocol on frozen wave-1 inputs and publish measured results.

**Architecture:** A new task-local binder and evidence reader feed independent residual and dependence operators. Predictions are persisted before the released evaluator opens annotations. Scene processes may run concurrently, with four BLAS threads each; CAL selection and transfer freeze are serial barriers. No neural model is loaded.

**Spec:** CODEX_FINAL_EXECUTION_EN.md, PROTOCOL_SPEC.json and MATH_AND_EDGE_CASES.md in this directory.

**Tech stack:** Existing ovimap-map Python, NumPy/SciPy, immutable JSON/NPZ receipts, existing released evaluator.

## Constraints

- Base 9c35333088551a9d4a67937936d4d9c347b2b207; task branch research/ovimap-paired-evidence-v1.
- No image/text forwards, model loads, downloads, temperature fitting, or geometry changes.
- Six controls, eleven new methods, four five-value CAL grids; one conditional composition.
- APall uses actual released thresholds; both ranking views and complete dataset pools.
- CAL global choices freeze before Replica evaluation. Deployment N0_UNCHANGED.
- Missing per-view inputs affect only dependent owners/methods; no fabricated evidence.

## Execution checklist

- [x] Read supplied algorithms, registry, source map, and relevant parent interfaces; isolate branch and archive original package.
- [x] Implement and check numerical operators in residuals.py, dependence.py and pair_graph.py. Tests exercise exact identity/mass, class permutation, finite-difference Jacobian, PSD, exhaustive simplex comparison, graph closed form and alias collapse.
- [x] Implement binding.py and evidence.py; bind input hashes once, recover genuine N/Q/F/O, pair common requests and recover actual masks/views. Reconstruct original B and OVR labels and full probabilities exactly.
- [x] Implement task-local evaluation.py and workflow.py; persist complete predictions, reuse exact evaluator identities with aliases, parallelize independent scenes without changing barriers.
- [x] Run CAL grids and complete pools; selection.py uses fixed practical bands. Run prescribed active combination if triggered and freeze every choice.
- [x] Execute eight Replica scenes and both ranking pools; compute eight seven-scene sensitivity pools for nominee and B.
- [x] diagnostics.py joins locked descriptors/predictions to annotations; preserve real matcher changes, coverage, calibration and stable adverse cases.
- [x] reporting.py renders four reports and cost/selection/mechanism tables from actual artifacts.
- [x] Primary review and targeted tests; normal-push implementation and compact bundle manifest.
- [ ] Post-commit full remote SHA comparison: authoritative final state is the external `publication/final.json`, avoiding a circular self-hash commit.

## Validation commands

From the task worktree:

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python docs/paper/static_ovmap/paired_evidence_v1/math_reference.py
/home/ww/miniconda3/envs/ovimap-map/bin/python -m pytest tests/paired_evidence -q
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_paired_evidence.py --phase all --resume --threads 4
```

Input recovery is annotation-free. Numerical tests cannot certify scientific improvements; the original evaluator and measured mechanism contrasts determine those. Prior instruction to proceed without questions supersedes skill approval menus; core scientific and integration work stays with the primary agent.
