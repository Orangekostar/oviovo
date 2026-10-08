# Source-preserving update implementation plan

> Execute inline with superpowers:executing-plans. The primary agent retains scientific design, implementation, review and integration authority.

**Goal:** Execute and publish the prescribed nine-condition, 26-scene source-preserving semantic update study.

**Architecture:** A narrow adapter reads immutable parent payloads and saved recognition records. Coarse acquisition uses the original FC worker environment; CPU decisions have no evaluator dependency. Real payloads, released scoring, diagnostics and publication follow separate resumable phases.

**Tech stack:** Inherited Python environments, NumPy/SciPy float64 decisions, original FP32 FC operators, released dataset pooling.

**Spec:** `docs/paper/static_ovmap/source_preserving_update_v1/spec/CODEX_FINAL_EXECUTION_EN.md` and its verbatim `PROTOCOL_SPEC.json`.

## Constraints

- Base `b000355eb8f91491e002df1499bd0170d91c35ef`; branch `research/ovimap-source-preserving-update-v1`.
- Parent roots and all inherited evidence stay read-only. Write only the new task root/worktree.
- No maps, observers, view searches, structural operations, AnyUp/NQ/frontend inference, fitting, downloads or cold timing calls.
- At most 832 coarse pools; encodings bounded by the actual required distinct image identities.
- One GPU worker; three evaluation workers; BLAS threads at most four; new-store free space at least 10 GiB.
- Nine methods, 234 logical rows, 18 exact ordered pools; deployment `N0_UNCHANGED`.
- Approximately ten focused tests, two real pilots; all phases implemented before full-outcome freeze.

## Task 1: Immutable inputs, decisions and history

Files: `source_preserving_update/{binding,evidence,decisions,outputs,selection}.py`, `tests/evaluation/test_source_preserving_update.py`.

- [ ] Write ten tests before implementation; run the focused file and retain its red result.
- [x] Implement `binding.bind(spec_path,parent_root,output_root,storage_root=None,gpu=None,path_map=None)` and `load_scene(binding,scene)` without the old structural controller.
- [x] `evidence.prepare_scene(binding,scene)` reconstructs the original D2 probabilities and unrestricted IR06/07 outputs, verifies actual arrays/ranks, locks every successful FULL observation and records potential-domain exclusions.
- [x] `decisions.source_components` and `update_probability` implement the fixed equations; `decide_scene` applies seven conditions on a common domain. `selection.select` uses all unrounded gates and deterministic ties.
- [x] `outputs.relabel_g1` calls the unchanged partition builder with `operations=[]`; owner/recovered/raw-zero/unknown parity is checked before prediction lock.
- [x] Run the focused tests and real source reconstruction; do not read new pooled outcomes.

## Task 2: Same-view coarse acquisition and pilots

Files: `source_preserving_update/{coarse_worker,orchestration}.py`, `scripts/evaluation/run_ovimap_source_preserving_update.py`.

- [x] Inventory every required parent frame/dense tensor, derive the actual region/image bounds.
- [x] Run one inherited FC worker: original RGB/image_tensor/signed_mask/region_vector/cosine_record, exact direct parent dense reuse before read-only ContentCache lookup.
- [x] Store input identities, precise C scores, attempted/successful work, unavailable representation reasons, failures and per-frame elapsed time separately.
- [x] Seal eligibility only when all required C views completed; never drop a failed paired view.
- [x] Run office1 and scene0011_00 through all nine decisions, real payload build and released scoring.

## Task 3: Frozen complete evaluation and diagnosis

Files: `source_preserving_update/{evaluation,analysis}.py`.

- [x] Implement complete actual-G1 registry, whole semantic confusion, official rank assertions and exact context/content aliases.
- [x] Implement exact ordered pools and all five metrics. Keep partial methods incomplete.
- [x] Implement all seven-versus-G1 and six fixed contrasts, geometric50/75 outcome ledgers, unique released matches, tied entry multiplicity, rank/per-class deltas and historical-to-matched changes.
- [ ] Freeze every implementation/config identity, then predict/evaluate/diagnose all 26 scenes.
- [ ] Select using unrounded gates and write measured next-stage assessment without new experiments.

## Task 4: Costs and publication

Files: `source_preserving_update/{costs,reporting,orchestration}.py`, four `SOURCE_UPDATE_*.md` reports, `artifacts/static_ovmap/source_preserving_update_v1/`.

- [x] Time resident CPU decisions only: Replica8, two reverse-order rounds, warmup once, 50 applications/sample, 16 samples/method. Separate actual acquisition costs.
- [ ] Generate exactly three tables in MD/CSV/JSON/booktabs from one result store, compact precise score/decision/diagnostic evidence and an explicit external dependency manifest.
- [x] Implement `all --resume` so unaffected leaves/reporting/publication continue on partial failures; zero exit only for complete science and verified ordinary push.
- [ ] Run the exact command below, commit code/real compact results, ordinary push, verify full local/remote SHA and write the external final receipt.

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_source_preserving_update.py --spec configs/static_ovmap/source_preserving_update_v1.json --parent-root /mnt/shared/ww/ovimap-minimal-instance-repair-v1/attempt_001 --output-root /mnt/shared/ww/ovimap-source-preserving-update-v1/attempt_001 --phase all --resume
```

Completion evidence: exact CLI exit 0; 234/18 coverage; strict source/history/output parity; real C execution; all ten relevant tests and two pilots; every diagnostic/selection/cost/report requirement; compact dependency-bearing artifacts; full verified publication SHA. A scientific negative result is complete only with this evidence.

Execution evidence before full-outcome freeze: 26/26 content-bound prepares; D2 and unrestricted historical IR06/07 actual payload parity; 584/584 successful same-view C pools with 378 direct dense hits, zero new image encodings; common domain 320/416; both pilots 9/9; ten tests passed plus affected selection test passed. The first red invocation failed on an import-path setup issue and is not claimed as behavioral-red proof. Full timing, pooled evaluation, diagnostics and publication remain subject to actual execution.
