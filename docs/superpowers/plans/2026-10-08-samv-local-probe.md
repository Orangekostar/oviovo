# SAM-V local probe implementation plan

> Execute inline with superpowers:executing-plans. Only pinned asset acquisition is delegated mechanically; the primary agent owns scientific implementation, model interfaces, review and integration.

**Goal:** Run and publish all six prescribed arms on four fixed complete maps, with real SAM2/SAM-V inference, paired FC controls, actual partition scoring and eight comparable segmentation-window timings.

**Architecture:** A narrow immutable adapter reads the completed source-update parent. Query planning receives predicted owners, source probabilities, measured observations and poses only. Isolated segmentation workers exchange exact masks with a shared row-wise lifter; a new output builder preserves the fixed surface while allowing prescribed ownership edits. Evaluation opens annotations only after all predictions are locked.

**Tech stack:** Existing controller and FP32 FC environments; separate Python3.10 SAM-V torch2.3.1 and SAM2 torch2.5.1 environments; NumPy/SciPy geometry and probabilities; inherited released evaluator.

**Spec:** `docs/paper/static_ovmap/samv_local_probe_v1/spec/CODEX_FINAL_EXECUTION_EN.md`, `IMPLEMENTATION_CONTRACTS.md`, `EVALUATION_AND_SELECTION.md`, verbatim `configs/static_ovmap/samv_local_probe_v1.json`.

## Global constraints

- Base a95c24d990cea57b14bb537e9acca95b95659bda; research/ovimap-samv-local-probe-v1; four scenes office1, room0, scene0011_00, scene0050_00.
- Actual physical storage is `/mnt/shared/capacity/node101/ww/ovimap-samv-local-probe-v1`; logical root is symlinked from the prescribed `/mnt/shared/ww/ovimap-samv-local-probe-v1`. At least30 GiB before assets; no older data removal.
- Maximum32 targets,128 reference observations,32 scientific SAM-V joint calls/192 frame slots,32 SAM2 tracks,128 FC region pools and64 new FC images. Engineering failures and eight extra timings are separate.
- One GPU worker at once; at most two CPU workers, four raycast/BLAS threads. No maps, CropFormer, AnyUp, N/Q inference, training or26-scene expansion.
- Default six chronological canonical JPEG95 views, bf16, microbatch1. A real pre-evaluation OOM permits one global four-frame replan only.
- Scientific source/config/resource freeze follows twelve narrow tests and two actual engineering pilots. No AP-driven profile, target, threshold, asset or class choice.
- Twenty-four main/four reference rows, twelve main/two reference exact ordered subset pools. All five metrics and complete target confusion. Deployment N0_UNCHANGED.
- Three table families/four reports and fixed-case mask contact sheets, compact real evidence under30 MiB. Ordinary authorized push, exact full CLI terminal status, external full-SHA receipt.

## Task1: Immutable binding and query plan

Files: `samv_local_probe/{binding,query_plan,geometry,common}.py`; `tests/evaluation/test_samv_local_probe.py`.

- [x] Write twelve boundary tests first, run and retain expected red, then implement and run the focused file.
- [x] `binding.bind(args)` verifies actual parent commit/publication/store/spec and four G1/D2 payloads. `binding.load_scene(binding,scene)` redirects inherited memo writes to this new task and caches resident immutable inputs.
- [x] `query_plan.observe(binding,scene)` verifies exact parent32-view geometry/camera/depth/operator identity or computes only missing selected observations with SourceRowProjector.
- [x] `query_plan.plan_scene(binding,scene,count)` inventories every positive valid-class owner, qualifies views, selects nominal4+4 with deterministic shortage filling, and generates largest-component interior points.
- [x] `geometry.choose_window(frames,qualified,anchor,count)` preserves both mandatory frames, applies fixed SE3 diversity over all representatives and orders original IDs. `geometry.edit_domain(xyz,faces,owners,targets)` freezes AABBs, boundary hops, selected/unowned permissions and prompt rows.
- [x] Canonical preparation uses exact Torch bilinear1024/truncate/JPEG95 bytes in the isolated model environment; query identity includes all ordered images/points/profile. GT never enters planning.

## Task2: Assets and real isolated inference

Files: `samv_local_probe/{assets,samv_worker,sam2_worker,segmentation}.py`; external assets/environments only.

- [x] Resolve only author stage2/SAM-H/VGGT/SAM2.1-L and exact gitlinks; record revision/SHA/license/failures. Assets do not include unrelated submodules or datasets.
- [x] Create isolated environments without touching inherited environments; validate frozen encoder coverage and every trainable tensor with author's partial loader.
- [x] `samv_worker.infer(query)` calls unchanged joint forward; split low-resolution panorama before interpolation, restore canonical/original shapes and test all positive anchor points.
- [x] `sam2_worker.infer(query)` uses prescribed overrides, identical JPEGs/points, forward then reverse from the same anchor and independent state per target. Capture coverage, masks/logits and real counters.
- [x] Run both models on the first locked target in each prescribed cohort pilot, validate real interfaces and output/lifting boundaries without scoring AP. Handle only genuine OOM with one global replan.
- [x] Commit complete implementation/spec after pilots and source/config/profile freeze; reuse exact successful pilot leaves and infer remaining selected targets once/model.

## Task3: Shared lifting, FC control and actual partitions

Files: `samv_local_probe/{lifting,readout,outputs}.py`.

- [x] `lifting.count_view_votes` counts one visibility/foreground vote per source row/view; invalid−1 never indexes a real row. Save raw and admitted supports separately.
- [x] `lifting.arbitrate` uses integer cross products, q>=2/3 additions, q<=1/3 deletions, ties retain old, domain/core protection and final prompt retention, without sequential cascades. Anchor-failed queries contribute no positive/negative proposals; raw failures remain diagnosed. Other admitted simultaneous donor transfers remain explicitly attributed.
- [x] Run only two mandatory original RGB frames per target and both OLD/NEW masks through the original FP32 FC signed pooling/v2 fallback. Retain raw availability; require both sources/two views and SAM-V anchor adherence for both semantic controls.
- [x] `readout.paired_labels` applies full-vocabulary equal cosine means and raw .01 margin with no fit or old raw-zero class veto.
- [x] `outputs.build` constructs real uniform-class exclusive owners and complete native_ranks. Structural arms inherit G1 classes. SV05 copies exact SV02 ownership and exact SV04 class decisions. No new IDs or old whole-unit builder.

## Task4: Complete subset scoring and attribution

Files: `samv_local_probe/{evaluation,diagnostics,selection}.py`.

- [x] Lock all predictions before annotations. Verify one actual G1 baseline scoring/trace/confusion parity per cohort. Key registries by actual owner-array digest and cover small/unknown/recovered owners in semantics.
- [x] `evaluation.evaluate` emits28 logical rows/14 complete ordered pools; content/context/rank-proven aliases only, no parent full-cohort metric reuse or per-scene AP averaging.
- [x] `diagnostics.analyze` reports whole-map class-agnostic maximum matching50/75, unique gained/lost GT, per-target fixed-reference IoU and best IoU, raw/admitted/final supports, suppressed rows/collisions/donor harms/fragmentation, OLD/NEW semantic WR/RW, tied official trace multiplicity and current-class rank changes.
- [x] `selection.select` implements exact all-five/two-cohort/D2 gates, tolerance-aware lexicographic ties and fixed simplicity. Mechanism flags remain separate; no expansion/deployment change.

## Task5: Costs, artifacts and publication

Files: `samv_local_probe/{costs,reporting,runner}.py`; `scripts/evaluation/run_ovimap_samv_local_probe.py`; four `SAMV_PROBE_*.md` reports.

- [x] Every scientific/engineering/failure/resume stage records actual work, model loading and wall separately; unavailable measurements remain null.
- [x] Run exactly eight extra cache-off resident segmentation-window timings. SAM2 round1 -> SAM-V round1 and reverse-window round2 -> SAM2 reverse-window round2 respects one-GPU residency and reversed method order. Reading canonical images through restored masks is timed; parity is checked outside.
- [x] Generate three canonical table families in four formats, compact masks/decisions/identities/dependency references, all-target contact sheets per scene and labeled success/failure pair when present. No licensed raw images/weights uploaded.
- [x] `runner.run(args)` implements every specified phase, content-validated resume and truthful partial propagation; no re-inference of unchanged completed leaves.
- [x] Execute exact prescribed full CLI to a real terminal status and audit every package requirement against actual evidence.

Publication is a separate post-commit gate: normal push, local/remote full SHA equality, and external `attempt_001/publication/final.json`. This record intentionally does not attest a future commit SHA.

Verification command:

```bash
PYTHONPATH=src:. /home/ww/miniconda3/envs/ovimap-map/bin/python -m pytest -q tests/evaluation/test_samv_local_probe.py
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_samv_local_probe.py --spec configs/static_ovmap/samv_local_probe_v1.json --parent-root /mnt/shared/ww/ovimap-source-preserving-update-v1/attempt_001 --output-root /mnt/shared/ww/ovimap-samv-local-probe-v1/attempt_001 --phase all --resume
```

## Verified execution progress (2026-10-08)

- Four scene bindings and exact D2 parity verified;32 target queries locked. All128 observer frames were reused with source/valid/camera/depth/operator identity checks; no new observation rays.
- Both isolated model environments installed and checked; actual SAM-V strict encoder and142 trainable-tensor coverage checks passed.
- Actual SAM2 and SAM-V pilots on office1/8 and scene0011_00/14 passed canonical-pixel, prompt, real-array lifting and output checks. Six-frame resource profile retained; no OOM fallback and no new AP.
- Controller CUDA library leakage into the SAM2 CUDA12.4 worker was fixed by removing inherited LD_LIBRARY_PATH only in the isolated subprocess. Actual pilot invocation then exited0; twelve boundary tests passed in1.05s.
- Engineering measurements and historical formal groups remain separate under artifacts/static_ovmap/resource_comparison_20261008. All32 SAM2 and32 SAM-V target leaves are now complete;15 new FC frames,31 exact parent dense hits,121 real region pools.
- All28 logical records and14 exact ordered two-scene pools completed, with18 physical new scene scorer calls and two actual baseline parity checks. Real no-evidence construction on all four complete maps exactly reproduces G1.
- Exactly8 extra formal window timings completed; all raw masks match scientific results. Both prescribed full all --resume CLI executions terminated0; completed segmentation invocation count stayed9 on resume.
- Scientific freeze1d9eca0 retained; explicit report/cost/no-evidence CLI amendment6d328053 preserves all numerical model/planning/lifting/readout/output/evaluation files.
- Three table families, four reports, all-target figures and actual compact masks/edits/traces delivered under18MiB at final review. Primary requirement and visual QA are recorded in artifacts/static_ovmap/samv_local_probe_v1/FINAL_REQUIREMENT_REVIEW.{md,json}.
- Scientific outcome COMPLETE_NO_PILOT_GAIN: retainG1 research reference, deploymentN0; no automatic full-cohort expansion. Publication completion is attested only by the external final receipt after normal push.
