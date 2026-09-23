# Authorized ScanNet execution — 2026-09-22

本页保留早期执行记录。当前 native-v10 实验与选择已完成，最终为 N0／NO_NET_GAIN；最新结果、交付状态和复现命令见 [结果报告](../MODULE_VALIDATION_RESULTS.md)、[交接文档](../MODULE_VALIDATION_HANDOFF.md)与 [progress.md](progress.md)。

Current update (2026-09-23 00:08 UTC): data preparation and all 12 CropFormer
caches are COMPLETE. Corrected native-v10 capture is running on GPU 2, following
the [query membership repair](QUERY_MEMBERSHIP_REPAIR.md). Native-v9 scene
artifacts remain historical and are excluded from the current scientific study.
All S/G/Q/selection/confirmation/report/release drivers are implemented; actual
execution remains IN_PROGRESS. The original 8/2/2/2 split is unchanged and
CONFIRM remains unopened.
See the [execution plan](../../../superpowers/plans/2026-09-22-scannet-scientific-drivers.md).

The data preparation and earlier resource-blocked execution below are retained
as historical evidence; their former GPU/implementation blockers are superseded.

Data preparation is COMPLETE. All 14 locked captures are downloaded; the 12 development scenes have 2,400 exported RGB/depth/pose slots and calibration matrices.

Data root: `/mnt/shared/ww/ovimap-module-validation-v1/data/scannet`.

The user's confirmation authorized these transfers. The preparation CLI reused literal release constants from `/home/ww/getscannet.py`; the original file remains unchanged. It did not execute its full-release branch, terms prompt or SSL override. All 84 scene files (six types per capture, 10,052,476,765 bytes), the label map and both official split lists are available. The final acquisition lock and bound 8 FIT / 2 CAL / 2 SELECT / 2 CONFIRM split agree with the [original acquisition plan](SCANNET_PREPARATION_20260922.md). No scene was replaced based on model outcomes.

Transfers used verified HTTPS, remote length/ETag checks and local SHA-256 receipts; local hashes are not official reference checksums. A real network failure exposed curl's internal retry behavior: retries restarted from the invocation's initial offset. Separate bounded invocations now preserve newly transferred bytes. The HTTPS regression test verifies offsets 13 then 30 following a deliberate disconnect.

Three v2 metadata counts exceed their reused v1 sensor counts by one: scene0107_00 (2366 vs 2365), scene0571_00 (2083 vs 2082), scene0445_00 (683 vs 682). In every case native step, end and all 200 selected IDs are identical. Export now records this provenance difference while rejecting any mismatch that changes the selected schedule. Original JPEG bytes, millimeter depths, calibrations and frame IDs are preserved.

Invalid poses are retained and logged, never backfilled: 73 scheduled slots in scene0639_00, 11 in scene0538_00 and 11 in scene0571_00. Thus 2,400 exported slots do not mean 2,400 valid mapping frames. The two confirmation scenes have raw files only: image export, visual inference and annotation-content evaluation have not run.

Validation:

- All 12 development annotation meshes match segmentation vertex order and category lookup.
- Pinned ScannetLoader checked all 200 slots of scene0547_00 and the first/last slots of every other development scene: 222 checked, 221 valid, one correctly skipped. Native BGR registration and depth-to-meter conversion pass.
- 116 scoped module tests, scoped Ruff and `git diff --check` pass.
- `attempt_005` at code `e1b4efc06d234c574ddc45ba8f5a50a61e311a58`: `bind=COMPLETE`, 10 bound assets, zero missing assets; `capture=BLOCKED_GPU_BUSY`. CUDA device 2 was occupied by another experiment (35,375 MiB). No new CropFormer/native mapping job started.

The new capture driver binds the instrumented native build, locked scene schedule, CropFormer checkpoint/config and local FP32 native SigLIP. Native perception uses the original mapping environment: transformers 4.49 resolves `SiglipImageProcessor`, whereas the alternate-model environment's 4.51 resolves `SiglipImageProcessorFast` for `use_fast=True`. Completed outputs are hash checked. Interrupted native replays are preserved before restarting a scene because the mapper has no TSDF checkpoint.

Run after the designated GPU is free; no new data authorization is needed:

```bash
cd /home/ww/crove/ovimap-module-validation
export OVIMAP_DATA_ROOTS=/mnt/shared/ww/ovimap-module-validation-v1/data/scannet
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_module_study.py \
  --spec docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json \
  --output-root /mnt/shared/ww/ovimap-module-validation-v1 --phase bind
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_module_study.py \
  --spec docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json \
  --output-root /mnt/shared/ww/ovimap-module-validation-v1 --phase capture
```

The capture leaf accepts `--resolved-config` and optional `--scenes` restricted to locked development scenes; confirmation scenes are rejected. Its real CropFormer/native replay still requires GPU validation. Remaining integration: ScanNet200 native text/readout/evaluator binding, S/G/Q execution, frozen selection and gated confirmation. Do not reuse historical NYU40 text embeddings as ScanNet200 labels.

Evidence: [execution receipt](../../../../artifacts/static_ovmap/module_validation_v1/scannet_execution_20260922.json). Earlier pre-authorization evidence remains unchanged.
