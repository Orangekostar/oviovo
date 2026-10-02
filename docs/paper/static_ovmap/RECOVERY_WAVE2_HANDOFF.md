# Recovery Wave 2 Handoff

Implementation: `IMPLEMENTATION_COMPLETE`. Scientific outcome: `RETAIN_BASELINE`. Publication is verified only by the external `publication/final.json` full-SHA comparison. Deployment: `N0_UNCHANGED`.

Task root: `/mnt/shared/ww/ovimap-recovery-wave2-v1/attempt_001`. Immutable parent: `/mnt/shared/ww/ovimap-backbone-wave1-v1/attempt_001`. Branch: `research/ovimap-recovery-wave2-v1`.

Completed new full maps: development 20, Replica 0. No baseline maps, new SAM/CropFormer/text forwards, training, checkpoint or temperature fits. Fresh own-map N/Q/FC regeneration is separately measured.

Frozen Replica package: `RW_B_D2`, coverage `8/8`, APall=11.7413%, AP50=24.4952%, AP25=37.9746%, mIoU=29.6911%, mAcc=37.4279%. Scientific status: `NO_NET_GAIN`. Replica results did not change the nomination.

Default Python: `/home/ww/miniconda3/envs/ovimap-map/bin/python`. FC Python: `/home/ww/miniconda3/envs/oviovo-radseg/bin/python`. Original FP32 model/operator/checkpoint identities and frozen category/template order are in `resolved_inputs.json`; native patch/build receipts are retained. The isolated upstream patch stack is capture, backbone, then recovery. Original binaries remain unchanged.

Use the existing shared artifacts listed with content hashes in `external_artifacts.json`; large RGB-D, models, dense caches, maps, TSDF and verbose native logs are external. These paths are not public download links. Restore the licensed original data/model access and hash-verify files. Initial binding accepts `--path-map` for relocated parent inputs; exact completed-task resume retains the immutable binding and original root aliases.

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python /mnt/shared/ww/ovimap-recovery-wave2-v1/worktree/scripts/evaluation/run_ovimap_recovery_wave2.py --phase all --resume
```

Mapping defaults to 2 x 8 CPU threads, evaluation to 3 x 4 BLAS threads, one model worker under the existing GPU lock. Complete receipts resume only with verified content and recorded producer identities. Failed map attempts restart at frame zero in a new retained directory.

`experiment_matrix.json` records measured, exact-equivalent, geometry-screened and non-nominated transfer leaves. Failure receipts and missing timings remain visible in `failure_and_costs.json`. Primary requirement audit and targeted production/native validation are under the compact `validation/` directory.
