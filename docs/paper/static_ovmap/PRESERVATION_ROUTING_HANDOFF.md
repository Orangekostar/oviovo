# Preservation routing handoff

Branch: `research/ovimap-preservation-routing-v1`. Base: `6767eb90c2fd6999621b356b87269c012821713a`. Parent: `/mnt/shared/capacity/node101/ww/ovimap-learned-object-readout-v1/attempt_001` (read-only). External capacity output: `/mnt/shared/capacity/node101/ww/ovimap-preservation-routing-v1/attempt_001`.

Resolved reproduction command (actual observed invocation is added after subprocess completion):

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python -u scripts/evaluation/run_ovimap_preservation_routing.py --spec configs/static_ovmap/preservation_routing_v1.json --parent-root /mnt/shared/capacity/node101/ww/ovimap-learned-object-readout-v1/attempt_001 --output-root /mnt/shared/capacity/node101/ww/ovimap-preservation-routing-v1/attempt_001 --gpu 0 --phase all --resume
```

Separate controller/FC/SAM2 environments are retained. GPU0 uses the inherited resource lock; NVML query has a driver/library mismatch, while real CUDA is verified by engineering. No global upgrades.

[Selected inference registry](../../../artifacts/static_ovmap/preservation_routing_v1/export/weights_registry.json) and [actual strict-load/sample-score roundtrip](../../../artifacts/static_ovmap/preservation_routing_v1/export/roundtrip.json). Use `load_exported(registry_file, method, device, FC_identity=..., text_identity=...)`; it loads relative small-weight dependencies and validates exact frozen FC/text identities. Initial MA references and Rstar bases are explicit and deduplicated. FC/text/foundation weights remain external.

[Input/path/hash binding](../../../artifacts/static_ovmap/preservation_routing_v1/source_binding.json), [H2 prelock](../../../artifacts/static_ovmap/preservation_routing_v1/H2_plan.json), [component curves](../../../artifacts/static_ovmap/preservation_routing_v1/training/), [parent audit](../../../artifacts/static_ovmap/preservation_routing_v1/parent_audit/receipt.json). Resume stores exact optimizer, LR step, sampler position and Python/NumPy/Torch/CUDA RNG; selected/last optimizer checkpoints remain external. Replayed/discarded work is separately recorded; unknown historical failure time is null.

Published artifacts exclude raw licensed scans, GT maps, original/predicted image masks, huge grids and foundation weights. Compact class decisions/embeddings and small heads are included. External publication/final.json is written only after normal push and full local/remote SHA equality; this commit does not claim its own future SHA.

Actual completed invocation (working directory `/mnt/shared/ww/ovimap-preservation-routing-v1/worktree`):

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python -u scripts/evaluation/run_ovimap_preservation_routing.py --spec configs/static_ovmap/preservation_routing_v1.json --parent-root /mnt/shared/ww/ovimap-learned-object-readout-v1/attempt_001 --output-root /mnt/shared/ww/ovimap-preservation-routing-v1/attempt_001 --gpu 0 --phase all --resume
```

Observed subprocess exit: **0**, terminal `COMPLETE_NO_2D_FOUNDATION`. [Actual exit observation](../../../artifacts/static_ovmap/preservation_routing_v1/execution_observed.json), [terminal receipt](../../../artifacts/static_ovmap/preservation_routing_v1/execution_last.json), [final report phase](../../../artifacts/static_ovmap/preservation_routing_v1/phases_report.json). This invocation resumed cached completed work; it did not add scientific optimizer updates.

The first controller reported its own wall clock and report-envelope error after successful training, while its external exit observer was lost. Its actual subprocess exit is unknown. The known later report-envelope failure and all subsequent observed exits remain archived. Original per-arm training clocks are retained; unavailable overwritten original stage clocks are disclosed, and nested clocks must not be summed.

[Final prompt completion review](PRESERVATION_ROUTING_FINAL_REVIEW.md).
