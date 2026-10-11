# Frozen FC routing handoff

Task: `ovimap-frozen-fc-routing-v1`; base `76db115aa7df652a0b2dc4575df52e6a74a63549`; fixed spec in `configs/static_ovmap/frozen_fc_routing_v1.json`.
Current science: `COMPLETE_NO_TARGET_GAIN`. Root `/mnt/shared/capacity/node101/ww/ovimap-frozen-fc-routing-v1/attempt_001`; worktree `/mnt/shared/capacity/node101/ww/ovimap-frozen-fc-routing-v1/worktree`; parent PR `/mnt/shared/capacity/node101/ww/ovimap-preservation-routing-v1/attempt_001`; LR `/mnt/shared/capacity/node101/ww/ovimap-learned-object-readout-v1/attempt_001`; source-update `/home/ww/ovimap-source-preserving-update-v1-storage/attempt_001`.
Controller `/home/ww/miniconda3/envs/ovimap-map/bin/python`; FC `/home/ww/miniconda3/envs/oviovo-radseg/bin/python`; single GPU `0` queued through the inherited resource file. Existing FP32 environments are unchanged; CUDA worked despite the pre-existing NVML warning.

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python -u scripts/evaluation/run_ovimap_frozen_fc_routing.py --spec configs/static_ovmap/frozen_fc_routing_v1.json --parent-root /mnt/shared/capacity/node101/ww/ovimap-preservation-routing-v1/attempt_001 --output-root /mnt/shared/ww/ovimap-frozen-fc-routing-v1/attempt_001 --gpu 0 --phase all --resume
```

Use the same command with any fixed `--phase` for recovery. Source, fixed split/bank/spec and training-code identities bind resume; preserve original failed attempts. `last.pt` keeps optimizer, sampler index and Python/NumPy/Torch/CUDA RNG. Saved scalar curves contain every completed update; replayed uncheckpointed updates are separate. Do not change scientific settings to recover a negative metric.

Training: four fresh identically initialized216,194-parameter heads, seed17,2000 updates each; all four seed29 only on a seed17 qualification. Exact inherited class-uniform original sampling,2/4/8 prefixes, first zero-LR update, FP32 AdamW and eta ramp. The explicit seven-key JSON whitelist governs the prose's erroneous count of eight. Base loss retains the formal0.1 student consistency term; no additional consistency/preservation/correspondence objective.

Ordinary FC region vectors use hard-mask/area-v2 fallback. Sparse inputs, when used, are exact tensors produced by the original loader, validated on real objects for bitwise inputs/outputs and input gradients. TRAIN/DEV/oldH/H2 use registered depth-grid RGB; regression banks retain their original preprocessing. No MA/R/learned768x768 value projection is an inference dependency.

Load interface: `static_ovmap.frozen_fc_routing.export.load_exported(registry_file, name, device, FC_identity=..., text_identity=...)`. Registry at `artifacts/static_ovmap/frozen_fc_routing_v1/export/weights_registry.json`; weights use relative paths and strict state loading. Reconstruct with the same ordered200-row FC text table and FrozenFC.phi. Raw scans/GT maps/dense caches/frozen backbone weights remain external. Artifact index maps compact source receipts to published files.

Completed science updates: 16000; new scene rows/pools: 286/22; missing active requirements: []. All applicable selected heads, full scalar curves, checkpoint metrics, compact embeddings/cosines, decisions, D0/D1 and post-lock mass/norm diagnostics are archived. D1 is GT_DIAGNOSTIC_ONLY and cannot nominate a model; undefined norm pairs are masked on both paired sides while ordinary FG4 retains full support. Real proposal purity is NOT_COMPUTED without exact independent site targets.

Tables have MD/CSV/JSON/LaTeX and explicit percent columns; machine evaluation receipts retain fractions. Recognition is correlated over conditions and repeated views. DEV nomination, benchmark-exposed exploratory recommendation and same-architecture repeat remain distinct. H2 has fixed names and is proposal recognition, not independent map AP. Old H is exposed. Runtimes have explicit resident/cache boundaries; resumed/nested stage clocks must not be summed into a cold end-to-end total, and no online FPS claim is made.

The actual all --resume process exit is observed by an external parent. After that process exits, terminal/ archives its final execution receipt, parent observation and complete stdout, with a separate terminal/index.json. Prior observations remain in observations/; they are attempt history, not the final exit proof. Publication/final.json is written externally only after normal push and full local/remote SHA equality. No commit embeds its own future SHA.
