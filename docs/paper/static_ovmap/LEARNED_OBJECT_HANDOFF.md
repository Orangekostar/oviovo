# Reproduction and handoff

TRAIN:24 families, 622 original base objects; 74 observed positive categories within160 base categories. 40 adapter-heldout categories remain in the full200-class prediction vocabulary.

Run from this worktree:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/ww/miniconda3/envs/ovimap-map/bin/python -u scripts/evaluation/run_ovimap_learned_object_readout.py --spec configs/static_ovmap/learned_object_readout_v1.json --parent-root /home/ww/ovimap-source-preserving-update-v1-storage/attempt_001 --observation-root /mnt/shared/capacity/node101/ww/ovimap-disagreement-query-v1/attempt_001 --output-root /mnt/shared/ww/ovimap-learned-object-readout-v1/attempt_001 --phase all --resume
```

FC worker environment: `/home/ww/miniconda3/envs/oviovo-radseg/bin/python`; GPU:0; inherited GPU lock is used.

Source binding and all resolved paths: [source_binding.json](../../../artifacts/static_ovmap/learned_object_readout_v1/source_binding.json). External raw/features/checkpoints: `/mnt/shared/capacity/node101/ww/ovimap-learned-object-readout-v1/attempt_001`.

Small selected FP32 head weights and configs: learned_heads.json and weights/. Load `ReadoutHead(checkpoint['config']['grouping'])`, then `head.load_state_dict(checkpoint['model'], strict=True)`; use the exact bound frozen FC phi and original frozen text. The weights contain no FC backbone or text encoder.

Each of the three table families has Markdown/CSV/JSON/LaTeX. Their numerical sources are the compact result store, H recognition, DEV selected-head receipts, per-object decisions and post-lock diagnostics.

runtime_profile/receipt.json records one cache-only replay with separate capture/readout clocks, zero new encodings and zero updates. It is reproduced by the predict worker on a fresh output root and reused on resume; it is not cold end-to-end latency. publication/package.json records packaging wall time separately.

The full CLI terminal exit and local/remote publication SHA are verified externally after the process exits; see publication/final.json in the external output root. The release commit never embeds its own future SHA.
