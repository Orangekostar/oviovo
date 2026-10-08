# Reproduction handoff

Freeze commit: `f7bd2a8112cd62f35e93d48d8950b820aac7f445`. Binding: `c425717ff20d74201710e62ea03241d06661b1a1fcf77851a2326c5964465f09`. Canonical store: `9b4fec25adb59776fee4ad322e917e00c8c5087d0ba6d778c6b42d85417b815d`.

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_minimal_instance_repair.py --spec configs/static_ovmap/minimal_instance_repair_v1.json --parent-root /mnt/shared/ww/ovimap-evidence-exploration-v1/attempt_001 --output-root /mnt/shared/ww/ovimap-minimal-instance-repair-v1/attempt_001 --phase all --resume
```

Logical output: `/mnt/shared/ww/ovimap-minimal-instance-repair-v1/attempt_001`; physical storage: `/home/ww/ovimap-minimal-instance-repair-v1-storage/attempt_001`. Large raster/mask tensors, licensed scans and model weights remain outside Git; manifests bind their hashes. No old runs were deleted.

Scientific costs: `{"AnyUp_QK_attempts": 411, "AnyUp_QK_computations": 411, "AnyUp_region_pools": 1805, "FC_frame_encoding_attempts": 126, "new_FC_frame_inputs": 126, "original_attention_applications": 9170, "parent_dense_cache_hits": 285, "query_chunks": 9170, "query_locations": 2294992, "union_FC_region_pool_attempts": 1}`; observer rasters=832; extra pilot AnyUp QK=1. Cold calls=64, counted separately. Recorded failures are preserved; unrecorded durations are null.

The initial complete implementation freeze preceded main acquisition. Corrective freezes subsequently record diagnostic tuple handling, scoped resume orchestration, protected-raw-zero class-change abstention, exact ordered pool aliasing and completed-timing resume. Their receipts are retained in the provenance directory; the supplied numerical protocol was unchanged. A uniform incumbent relabel that would change an original raw-zero row keeps the original class, including on an enlarged host support.

Evaluation uses the original controller environment; FC/AnyUp uses the inherited FC environment. This avoids invoking the released NumPy evaluator in NumPy 2.4. No environment upgrades or GT input to prediction were used.
