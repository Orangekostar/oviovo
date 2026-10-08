# Reproducible handoff

Execution: SCIENCE_COMPLETE; reporting: TABLES_COMPLETE. Scene-method rows 234/234; complete pools 18/18.

Selection: COMPLETE_NO_TARGET_GAIN; selected SU01_G1; passing candidates []; material target False.

Parent store 9b4fec25adb59776fee4ad322e917e00c8c5087d0ba6d778c6b42d85417b815d; source binding b50d0eaf026bbcbb0295b3eb44085fd44ef43f8115fea582d4b9b86f77c54545; science identity ecf676e26bed78cbe05b14e462160f1fe0be70769656f087a33f6914bf4a99b6; result-store identity c6e13f2182f03752a3a4bd278fe3e69ad28a2fe0f7b92a23bbd1353379b06a17.

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_source_preserving_update.py --spec configs/static_ovmap/source_preserving_update_v1.json --parent-root /mnt/shared/ww/ovimap-minimal-instance-repair-v1/attempt_001 --output-root /mnt/shared/ww/ovimap-source-preserving-update-v1/attempt_001 --phase all --resume
```

Use the inherited controller/FC environments and external dependency_manifest.json. --storage-root, --path-map and --gpu provide explicit relocation. Resume validates content and archives only changed new-task descendants. Original experiments are read-only.

Exactly three main table families are generated from result_store.json. source_scores.npz preserves precise real N/Q/F/D2/A/C arrays; paired_evidence_manifest and per-scene decisions/coarse/diagnostics retain masks, identities and full distributions. decision_vectors.npz deduplicates float64 vectors losslessly; reporting.unpack_vectors restores per-scene decisions and verifies original vector and decision identities. source_binding.json.gz is the exact source binding. Large payloads/dense/RGBD/weights remain external.

The packaging-only post-outcome correction and original pre-outcome freeze are published separately. Scientific producers, predictions, scores and measured CPU samples were retained exactly; no result-conditioned method or timing change was made.

Historical unrestricted IR06/IR07 replay and their original-to-matched metric/coverage differences are in result_store.diagnostics.historical_to_matched, with actual prediction and scoring identities. No historical score is substituted for a new matched method.

Publication proof is external publication/final.json; it records the full local/remote commit hashes and observed full-command exit. No commit contains its own commit hash.

Replica8 and CF18 were previously exposed. CF18 comprises 18 captures from seven physical families; this is neither untouched confirmation nor full ScanNet200 validation. No significance or monotone-accuracy claim is supported. Deployment: N0_UNCHANGED.
