# Execution handoff

Ongoing scientific execution; no terminal completion is claimed.

Resolve `$PY_CONTROL`, `$SEMANTIC_ROOT`, `$STRUCTURE_ROOT`, and `$OUT` from the local task_paths.json and binding.json. Private machine paths remain in the local execution receipts.

```bash
PYTHONPATH=src:. "$PY_CONTROL" scripts/evaluation/run_ovmap_benchmark_completion_v2.py --semantic-root "$SEMANTIC_ROOT" --structure-root "$STRUCTURE_ROOT" --output-root "$OUT" --resume --phase benchmark_predict
PYTHONPATH=src:. "$PY_CONTROL" scripts/evaluation/run_ovmap_benchmark_completion_v2.py --semantic-root "$SEMANTIC_ROOT" --structure-root "$STRUCTURE_ROOT" --output-root "$OUT" --resume --phase benchmark_score
PYTHONPATH=src:. "$PY_CONTROL" scripts/evaluation/run_ovmap_benchmark_completion_v2.py --semantic-root "$SEMANTIC_ROOT" --structure-root "$STRUCTURE_ROOT" --output-root "$OUT" --resume --phase baselines
PYTHONPATH=src:. "$PY_CONTROL" scripts/evaluation/run_ovmap_benchmark_completion_v2.py --semantic-root "$SEMANTIC_ROOT" --structure-root "$STRUCTURE_ROOT" --output-root "$OUT" --resume --phase statistics
PYTHONPATH=src:. "$PY_CONTROL" scripts/evaluation/run_ovmap_benchmark_completion_v2.py --semantic-root "$SEMANTIC_ROOT" --structure-root "$STRUCTURE_ROOT" --output-root "$OUT" --resume --phase tables
PYTHONPATH=src:. "$PY_CONTROL" scripts/evaluation/run_ovmap_benchmark_completion_v2.py --semantic-root "$SEMANTIC_ROOT" --structure-root "$STRUCTURE_ROOT" --output-root "$OUT" --resume --phase review
```

One inherited physical GPU0 and the original visual GPU lock; at most3 CPU scorers and BLAS1. Completed source/model/readout outputs are reused. Do not launch a duplicate streaming driver while its process marker is live.

A separate single CPU score_recovery monitor repairs only recorded absent-GT auxiliary recall failures. It reuses frozen predictions/GT, preserves the original failure logs, and starts no GPU workers. The pinned author source and complete AP AST remain unchanged; the added auxiliary AR initialization and actual existing-scene AP parity are recorded in review/G3_evaluator_auxiliary*.json.

MaskClustering Replica now has a full original-stage driver, native prediction freeze, independent score-only GT conversion, full class-by-IoU author AP agreement, and common pooled evaluation. Actual per-scene missing files are in external/MaskClustering/pipeline/replica48/*/availability.json. The independent environment, authentic Mask2Former576d0b/CLIP-H14 weights and author instance-GT/evaluation point domain remain required. Synthetic format checks never enter its formal pool. Run baselines after the assets are bound; a real first-scene G3 failure stops later GPU scenes.

OpenMask3D Replica now has the original arbitrary-scene proposal/features/text/query driver, its own actual-stage prediction freeze, and the shared independent full-GT/author-AP/pooling adapter. It preserves120 queries, DBSCAN0.95, SAM ViT-H and CLIP-L14@336. Actual missing files are in external/OpenMask3D/pipeline/replica48/*/availability.json. Its independent environment, arbitrary-scene checkpoint, SAM/CLIP weights and authentic colored evaluation PLY/instance tuple remain necessary. CPU connection tests do not establish real Replica G3 or full-method results.

Remaining logical jobs: 129. Exact jobs and missing dependencies: `remaining_jobs.json`. Completed job costs and planning ranges: `resource_ledger.json`, `cost_estimate.json`. Training updates remain0; native reconstruction, CropFormer, SigLIP, FC encoding and frozen head inference are real new work.

## Recorded execution

Counters below describe the named sealed receipts. Live jobs may have additional unsealed consumption.

Resource ledger identity: `a365e13b745f7ae4f9510ec6442c1a15f5f30a713d8978f2a1a1284fd4d68b94`.

| Counter | Recorded count |
|---|---:|
| `calibration_GPU_head_calls` | 0 |
| `calibration_RGB_encodings` | 0 |
| `engineering_MV_head_calls` | 8 |
| `legacy_new_MV_head_calls` | 2868 |
| `new_training_updates` | 0 |
| `standard_CropFormer_frames` | 62400 |
| `standard_FC_image_encodings` | 48308 |
| `standard_FC_region_poolings` | 159436 |
| `standard_MV_head_calls` | 43680 |
| `standard_N_crop_encodings` | 1330152 |
| `standard_Q_crop_encodings` | 373236 |

Unavailable measurements: whole_pipeline_peak_GPU_memory, isolated_FC_module_latency, isolated_CropFormer_peak_memory, true_online_throughput.

CPU calibration receipt: `73283e52268234d4503f5c349ee1634f37c529dc6d76e2ed16ee73209fb08f80`.

| Calibration counter | Recorded count |
|---|---:|
| `antipodal_fallback_realizations` | 0 |
| `degenerate_nmix_fallback_realizations` | 0 |
| `logical_candidate_rows` | 1331 |
| `new_GPU_workers` | 0 |
| `new_head_calls` | 0 |
| `new_image_encodings` | 0 |
| `new_training_updates` | 0 |
| `physical_CAP_text_matmuls` | 30 |
| `physical_CPU_candidate_predictions` | 1111 |
| `physical_NMIX_text_matmuls` | 24 |
| `physical_base_score_passes` | 100 |
| `reused_original_MV_predictions` | 220 |

Successful standard scene scoring retained in the current coverage:

| Scope | Logical result rows | Unique scene scorer inputs | Author-entry checks |
|---|---:|---:|---:|
| replica48 | 0 | 0 | 0 |
| scannet200_val | 2496 | 2481 | 1 |

Unique scene scorer inputs count successful per-scene evaluations after exact prediction aliases. Failed/interrupted attempts, pooled rescoring and engineering checks are excluded; this is not a total historical CPU-call count.

Legacy coverage: 176/208 logical scene rows; 12/16 pools. Logical rows do not measure physical scorer calls.

## Table fields

Each table includes the Method / setting row label and the following data columns. Per-cell units, result keys and aggregation formulas are in `cell_provenance.jsonl`.

Table-definition identity: `2081e99b620d9f45732ccc64d5b7b41550aaa25e42337d8685273faaafea6849`.

| Table | Title | Data columns in order |
|---|---|---|
| T1 | Full external methods on official benchmarks | ScanNet AP; AP50; AP25; Replica AP; AP50; AP25 |
| T2 | Global all-instance source and calibration ablation | MF; AC; ScanNet AP; AP50; Replica AP; AP50 |
| T3 | Calibration operators on the same frozen MV model | Base; Novel; H |
| S1 | Protocol and complete method identities | Proposal; Semantic source; Information condition |
| S2 | Legacy fixed-eligible and standard results | AP; AP50; AP25; mIoU; mAcc |
| S3 | Actual available-source fusion | Owners; N weight; Q weight; F weight; MFoff |
| S4 | Historical structural controls and two-level ablation | GT Base; GT Novel; GT All; REAL AP; REAL AP50 |
| S5 | Per-seed RAW and LOFO calibration | Base; Novel; H; Base micro; Novel micro; H micro |
| S6 | Class-independent proposal support | All owners; F supported; Zero F; One F; Valid GT; Matched GT; Recall |
| S7 | Calibration corrections and damage on common geometric matches | Matched; Changed; Corrected; Harmed; Correct before; Correct after |
| S8 | Measured execution and cache costs | Value; Unit; Scope |

The original deployment/N0 has not been replaced. Current partial results cannot establish a complete benchmark win or a CVPR novelty claim.
