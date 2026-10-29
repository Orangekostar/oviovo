# Execution handoff

Current coverage and remaining dependencies are recorded below; terminal completion is determined by publication/final.json.

Resolve `$PY_CONTROL`, `$SEMANTIC_ROOT`, `$STRUCTURE_ROOT`, and `$OUT` from the local task_paths.json and binding.json. Private machine paths remain in the local execution receipts.

```bash
PYTHONPATH=src:. "$PY_CONTROL" scripts/evaluation/run_ovmap_benchmark_completion_v2.py --semantic-root "$SEMANTIC_ROOT" --structure-root "$STRUCTURE_ROOT" --output-root "$OUT" --resume --phase benchmark_predict
PYTHONPATH=src:. "$PY_CONTROL" scripts/evaluation/run_ovmap_benchmark_completion_v2.py --semantic-root "$SEMANTIC_ROOT" --structure-root "$STRUCTURE_ROOT" --output-root "$OUT" --resume --phase benchmark_score
PYTHONPATH=src:. "$PY_CONTROL" scripts/evaluation/run_ovmap_benchmark_completion_v2.py --semantic-root "$SEMANTIC_ROOT" --structure-root "$STRUCTURE_ROOT" --output-root "$OUT" --resume --phase baselines
PYTHONPATH=src:. "$PY_CONTROL" scripts/evaluation/run_ovmap_benchmark_completion_v2.py --semantic-root "$SEMANTIC_ROOT" --structure-root "$STRUCTURE_ROOT" --output-root "$OUT" --resume --phase statistics
PYTHONPATH=src:. "$PY_CONTROL" scripts/evaluation/run_ovmap_benchmark_completion_v2.py --semantic-root "$SEMANTIC_ROOT" --structure-root "$STRUCTURE_ROOT" --output-root "$OUT" --resume --phase tables
PYTHONPATH=src:. "$PY_CONTROL" scripts/evaluation/run_ovmap_benchmark_completion_v2.py --semantic-root "$SEMANTIC_ROOT" --structure-root "$STRUCTURE_ROOT" --output-root "$OUT" --resume --phase review
```

GPU selection and locks follow the current asset configuration and actual job receipts; at most3 CPU scorers and BLAS1. Historical allocation limits and measured consumption are recorded in resource_ledger.json. Completed source/model/readout outputs are reused. Do not launch a duplicate streaming driver while its process marker is live.

Past recovery attempts and original failures remain in their actual runtime receipts. The pinned author source and complete AP AST remain unchanged; the added auxiliary AR initialization and actual existing-scene AP parity are recorded in review/G3_evaluator_auxiliary*.json.

Full external method variants, assets and exact commands are in external/<method>/recipe.json. The following coverage comes from current official whole-benchmark pools and compatible author references. Locally scored complete author predictions are distinguished from locally inferred predictions in each recipe. Synthetic format checks never count as formal completed scenes.

| Method | Benchmark | Evidence | Locally scored scenes |
|---|---|---|---:|
| OVI-MAP | replica48 | LOCAL_FULL | 8/8 |
| OVI-MAP | scannet200_val | LOCAL_FULL | 312/312 |
| OpenMask3D | replica48 | LOCAL_FULL | 8/8 |
| OpenMask3D | scannet200_val | AUTHOR_REPORTED_COMPATIBLE | 0/312 |
| Open3DIS | replica48 | LOCAL_FULL | 8/8 |
| Open3DIS | scannet200_val | LOCAL_FULL | 312/312 |
| MaskClustering | replica48 | LOCAL_FULL | 8/8 |
| MaskClustering | scannet200_val | AUTHOR_REPORTED_COMPATIBLE | 0/312 |

Remaining logical jobs: 65. Exact jobs and missing dependencies: `remaining_jobs.json`. Completed job costs and planning ranges: `resource_ledger.json`, `cost_estimate.json`. Training updates remain0; native reconstruction, CropFormer, SigLIP, FC encoding and frozen head inference are real new work.

## Recorded execution

Counters below describe the named sealed receipts. Live jobs may have additional unsealed consumption.

Resource ledger identity: `cae9a0342f204cac10ebb1ed265bfdd5c0e18992a64d0c9d4fd8731bd6c46de3`.

| Counter | Recorded count |
|---|---:|
| `calibration_GPU_head_calls` | 0 |
| `calibration_RGB_encodings` | 0 |
| `engineering_MV_head_calls` | 8 |
| `legacy_new_MV_head_calls` | 2868 |
| `new_training_updates` | 0 |
| `standard_CropFormer_frames` | 64000 |
| `standard_FC_image_encodings` | 49408 |
| `standard_FC_region_poolings` | 162573 |
| `standard_MV_head_calls` | 44504 |
| `standard_N_crop_encodings` | 1354746 |
| `standard_Q_crop_encodings` | 382836 |

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
| replica48 | 32 | 32 | 0 |
| scannet200_val | 2496 | 2481 | 1 |

Unique scene scorer inputs count successful per-scene evaluations after exact prediction aliases. Failed/interrupted attempts, pooled rescoring and engineering checks are excluded; this is not a total historical CPU-call count.

Legacy coverage: 176/208 logical scene rows; 12/16 pools. Logical rows do not measure physical scorer calls.

## Table fields

Each table includes the Method / setting row label and the following data columns. Per-cell units, result keys and aggregation formulas are in `cell_provenance.jsonl`.

Table-definition identity: `d24eabd9ba073b866fc0d5b9a08af0a8b875d637234b1695d447bcf94788570d`.

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

Final publication preparation verified104 distinct source files, 207 software tests and17 subtests. Standard full-class score artifacts cover2528/2560 cells; legacy artifacts cover176/208. Execution evidence uses a validated, finite merged profile. Derived Git export is limited to512MiB and40MiB per blob; overflow remains intact under $OUT/publication/release_assets when existing same-repository release access is unavailable. artifact_index.json gives every part, encoding and hash; local_artifacts.json resolves original local paths.

For publication use the original CLI with `--resume --phase publish --asset-config "$ASSET_CONFIG" --output-root "$OUT"`. For unchanged terminal recovery use the same interpreter and parameters with `--resume` and no phase override. Expected partial exit is20 and stdout must report `new_workers=0 scientific_imports=0 writes=0`. Resolve ASSET_CONFIG and interpreter from task_paths.json; retain existing explicit roots and GPU parameters. New sourced class-role mapping requires a new asset configuration and explicit `--refresh-assets`; ordinary unchanged resume does not retry blocked scientific jobs.

Verified scientific commit C1: `dd82c2e0e795ab0becf2281431ff1ab8860870cd`. [Immutable results](https://github.com/Orangekostar/oviovo/blob/dd82c2e0e795ab0becf2281431ff1ab8860870cd/docs/paper/static_ovmap/benchmark_completion_v2/RESULTS.md).
The C2 full SHA and actual post-publication resume receipt remain at $OUT/publication/final.json.
