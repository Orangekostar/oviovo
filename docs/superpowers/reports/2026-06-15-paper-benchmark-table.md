# Paper Benchmark Table Snapshot

Date: 2026-06-15

Benchmark target: Replica `room0`, 200 frames, stride 10, unless noted.

## Main Dual-Memory Ablation

| Method | Online s/frame | Online total s | Finalization s | Eval/IO s | Total incl. eval s | mIoU | f-mIoU | Objects | Final local pts | Peak local pts | Bounded backlog |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Eager local-PCD baseline | 4.378 | 875.53 | n/a | 129.27 | 1004.80 | 0.405 | 0.361 | 57 | 691,517 | n/a | Yes |
| Naive lazy local-PCD diagnostic | 3.513 | 702.64 | n/a | 671.35 | 1373.99 | 0.408 | 0.373 | 57 | 85,371,124 | n/a | No |
| Bounded local-PCD pool | 4.004 | 800.74 | 3.45 | 135.42 | 939.61 | 0.406 | 0.365 | 57 | 370,723 | 1,241,120 | Yes |
| Bounded pool + association index | 3.389 | 677.76 | 4.05 | 124.40 | 806.22 | 0.406 | 0.365 | 57 | 370,723 | 1,241,120 | Yes |

Use `Bounded local-PCD pool` as the conservative main-method row. Use `Bounded pool + association index` as a speed ablation unless the association-index path is promoted as part of the final method.

## Existing-Method Local Results

| Method | Scene | Frames | Runtime s/frame | mIoU | f-mIoU | Other metric | Objects | Fairness status |
|---|---|---:|---:|---:|---:|---|---:|---|
| DualMap | Replica room0 | 200, stride 10 | 0.242 | 0.276 | 0.716 | mAcc 0.467 | 62 | Runtime/eval available, but not same dual-memory reporting contract |
| HOV-SG | Replica office0 | n/a | n/a | 0.222 | 0.298 | pAcc 0.305 | n/a | Existing output is office0, not room0 |
| OVI-MAP local reproduction | Replica room0 | 200, stride 10 | 4.870 | 0.333 | 0.597 | mAcc 0.382; inst mIoU 0.459; AP50 0.343 | 77 | CropFormer masks + compatible Replica geometric masks; official depth segmentation was not built |
| OVO / OVOMap | Replica room0 | 200, stride 10 | 2.973 | 0.165 | 0.352 | mAcc 0.269; f-mAcc 0.428 | 609 | Local CUDA0 reproduction; precomputed SAM masks; ViT-H-14 local checkpoint |
| ConceptGraphs | Replica room0 | 200, stride 10 | 1.741 | 0.038 | 0.033 | mAcc 0.120 | 20 | Local CUDA1 reproduction; YOLO-World + MobileSAM detections; AABB IoU mapping fallback because PyTorch3D was unavailable |
| Ours / OVI-OVO | Replica room0 | 200, stride 10 | 3.389 | 0.406 | 0.365 | bounded pool + association index; eval/IO 124.40 s | 57 | Current best online version under the dual-memory reporting contract |

## Paper-Ready LaTeX

```tex
\begin{table*}[t]
\centering
\caption{Runtime and semantic mapping quality on Replica room0 with 200 input frames at stride 10. Finalization and evaluation/IO are reported separately to avoid hiding deferred work.}
\begin{tabular}{lrrrrrrr}
\toprule
Method & Online s/f & Online s & Final. s & Eval/IO s & mIoU & f-mIoU & Bounded \\
\midrule
Eager local-PCD & 4.378 & 875.53 & -- & 129.27 & 0.405 & 0.361 & Yes \\
Naive lazy local-PCD & 3.513 & 702.64 & -- & 671.35 & 0.408 & 0.373 & No \\
Bounded local-PCD pool & 4.004 & 800.74 & 3.45 & 135.42 & 0.406 & 0.365 & Yes \\
Bounded pool + assoc. index & 3.389 & 677.76 & 4.05 & 124.40 & 0.406 & 0.365 & Yes \\
\bottomrule
\end{tabular}
\end{table*}
```

```tex
\begin{table}[t]
\centering
\caption{Local baseline results under the Replica room0 200-frame stride-10 protocol. Runtime reports online mapping/inference time where available.}
\begin{tabular}{lrrrrl}
\toprule
Method & Runtime s/f & mIoU & f-mIoU & Obj. & Status \\
\midrule
DualMap & 0.242 & 0.276 & 0.716 & 62 & Available, different contract \\
HOV-SG & -- & 0.222 & 0.298 & -- & office0 only \\
OVI-MAP & 4.870 & 0.333 & 0.597 & 77 & Local reproduction, compatible geom. masks \\
OVO/OVOMap & 2.973 & 0.165 & 0.352 & 609 & Local reproduction \\
ConceptGraphs & 1.741 & 0.038 & 0.033 & 20 & Local reproduction, AABB fallback \\
Ours/OVI-OVO & 3.389 & 0.406 & 0.365 & 57 & Best online variant \\
\bottomrule
\end{tabular}
\end{table}
```

## Source Files

- Eager local-PCD baseline: `outputs/tmp_validation/20260614_structural_fast_s10_200f_fasteval/20260614063946_room0_checkpointed_s10_200f`
- Naive lazy local-PCD diagnostic: `outputs/tmp_validation/20260614_structural_v2_fast_s10_200f/20260614073450_room0_checkpointed_s10_200f`
- Bounded local-PCD pool: `outputs/tmp_validation/20260614_bounded_pool_s10_200f/20260614153952_room0_checkpointed_s10_200f`
- Bounded pool + association index: `outputs/tmp_validation/20260614_assoc_index_no_voxel_pool_s10_200f/20260614230350_room0_checkpointed_s10_200f`
- DualMap: `/home/ww/vv/paper2/DualMap/output/benchmark_20260615_existing_methods/eval/replica_room_0/results.json` and `/home/ww/vv/paper2/DualMap/output/runtime_results/room0_dualmap_200f_s10/summary.json`
- HOV-SG: `/home/ww/vv/paper2/HOV-SG/output_test_hovsg/office0_iou_quantitative.json`
- OVI-MAP local reproduction:
  - Mapping output: `/home/ww/vv/paper2/OVI-MAP/output/benchmark_20260615_ovimap/mapping_room0_200f_s10`
  - Evaluation output: `/home/ww/vv/paper2/OVI-MAP/output/benchmark_20260615_ovimap/eval_ovimap_room0_200f_s10`
  - Logs: `/home/ww/vv/paper2/OVI-MAP/output/benchmark_20260615_ovimap/logs`
  - Notes: CropFormer official weights were used for instance masks. Replica geometric masks were generated by `scripts/generate_replica_temp_geometrics.py` because the official `depth_segmentation_py` target did not build in the local Docker environment.
- OVO / OVOMap local reproduction:
  - Mapping output: `/home/ww/vv/paper2/OVO/data/output/Replica/20260615_room0_s10_200f_localclip_cuda0/room0_s10_200f`
  - Evaluation output: `/home/ww/vv/paper2/OVO/data/output/Replica/20260615_room0_s10_200f_localclip_cuda0/replica/statistics.txt`
  - Logs: `/home/ww/vv/paper2/OVO/data/output/Replica/20260615_room0_s10_200f_localclip_cuda0_full.log` and `/home/ww/vv/paper2/OVO/data/output/Replica/20260615_room0_s10_200f_localclip_cuda0_segment_eval.log`
  - Notes: Run used CUDA0, local ViT-H-14 checkpoint, and precomputed SAM masks. Mapping took 594.51 s for 200 frames; segment/eval补跑 took 29.61 s. Evaluation emitted 609 predicted masks; the saved map checkpoint contains 646 instance ids.
- ConceptGraphs local reproduction:
  - Detection runtime: `/home/ww/vv/paper2/OVO/data/input/Datasets/Replica/room0_s10_200f/exp_benchmark_room0_s10_200f/runtime_summary.json`
  - Mapping runtime: `/home/ww/vv/paper2/OVO/data/input/Datasets/Replica/room0_s10_200f/pcd_saves/full_pcd_yolo_room0_s10_200f_benchmark_room0_s10_200f_post_runtime.json`
  - Mapping output: `/home/ww/vv/paper2/OVO/data/input/Datasets/Replica/room0_s10_200f/pcd_saves/full_pcd_yolo_room0_s10_200f_benchmark_room0_s10_200f_post.pkl.gz`
  - Evaluation output: `/home/ww/vv/lifelongmap_benchmack/concept-graphs/outputs/benchmark_room0_s10_200f/eval.json`
  - Logs: `/home/ww/vv/lifelongmap_benchmack/concept-graphs/benchmark_room0_s10_detections_full.log`, `/home/ww/vv/lifelongmap_benchmack/concept-graphs/benchmark_room0_s10_mapping_full.log`, and `/home/ww/vv/lifelongmap_benchmack/concept-graphs/benchmark_room0_s10_eval_full.log`
  - Notes: Run used CUDA1. Detection took 260.58 s and mapping took 87.64 s by internal runtime summaries. `/usr/bin/time` wall time including load/startup was 282.36 s detection, 93.59 s mapping, and 25.22 s eval. Mapping used the project's AABB IoU path because PyTorch3D was unavailable for the default accurate 3D overlap path.
