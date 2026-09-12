# Frozen room1 confirmation

STATIC_CONFIRMATION_STATUS=COMPLETE

All 11 selected methods and direct controls were predicted before opening room1
GT, then evaluated on the same 6,421,401 source rows. No retuning was performed.
The selection hash and method list are recorded in `room1_confirmation_status.json`;
source and metric invariants passed in `room1_confirmation_invariants.json`.
This is one held-out static scene, not evidence of dynamic cross-environment generalization.

| Method | mIoU | f-mIoU | CA-AP25 | CA-AP50 |
|---|---:|---:|---:|---:|
| B_SEM_OVI_NATIVE | 0.399250 | 0.704025 | 0.485494 | 0.209150 |
| B_SEM_CROVE_S0 | 0.423211 | 0.690873 | 0.485494 | 0.209150 |
| B_SEM_CROVE_S2 | 0.444857 | 0.708311 | 0.485494 | 0.209150 |
| MV_DIVERSE_MEAN | 0.351038 | 0.555729 | 0.485494 | 0.209150 |
| MV_QUALITY | 0.355117 | 0.555948 | 0.485494 | 0.209150 |
| S2_GRAPH_GEOM | 0.446689 | 0.709982 | 0.485494 | 0.209150 |
| S2_GRAPH_BOUNDARY | 0.446672 | 0.709970 | 0.485494 | 0.209150 |
| INST_PAIRWISE | 0.399250 | 0.704025 | 0.285446 | 0.116899 |
| INST_CONSENSUS_OWNER | 0.399250 | 0.704025 | 0.505323 | 0.227223 |
| ADAPTER_CLIP_MEAN | 0.411792 | 0.719303 | 0.485494 | 0.209150 |
| ADAPTER_CLIP_LEARNED | 0.345663 | 0.652878 | 0.485494 | 0.209150 |

All methods preserve F5=0.9307395484. M1/M2/M4 preserve native class-agnostic
AP; both M3 owner-only methods preserve native mIoU and f-mIoU.

M1 QUALITY improves its diverse-view direct control by 0.004079 mIoU but falls
0.044133 below native. M2's frozen boundary graph improves S2 by 0.001815 mIoU,
while adding boundaries to its geometry-graph control changes mIoU by -0.000017.
The latter is reported without replacing the frozen representative after confirmation.
M3 consensus improves native CA-AP50 by 0.018072 and pairwise by 0.110324;
its positive room1 result contrasts with its negative room0 native comparison.
M4's official learned head falls 0.066129 mIoU below its matched mean-pooling
control, so the DEV improvement over that control did not transfer to room1.

Exact metrics, prediction-stage timing, owner changes and selection binding are
retained in each `room1_<method>.json`. Shared preparation costs are not standalone
per-method costs. No confidence intervals or significance claims are inferred
from this single confirmation scene.
