# Source bindings and epistemic boundaries

Reviewed on 2026-09-20. Repository content was read through the connected GitHub tool. Source references below are immutable where the reviewed commit was available. They establish existing behavior, not correctness or future performance of the proposed module. Paper mechanisms inherited from the preceding review are identified separately; no full third-party neural pipeline was executed here.

## A. User project: Orangekostar/oviovo @ b6455520c758a3413988e0827b3e1f34667bdfd1

Base URL: https://github.com/Orangekostar/oviovo/tree/b6455520c758a3413988e0827b3e1f34667bdfd1

| Source | Verified use and source identity |
|---|---|
| [T1 asset configuration](https://github.com/Orangekostar/oviovo/blob/b6455520c758a3413988e0827b3e1f34667bdfd1/configs/evaluation/ovimap_t1_attribution_v1.json) | Actual native mesh, S1a/B0, T0 receipts, evaluator/projection and text paths. Blob `3c03c1cdcaa72dae44eac21c327794033c06f01b`. Paths recorded in a receipt are not proof of current machine readability. |
| [OD asset configuration](https://github.com/Orangekostar/oviovo/blob/b6455520c758a3413988e0827b3e1f34667bdfd1/configs/evaluation/ovimap_object_decoupling_v1.json) | Native local model, processor, RGB receipt, earlier inference restrictions. Blob `6df3bc834fdabcd7b45f0db03f71ed9667d43b1a`. |
| [Last semantic configuration](https://github.com/Orangekostar/oviovo/blob/b6455520c758a3413988e0827b3e1f34667bdfd1/configs/evaluation/ovimap_sf_ovi_semantics_v1.json) | Fixed H geometry/ranks, last reverse-transfer and support-gate definitions. Blob `d09ac2012e382c42b57356861801a20f9b7d288c`. |
| [Semantic handoff](https://github.com/Orangekostar/oviovo/blob/b6455520c758a3413988e0827b3e1f34667bdfd1/docs/paper/static_ovmap/SF_OVI_SEMANTIC_HANDOFF.md) | Reported4restored/10lost for SF transfer, zero new inference and explicit identical-payload reuse. Blob `b4d416ced0c143d58d9cd7579dbf45f7fe9e674b`. This bundle does not independently rerun those experiments. |
| [candidate_semantics.py](https://github.com/Orangekostar/oviovo/blob/b6455520c758a3413988e0827b3e1f34667bdfd1/src/static_ovmap/candidate_semantics.py) | Real native six-crop construction, old1024-D encoder, separate pregate scores vs gated final class. Blob `fb96709c91244ba981fb92904c40658364dca3c9`. |
| [ownership_evidence.py](https://github.com/Orangekostar/oviovo/blob/b6455520c758a3413988e0827b3e1f34667bdfd1/src/static_ovmap/ownership_evidence.py) | Existing positive-local-PNG restriction and depth-visible projection; not equivalent to native union ROI. Blob `80a8e07d5c8feab7ffcc6a97b828ffe0c037ec4c`. |
| [fixed-geometry evaluator](https://github.com/Orangekostar/oviovo/blob/b6455520c758a3413988e0827b3e1f34667bdfd1/scripts/evaluation/evaluate_static_sf_ovi_semantics.py) | Actual released-evaluator adapter, payload keys, strict protocol checks, hardcoded zero new inference inappropriate for this study. Blob `949b8b4e759e2b99b2464e65675ee44cd71a1b46`. |
| [attribution_objects.py](https://github.com/Orangekostar/oviovo/blob/b6455520c758a3413988e0827b3e1f34667bdfd1/src/static_ovmap/attribution_objects.py) | Reuse target for geometric correspondence and actual released matching traces, already reviewed in the preceding studies. Codex must inspect actual signatures; the new instruction does not fabricate a new existing helper signature. |

Historical details are in `REPLICA8_RESULTS.md`, `R4_RESULTS.md`, `R5_RESULTS.md`, `T1_ATTRIBUTION_RESULTS.md`, `LOCAL_OWNERSHIP_RESULTS.md`, `OBJECT_DECOUPLING_RESULTS.md` and `SF_OVI_SEMANTIC_RESULTS.md` at that same commit. They motivated the preceding review, not new measurements in this task. The attached `THREE_INNOVATIONS_REVIEW_ZH.md` is the proposal source and has been read; it is not runtime authority.

## B. Official OVI-MAP @ f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424

| Source | Actually checked behavior |
|---|---|
| [panoptic_mapping_.py](https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/scripts/panoptic_mapping_.py) | Main loop and worker path (requested lines1–570). Global bbox, union local/global mask, native combine, delayed feature collection, selected-query metadata and final top10 retention. Blob `c0858b8d10b6b863cecb0e9ea6de7643212a1d19`. |
| [view_selection.py](https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/scripts/view_selection.py) | Technical admission and coverage/area selection; mutation before late visibility rejection; query selection is not a strict total query cap. Blob `cf10338e1df3a6d9d32a4fac401422a0048a3c2d`. |
| [global_segment_map_py.cpp](https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm/src/global_segment_map_py.cpp) | Inspected constructor and integration/clear/mesh sections (1–200 and365–600): active association flags, SegGraph modes3/4/6/7, segment labels and merge data before clear, mesh flags. Blob `14e66cac11aff614d15bcca0a7adb8062d59512f`. This is not a claim to have audited every C++ branch. |
| [global_segment_map_py.h](https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm/include/consistent_mapping/global_segment_map_py.h) | Existing wrappers/member ownership (1–270), confirms NEW export APIs must be added; no existing arbitrary-owner setter was claimed. Blob `42063112253235ff36e920fc0f085f112521748e`. |
| [segment_graph.cpp](https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/mapping_ros_ws/src/consistent_panoptic_mapping/global_segment_map/src/segment_graph.cpp) | Preceding review: accumulated confidence and instance extraction, blob `a3c82bd74863f7e03b3bf224eb3b2bd9404a5aa0`; active-path applicability must be bound at runtime. |

Paper/source context: https://arxiv.org/abs/2603.26541 ; https://arxiv.org/html/2603.26541v1 ; https://ovi-map.github.io/
The manuscript's ablations motivate retaining the strong front end and language model. Their numbers are not executable success targets or additive contributions. No PDF visual inspection or new benchmark execution is claimed in this bundle.

## C. Frozen model candidates

- Google SigLIP2-L official model card: https://huggingface.co/google/siglip2-large-patch16-384 . Its image/text feature interfaces were checked. Model size/revision/file access must be bound on the executing machine; the exact mutable HF main is not silently hard-coded. This is a **2025** model used as a strong control, not a2026 novelty claim.
- WOW author repository: https://github.com/AAwcAA/WOW-Seg-Meta ; reviewed main resolves to `bfc6f2424c47097e70a641c6a8016319cac192cb` on2026-09-20.
- [WOW region helper](https://github.com/AAwcAA/WOW-Seg-Meta/blob/bfc6f2424c47097e70a641c6a8016319cac192cb/demo/wow_inference.py): full relevant preparation/chat code read, blob `c298466003cbe4257f585cf1e4ef1c8acabf936e`; confirms actual16x16 mask preprocessing, category-only prompt and per-example fallbacks that this trial must disable.
- [WOW README](https://github.com/AAwcAA/WOW-Seg-Meta/blob/bfc6f2424c47097e70a641c6a8016319cac192cb/README.md): blob `46fd724553e09596fab7361d5333f0129cd85a39`; official weight link and region-classification/name-mapper examples. Author reports ICLR2026 acceptance and released weights.
- Official weight identifier: https://huggingface.co/AAwcAA/WOW-Seg . **Not loaded/verified in the planning environment.** A public weight link is not evidence of successful local download, compatible architecture or sufficient device memory.
- Fixed name-mapper identifier: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 ; used by the author examples. Bind revision and preprocess at execution; do not replace it opportunistically based on accuracy.

## D. Ideas inherited from the preceding review — NOT runtime dependencies

| Idea source | Reference | Narrow role / not claimed |
|---|---|---|
| LEGO | https://github.com/WHU-USI3DV/LEGO ; https://arxiv.org/abs/2608.10057 | Cross-view structural granularity inspired G's complete hypotheses. No Gaussian training or complete LEGO reproduction. |
| GeoGuide | https://arxiv.org/abs/2603.26260 | Whole-instance and relation constraints motivated partition scoring. No verified deployable code dependency is assumed. |
| OVRCOAT | https://github.com/nickormushev/OVRCOAT ; https://arxiv.org/abs/2603.21386 | Distinguishing objectness from region recognition motivates separate semantic evidence. Its model is not being installed. |
| APPLE | https://timschneider42.github.io/apple/ ; https://github.com/TimSchneider42/apple | Task-loss-driven acquisition inspiration; author identifies ICLR2026. Q uses a small regression head, NOT APPLE reinforcement learning. |
| COVER/NBV-Gym | https://github.com/chengine/nbv_gym | Candidate/score/selection separation and equal-budget controls. No new trajectory, Gaussian system or native COVER reproduction. |

## E. What remains unknown, and is not hidden by the word final

Authorized independent dataset availability; exact historical capture completeness; actual loaded native extension and its numerical label-export path; WOW weight compatibility; amount of independent GAIN/HARM/partition/query training evidence; any net module gain; holdout generalization. The execution contract provides concrete resolution/status rules for these unknowns. It does not turn them into proven facts. No promised AP/mIoU gain appears in the contract.
