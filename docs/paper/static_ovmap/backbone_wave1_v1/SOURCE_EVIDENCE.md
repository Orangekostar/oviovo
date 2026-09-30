# Source evidence and literature binding

Access/review date: 2026-09-30. Source-code ranges below are the actual inspected ranges, not citation line numbers. The review is scoped, not a claim of compiling or auditing every line of either repository. New function/file names in the contracts are proposed interfaces.

## A. User repository (pinned)

Commit: `1ce806b22c63943843300006b5b1034ec9e1cb0b`. The task branch ref was read through the connected GitHub tool and matched this commit.

| Anchor | File / inspected range | Evidence used |
|---|---|---|
| S01 | [docs/paper/static_ovmap/PAIRED_EVIDENCE_RESULTS.md](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/docs/paper/static_ovmap/PAIRED_EVIDENCE_RESULTS.md) — 1–120 | Historical official scores; actual APall .50–.90; exposure and nomination. |
| S02 | [docs/paper/static_ovmap/PAIRED_EVIDENCE_HANDOFF.md](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/docs/paper/static_ovmap/PAIRED_EVIDENCE_HANDOFF.md) — full | Parent run and release/restore paths. |
| S03 | [configs/evaluation/ovimap_module_scannet_runtime.json](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/configs/evaluation/ovimap_module_scannet_runtime.json) — full | Real environments, upstream tree, native-v10 binary and CropFormer assets. |
| S04 | [src/static_ovmap/module_validation/scannet_runtime.py](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/src/static_ovmap/module_validation/scannet_runtime.py) — 1–260 | Capture command mode4; strict old patch binding; resource and reuse boundaries. |
| S05 | [src/static_ovmap/module_validation/scannet_study.py](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/src/static_ovmap/module_validation/scannet_study.py) — 1–260 | Original native readout, color painting, fresh projection and frozen-anchor relabeling. |
| S06 | [src/static_ovmap/module_validation/native_capture.py](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/src/static_ovmap/module_validation/native_capture.py) — 1–240, 760–1060 | Request identity, current-frame snapshots, raycast-owner lineage and masks. |
| S07 | [src/static_ovmap/module_validation/query_study.py](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/src/static_ovmap/module_validation/query_study.py) — 1–180 | Current-frame causal Q replay and charged feature dispatch. |
| S08 | [src/static_ovmap/module_validation/semantic_study.py](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/src/static_ovmap/module_validation/semantic_study.py) — 1–255 | Static cap128, top3 area requests and final segment-lineage reconciliation. |
| S09 | [src/static_ovmap/a7_evidence_upgrade/region_adapter.py](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/src/static_ovmap/a7_evidence_upgrade/region_adapter.py) — 1–125 | Pinned mask-pooling operators, image/mask conventions and FC backbone. |
| S10 | [src/static_ovmap/a7_evidence_upgrade/region_worker.py](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/src/static_ovmap/a7_evidence_upgrade/region_worker.py) — 1–225 | Whole-frame encoding reuse; pooled request features; original static source contract. |
| S11 | [src/static_ovmap/m2_reviewer_study/evaluation.py](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/src/static_ovmap/m2_reviewer_study/evaluation.py) — 1–240 | Geometry-bound evaluator, official current-class area ranks and complete dataset pool. |
| S12 | [src/static_ovmap/module_validation/evaluation.py](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/src/static_ovmap/module_validation/evaluation.py) — 1–205 | PredictionPayload/GeometryIdentity constraints and valid branch tags. |
| S13 | [third_party_patches/ovimap/module_validation_v1/ovimap_module_validation_v1.patch](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/third_party_patches/ovimap/module_validation_v1/ovimap_module_validation_v1.patch) — 1–230 | Existing exportStudyFrameState/TsdfState/SurfaceLabels interface layer. |
| S14 | [docs/paper/static_ovmap/A7_WAVE1_HANDOFF.md](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/docs/paper/static_ovmap/A7_WAVE1_HANDOFF.md) — full | SAM2/FC/OVR assets and code pins; SAM3 access block; full OVR wrapper not run. |
| S15 | [docs/paper/static_ovmap/MODULE_VALIDATION_HANDOFF.md](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/docs/paper/static_ovmap/MODULE_VALIDATION_HANDOFF.md) — full | Historical4 development roles, native-v10 restoration, excluded old confirm scenes. |
| S16 | [src/static_ovmap/paired_evidence_study/binding.py](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/src/static_ovmap/paired_evidence_study/binding.py) — 1–220 | Old task branch and exact-scene checks; do not reuse with fake identities. |
| S17 | [src/static_ovmap/a7_evidence_upgrade/binding.py](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/src/static_ovmap/a7_evidence_upgrade/binding.py) — 1–240 | Parent reviewer/source/calibration receipt chain. |
| S18 | [scripts/evaluation/run_ovimap_paired_evidence.py](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/scripts/evaluation/run_ovimap_paired_evidence.py) — 1–145 | Old runner explicitly disablesCUDA and is unsuitable for new capture. |
| S19 | [docs/paper/static_ovmap/paired_evidence_v1/PROTOCOL_SPEC.json](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/docs/paper/static_ovmap/paired_evidence_v1/PROTOCOL_SPEC.json) — 1–190 | Historical CAL/Replica IDs and frozen source-temperature policy. |

## B. Upstream (pinned)

Commit: `f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424`. Read together with the user repository native-v10 patch, not a pristine upstream binary.

| Anchor | File / inspected range | Evidence used |
|---|---|---|
| U01 | [scripts/utils/common_scannet_nyu.py](https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/scripts/utils/common_scannet_nyu.py) — 110–285 | Sequential mutation; .9/.5/.2 thresholds; initial100-pixel removal; score1.0. |
| U02 | [scripts/panoptic_mapping_.py](https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/scripts/panoptic_mapping_.py) — 230–580 | Real insertion/raycast/view-selection order; skip-feature metadata versus missing final pickle. |
| U03 | [mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm/src/global_segment_map_py.cpp](https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm/src/global_segment_map_py.cpp) — 200–465, 585–775 | Actual Python-facing integrateFrame and raycast forwarding. |
| U04 | [mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm/src/label_tsdf_confidence_integrator.cpp](https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm/src/label_tsdf_confidence_integrator.cpp) — 390–700 | Fragment candidates/decisions and mode4 count+graph update. |
| U05 | [mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm/src/label_tsdf_confidence_integrator.cpp](https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm/src/label_tsdf_confidence_integrator.cpp) — 720–1035 | Existing current-instance grouping and graphmode3 confidence construction. |
| U06 | [mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm/src/label_tsdf_confidence_integrator.cpp](https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm/src/label_tsdf_confidence_integrator.cpp) — 1035–1450 | First-fragment object decision; unused ratio predicate; alias merge execution. |
| U07 | [mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm/src/label_tsdf_confidence_integrator.cpp](https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm/src/label_tsdf_confidence_integrator.cpp) — 1630–1875 | Read-only instance raycaster versus distinct mutating panoptic routine. |
| U08 | [mapping_ros_ws/src/consistent_panoptic_mapping/global_segment_map/src/semantic_instance_label_fusion.cc](https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/mapping_ros_ws/src/consistent_panoptic_mapping/global_segment_map/src/semantic_instance_label_fusion.cc) — 1–170, 190–360 | Mode4 ordinary lookup uses count maps, not mode3 graph components. |
| U09 | [mapping_ros_ws/src/consistent_panoptic_mapping/global_segment_map/src/segment_graph.cpp](https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/mapping_ros_ws/src/consistent_panoptic_mapping/global_segment_map/src/segment_graph.cpp) — 1–220, 265–480 | Graph accumulation and re-extraction already exist. |
| U10 | [scripts/options.py](https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/scripts/options.py) — 1–165 | Help text is incomplete relative to compiled dispatch; not an authoritative mode switch. |

## C. SAM2 executable interface

[Pinned SAM2VideoPredictor](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py), ranges20–165 and510–680: `init_state`/CPU offload, object state, and inclusive `propagate_in_video` bounds. The existing run receipt records this same code revision. API availability is not proof that local weights, CUDA and permissions are currently usable; Codex must perform the scoped real preflight.

## D. Literature-to-experiment mapping

OnlinePG was re-read in its author manuscript, section3.3: [arXiv HTML](https://arxiv.org/html/2603.18510v1). It uses distinct directional containment information and mutual bipartite correspondences. Our geometry-only depth-conditioned adaptation is NOT a full reproduction of its Gaussian representation, semantic costs, sliding-window graph or map update.

| Roadmap reference (formal venue year) | Role in this executable wave |
|---|---|
| OnlinePG — CVPR2026 | Motivates B05; actual independent directional costs are explicitly adapted, not copied as a claimed full implementation. |
| SAM3 — ICLR2026 | Context for detection/tracking separation; deferred due to scope and historical access block. |
| OVRCOAT — CVPR2026 | B04 backlog; distinguishes candidate/objectness from the already-tested region-only classifier. |
| GeoSAM2 — CVPR2026 | B12 local part-candidate backlog; not a full-scene object replacement in this wave. |
| Efficient-SAM2 — ICLR2026 | B11 efficiency follow-up only after propagation has useful map effects. |
| CompetitorFormer — CVPR2026 | B10 trained3D proposal follow-up; not implemented by renaming a greedy matcher. |
| GeoGuide — CVPR2026 | B10 learned geometric-consistency follow-up. |
| GeoPurify — ICLR2026 | B10 geometric-distillation follow-up, not ordinary hand smoothing. |
| The Midas Touch for Metric Depth — CVPR2026 | B08 optional depth-hole work; partial code release is not full reproduction. |
| TGSFormer — CVPR2026 | Long-term memory/completion context, outside current observed-surface protocol. |
| Ov3R — CVPR2026 | Separate whole-system RGB reconstruction follow-up, not a drop-in RGB-D backend. |

SAM2.1 is the existing public implementation/control used now; it is not included in the count of 2026 papers.

## E. Inherited bibliography

The following source links are retained from the user-supplied 2026-09-30 roadmap. This section preserves that prior literature work and does not imply a new full reproduction or line-by-line review of all11 repositories in this turn. The earlier mode3-only assertion in that roadmap is explicitly superseded by U04/U08 and the corrected execution instruction.

- https://openaccess.thecvf.com/content/CVPR2026/html/Kormushev_Mitigating_Objectness_Bias_and_Region-to-Text_Misalignment_for_Open-Vocabulary_Panoptic_Segmentation_CVPR_2026_paper.html
- https://github.com/nickormushev/OVRCOAT
- https://proceedings.iclr.cc/paper_files/paper/2026/hash/e0982cbc81401df3430ee1ff780dc7a2-Abstract-Conference.html
- https://github.com/facebookresearch/sam3
- https://openaccess.thecvf.com/content/CVPR2026/html/Deng_GeoSAM2_Unleashing_the_Power_of_SAM2_for_3D_Part_Segmentation_CVPR_2026_paper.html
- https://github.com/VAST-AI-Research/GeoSAM2
- https://proceedings.iclr.cc/paper_files/paper/2026/hash/aaa0ac4253da75faf9b0dc0dda062612-Abstract-Conference.html
- https://github.com/jingjing0419/Efficient-SAM2
- https://openaccess.thecvf.com/content/CVPR2026/html/Zhai_OnlinePG_Online_Open-Vocabulary_Panoptic_Mapping_with_3D_Gaussian_Splatting_CVPR_2026_paper.html
- https://arxiv.org/html/2603.18510
- https://openaccess.thecvf.com/content/CVPR2026/html/Wang_CompetitorFormer_Mitigating_Query_Conflicts_for_3D_Instance_Segmentation_via_Competitive_CVPR_2026_paper.html
- https://github.com/DuanchuWang/CompetitorFormer
- https://openaccess.thecvf.com/content/CVPR2026/html/Tao_GeoGuide_Hierarchical_Geometric_Guidance_for_Open-Vocabulary_3D_Semantic_Segmentation_CVPR_2026_paper.html
- https://arxiv.org/html/2603.26260
- https://proceedings.iclr.cc/paper_files/paper/2026/hash/039bc8e424e1fc196b4203555b54ebc4-Abstract-Conference.html
- https://github.com/tj12323/GeoPurify
- https://openaccess.thecvf.com/content/CVPR2026/html/Ma_The_Midas_Touch_for_Metric_Depth_CVPR_2026_paper.html
- https://github.com/HenryMaxixi/MTD
- https://openaccess.thecvf.com/content/CVPR2026/html/Qian_TGSFormer_Scalable_Temporal_Gaussian_Splatting_for_Embodied_Semantic_Scene_Completion_CVPR_2026_paper.html
- https://arxiv.org/html/2512.00300
- https://openaccess.thecvf.com/content/CVPR2026/html/Gong_Ov3R_Open-Vocabulary_Semantic_3D_Reconstruction_from_RGB_Videos_CVPR_2026_paper.html
- https://github.com/ZoranGong/Ov3R
- https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/scripts/panoptic_mapping_.py
- https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/scripts/utils/common_scannet_nyu.py
- https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/scripts/options.py
- https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/mapping_ros_ws/src/consistent_panoptic_mapping/global_segment_map/src/segment_graph.cpp
- https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm/src/label_tsdf_confidence_integrator.cpp
- https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/docs/paper/static_ovmap/PAIRED_EVIDENCE_RESULTS.md
- https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/src/static_ovmap/paired_evidence_study/residuals.py
- https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/src/static_ovmap/m2_reviewer_study/evaluation.py
- https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/src/static_ovmap/module_validation/scannet_study.py
- https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/src/static_ovmap/module_validation/query_study.py

Additional verified source: [composition_study/execution.py](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/src/static_ovmap/composition_study/execution.py), inspected1–85 and170–300; the query job identity explicitly binds budget200 and reads the actual frozen checkpoint/text records.

Native patch parent independently confirmed in the legacy [module-validation specification](https://github.com/Orangekostar/oviovo/blob/1ce806b22c63943843300006b5b1034ec9e1cb0b/docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json), inspected1–145: `official_commit` is exactly the same pinned f8f7bcd full SHA. This avoids accidentally layering the capture patch on a different upstream revision.
