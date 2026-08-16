# Literature Search: Causal Temporal Identity for Dynamic Mapping

Search date: 2026-08-16

## Decision

The closest defensible improvement is not a generic stronger re-identification
frontend. It is a causal, role-separated identity decision for an online
current-state map:

1. active continuation is locally geometry-gated;
2. dormant reactivation alone may use a wider appearance/semantic gate;
3. a new persistent identity is admitted only after repeated support; and
4. dynamic state requires stable motion evidence rather than one noisy centroid.

This choice is driven by the observed CROVE failure pattern: many one-frame
identities, excessive dynamic entities, and weak change/background recovery.
It also avoids direct overlap with OASIS-Map's multi-session dense semantic
correspondence backend.

## Priority Papers

| Work | Year | Primary source | Evidence used | Decision |
| --- | ---: | --- | --- | --- |
| Khronos | 2024 | [RSS paper](https://www.roboticsproceedings.org/rss20/p081.pdf) | Separates fast short-term dynamics from slower long-term change reasoning. | Closest dynamic baseline; retain. |
| OASIS-Map | 2026 | [arXiv](https://arxiv.org/abs/2607.14899) | Identifies cross-visit object association under partial views, occlusion, and imperfect segmentation as the central difficulty. | Closest recent threat; retain and differentiate online causal setting. |
| OVI-MAP | 2026 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Deng_OVI-MAP_Open-Vocabulary_Instance-Semantic_Mapping_CVPR_2026_paper.html) | Stable class-agnostic instance reconstruction and geometry-first association support a strong static identity foundation. | Retain; motivates preserving T1 and local geometric continuation. |
| DualMap | 2025 | [arXiv](https://arxiv.org/abs/2506.01950) | Object-level status checks and dual map roles support lightweight online change updates. | Retain baseline/context. |
| Panoptic Multi-TSDFs | 2022 | [arXiv](https://arxiv.org/abs/2109.10165) | Uses objects as change units and separate volumetric submaps for long-term consistency. | Retain representation prior. |
| Fusion++ | 2018 | [arXiv](https://arxiv.org/abs/1808.08378) | Per-object TSDFs and existence probability explicitly address spurious detections. | Retain; motivates delayed persistent admission. |
| EM-Fusion | 2019 | [CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Strecke_EM-Fusion_Dynamic_Object-Level_SLAM_With_Probabilistic_Data_Association_ICCV_2019_paper.html) | Probabilistic pixel-object data association is central to robust dynamic reconstruction. | Retain mechanism context. |
| ODAM | 2021 | [CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Li_ODAM_Object_Detection_Association_and_Mapping_Using_Posed_RGB_Video_ICCV_2021_paper.html) | Frame-to-global object association uses multi-view constraints. | Retain association context. |
| Living Scenes | 2024 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_Living_Scenes_Multi-object_Relocalization_and_Reconstruction_in_Changing_3D_Environments_CVPR_2024_paper.html) | Long-term object relocalization and reconstruction require identity consistency across sparse revisits. | Retain long-term context. |
| Consistent Instance Field | 2026 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_Consistent_Instance_Field_for_Dynamic_Scene_Understanding_CVPR_2026_paper.html) | Explicitly separates visibility from persistent identity in a probabilistic representation. | Retain conceptual support; different offline/neural setting. |
| GaME | 2026 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Yugay_Gaussian_Mapping_for_Evolving_Scenes_CVPR_2026_paper.html) | Stale observations damage evolving-scene geometry and semantics; keyframe state must be updated. | Retain adjacent evolving-scene evidence. |
| MaskFusion | 2018 | [arXiv](https://arxiv.org/abs/1804.09194) | Instance masks support object-level tracking and independent reconstruction. | Retain historical context. |

## Screened But Secondary

| Work | Primary source | Reason not central |
| --- | --- | --- |
| Co-Fusion | [arXiv](https://arxiv.org/abs/1706.06629) | Strong dynamic reconstruction history, but not long-term open-vocabulary current-state maintenance. |
| ReFusion | [arXiv](https://arxiv.org/abs/1905.02082) | Residual/free-space dynamic rejection mainly protects static reconstruction. |
| MID-Fusion | [arXiv](https://arxiv.org/abs/1812.07976) | Object-level dynamic SLAM, but closed-set and short-term. |
| ConceptGraphs | [arXiv](https://arxiv.org/abs/2309.16650) | Relevant open-vocabulary object association baseline, but assumes static accumulation. |
| Hydra | [arXiv](https://arxiv.org/abs/2201.13360) | Useful real-time scene-graph architecture context, not the closest change-maintenance mechanism. |
| DovSG | [arXiv](https://arxiv.org/abs/2410.11989) | Dynamic open-vocabulary scene graph for manipulation; evaluation goal differs. |

## Novelty Boundary

The claim must remain narrow: CROVE performs causal online current-state
maintenance from past and present frames only. It should not claim the strongest
cross-session semantic correspondence, global 4D reconstruction, or offline
identity optimization. Novelty remains contingent on experiments showing that
role-separated identity admission improves current-state and dynamic metrics
without reducing the frozen static readout.
