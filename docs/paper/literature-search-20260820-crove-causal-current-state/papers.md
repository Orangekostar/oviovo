# Literature Search: Causal Current-State Open-Vocabulary Mapping

Date: 2026-08-20
Search purpose: closest-work and novelty-risk diagnosis for CROVE
Target venue/family: AAAI, AI/ML/CV model-method paper
Source-quality policy: primary-source preference and CCFA exclusions applied

## Summary

The original headline, "online open-vocabulary mapping that reflects the
current scene rather than accumulated history," is already covered. Most
directly, RSS 2026 SuperMap is a real-time, open-vocabulary, instance-level
spatio-temporal SLAM system that maintains identities through occlusion and
prunes stale map content using depth-residual visibility states, log-odds
updates, and Bayesian semantic fusion. DynaMem also adds and removes
open-vocabulary voxels online; DualMap and DovSG update open-vocabulary maps in
changing scenes; and Where Did I Leave My Glasses? explicitly maintains a
current semantic map with probabilistic object stationarity. CROVE must not
claim this problem formulation or this capability bundle as its novelty.

The closest mechanism threats are also strong. Khronos factorizes short- and
long-term dynamics; POCD models object stationarity probabilistically; OASIS-Map
uses dense semantic correspondence for incremental cross-session identity and
change; and Consistent Instance Field explicitly separates visibility from
persistent identity. A dual-center trajectory fix alone is useful engineering
but not an AAAI-level central contribution.

The remaining defensible route is narrower: **observability- and
role-conditioned evidence routing**. CROVE can formalize that evidence may
update only the state variable it observes: free-space rays may update current
existence and ownership but not identity; local geometric continuity may update
an active pose but not widen dormant re-identification; qualified appearance
may preserve identity across a geometry epoch but not modify fused static
geometry; and a current pose may drive temporal readout while the fused centroid
continues to drive the cumulative map. This route is a literature-grounded
inference, not yet an established novelty claim.

## Paper Table

| # | Title | Year | Venue/source | Link | Type | Insight | Completeness | Numeric evidence | Overall | Relevance |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | SuperMap: A Spatio-Temporal SLAM System for Visual-Language Navigation | 2026 | RSS | [paper](https://www.roboticsproceedings.org/rss22/p052.pdf) | system/tool | 4 | 4 | 3 | Risk | Most direct capability and mechanism threat: real-time open-vocabulary 4D instance mapping, 3D-aware tracking, visibility-conditioned log-odds removal, and Bayesian semantics. |
| 2 | Khronos: A Unified Approach for Spatio-Temporal Metric-Semantic SLAM in Dynamic Environments | 2024 | RSS | [paper](https://www.roboticsproceedings.org/rss20/p081.html) | pure method | 5 | 5 | 5 | A | Closest protocol-valid dynamic-map baseline; active window, global reconciliation, and short/long-term factorization threaten broad temporal claims. |
| 3 | DynaMem: Online Dynamic Spatio-Semantic Memory for Open World Mobile Manipulation | 2025 | ICRA | [paper](https://arxiv.org/abs/2411.04999) | pure method | 4 | 4 | 4 | Risk | Directly covers causal online open-vocabulary voxel addition/removal and abstaining object localization in changing scenes. |
| 4 | Where Did I Leave My Glasses? Open-Vocabulary Semantic Exploration in Real-World Semi-Static Environments | 2026 | IEEE RA-L | [paper](https://arxiv.org/abs/2509.19851) | pure method | 4 | 4 | 4 | Risk | Directly claims current-state semantic-map maintenance, probabilistic stationarity, re-identification, and active revisitation. |
| 5 | OASIS-Map: Object-Level Change Detection in Multi-Session Mapping using Semantic Correspondence Matching | 2026 | arXiv, under review | [paper](https://arxiv.org/abs/2607.14899) | pure method | 5 | 3 | 3 | Risk | Closest identity/change threat; dense patch correspondence handles partial views and defers uncertain change decisions. |
| 6 | DualMap: Online Open-Vocabulary Semantic Mapping for Natural Language Navigation in Dynamic Changing Scenes | 2025 | IEEE RA-L | [paper](https://arxiv.org/abs/2506.01950) | pure method | 4 | 4 | 4 | A | Direct dynamic open-vocabulary online baseline with abstract/concrete map roles and online relocation updates. |
| 7 | Dynamic Open-Vocabulary 3D Scene Graphs for Long-Term Language-Guided Mobile Manipulation | 2025 | IEEE RA-L | [paper](https://arxiv.org/abs/2410.11989) | system/tool | 3 | 4 | 3 | B | Maintains a locally updated open-vocabulary scene graph in long-term changing environments. |
| 8 | OpenIN: Open-Vocabulary Instance-Oriented Navigation in Dynamic Domestic Environments | 2025 | IEEE RA-L | [paper](https://arxiv.org/abs/2501.04279) | pure method | 3 | 3 | 3 | B | Updates carrier relationships for moved object instances; relevant to downstream dynamic-query positioning. |
| 9 | OVI-MAP: Open-Vocabulary Instance-Semantic Mapping | 2026 | CVPR | [paper](https://openaccess.thecvf.com/content/CVPR2026/html/Deng_OVI-MAP_Open-Vocabulary_Instance-Semantic_Mapping_CVPR_2026_paper.html) | pure method | 4 | 5 | 5 | A | Strongest static instance-semantic foundation and T1 baseline; decouples instance reconstruction from semantics. |
| 10 | Consistent Instance Field for Dynamic Scene Understanding | 2026 | CVPR | [paper](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_Consistent_Instance_Field_for_Dynamic_Scene_Understanding_CVPR_2026_paper.html) | pure method | 5 | 5 | 5 | Risk | Explicit visibility/identity disentanglement is a direct conceptual threat, although its offline neural-field setting differs. |
| 11 | Gaussian Mapping for Evolving Scenes | 2026 | CVPR | [paper](https://openaccess.thecvf.com/content/CVPR2026/html/Yugay_Gaussian_Mapping_for_Evolving_Scenes_CVPR_2026_paper.html) | pure method | 4 | 5 | 5 | A | Removes stale observations through keyframe adaptation in evolving scenes; adjacent geometric-currentness threat. |
| 12 | Living Scenes: Multi-object Relocalization and Reconstruction in Changing 3D Environments | 2024 | CVPR | [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_Living_Scenes_Multi-object_Relocalization_and_Reconstruction_in_Changing_3D_Environments_CVPR_2024_paper.html) | pure method | 5 | 5 | 5 | A | Strong multi-session instance matching, pose recovery, and reconstruction baseline on 3RScan. |
| 13 | POCD: Probabilistic Object-Level Change Detection and Volumetric Mapping in Semi-Static Scenes | 2022 | RSS | [paper](https://www.roboticsproceedings.org/rss18/p013.html) | method + benchmark | 4 | 5 | 5 | A | Establishes probabilistic stationarity plus TSDF change evidence; prevents claiming generic object-belief updates. |
| 14 | Panoptic Multi-TSDFs: A Flexible Representation for Online Multi-resolution Volumetric Mapping and Long-term Dynamic Scene Consistency | 2022 | ICRA | [paper](https://arxiv.org/abs/2109.10165) | pure method | 4 | 5 | 4 | A | Establishes object-level multi-TSDF representation and online long-term consistency. |
| 15 | ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning | 2024 | ICRA | [paper](https://arxiv.org/abs/2309.16650) | pure method | 4 | 5 | 4 | A | Core open-vocabulary object association and scene-graph baseline, but primarily static/offline in the relevant comparison. |
| 16 | EM-Fusion: Dynamic Object-Level SLAM With Probabilistic Data Association | 2019 | ICCV | [paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Strecke_EM-Fusion_Dynamic_Object-Level_SLAM_With_Probabilistic_Data_Association_ICCV_2019_paper.html) | pure method | 4 | 5 | 5 | B | Historical anchor for object-level probabilistic association and independent dynamic reconstruction. |

Scores use the CCFA 1-5 anchors. `Risk` marks a paper that can invalidate a
central novelty claim even when its evaluation setting differs.

## Closest-Work Clusters

### Online open-vocabulary current-state memory

- Representatives: SuperMap, DynaMem, Where Did I Leave My Glasses?, DualMap,
  DovSG.
- Already solved: online open-vocabulary updates for moved, appeared, or removed
  objects and downstream object search/navigation.
- Under-tested: matched-protocol dense current-map quality, exact static-map
  non-interference, and typed attribution of each update. SuperMap already
  distinguishes observable, unobservable, and disappeared geometry, so
  occlusion handling alone is not a gap.
- CROVE consequence: current-state maintenance is the task, not the novelty.

### Spatio-temporal identity and change

- Representatives: SuperMap, Khronos, OASIS-Map, Living Scenes, POCD.
- Already solved: multi-timescale state, probabilistic stationarity,
  cross-session association, pose changes, and deferred change decisions.
- Under-tested: strict per-frame causality with open-vocabulary semantics and a
  state-specific evidence-admissibility contract.
- CROVE consequence: do not claim generic identity persistence, geometry epochs,
  Bayesian lifecycle state, or long-term change detection as new in isolation.

### Visibility, ownership, and stale geometry

- Representatives: DynaMem, Panoptic Multi-TSDFs, GaME, Consistent Instance
  Field.
- Already solved: ray-based stale-voxel removal, object-level submaps, stale
  keyframe replacement, and visibility/identity disentanglement.
- Remaining mechanism gap: an auditable online rule that prevents evidence for
  one latent variable from silently mutating another and proves specific safety
  invariants under occlusion and rejected motion.
- CROVE consequence: reversible ownership is credible only as part of the full
  evidence-routing mechanism and with a direct naive-removal comparison.

### Static open-vocabulary mapping

- Representatives: OVI-MAP, ConceptGraphs.
- Already solved: strong incremental instance reconstruction and open-vocabulary
  semantics in mostly static settings.
- Remaining gap: preserving this cumulative capability exactly while a temporal
  current-state branch revokes stale authority.
- CROVE consequence: T1 is a non-interference theorem/guardrail, not the main
  paper contribution.

## Opportunity Map

| Cluster | Status | Open gap | CROVE direction | Evidence required | Risk |
| --- | --- | --- | --- | --- | --- |
| Current-state open-vocabulary mapping | covered central claim | No explicit cross-state evidence compatibility contract | Reframe task as prior art and contribute observability-conditioned inference | Direct SuperMap/DynaMem/WMG/DualMap mechanism comparison | High |
| Identity and geometry | crowded but open | Active continuation, dormant reactivation, birth, and current pose have asymmetric evidence requirements | Role-conditioned association plus deferred persistent admission | Same-class lookalike and moved-object stress tests | Medium-high |
| Visibility and removal | crowded but open | Single-frame deletion can be brittle; occlusion must be non-evidence | Multi-view visible-absence transition with an occlusion-safety invariant | DynaMem-style ray deletion ablation and false-removal rate | Medium |
| Static/current tension | mechanism gap | Fused geometry is useful history but a biased current-pose estimator | Dual center/readout with exact cumulative non-interference | Localization ablation, trajectory error, byte-exact T1 gate | Medium |
| Evidence attribution | mechanism gap | Existing systems combine signals without specifying which latent state each may update | Typed causal evidence ledger and admissibility matrix | Component interventions, invariant tests, deterministic replay | Medium, novelty needs explicit formulation |

## Recommended Core Direction

Use **CROVE: Causal Role-conditioned Object-Visibility Evidence** as the method
principle, subject to final naming review. The key representation contains four
separate state variables: persistent identity, current geometry epoch, current
existence/readout validity, and cumulative static geometry. Each evidence token
has causal provenance and an admissible target state. The current A5/A6
role-separated identity, visible-absence lifecycle, reversible ownership, and
dual-center design then become consequences of one rule rather than a list of
independent modules.

The publishable claim is not that CROVE is the first dynamic open-vocabulary
map. It is that state-compatible evidence routing prevents three measurable
failure classes of current systems: occlusion-induced removal, lookalike-induced
identity leakage, and fused-centroid-induced trajectory bias, while preserving
the cumulative static map exactly. SuperMap prevents claiming 3D-aware
association, visibility-conditioned deletion, Bayesian label fusion, or their
combination as novel; CROVE must instead show that a declared admissibility
relation prevents cross-state contamination and preserves a separate frozen
cumulative readout.

## Mechanism-Level Novelty Stress Test

The table records what each primary paper explicitly formulates or evaluates.
`Not reported` does not prove that a capability is absent. The CROVE row is a
proposed claim boundary, not a completed result.

| Method | Stream/update setting | Identity/change mechanism | Visibility/existence mechanism | Geometry/currentness mechanism | Remaining distinction for CROVE |
| --- | --- | --- | --- | --- | --- |
| SuperMap | Real-time RGB-D/LiDAR stream with a 4D instance scene graph | 3D-to-2D motion-compensated tracking, reactivation, geometric consistency, and Bayesian label fusion | Depth residual labels points observable, unobservable, or disappeared; dynamic points are penalized by log-odds | A single evolving global object map is pruned and updated online | No explicit state/evidence admissibility relation, deterministic provenance ledger, or byte-exact frozen cumulative readout is reported; direct mechanism overlap is otherwise high |
| Khronos | Online active window plus slower global reconciliation | Associates temporal fragments into persistent objects | Uses representative free-space rays to infer presence intervals | Optimizes fragment poses and reconstructs scene state over time | No explicit cross-state evidence-admissibility contract or exact frozen cumulative readout is reported |
| DynaMem | Online open-vocabulary voxel memory | Object identity is not its primary map state | Removes a voxel when it projects in front of the current measured depth within a bounded frustum | Currentness comes from adding and removing voxels | Direct naive-removal comparator; CROVE cannot claim online current-state updating itself |
| Where Did I Leave My Glasses? | Online semantic map maintenance with active revisit | Semantic/geometric association, missing-object library, and re-identification | Bayesian object stationarity drives translation/removal decisions | Object pose is updated after qualified reappearance | Direct probabilistic-lifecycle comparator; CROVE must distinguish typed state mutation rather than generic stationarity |
| DualMap | Online open-vocabulary mapping during navigation | Concrete-map association and object-status checks update queried objects | A stability check removes long-unobserved noisy objects; explicit occlusion-safe existence inference is not reported | Its global abstract map supports candidate selection and local concrete map supports goal reaching | Its two maps have planning/detail roles, not CROVE's cumulative/current readout roles; the word `dual` is not a novelty distinction |
| POCD | Object-level semi-static volumetric mapping | Probabilistic object stationarity and change | Free-space and object evidence update stationarity | Maintains object-level volumetric state | Prevents claiming generic existence belief or object-level change reasoning |
| OASIS-Map | Incremental multi-session mapping | Dense semantic correspondence and deferred object-change decisions | Treats partial views and uncertainty during change detection | Relocalizes moved objects across visits | Direct identity comparator; CROVE's defensible scope is per-frame causal inference without dense cross-session optimization |
| Consistent Instance Field | Offline learned 4D neural field | Conditional instance distribution represents persistent identity | Occupancy probability is separated from conditional identity | Deformable Gaussians jointly model space-time geometry | Prevents claiming visibility/identity separation; CROVE must instead establish online state-specific transition guarantees |
| CROVE, proposed | Strict-prefix posed RGB-D stream | Role-conditioned active continuation, dormant reactivation, and delayed birth | Only visible free space may reduce existence; occlusion and unknown depth are neutral | Current pose/epoch drives temporal readout while fused geometry drives cumulative output | One auditable admissibility rule governs which evidence may mutate identity, existence, ownership, epoch, and readout; novelty and benefit remain to be proven |

The mechanism claim survives this stress test only if the paper establishes all
of the following together:

1. Evidence is represented with observable support and causal provenance, not
   merely as an untyped confidence score.
2. A declared admissibility relation prevents evidence for one latent variable
   from mutating incompatible state.
3. The restrictions imply executable safety properties for occlusion,
   rejected motion, identity reactivation, and cumulative-map non-interference.
4. Matched-front-end ablations show that violating each restriction causes its
   predicted failure class, including a SuperMap-style untyped
   visibility/log-odds update and a DynaMem-style ray-deletion alternative.
5. Held-out current-state results improve over protocol-valid dynamic baselines.

Absent any one of these conditions, present the mechanism as a robust systems
design rather than a novel inference formulation.

## Benchmark And Dataset Candidates

| Name | Link | Purpose | Metrics | Required baselines | Main risk |
| --- | --- | --- | --- | --- | --- |
| TESSE-CD | [Khronos paper](https://www.roboticsproceedings.org/rss20/p081.html) | Main causal dynamic-map evaluation | Obj./Dyn./Chg. F1, current mIoU, ghost, BG F@5cm, recovery | Khronos, Panoptic Mapping, frozen open-vocabulary maps | Only two scenes; must avoid development leakage |
| SuperMap six-object change sequence | [RSS paper](https://www.roboticsproceedings.org/rss22/p052.pdf) | Direct mechanism sanity check if assets/code become available | appeared/disappeared object recall and change recall | SuperMap, DualMap | Public repository currently contains setup/docs but not the promised benchmark implementation; do not report a rerun without released assets |
| 3RScan/RIO | [CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Wald_RIO_3D_Object_Instance_Re-Localization_in_Changing_Indoor_Environments_ICCV_2019_paper.html) | Cross-visit identity and moved-object stress | moved/static association, ID switches, pose error | OASIS-Map, Living Scenes, ConceptGraphs | Multi-session rather than per-frame online protocol |
| Replica | [project](https://github.com/facebookresearch/Replica-Dataset) | Static capability guardrail | mIoU, mAcc, f-mIoU, AP, F@5cm | OVI-MAP, ConceptGraphs, OpenFusion | Cannot support dynamic claims |
| DynaBench | [DynaMem paper](https://arxiv.org/abs/2411.04999) | Query/update behavior in dynamic memory | localization accuracy and absent-query behavior | DynaMem query variants | Protocol/code compatibility must be checked |

## Citation And Positioning Cautions

- Do not use “first online dynamic open-vocabulary map,” “first current-state
  semantic memory,” or “first separation of visibility and identity.”
- Cite SuperMap in the Introduction and distinguish its single evolving 4D map,
  visibility-conditioned log-odds update, and Bayesian semantic fusion from
  CROVE's typed mutation contract and frozen cumulative branch.
- Cite DynaMem and Where Did I Leave My Glasses? in the Introduction, not only
  Related Work; they directly establish the problem.
- Treat OASIS-Map as concurrent arXiv work with strong mechanism overlap and no
  transferred cross-protocol result claims.
- Explain the difference between CROVE's cumulative/current readouts and
  DualMap's abstract/concrete maps; the word “dual” is not a distinction.
- Use OVI-MAP only for the frozen static foundation; it does not validate
  temporal maintenance.
