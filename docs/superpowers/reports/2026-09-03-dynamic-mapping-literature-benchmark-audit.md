# Dynamic Mapping Literature and Benchmark Audit

Date: 2026-09-03

Status: B0/B1 evidence frozen before any CROVE 3RScan or Panoptic Flat pilot
result is viewed or aggregated.

Purpose: distinguish a weak mapper from unequal semantic inputs, evaluator or
aggregation drift, and a mismatch between full historical 4D reconstruction
and CROVE's causal current-state claim.

Source policy: official proceedings, author project pages, arXiv records, and
exact official repository commits only. MDPI and untraceable secondary sources
were excluded. Numerical claims below are included only where the primary page
defines them; no cross-protocol paper number is reused as a local result.

## Audit Outcome Before Pilots

1. TESSE-CD is the only reviewed benchmark that combines continuous
   short-term motion, out-of-view long-term changes, full geometry, and an
   online spatio-temporal mapper. It remains necessary evidence.
2. The RSS 2024 release commit contains no `khronos_eval` files. The first
   public evaluator lineage is post-release, and the current public evaluator
   does not by itself prove equivalence to the paper's Table I implementation.
3. The current P5 manifest binds the local temporal importer but not the
   evaluator script/binary closure or dirty Khronos source state. Its exact
   artifact evaluator identity is therefore unproven.
4. 3RScan provides the strongest exact real-world cross-session identity and
   object-change attribution, but is temporally sparse and its standard
   pipelines are offline.
5. Panoptic Flat provides the most direct controlled current-surface test, but
   is synthetic, closed-set, and has a small baseline ecosystem.
6. ReScene4D/stmetrics makes 3RScan temporal instance evaluation much more
   useful through t-AP/t-REC, but its released checkpoint was still marked
   `coming soon` at the frozen source commit.
7. SuperMap directly overlaps the broad current open-vocabulary mapping claim.
   CROVE must claim a narrower, testable evidence-routing or non-interference
   mechanism rather than first current-state semantic mapping.

These findings support a split evidence package, not benchmark replacement:
TESSE official/full history, TESSE current diagonal, 3RScan long-term identity,
and Flat controlled current maintenance must remain separately labeled until
B2-B8 establish protocol validity and pilot reproducibility.

## B0 Repository and Evaluator Identity

| Identity | Frozen value | Consequence |
| --- | --- | --- |
| CROVE base | `668aefc49034ef97d090b811b1c3f291ecafbd66` | Historical A6/P5/P6 evidence remains immutable. |
| Khronos RSS release | `742227a88de8b2ac23ac54d719b321c3af88dc75` | Release tree has zero files under `khronos_eval/`. |
| First historical public evaluator | `366062a32ab69bf20ad2062708f3a145351cc287` | Post-release code generated the full triangular robot-time/query-time grid. |
| Current evaluator lineage introduction | `5bf72d97857e746c875c34a3ff44d7abf86bb86b` | Post-release public implementation; it writes diagonal-equivalent repeated rows in the inspected loop. |
| Latest official Khronos HEAD | `63faadde6ed92220e78fb2f6ca86dcc54bb5cf9e` | Frozen public source for later parity; not assumed to be the paper evaluator. |
| Current P5 artifact evaluator | `UNPROVEN_ARTIFACT_EVALUATOR_IDENTITY` | Importer source/binary are bound, but evaluator closure, repository commit, and local patch set are not. |
| Local 3RScan metadata | SHA-256 `674a00f50f76b198b9de44efd86c390fea3da37ba8f12cf8ccd00045e265fa64` | Metadata is available for deterministic protocol selection. |
| Local 3RScan validation list | SHA-256 `002229133d4dbc311a01994b8ea31ecac0b662511864493efed9b4febcb0a4e8` | Visit membership can be frozen independently of method output. |
| Panoptic Flat assets | Not found in approved local roots at B0 | B8 must attempt official access, then report `BLOCKED_DATASET_ACCESS` if unavailable. |

The source registry at
`configs/evaluation/external_benchmark_sources.json` is normative for repository
commits and licenses. In particular, same repository ownership is not evidence
that release, latest, and artifact evaluator implementations are identical.

## Search Method

Safe public queries covered: dynamic metric-semantic SLAM; evolving-scene RGB-D
mapping; multi-session object change detection; 3RScan temporal instance
segmentation; open-vocabulary spatio-temporal map; current-state scene memory;
dynamic indoor mapping datasets. Candidates were deduplicated by title and
verified against primary pages. The 12 retained works cover anchor datasets,
protocol-valid baselines, close mechanism threats, and new dataset maturity.

## Evidence Cards

### LIT-RIO-2019 — RIO / 3RScan

- **Paper:** [RIO: 3D Object Instance Re-Localization in Changing Indoor
  Environments](https://openaccess.thecvf.com/content_ICCV_2019/html/Wald_RIO_3D_Object_Instance_Re-Localization_in_Changing_Indoor_Environments_ICCV_2019_paper.html),
  ICCV 2019; method + benchmark.
- **Official code:** `SRC-3RSCAN`, commit
  `12f040d3dc849e394ed366acd7b54e63acb49205`.
- **Dataset:** 3RScan, 1,482 reconstructions of 478 naturally changing indoor
  environments across repeated visits.
- **Sensor/input:** calibrated RGB-D frames, camera intrinsics and 6DoF poses,
  textured meshes, semantic-instance meshes, and global scan alignments.
- **Temporal regime:** sparse multi-session long-term change.
- **Online:** no; the canonical task compares completed scans.
- **Open-vocabulary:** no; class and instance annotations are fixed.
- **GT semantics:** yes.
- **GT instances:** yes, dense mesh instance annotations.
- **Cross-time identity GT:** yes, fixed instance IDs across scans plus
  ambiguity groups.
- **Change-type GT:** yes for rigid, nonrigid, and removed instances; additions
  require comparison-derived handling.
- **Geometry GT:** yes, per-visit reconstructed meshes and scan transforms.
- **Metrics:** RIO object re-localization accuracy under pose thresholds;
  keypoint precision, recall, F1, and top-k retrieval.
- **Error provenance:** exact per object, correspondence, transform, and visit.
- **Baseline reproducibility:** high for metadata/readers; data access is gated
  by the dataset terms but local metadata and validation assets are present.
- **Match to CROVE claim:** strong for long-term identity and moved-object
  attribution, weak for continuous online dynamics.
- **Quality:** insight 5/5, completeness 5/5, numeric evidence 5/5, overall A.

### LIT-PANOPTIC-MULTITSDFS-2022 — Panoptic Multi-TSDFs

- **Paper:** [Panoptic Multi-TSDFs: a Flexible Representation for Online
  Multi-resolution Volumetric Mapping and Long-term Dynamic Scene
  Consistency](https://arxiv.org/abs/2109.10165), ICRA 2022; pure method.
- **Official code:** `SRC-PANOPTIC-MAPPING`, commit
  `3926396d92f6e3255748ced61f5519c9b102570f`.
- **Dataset:** Panoptic Flat and selected RIO/3RScan sequences.
- **Sensor/input:** RGB-D, camera pose, and GT or predicted panoptic masks.
- **Temporal regime:** continuous online mapping across two long-term visits.
- **Online:** causal mapping; run 1 is loaded before run 2.
- **Open-vocabulary:** no, but the raw RGB-D stream permits an adapted frontend.
- **GT semantics:** yes in the Flat-GT condition.
- **GT instances:** yes in Flat panoptic labels.
- **Cross-time identity GT:** partial; changed-object labels exist, while the
  published metric is surface-centric rather than an identity metric.
- **Change-type GT:** yes: new, removed, and modified objects.
- **Geometry GT:** yes, complete simulated geometry at each time.
- **Metrics:** Mean Absolute Distance (MAD), map size, coverage defined by GT
  points within 5 cm, and map-query time.
- **Error provenance:** recoverable at surface/submap level, but the published
  curves are aggregate.
- **Baseline reproducibility:** medium; code is public, while its ROS 1 stack
  is old and the original dataset page moved.
- **Match to CROVE claim:** strong for controlled current geometry and stale
  surface removal; partial for semantics and identity.
- **Quality:** insight 4/5, completeness 5/5, numeric evidence 4/5, overall A.

### LIT-OBJECTS-CAN-MOVE-2022 — Objects Can Move

- **Paper:** [Objects Can Move: 3D Change Detection by Geometric
  Transformation Consistency](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6194_ECCV_2022_paper.php),
  ECCV 2022; pure method.
- **Official code:** `SRC-OBJECTS-CAN-MOVE`, commit
  `41cf524652d1c08c9d1e595a01c193ebb71f36a7`; no top-level license.
- **Dataset:** 3RScan validation split converted to object-change GT.
- **Sensor/input:** aligned reconstructed meshes, rendered depth differences,
  supervoxels, geometric features, and scan transforms.
- **Temporal regime:** sparse pairwise multi-session.
- **Online:** no; graph optimization follows complete-scan preprocessing.
- **Open-vocabulary:** no semantic category assumption, but not a native
  language/open-vocabulary system.
- **GT semantics:** used by the dataset conversion, not required by the core
  geometric change proposal.
- **GT instances:** yes for evaluation.
- **Cross-time identity GT:** yes through 3RScan object correspondences.
- **Change-type GT:** changed versus unchanged, with rigid motion emphasized;
  added objects are optional in the distributed conversion script.
- **Geometry GT:** yes, aligned meshes and object transforms.
- **Metrics:** mean IoU, mean recall, mean accuracy, and mean completeness.
- **Error provenance:** exact changed points/objects are recoverable.
- **Baseline reproducibility:** medium-low; public scripts are hard-coded,
  RANSAC remains stochastic, several external build steps are manual, and the
  repository license is undeclared.
- **Match to CROVE claim:** partial; useful for moved-object diagnosis but not
  causal online maintenance.
- **Quality:** insight 4/5, completeness 4/5, numeric evidence 4/5, overall B.

### LIT-KHRONOS-2024 — Khronos

- **Paper:** [Khronos: A Unified Approach for Spatio-Temporal Metric-Semantic
  SLAM in Dynamic Environments](https://www.roboticsproceedings.org/rss20/p081.html),
  RSS 2024; method + benchmark.
- **Official code:** `SRC-KHRONOS`, latest commit
  `63faadde6ed92220e78fb2f6ca86dcc54bb5cf9e`; RSS release and evaluator
  identities are separately frozen in the registry.
- **Dataset:** TESSE-CD Apartment and Office plus real-robot demonstrations.
- **Sensor/input:** RGB-D surfaces, object/semantic segments, and GT or Kimera
  poses; Table I uses simulator GT semantics.
- **Temporal regime:** both short-term motion and out-of-view long-term change.
- **Online:** causal estimation, but the official 4D objective evaluates every
  belief time `t <= T` at each robot time `T`.
- **Open-vocabulary:** adapted SAM+CLIP OpenSet condition is reported separately
  from Table I's GT-semantic system-component condition.
- **GT semantics:** yes for Table I.
- **GT instances:** yes in TESSE evaluator annotations.
- **Cross-time identity GT:** yes through the time-indexed object/DSG GT.
- **Change-type GT:** yes for newly appeared, disappeared, persistent, and
  dynamic objects.
- **Geometry GT:** yes for background and object surfaces.
- **Metrics:** Background, Objects, Dynamics, and Changes precision, recall,
  and F1; the 4D score integrates across robot time and belief time.
- **Error provenance:** aggregate in released CSVs; exact TP/FP/FN node
  attribution is not present in current official outputs.
- **Baseline reproducibility:** medium-low for the paper protocol: code/data are
  public, but the RSS release lacks `khronos_eval` and current evaluator code is
  post-release.
- **Match to CROVE claim:** partial; contains the current diagonal but its
  headline task is broader historical 4D belief reconstruction.
- **Quality:** insight 5/5, completeness 5/5, numeric evidence 5/5, overall A.

### LIT-DOVSG-2025 — DovSG

- **Paper:** [Dynamic Open-Vocabulary 3D Scene Graphs for Long-term
  Language-Guided Mobile Manipulation](https://arxiv.org/abs/2410.11989), IEEE
  RA-L 2025; system/tool.
- **Official code:** `SRC-DOVSG`, commit
  `b355987a1ca586f7756f025820fddc24166d75af`; no top-level license.
- **Dataset:** four real rooms with manually introduced minor adjustment,
  appearance, and positional-shift trials.
- **Sensor/input:** RealSense D455 RGB-D, estimated poses, VLM features, and
  interactive robot observations.
- **Temporal regime:** sequential tasks with long-term human-induced changes.
- **Online:** local graph updates are causal during task execution, after
  substantial room-scan preprocessing.
- **Open-vocabulary:** native.
- **GT semantics:** human-built GT scene graphs for evaluation.
- **GT instances:** task-relevant objects only, not dense map instances.
- **Cross-time identity GT:** task-level correspondence, not a released dense
  benchmark contract.
- **Change-type GT:** minor adjustment, appearance, and positional shift.
- **Geometry GT:** no complete current-surface GT.
- **Metrics:** Scene Change Detection Accuracy (SCDA), Scene Graph Accuracy
  (SGA), Task Success Rate, update time, and memory.
- **Error provenance:** recoverable per trial/task, not per map association.
- **Baseline reproducibility:** medium-low; code and example recordings are
  public but the full robot stack is complex and the repository license is
  undeclared.
- **Match to CROVE claim:** partial for dynamic open-vocabulary local update;
  weak for dense current-map geometry.
- **Quality:** insight 3/5, completeness 4/5, numeric evidence 3/5, overall B.

### LIT-GAME-2026 — Gaussian Mapping for Evolving Scenes

- **Paper:** [Gaussian Mapping for Evolving
  Scenes](https://openaccess.thecvf.com/content/CVPR2026/html/Yugay_Gaussian_Mapping_for_Evolving_Scenes_CVPR_2026_paper.html),
  CVPR 2026; pure method.
- **Official code:** `SRC-GAME`, commit
  `1c971d65d29952789fa4a2ebb342580d41acbbff`.
- **Dataset:** Panoptic Flat and selected Aria Digital Twin sequences.
- **Sensor/input:** posed RGB-D with image masks; online 3D Gaussian mapping.
- **Temporal regime:** evolving scenes with changes both inside and outside FoV.
- **Online:** causal single-camera RGB-D mapping.
- **Open-vocabulary:** not a native evaluation target; an adapted semantic
  frontend is possible.
- **GT semantics:** available in Flat inputs, not the primary target.
- **GT instances:** masks are inputs; identity is not the headline metric.
- **Cross-time identity GT:** partial through Flat/Aria annotations.
- **Change-type GT:** available for Flat changes.
- **Geometry GT:** depth and held-out views support current reconstruction.
- **Metrics:** PSNR, SSIM, LPIPS, and depth L1 error.
- **Error provenance:** recoverable per frame/keyframe; reported summaries are
  aggregate rendering metrics.
- **Baseline reproducibility:** medium; code/configs and refreshed Flat/Aria
  access are public, but the rasterizer is nondeterministic and bundled
  components have separate non-commercial terms.
- **Match to CROVE claim:** strong adjacent evidence for stale-observation
  removal and current geometry, partial for object identity.
- **Quality:** insight 4/5, completeness 5/5, numeric evidence 5/5, overall A.

### LIT-RESCENE4D-2026 — ReScene4D

- **Paper:** [ReScene4D: Temporally Consistent Semantic Instance Segmentation
  of Evolving Indoor 3D Scenes](https://arxiv.org/abs/2601.11508), CVPR 2026;
  method + benchmark.
- **Official code:** `SRC-RESCENE4D` and `SRC-STMETRICS`, commits
  `fb2fe42eb8f1e926567c48eea9acb874e608ee10` and
  `640e34c2dd15c8e1a5061f4e66aa4fb6a5da9a5f`.
- **Dataset:** 3RScan reformulated as temporally sparse 4D semantic instance
  segmentation.
- **Sensor/input:** registered multi-timestep point clouds and semantic-instance
  targets.
- **Temporal regime:** sparse multi-session.
- **Online:** no; joint multi-timestep inference and training are offline.
- **Open-vocabulary:** no, fixed RIO class specifications.
- **GT semantics:** yes.
- **GT instances:** yes.
- **Cross-time identity GT:** yes; temporal identity consistency is the task.
- **Change-type GT:** yes through auxiliary labels in `stmetrics` dataset specs.
- **Geometry GT:** per-visit meshes/point clouds, not continuous surface truth.
- **Metrics:** mAP/AP, temporal AP (t-AP/t-mAP), per-timestep AP, temporal recall
  (t-REC), change-label recall, and temporal feature similarity (t-SIM).
- **Error provenance:** exact per instance, class, timestep, and auxiliary
  change label.
- **Baseline reproducibility:** medium; source and metrics are released, but the
  frozen ReScene4D README still marks checkpoints as coming soon.
- **Match to CROVE claim:** strong for identity attribution, partial for causal
  mapping and weak for short-term motion.
- **Quality:** insight 5/5, completeness 4/5, numeric evidence 5/5, overall A.

### LIT-OASIS-MAP-2026 — OASIS-Map

- **Paper:** [OASIS-Map: Object-Level Change Detection in Multi-Session
  Mapping using Semantic Correspondence
  Matching](https://arxiv.org/abs/2607.14899), 2026 arXiv/under review; pure
  method.
- **Official code:** official project page says `Code Soon` as of the retrieval
  date; no exact public code commit is available.
- **Dataset:** 3RScan, an RGB-LiDAR car park, and an outdoor market.
- **Sensor/input:** RGB-D or RGB-LiDAR multi-session maps and dense semantic
  image correspondences.
- **Temporal regime:** sparse multi-session long-term change.
- **Online:** incremental association within a revisit; dense cross-session
  backend processing is not a per-frame online benchmark.
- **Open-vocabulary:** adapted semantic correspondence, not a native
  open-vocabulary label evaluation.
- **GT semantics:** yes where dataset annotations support it.
- **GT instances:** yes for 3RScan object association.
- **Cross-time identity GT:** yes in 3RScan.
- **Change-type GT:** appear, disappear, static, moved, and unknown.
- **Geometry GT:** aligned scans and real-session maps; completeness varies by
  scenario.
- **Metrics:** change-detection F1 and moved-object association F1.
- **Error provenance:** exact per associated object/change when annotations are
  available.
- **Baseline reproducibility:** low at freeze time because code is announced
  but unavailable; paper and project descriptions are inspectable.
- **Match to CROVE claim:** strong for long-term identity/change, weak for
  continuous current-state mapping.
- **Quality:** insight 5/5, completeness 3/5, numeric evidence 3/5, overall Risk.

### LIT-DAAAM-2026 — Describe Anything Anywhere At Any Moment

- **Paper:** [Describe Anything Anywhere At Any
  Moment](https://openaccess.thecvf.com/content/CVPR2026/html/Gorlo_Describe_Anything_Anywhere_At_Any_Moment_CVPR_2026_paper.html),
  CVPR 2026; system/tool.
- **Official code:** `SRC-DAAAM`, commit
  `ebd6e3b89763849eb7bce7f75e8d08550ea7344a`.
- **Dataset:** NaVQA, extended OC-NaVQA, and SG3D.
- **Sensor/input:** RGB, segmentation/tracking, captions, depth/pose grounding,
  and Hydra 4D scene graphs.
- **Temporal regime:** continuous long-horizon spatio-temporal memory.
- **Online:** causal real-time scene-graph construction with batched semantic
  workers.
- **Open-vocabulary:** native localized captioning and language grounding.
- **GT semantics:** benchmark question/scene annotations, not dense mapping GT.
- **GT instances:** task annotations only.
- **Cross-time identity GT:** temporal questions exist, but no dense
  object-correspondence benchmark contract is exposed.
- **Change-type GT:** no standardized moved/added/removed map labels.
- **Geometry GT:** supports grounding position error, not current-surface GT.
- **Metrics:** OC-NaVQA question accuracy, position error, temporal error, and
  SG3D task-grounding accuracy.
- **Error provenance:** recoverable per question/task; aggregate for map state.
- **Baseline reproducibility:** medium; code is available with a complex ROS 2,
  model, dataset, and scene-graph dependency chain.
- **Match to CROVE claim:** partial for real-time open-vocabulary memory; weak
  for explicit current-surface maintenance.
- **Quality:** insight 4/5, completeness 4/5, numeric evidence 4/5, overall A.

### LIT-CIF-2026 — Consistent Instance Field

- **Paper:** [Consistent Instance Field for Dynamic Scene
  Understanding](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_Consistent_Instance_Field_for_Dynamic_Scene_Understanding_CVPR_2026_paper.html),
  CVPR 2026; pure method.
- **Official code:** no official repository is linked from the frozen primary
  record; paper implementation details are available.
- **Dataset:** HyperNeRF and Neu3D dynamic videos.
- **Sensor/input:** monocular or synchronized multi-view RGB and generated
  instance masks; deformable 3D Gaussians.
- **Temporal regime:** dense short-sequence dynamics.
- **Online:** no; per-scene optimization is offline.
- **Open-vocabulary:** adapted Grounded-DINO text queries are evaluated.
- **GT semantics:** pseudo-GT instance masks generated with DEVA.
- **GT instances:** pseudo-GT, restricted to consistently visible instances in
  the multi-view setting.
- **Cross-time identity GT:** generated tracking masks, not independent physical
  identity annotations.
- **Change-type GT:** no.
- **Geometry GT:** image/depth view supervision, not a complete current-surface
  map benchmark.
- **Metrics:** mean pixel accuracy within instances (mAcc-pix), mean instance
  accuracy (mAcc-inst), mIoU, open-vocabulary query mAcc/mIoU, and PSNR.
- **Error provenance:** per frame/instance under pseudo-GT; no
  appeared/disappeared attribution.
- **Baseline reproducibility:** medium-low; the protocol is detailed but no
  official code commit is source-bound here and some baselines are reimplemented.
- **Match to CROVE claim:** strong conceptual threat to visibility/identity
  disentanglement, weak protocol match to online RGB-D mapping.
- **Quality:** insight 5/5, completeness 5/5, numeric evidence 5/5, overall Risk.

### LIT-SUPERMAP-2026 — SuperMap

- **Paper:** [SuperMap: A Spatio-Temporal SLAM System for Visual-Language
  Navigation](https://www.roboticsproceedings.org/rss22/p052.html), RSS 2026;
  system/tool.
- **Official code:** `SRC-SUPERMAP`, commit
  `ec95b1d50a458645b2669836e2c431f1e957bbc6`; frozen tree is documentation-only
  and has no top-level license.
- **Dataset:** static semantic mapping datasets plus a real two-traversal
  appeared/disappeared-object sequence and real-robot navigation.
- **Sensor/input:** continuous RGB-D or RGB-LiDAR, IMU/pose estimation, 2D
  instance masks, and language labels.
- **Temporal regime:** continuous short-term tracking and long-term revisits.
- **Online:** causal, onboard mapping.
- **Open-vocabulary:** native asynchronous open-vocabulary perception.
- **GT semantics:** human annotations for evaluation, not runtime oracle input.
- **GT instances:** yes for evaluated object maps.
- **Cross-time identity GT:** annotated object identities in the reported
  sequence, but no public benchmark schema was found.
- **Change-type GT:** appeared and disappeared objects.
- **Geometry GT:** no complete current-surface ground truth is documented.
- **Metrics:** class-level mAP@50/mAP@25, instance-level mAP@50/mAP@25,
  appeared/disappeared detection and change recall, and object-map precision,
  recall, and F1.
- **Error provenance:** per object in the small change study; benchmark artifact
  availability is incomplete.
- **Baseline reproducibility:** low at the frozen commit because implementation
  and benchmark assets are not yet released despite the paper's release claim.
- **Match to CROVE claim:** very strong; directly covers current
  open-vocabulary object-map maintenance and stale-content pruning.
- **Quality:** insight 4/5, completeness 4/5, numeric evidence 3/5, overall Risk.

### LIT-DYNAMICTHOR-2026 — DynamicTHOR

- **Paper:** [DynamicTHOR: A Scalable Dataset of Human-Centric Dynamic Scenes
  for Embodied AI](https://www.nature.com/articles/s41597-026-07201-7),
  Scientific Data 2026; pure benchmark.
- **Official code:** `SRC-DYNAMICTHOR`, commit
  `bed7e573e579edbc1552c9389dd9fdaa7f4d3c2b`.
- **Dataset:** 100 ProcTHOR houses with 50 synthetic residents, schedules,
  timestamped object placements, and navigation tasks.
- **Sensor/input:** AI2-THOR simulator state and rendered embodied observations;
  object poses are available at every timestamp.
- **Temporal regime:** long-horizon event-driven object rearrangement.
- **Online:** simulator can expose states causally, but the released primary
  task is dynamic semantic-goal navigation rather than mapping.
- **Open-vocabulary:** no native map evaluation; language activities can support
  an adaptation.
- **GT semantics:** yes from ProcTHOR object metadata.
- **GT instances:** yes in simulator state.
- **Cross-time identity GT:** yes through persistent simulated object records.
- **Change-type GT:** relocation events are explicit; a standard
  moved/added/removed mapping taxonomy is not provided.
- **Geometry GT:** simulator geometry and object poses exist, but no released
  current-surface mapping evaluator is defined.
- **Metrics:** technical-validation Believability, Comprehensiveness, and
  Diversity scores; navigation code emits task success measures.
- **Error provenance:** exact simulator state is available, while a mapping
  attribution protocol would be custom.
- **Baseline reproducibility:** medium for dataset generation/navigation; low
  for CROVE mapping because a new adapter and metric contract are required.
- **Match to CROVE claim:** partial as a future causal dynamics source, weak as
  a mature current-map benchmark.
- **Quality:** benchmark-quality note: broad controllable object-state coverage
  and public code/data, but synthetic behavior and no accepted mapping metric;
  overall B.

## Closest-Work Clusters

| Cluster | Already covered | Remaining diagnostic gap | CROVE consequence |
| --- | --- | --- | --- |
| Full spatio-temporal mapping | Khronos models short- and long-term dynamics and revises historical beliefs. | Exact release/evaluator identity and current-diagonal agreement. | Keep TESSE, but split official 4D and current-state claims. |
| Sparse identity/change | 3RScan, ReScene4D, Objects Can Move, and OASIS-Map expose cross-visit identity and change. | Strict-prefix mapper execution from RGB-D and exact CROVE ID failures. | Use 3RScan for diagnosis, not continuous-dynamics replacement. |
| Controlled current geometry | Panoptic Multi-TSDFs and GaME update stale geometry on Flat. | Matched A6/P5/P6-C current-object and stale-surface evaluation. | Use Flat as a mechanism pilot if assets reproduce. |
| Open-vocabulary temporal memory | DovSG, DAAAM, and SuperMap update language-grounded memories. | Dense current-map quality under matched inputs and exact non-interference. | Broad novelty is covered; narrow the paper claim. |
| Visibility and identity | Consistent Instance Field explicitly separates visibility from identity. | Online causal state-transition evidence rather than offline field fitting. | Do not claim the separation alone as novel. |

## Benchmark Candidate Matrix

| Benchmark | Task alignment | What it can prove | What it cannot prove |
| --- | --- | --- | --- |
| TESSE official | Both short/long-term; full historical 4D belief | Unified online spatio-temporal performance after protocol parity | A pure current-state mechanism in isolation |
| TESSE current diagonal | Same sequence/GT, `belief_time == robot_time` | Whether current-state ranking diverges from full history | Equivalence to the RSS Table I score |
| 3RScan + stmetrics | Real sparse cross-session instances and changes | Exact identity, moved/removed/reactivation error | Short-term moving-object behavior |
| Panoptic Flat | Controlled two-run simulated geometry/change | Stale geometry, current surface, recovery and change sensitivity | Broad real-world or open-vocabulary generalization |
| DynamicTHOR | Scalable event-driven simulated changes | Future data generation and embodied downstream stress | A mature dynamic mapping result today |

## Quality and Positioning Risks

- Highest protocol risk: current TESSE artifacts do not prove which evaluator
  source produced them.
- Highest fairness risk: Khronos Table I uses simulator GT semantics while
  CROVE headline rows use an open-vocabulary frontend.
- Highest task risk: full 4D integration rewards retrospective reconstruction
  outside a bounded current-state claim.
- Highest novelty risk: SuperMap already formulates online open-vocabulary
  object-map update with visibility-conditioned stale-content pruning.
- Highest replacement risk: 3RScan and Flat are complementary, not direct
  substitutes; promotion requires adapters, baselines, and causal pilots.

## Source Checklist

- Public queries used: complete.
- Primary-source and source-quality policy: applied.
- Deduplication by title: complete.
- Twelve retained works classified and scored: complete.
- Dataset/baseline/metric fields: complete.
- Numerical claims copied into local results: none.
- Unknowns retained explicitly: Khronos paper evaluator source, current P5
  evaluator identity, OASIS-Map code, ReScene4D checkpoint, SuperMap code/assets,
  and local Flat access.

Handoff: B2 must first close the Khronos paper-protocol identity/reproduction
gate. B3 then compares local aggregation against a frozen post-release upstream
implementation. Pilot results remain unread until the suitability scorecard and
this report are committed.
