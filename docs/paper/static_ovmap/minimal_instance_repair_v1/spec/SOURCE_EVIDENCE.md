# Source evidence and decision ledger

Read date: 2026-10-07. Source pin: `247e1e9782d43e882589bd9ab0015d513c200e49`.
This review inspected code/report content. It did not mount the server scan/weight files or rerun inference.

| Key | Pinned file | Functions/range | Observed fact and resulting design |
|---|---|---|---|
| S01 | [docs/paper/static_ovmap/EVIDENCE_EXPLORATION_CLAIMS.md](https://github.com/Orangekostar/oviovo/blob/247e1e9782d43e882589bd9ab0015d513c200e49/docs/paper/static_ovmap/EVIDENCE_EXPLORATION_CLAIMS.md) | entire report | Residual-mask diagnosis; 131/132 and 37/42; new semantic mechanisms did not meet the gate. |
| S02 | [src/static_ovmap/evidence_exploration/analysis.py](https://github.com/Orangekostar/oviovo/blob/247e1e9782d43e882589bd9ab0015d513c200e49/src/static_ovmap/evidence_exploration/analysis.py) | baseline_scene | Best IoU is computed on actually exported candidate masks in target space; it is not a full-raw-owner ceiling. |
| S03 | [src/static_ovmap/recovery_wave2/recovery_registry.py](https://github.com/Orangekostar/oviovo/blob/247e1e9782d43e882589bd9ab0015d513c200e49/src/static_ovmap/recovery_wave2/recovery_registry.py) | build_registry | Active painted owners are excluded; residual K; source-row cutoff 100 and cap128. |
| S04 | [src/static_ovmap/module_validation/native_capture.py](https://github.com/Orangekostar/oviovo/blob/247e1e9782d43e882589bd9ab0015d513c200e49/src/static_ovmap/module_validation/native_capture.py) | NativeCaptureSession.before_insertion; finalize; FrameObservation.to_manifest | Actual captured aligned pre-insertion panoptic PNG, RGB, metric depth and final mesh; do not confuse NativeScenePack with NativeCaptureSession artifacts. |
| S05 | [src/static_ovmap/module_validation/boundary_jobs.py](https://github.com/Orangekostar/oviovo/blob/247e1e9782d43e882589bd9ab0015d513c200e49/src/static_ovmap/module_validation/boundary_jobs.py) | _frame_arrays; geometry_smoke | Saved 2D panoptic and depth are consumable; old geometry smoke is not a new full experiment. |
| S06 | [src/static_ovmap/module_validation/entity_hypotheses.py](https://github.com/Orangekostar/oviovo/blob/247e1e9782d43e882589bd9ab0015d513c200e49/src/static_ovmap/module_validation/entity_hypotheses.py) | build_surface_graph; build_frame_leaf_evidence; project_frame_leaf_evidence; build_conflict_groups | Existing graph/evidence primitives; old point projector uses .05m and different leaf/frame logic, so do not execute unchanged as this method. |
| S07 | [src/static_ovmap/runtime_parity/views.py](https://github.com/Orangekostar/oviovo/blob/247e1e9782d43e882589bd9ab0015d513c200e49/src/static_ovmap/runtime_parity/views.py) | build_views; CallLoader | R2 producer/scientific identities, completed frame checks, same-call RGB reuse; retain old G1 evidence independently. |
| S08 | [src/static_ovmap/runtime_parity/kernels.py](https://github.com/Orangekostar/oviovo/blob/247e1e9782d43e882589bd9ab0015d513c200e49/src/static_ovmap/runtime_parity/kernels.py) | ExactProjector; retain_top3 | Verified ROI infrastructure; new arbitrary-source-row observer needs triangle-hit mapping, not raw owner raster substitution. |
| S09 | [src/static_ovmap/cvpr_compact/outputs.py](https://github.com/Orangekostar/oviovo/blob/247e1e9782d43e882589bd9ab0015d513c200e49/src/static_ovmap/cvpr_compact/outputs.py) | construct_output; build_method_outputs | Old append-only output cannot merge or attach raw candidates; do not remove its safeguards to fake new support. |
| S10 | [src/static_ovmap/module_validation/evaluation.py](https://github.com/Orangekostar/oviovo/blob/247e1e9782d43e882589bd9ab0015d513c200e49/src/static_ovmap/module_validation/evaluation.py) | GeometryIdentity; PredictionPayload; validate_prediction_invariants | Geometry identity does not include owners; COMBO accepts new exclusive owner arrays and one class per owner; G branch imposes a different owned-support invariant. |
| S11 | [src/static_ovmap/m2_reviewer_study/evaluation.py](https://github.com/Orangekostar/oviovo/blob/247e1e9782d43e882589bd9ab0015d513c200e49/src/static_ovmap/m2_reviewer_study/evaluation.py) | official_view; SceneEvaluator; pool | Evaluator caches masks on construction under owner filenames; partition-specific roots and actual owner registry are mandatory. Official ranking is class-area-based and whole cohort is pooled. |
| S12 | [src/static_ovmap/composition_study/evaluation.py](https://github.com/Orangekostar/oviovo/blob/247e1e9782d43e882589bd9ab0015d513c200e49/src/static_ovmap/composition_study/evaluation.py) | load_targets | Original target projection and annotation identities depend on the source manifest; preserve target mapping while changing output partition. |
| S13 | [src/static_ovmap/backbone_wave1/readouts.py](https://github.com/Orangekostar/oviovo/blob/247e1e9782d43e882589bd9ab0015d513c200e49/src/static_ovmap/backbone_wave1/readouts.py) | fuse_readout | Returns actual D2 probabilities/audit; suitable for predeclared incumbent low-margin selection, not a GT selection. |
| S14 | [src/static_ovmap/evidence_exploration/binding.py](https://github.com/Orangekostar/oviovo/blob/247e1e9782d43e882589bd9ab0015d513c200e49/src/static_ovmap/evidence_exploration/binding.py) | bind; load_binding | Actual parent runtime/v2 source chain and baseline prediction manifests; old authoritative spec is not editable for a new task. |
| S15 | [src/static_ovmap/evidence_exploration/acquisition.py](https://github.com/Orangekostar/oviovo/blob/247e1e9782d43e882589bd9ab0015d513c200e49/src/static_ovmap/evidence_exploration/acquisition.py) | load_anyup; project_raw; validate_neutral | Paper checkpoint, frozen FP32, raw dense pooling then original visual head; full real neutral test exists, do not repeat it on all scenes. |
| S16 | [src/static_ovmap/evidence_exploration/timing.py](https://github.com/Orangekostar/oviovo/blob/247e1e9782d43e882589bd9ab0015d513c200e49/src/static_ovmap/evidence_exploration/timing.py) | stream_selected; cold_call | Ordinary AnyUp single-variant execution exists; latest boundary ends at constructed payload, not legacy full file export. |
| S17 | [src/static_ovmap/evidence_exploration/diagnostic_details.py](https://github.com/Orangekostar/oviovo/blob/247e1e9782d43e882589bd9ab0015d513c200e49/src/static_ovmap/evidence_exploration/diagnostic_details.py) | compare_entries; paired_analysis | Unique GT changes, tied score entries and FP multiplicity are different; retain their distinction. |
| S18 | [src/static_ovmap/cvpr_compact/area_fallback.py](https://github.com/Orangekostar/oviovo/blob/247e1e9782d43e882589bd9ab0015d513c200e49/src/static_ovmap/cvpr_compact/area_fallback.py) | pool_region; region_vector | Area fallback already exists. Normal path unchanged; do not call it the novel component of this study. |

## External primary sources and appropriate attribution

- **Open3D 0.19.0 RaycastingScene**: https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html
  `cast_rays` returns t_hit, primitive_ids and primitive_uvs; ray-direction norm controls t units.
  Documented interpolation uses (1-u-v,u,v) with triangle vertices (0,1,2). This supports the
  new observer interface, not a claim that maximal-barycentric categorical assignment is an
  established accuracy improvement. This assignment is our explicit finite-grid choice.
- **MaskClustering (CVPR 2024)**: https://arxiv.org/abs/2401.07745
  View-consensus mask graph clustering is prior art. Our proposal must not claim to invent
  multi-view mask consensus; it tests local edits of an already formed semantic map.
- **MV3DIS (2026 paper)**: https://arxiv.org/abs/2604.08916
  Multi-view mask matching with 3D guidance and depth-informed refinement is prior art.
  The new experiments are not its complete reproduction and cannot reject that paper from
  a negative result on our small repair operator. Venue detail should be checked against
  the authors' record when preparing the manuscript bibliography.
- **AnyUp code pin**: https://github.com/wimmerth/anyup/tree/351807a9c4287368732cc247f26c7c81c9139af4
  `model.py`, `hubconf.py`, `layers/attention/chunked_attention.py` were read in the preceding
  evidence-exploration source audit and are bound by that existing task's asset manifest.
  Use the original paper checkpoint, not multi_backbone; strict verification is repeated
  only for the exact files consumed if their identity has changed.
  Paper checkpoint: https://github.com/wimmerth/anyup/releases/download/checkpoint/anyup_paper.pth
  SHA256: `9d035c0f27114a6f32bdd3d8ed93b6cd39dad4b8f8bf94e69fddbfbf022901b2`.
  Size: 3,540,612 bytes. Original license attribution retained by the existing adapter.

No new SenseFuse, Uni3D, VLA, active robot, LoRA or paper-specific mask generator is installed
in this task. They are outside the selected research scope.

## Fact / hypothesis separation

**Confirmed by read code:** available inherited data schemas; old residual restriction;
fixed-surface payload capability; per-partition evaluation requirements; existing neural
operators and timed-return boundary.

**Not independently confirmed on the server in this response:** actual availability of
all selected panoptic PNGs and dense caches; exact new observer masks; candidate counts
and vote quality under the new sampling; new latency; target improvements.

**Proposed, not previously validated:** .20m/15-degree grouping, 32 frames, .08m neighbor
range, evidence thresholds, maximum edit sizes, boundary tests, and no-decrease target.
All are frozen exploratory choices in PROTOCOL_SPEC.json. Do not write them as published
optimal values or claim the safeguards prove AP cannot fall.

**Source-dependent open binding:** actual physical root, GPU selection, FC interpreter,
model/text file paths and current cache locations come from immutable parent manifests,
not guessed paths or a public repository default. Missing assets are reported precisely.
