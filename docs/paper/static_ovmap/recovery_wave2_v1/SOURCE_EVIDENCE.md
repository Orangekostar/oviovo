# Source evidence and implementation map

Planning source anchor: `Orangekostar/oviovo@c21297413954ecb7d706d1050a8e07822938ea6f`.
The branch ref was read again for this task and returned the same full SHA.
Source text was inspected through the connected GitHub reader. Some exact relevant ranges
were already present in this conversation; new reads include cache persistence, request records,
native patch integration, mapping and saved SAM output contracts. No server arrays/model were run here.

| Key | Inspected file/symbol | Supported planning fact |
|---|---|---|
| E01 | [src/static_ovmap/module_validation/scannet_study.py](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/src/static_ovmap/module_validation/scannet_study.py) — `native_readout; native_surface_readout; build_native_prediction; native_ranks` | Native needs >=2 observations; only AVAILABLE colors paint eligible output; raw and painted owners differ. |
| E02 | [src/static_ovmap/backbone_wave1/readouts.py](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/src/static_ovmap/backbone_wave1/readouts.py) — `fuse_readout; build_sources; evaluate_map` | FC/source loops depend on the native painted registry; all new maps use fresh anchors. |
| E03 | [src/static_ovmap/module_validation/semantic_study.py](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/src/static_ovmap/module_validation/semantic_study.py) — `prepare_semantic_manifest; reconcile_semantic_request` | Original <=128 targets / <=3 views; all segments must resolve to one final owner. Use raw registry for new U branch. |
| E04 | [src/static_ovmap/backbone_wave1/semantic_readout.py](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/src/static_ovmap/backbone_wave1/semantic_readout.py) — `reconstruct_native; run_native_query` | Native request receipts retain physical feature keys; query_scores contain final-reconciled owners before painted-registry filtering. |
| E05 | [src/static_ovmap/backbone_wave1/features.py](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/src/static_ovmap/backbone_wave1/features.py) — `TensorEncoderCache` | Stores raw encoder vectors for exact tensors; caller normalization must be preserved. |
| E06 | [src/static_ovmap/backbone_wave1/region_readout.py](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/src/static_ovmap/backbone_wave1/region_readout.py) — `run_region` | NOW saves whole-frame dense features, pooled regions, text and provenance; fresh mask can reuse exact dense encoding. |
| E07 | [src/static_ovmap/a7_evidence_upgrade/region_adapter.py](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/src/static_ovmap/a7_evidence_upgrade/region_adapter.py) — `image_tensor; signed_mask; region_vector; load_model` | Frozen ConvNeXt 800/1333 preprocessing, signed masks and pinned OVR operators. Not full FC-CLIP panoptic inference. |
| E08 | [src/static_ovmap/backbone_wave1/association.py](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/src/static_ovmap/backbone_wave1/association.py) — `directional_assignments; plan_objects` | Old solver has one-to-one capacities and unmatched planned_owner=0. |
| E09 | [third_party_patches/ovimap/backbone_wave1_v1/backbone_wave1_v1.patch](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/third_party_patches/ovimap/backbone_wave1_v1/backbone_wave1_v1.patch) — `beginBackboneAssociation; candidate/fresh-group/count/alias changes` | Zero plan allocates fresh owner; a complete native fallback needs changes at all integration layers. |
| E10 | [src/static_ovmap/backbone_wave1/frontend_sam2.py](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/src/static_ovmap/backbone_wave1/frontend_sam2.py) — `paired_rasters; _raster; run_paired_frontend` | Partial-overlap discovery rule; saved masks and RAW winners, not full logits; 5-frame causal chunks. |
| E11 | [src/static_ovmap/backbone_wave1/mapping_hooks.py](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/src/static_ovmap/backbone_wave1/mapping_hooks.py) — `BackboneMappingHook` | Real preinsert probe and recipe hooks; new after-frame callback needed for own-map S2 history. |
| E12 | [src/static_ovmap/backbone_wave1/maps.py](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/src/static_ovmap/backbone_wave1/maps.py) — `mapping_job; run_map` | Exact schedule, isolated binary, deferred features; do not call old all-runner to execute new matrix. |
| E13 | [src/static_ovmap/m2_reviewer_study/evaluation.py](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/src/static_ovmap/m2_reviewer_study/evaluation.py) — `SceneEvaluator; official_view; pool` | Masks enumerate input registry; exact ordered pooling/current-class area ranks; new U universe must be included. |
| E14 | [src/static_ovmap/backbone_wave1/diagnostics.py](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/src/static_ovmap/backbone_wave1/diagnostics.py) — `instance_diagnostics; diagnose_map; compare_maps` | Raw geometry diagnosis; current old function waits for all readouts, requiring a new staged adapter. |
| E15 | [artifacts/static_ovmap/backbone_wave1_v1/raw_geometry_pools.json](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/artifacts/static_ovmap/backbone_wave1_v1/raw_geometry_pools.json) — `development pools` | BIDIR fragmentation and R50 loss motivate fail-native / capacity controls, not guaranteed recovery. |
| E16 | [artifacts/static_ovmap/backbone_wave1_v1/spatially_aligned_availability_changes.json](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/artifacts/static_ovmap/backbone_wave1_v1/spatially_aligned_availability_changes.json) — `office2/office4 predicted support correspondences` | Nearly unchanged predicted geometry can change native/F availability; these are prediction-prediction IoUs, not GT proof. |
| E17 | [docs/paper/static_ovmap/BACKBONE_WAVE1_RESULTS.md](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/docs/paper/static_ovmap/BACKBONE_WAVE1_RESULTS.md) — `official development/Replica results` | Same-cohort Native+D2 remains AP reference; raw geometry does not establish stronger new backbone. |
| E18 | [docs/paper/static_ovmap/BACKBONE_WAVE1_HANDOFF.md](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/docs/paper/static_ovmap/BACKBONE_WAVE1_HANDOFF.md) — `roots, runtime, patch stack, measured limitations` | Actual restoration/build context; model times include CPU/I/O and data are exposed. |
| E19 | [src/static_ovmap/module_validation/native_capture.py](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/src/static_ovmap/module_validation/native_capture.py) — `RegionRequest; FrameObservation` | Recorded identity/lineage, target masks, bbox, image and map version contracts. |
| E20 | [src/static_ovmap/paired_evidence_study/residuals.py](https://github.com/Orangekostar/oviovo/blob/c21297413954ecb7d706d1050a8e07822938ea6f/src/static_ovmap/paired_evidence_study/residuals.py) — `pool; simple_dependence` | Actual grouped fusion and availability behavior; D2 is .25/.25/.5 only with all three sources. |

## What was not verified in this planning turn

- No target-server filesystem/GPU access, no model loading, no C++ compilation, and no new benchmark.
- No complete reread of all large raw provenance/geometry arrays; source availability counts for new U arms are unknown until binding.
- GitHub source inspection does not establish that every external cache still exists.
- No assertion that proposed rules are new to the entire literature or guarantee a metric gain.

## Prior instruction change

The supplied 2026-09-30 IMPLEMENTATION_CONTRACTS.md C8 intentionally preserved native-painted
eligibility during the geometry comparison. U now changes that policy under a new identity.
The old control remains intact. The new recovery branch must not silently rewrite the old experiment.

## Proposed code-to-task mapping

| New task-local module | Reuse/inspect | New responsibility |
|---|---|---|
| recovery_registry.py | E01/E03/E19 | Independent RAW candidate registry; no GT; append-only support |
| recovery_sources.py | E04/E05/E06/E07 | One native/Q cached source; matched or multi-view FC; capped misses |
| weight_readout.py/export.py | E02/E13/E20 | Exact endpoints and missing-source behavior; expanded payloads/masks |
| association.py/native_patch.py | E08/E09/E11/E12 | Per-group native fallback and compatible many-to-one real integration |
| sam_completion.py/mapping_hooks.py | E10/E11/E19 | Binary/winner crop priority; own-past known-foreign filtering |
| evaluation.py/selection.py | E13/E14/E15/E17 | Official metrics, raw screen, pre-Replica selection |
| reporting.py/workflow.py | E12/E18 | Dynamic event/cost reports, resume, compact release, SHA check |

All new paths/function names in the specification are development requirements, not claims they already exist.
