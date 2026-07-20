# OVIV2 ScanNet200 Stage4 Design

## Purpose

Complete the four remaining OVIV2 ScanNet200-5 Table 1 cells with an independently reproducible OVIV2 run. The run may reuse the same released model code and weights used by OVIV2 on Replica, but it must not consume detections, maps, embeddings, labels, or other intermediate artifacts produced by a competing baseline.

This design supersedes the ScanNet access blocker in `2026-07-19-benchmark-repair-oviv2-scannet-design.md`. Publisher-approved data is now present and validated for the five frozen scenes.

## Fixed Scope

The frozen scene order is:

1. `scene0011_00`
2. `scene0050_00`
3. `scene0231_00`
4. `scene0378_00`
5. `scene0518_00`

Every scene uses source frame 0 with stride 10. Frames with an invalid pose are excluded before any model runs. This yields 238, 465, 444, 190, and 147 frames respectively, for 1,484 frames total; `scene0050_00/4650` is the only excluded stride-10 frame.

The headline mapping configuration retains the selected Replica Stage3 parameters, including 5 cm voxels, cached dense probabilities, dense entropy power 16, integration radius 6 m, top-k 4, and fused semantic entity scale 0.49. Dataset paths, vocabulary, image geometry, frame IDs, output paths, and capacity controls are dataset-specific and are not tuning degrees of freedom.

## Non-Goals

- Do not reuse the existing ConceptGraphs ScanNet detection cache or any other baseline intermediate output.
- Do not tune mapping or fusion hyperparameters on the five evaluation scenes.
- Do not use ScanNet semantic labels, instance IDs, class presence, or axis-aligned GT meshes during frontend inference or mapping.
- Do not overwrite Replica outputs, existing baseline outputs, or previous result manifests.
- Do not claim official ScanNet leaderboard performance; this is the frozen five-scene paper split only.
- Do not populate a table token until the result finalizer reports `VERIFIED`.

## Input Contract

The existing raw ScanNet manifest remains the authority for each scene's `.sens`, metadata, mesh, label mesh, segments, and aggregation hashes. A Stage4 input manifest adds only derived RGB-D execution inputs:

- explicit selected source frame IDs;
- SHA-256 for every selected color image, depth image, and pose file;
- SHA-256 for depth and color intrinsics;
- color resize rule and depth scale;
- official axis-aligned evaluation PLY and metadata hashes;
- the ordered official ScanNet200 class names and numeric class IDs.

The loader accepts only the frozen explicit source-frame list. Color is deterministically resized from 1296x968 to 640x480 and paired with the native 640x480 depth map. Projection uses `intrinsic_depth.txt`, whose calibration matches the resized color view, and converts depth millimeters with scale 1000. Poses must be finite invertible 4x4 camera-to-world matrices.

The 200-class vocabulary is materialized as a tracked JSON asset derived from the official constants. Its class order is shared by the frontend, RADSeg worker, mapper, evaluator, and result finalizer. `wall`, `floor`, and `ceiling` remain structure classes generated from depth rather than object detector instances.

## Independent Frontend

The Stage4 frontend runs the OVIV2 YOLO-World plus MobileSAM pipeline directly on the frozen Stage4 RGB view. It uses the same pinned model identities and cache schema as Replica, with the ScanNet200 object vocabulary. Each cache file is named by contiguous cache index while its manifest records the corresponding original ScanNet source frame ID.

Frontend validation requires:

- exact scene, frame count, source-frame order, mask shape, and class order;
- finite boxes and confidences and valid class IDs;
- one model-generated cache file for every selected frame;
- hashes for the runner, configuration, class file, YOLO-World, MobileSAM, CLIP weights, input manifest, and every cache file;
- cache generation in a fresh OVIV2 output root, with no path under a baseline result directory.

The frontend runs on available GPUs. A completed valid prefix may be resumed, but files that fail hash or schema validation are regenerated rather than trusted.

## Dense Semantics

The existing RADSeg plus SAM worker is generalized from a fixed 41-class Replica vocabulary to a manifest-declared class count. The ScanNet run uses all 200 ordered classes, sample stride 4, top-k 4, pinned RADIO and SigLIP2 identities, and the same SAM refinement mode used by the selected Replica run.

Dense cache manifests bind source-frame IDs, input shape, vocabulary hash, prompt hash, inference configuration hash, model hashes, cache prefix hash, and per-frame file hashes. Cache production uses GPUs and supports validated prefix resume. The mapper consumes no dense file outside the manifest-bound prefix.

## Mapping

A ScanNet dataset adapter implements the existing `Frame` contract without changing the OVIV2 runtime. The Stage4 runner shares the Stage3 runtime configuration builder, frontend observation adapter, depth-structure frontend, TSDF/evidence/ownership/entity pipeline, checkpoint format, semantic fusion, and labeled meshing.

Mapping runs in native ScanNet sensor-world coordinates. It writes owner, dense, and fused meshes plus the immutable voxel snapshot, entity registry, timings, cache audit records, and run manifest. The fused head at entity scale 0.49 is the headline semantic output. Instance geometry remains the OVIV2 ownership output; dense semantics do not create or split instances.

A 20-frame `scene0011_00` smoke run must prove non-empty frontend observations, TSDF blocks, entities, dense updates, mesh vertices, and finite transforms before full preprocessing or mapping is promoted. The smoke output is diagnostic and cannot populate benchmark tokens.

## Evaluation

Evaluation is a separate process and is the first component allowed to read official labels. Predicted geometry is transformed by the scene's `axisAlignment` matrix into the coordinates of `scannet200_official/val/<scene>.ply`.

Two prediction views are evaluated independently:

- semantic points are grouped by fused per-vertex semantic ID and mapped to the official 200 class names;
- instance points are grouped by positive OVIV2 entity ID and ranked by deterministic entity confidence.

The shared static evaluator computes semantic mIoU, mAcc, frequency-weighted mIoU, class-agnostic AP25/AP50, and geometry F5 at 5 cm. This separation prevents semantic partitions from being counted as object instances. The evaluator records per-class metrics, counts, alignment matrix, unmatched ratio, and hashes of every input and output.

The final aggregate is an unweighted macro average over exactly the five frozen scenes. A finalizer reruns every scene evaluation in a fresh directory and requires byte-identical outputs, identical algorithm/model/vocabulary contracts across scenes, exact artifact sets, a clean repository commit, and complete checksums. It then emits a `VERIFIED` result compatible with the benchmark importer.

## Failure Handling

- Missing, changed, symlinked, malformed, non-finite, or out-of-order inputs stop preflight before output creation.
- Any invalid selected pose is excluded only by the frozen input manifest; the runner cannot silently skip frames.
- Frontend or dense worker failure preserves logs and the last verified prefix, without publishing a completion manifest.
- TSDF capacity exhaustion, empty maps, zero dense updates, or empty evaluation domains are hard failures.
- Coordinate alignment is tested numerically; a prediction-to-GT mismatch cannot be bypassed with a larger evaluation radius.
- Interrupted mapping may restart from validated immutable caches, but the headline run begins in a fresh output directory.
- A failed repeat evaluation leaves the table tokens `UNFILLED`.

## Testing Strategy

All production behavior follows red-green-refactor.

Dataset tests cover deterministic RGB resizing, depth scale, explicit source IDs, invalid poses, intrinsics, missing files, path escape, and frame hash mismatches. Frontend tests cover command construction, independent output roots, 200-class ordering, source-frame binding, malformed cache payloads, and resume validation. Dense tests cover variable class count, 200-class metadata, top-k bounds, vocabulary mismatch, prefix integrity, and resume.

Runner tests use small synthetic RGB-D scenes to verify the shared Stage3 configuration, cache-index/source-ID separation, no GT access in mapping, non-empty artifacts, and deterministic manifests. Evaluator tests use hand-constructed aligned point clouds to verify axis alignment, semantic/instance view separation, exact 5 cm boundaries, AP ranking, and macro aggregation. Finalizer and importer tests reject incomplete scenes, dirty provenance, changed artifacts, non-identical repeated evaluation, and direct table edits.

## Acceptance Criteria

1. The Stage4 input manifest validates all 1,484 selected frames and the official five-scene GT without reading labels during mapping.
2. Every frontend and dense cache is independently generated for OVIV2 and fully hash-bound.
3. The 20-frame smoke run passes all geometry, semantic, instance, and coordinate gates.
4. All five full runs share the frozen algorithm, model, vocabulary, and evaluation contracts and terminate without OOM or capacity exhaustion.
5. Fresh repeated evaluation is byte-identical for all five scenes.
6. The final result is `VERIFIED` and provides finite ScanNet200-5 mIoU, AP25, AP50, and F5 values.
7. The benchmark importer fills exactly `T1_OVIV2_SCANNET5_MIOU`, `T1_OVIV2_SCANNET5_AP25`, `T1_OVIV2_SCANNET5_AP50`, and `T1_OVIV2_SCANNET5_F5` from that result.
8. Focused and full OVIV2/evaluation/table tests pass from a clean commit.

