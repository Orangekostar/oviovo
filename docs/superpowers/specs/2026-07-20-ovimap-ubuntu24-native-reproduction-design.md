# OVI-MAP Ubuntu 24.04 Native Reproduction Design

## Goal

Reproduce the released OVI-MAP Replica pipeline on the current Ubuntu 24.04 host, repair the invalid CropFormer-to-OVI-MAP interface used by the historical run, and publish only metrics supported by released or independently verified evaluators.

The historical OVI-MAP result remains immutable failure evidence. New outputs use a new run ID and do not overwrite old masks, maps, features, evaluations, or manifests.

## Existing Local Evidence

`/home/ww/vv/paper2/OVI-MAP` contains the previous local reproduction workspace, catkin build trees, source changes, and a Python 3.10 `consistent_gsm` extension. It confirms that OVI-MAP was previously built locally with a reduced dependency surface:

- `gsm_node` and RViz-only visualization dependencies were removed from the Python mapping path;
- the relevant mapping targets were compiled as C++17;
- system PCL/OpenGL linking and OpenCV/PCL catkin compatibility changes were applied;
- dedicated Python 3.10 build/devel directories were used.

The old runtime is not reusable as-is. `/home/ww/miniconda3/envs/ovi-map` no longer exists, ROS commands are absent, and the old extension currently has unresolved ROS-era PCL, VTK, Boost, glog, protobuf, and FLANN libraries. The old `.so` is therefore evidence of a successful build procedure, not a valid current binary.

## Reproduction Boundary

The reproduction runs directly on Ubuntu 24.04. It does not use an Ubuntu 20.04 container and does not silently switch to a different host image.

Two isolated environments separate the incompatible frontend and mapping requirements:

1. `ovimap-cropformer`: Python 3.8, PyTorch 2.1.1, torchvision 0.16.1, CUDA 12.1, and the official CropFormer HoRNet configuration and checkpoint.
2. `ovimap-map`: Python 3.11, the current RoboStack Noetic ABI, PyTorch 2.1.1, transformers 4.49, and the smallest Python/catkin dependency set that can rebuild and import the released OVI-MAP targets on Ubuntu 24.04. Transformers 4.49 requires Python 3.9 or newer, so it belongs to this SigLIP mapping environment rather than the Python 3.8 CropFormer environment. Its exact package set is frozen after the import smoke test passes.

CUDA compiler and runtime packages are installed inside the frontend environment. The host CUDA toolkit is not assumed. CropFormer must import and execute the compiled CUDA `MSDeformAttn` operator; the pure-PyTorch deformable-attention fallback is forbidden.

ROS is a build dependency, not a runtime workflow. The design reuses the reduced `paper2` build path and does not require roscore, rosbag playback, RViz, or ROS nodes. If the retained source still requires catkin metadata or ROS message packages, they are provided inside the isolated mapping environment. No system-wide ROS Noetic installation is assumed on Ubuntu 24.04.

## Source and Compatibility Policy

The official OVI-MAP and Entity/CropFormer commits, model checkpoints, and Replica inputs are hash-bound before execution. A fresh build copy is created from those sources. The dirty `paper2` checkout is read-only reference material and is never cleaned or overwritten.

Compatibility changes may affect only build configuration, compiler/API compatibility, or diagnostic output. Changes to association thresholds, geometry integration, view selection, semantic scoring, or evaluation equations are prohibited.

Every compatibility patch is recorded with:

- source file and exact diff;
- reason it is required on Ubuntu 24.04;
- whether it changes runtime behavior;
- focused compile or regression evidence.

The two PyTorch dispatch edits using `scalar_type()` are allowed because they are upstream API compatibility fixes. Disabling the CUDA operator or substituting a different segmentation model is not allowed.

## CropFormer Output Contract

The released CropFormer demo is adapted only to export the OVI-MAP input contract. For each frame it writes one lossless single-channel instance-ID PNG with:

- background ID fixed to zero;
- positive, frame-local, contiguous instance IDs;
- integer dtype with no color-palette conversion;
- original Replica image resolution;
- deterministic overlap resolution derived from the released prediction ordering.

The exporter also writes per-frame diagnostics: instance count, unique IDs, foreground coverage, largest-instance coverage, and checkpoint/config hashes. A one-frame regression fixture verifies that visualization RGB images cannot be accepted as instance masks.

## Mapping Interface Audit

The previous run produced corrupted color records and full-frame semantic query boxes. The new run therefore treats the CropFormer-to-mapper and mapper-to-feature interfaces as hard contracts.

For every audited frame, the runner records:

- input mask ID histogram and foreground coverage;
- mapper `getInstanceColor` output for each active global instance;
- backend RGB value written to the instance-color log;
- raycast instance-ID map shape, dtype, unique IDs, and coverage;
- the bounding box derived from each raycast instance mask;
- counts of raw detections, associated instances, eligible feature instances, and semantic queries.

The run fails before feature extraction when RGB channel order differs between the mapper and log, a raycast output is incorrectly marshalled, or all non-empty boxes become full-frame. Diagnostics may be added at the Python/C++ boundary, but the mapping algorithm remains unchanged.

## Execution Gates

Execution expands only after the preceding gate passes:

1. Environment gate: import PyTorch, compiled CropFormer CUDA operator, `consistent_gsm`, and `depth_segmentation_py`; record versions and linked-library audit.
2. Frontend gate: run one Replica frame and validate the exported mask contract.
3. Mapping gate: run one frame through OVI-MAP and validate color, raycast, and bbox contracts.
4. Room gate: run the official 200-frame, start-0/end-2000/step-10 protocol on `room0`; generate mapping, features, and complete diagnostics.
5. Dataset gate: run the same protocol on all eight Replica scenes only after `room0` passes.
6. Evaluation gate: finalize metrics only after all scene artifacts and manifests pass hash and protocol audits.

A failed gate preserves its command, environment manifest, stdout/stderr, diagnostics, and exit code. It does not publish benchmark values.

## Evaluation and Table Policy

Replica semantic results use the released Replica-51 preprocessing and `eval_sem_seg.py` behavior. Metric names remain exactly those produced by the evaluator.

The released `eval_inst_seg.py` precision/recall output is diagnostic and must not be renamed to the paper's Table 2 AP metrics. Table 2 instance AP25/AP50/AP75 remain `UNFILLED` unless an evaluator is obtained or reconstructed and independently validated against hand-computed fixtures and the paper protocol. Paper values are never copied into reproduced-result cells.

Metrics not defined by the released OVI-MAP protocol, including the benchmark's neutral f-mIoU/F5 fields, also remain `UNFILLED` unless produced by a separately named neutral evaluator. Any such neutral results are reported separately from paper reproduction results.

Only a passing, hash-bound result manifest may update OVI-MAP benchmark tokens. The existing quarantined historical result remains unchanged.

## Testing Strategy

Production changes follow test-first development. Focused tests cover:

- mask exporter dtype, contiguous IDs, overlap ordering, and deterministic output;
- rejection of visualization/color masks;
- CUDA operator availability with no fallback path;
- RGB/BGR color-log round trip;
- raycast array shape and integer-ID preservation across pybind;
- bbox calculation for sparse, empty, and full-frame masks;
- official frame range and eight-scene manifest construction;
- evaluator output parsing without metric renaming;
- refusal to bind results when any gate, hash, scene, or evaluator contract is missing.

## Acceptance Criteria

1. The CropFormer CUDA extension compiles and executes on the Ubuntu 24.04 host without the PyTorch fallback.
2. Both OVI-MAP Python extensions import from a newly frozen environment with no unresolved shared libraries.
3. A one-frame run proves exact instance-ID, color-log, raycast, and bbox preservation.
4. `room0` completes 200 frames without the historical channel-collapse or all-full-frame-box failure.
5. All eight Replica scenes complete the frozen paper frame protocol with immutable manifests.
6. Released Replica-51 evaluation is reproducible from recorded commands and hashes.
7. Only verified evaluator outputs populate the benchmark; unavailable Table 2 AP metrics remain explicitly unfilled.
8. No pre-existing dirty project file or `paper2` source is overwritten, cleaned, or staged.
