# Source evidence — checked 2026-10-08

The parent and model repository revisions were resolved through the connected GitHub reader.
New algorithm constants and thresholds are proposed design choices, not facts copied from these sources.
No new inference or experiment execution took place while preparing this package.

| ID | Source / location | What it constrains |
|---|---|---|
| S01 | [SAM-V paper](https://arxiv.org/html/2609.25490v1) — Sections 3.1–3.4, 4.1, appendix evaluation definitions | SAM/VGGT geometry and prompt fusion; trained modules, not our contribution. |
| S02 | [SAM-V PDF figures](https://arxiv.org/pdf/2609.25490) — PDF pages 4–5 inspected visually | Architecture and training/input protocol; not our 3D mask-AP protocol. |
| S03 | [Author setup / checkpoint declaration](https://github.com/gong208/SAM-V/blob/33fab24c1d0d21ac56f4e471abf30c6de3e7018b/README.md) — lines 1–106 | Stage-2 checkpoint plus frozen bases; only required submodules; release pointer not a successful download. |
| S04 | [Requirements](https://github.com/gong208/SAM-V/blob/33fab24c1d0d21ac56f4e471abf30c6de3e7018b/requirements.txt) — complete file | Python3.10/torch2.3.1/torchvision0.18.1/timm1.0.22 isolated from other workers. |
| S05 | [Partial checkpoint verifier](https://github.com/gong208/SAM-V/blob/33fab24c1d0d21ac56f4e471abf30c6de3e7018b/utils/checkpoint.py) — load_partial_checkpoint / load_slim_checkpoint | Trainable tensor coverage must be validated; no silent shape-filter load. |
| S06 | [Author benchmark hazards](https://github.com/gong208/SAM-V/blob/33fab24c1d0d21ac56f4e471abf30c6de3e7018b/benchmarks/compare_baseline_sam2.py) — lines 1–230, 500–835 | GT-derived prompts and frame pools; main clamps frame_count>=16; cannot call its dataset CLI. |
| S07 | [Model inference conversion](https://github.com/gong208/SAM-V/blob/33fab24c1d0d21ac56f4e471abf30c6de3e7018b/benchmarks/compare_baseline_sam2.py) — lines 389–505 | Single-mask forward; split low-res panorama by view; JPEG input discrepancy must be removed. |
| S08 | [SAM2 propagation](https://github.com/gong208/SAM-V/blob/33fab24c1d0d21ac56f4e471abf30c6de3e7018b/benchmarks/compare_baseline_sam2.py) — lines 500–558 | Anchor points, forward then reverse, reset state, original object ID mapping. |
| S09 | [Joint model interface](https://github.com/gong208/SAM-V/blob/33fab24c1d0d21ac56f4e471abf30c6de3e7018b/model/sam_vggt_model.py) — lines 400–692 | SAM frame chunks; whole-group VGGT; prompt frame/point features; forward output is logits. |
| S10 | [SAM2 package requirement](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/setup.py) — lines 1–140 | torch>=2.5.1 incompatible with simply pip-installing into fixed SAM-V2.3.1 env. |
| S11 | [SAM2 constructor settings](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/build_sam.py) — lines 80–155 | Explicit postprocessing profile, no VOS compile; no silent optional hole-fill path. |
| S12 | [SAM2 author weight address](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/checkpoints/download_ckpts.sh) — complete file | Only download large SAM2.1 checkpoint; do not run all-four download script. |
| S13 | [Our parent data adapter](https://github.com/Orangekostar/oviovo/blob/a95c24d990cea57b14bb537e9acca95b95659bda/src/static_ovmap/source_preserving_update/binding.py) — load_scene; bind | Actual G1/D2, camera/mesh, N/Q/F and identities; old bind/all remains read-only. |
| S14 | [Our independent observer](https://github.com/Orangekostar/oviovo/blob/a95c24d990cea57b14bb537e9acca95b95659bda/src/static_ovmap/minimal_instance_repair/observations.py) — SourceRowProjector / representative_rows / pose_banks | Measured depth with complete-mesh occlusion and barycentric source-row mapping. |
| S15 | [Old output restrictions](https://github.com/Orangekostar/oviovo/blob/a95c24d990cea57b14bb537e9acca95b95659bda/src/static_ovmap/minimal_instance_repair/outputs.py) — construct_partition | Whole-unit/raw-zero/core guards would suppress intended new row-wise experiment. |
| S16 | [Existing real FC fallback](https://github.com/Orangekostar/oviovo/blob/a95c24d990cea57b14bb537e9acca95b95659bda/src/static_ovmap/cvpr_compact/area_fallback.py) — pool_region / region_vector; content also inspected in prior context | Hard support then area fallback, followed by original visual projection; not AnyUp. |
| S17 | [Released partition scorer](https://github.com/Orangekostar/oviovo/blob/a95c24d990cea57b14bb537e9acca95b95659bda/src/static_ovmap/minimal_instance_repair/evaluation.py) — partition_evaluator; available source inspected in prior context | Actual partition-specific registry and ordered whole-cohort pooling. |
| S18 | [Source update results](https://github.com/Orangekostar/oviovo/blob/a95c24d990cea57b14bb537e9acca95b95659bda/docs/paper/static_ovmap/SOURCE_UPDATE_RESULTS.md) — full three-table report in conversation source context | No upgraded method; same-view coarse FC is a necessary control, not a guaranteed better classifier. |
| S19 | [Scope diagnosis](https://github.com/Orangekostar/oviovo/blob/a95c24d990cea57b14bb537e9acca95b95659bda/artifacts/static_ovmap/minimal_instance_repair_v1/provenance/diagnostics/summary.json) — full diagnostic summary in conversation source context | R/K matchability limitations motivate testing new support, not promise a positive result. |

## Exact dependency revisions

- SAM-V: `33fab24c1d0d21ac56f4e471abf30c6de3e7018b`.
- Patched sam-hq: `8c0ef8de98c5b84f3c050339d91e953c7249f10e`.
- VGGT submodule: `44b3afbd1869d8bde4894dd8ea1e293112dd5eba`.
- SAM2 submodule: `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
Gitlinks verified from [submodule inventory](https://api.github.com/repos/gong208/SAM-V/contents/submodules?ref=33fab24c1d0d21ac56f4e471abf30c6de3e7018b).

## Prerequisites not attested by the assistant

- The stage-2 author HF pointer is `Frank-Gong123/SAM-V/sam_v_stage2.pth`. Browser reads of its HF page failed during preparation; no file length/SHA256 or successful load is asserted. Resolve revision, hash and full module coverage on the server.
- SAM/VGGT/SAM2 weights may already exist on shared storage; local availability and bytes must be checked. A basename/hash-looking filename is not a SHA256.
- The public [VGGT-1B model page](https://huggingface.co/facebook/VGGT-1B/tree/main) lists model.pt and a CC-BY-NC-4.0 license. Read the actual licenses of all adopted assets before use; do not republish them.
- The assistant container could not resolve raw.githubusercontent.com for a programmatic copy. Source inspection used the GitHub connector instead; there was no local model checkout or execution.
- Four-scene data/caches are referenced by completed studies, not independently verified as present in the current remote filesystem. Exact relocation is allowed; inventing missing data is not.
- A40 execution with six views is an intended starting profile, not a measured memory claim. Only the specified prefreeze real-OOM four-view fallback is permitted.

## Attribution

SAM-V is a pretrained external segmentor, SAM2 an external control, and FC the inherited recognizer. The proposed contribution under test is map-driven local querying and controlled use of resulting masks. Do not assert novelty of SAM-V fusion, a new segmentation decoder, a new FC pooling method or a new released evaluation metric.
