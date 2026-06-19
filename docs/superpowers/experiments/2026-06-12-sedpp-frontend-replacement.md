# SED++ Frontend Replacement Experiment

Date: 2026-06-12

## Setup

- Scene: Replica room0
- Frames: 200
- Stride: 10
- Runner: `scripts/run_room0_checkpointed_eval.py --mode build-export`
- Backend: `proposal.backend=sedpp`
- Anchor frontend: disabled
- Anchor-guided SAM: disabled
- SED weights: `/home/ww/vv/paper2/SED/weights/sed_model_large.pth`
- OpenCLIP weights: local ConvNeXt-L checkpoint under `/home/ww/vv/paper2/SED/weights`

Reference baseline from the implementation plan:

- Run: `outputs/tmp_validation/20260612_wwai_online_yoloworld_sam_baseline_s10_200f`
- mIoU: `0.5190`
- f-mIoU: `0.6052`
- Wall: `40:37.90`
- `proposal_generation` mean: `10.4371s`
- `anchor_guided_sam_fusion` mean: `9.0352s`

## Runs

| Run | Config | mIoU | f-mIoU | Wall | Mapping sec/frame | Objects | Proposal gen mean | SED inference mean | SED postprocess mean | RuntimeVis mean | Association mean | Object update mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cap40 200f | `configs/replica_sedpp_prob_cap40_4090.yaml` | `0.2213` | `0.4488` | `26:05.58` | `7.4632s` | `50` | `0.8969s` | `0.1151s` | `0.7817s` | `1.2254s` | `1.0724s` | `3.2534s` |
| cap80 200f | `configs/replica_sedpp_prob_lowthr_4090.yaml` | `0.3668` | `0.5618` | `42:22.95` | `12.3269s` | `176` | `0.8069s` | `0.1148s` | `0.6919s` | `3.7005s` | `2.1091s` | `4.5083s` |

20-frame probes:

| Run | mIoU | f-mIoU | Mapping sec/frame | Objects | Raw proposal mean | Patch mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cap40 20f | `0.0922` | `0.2854` | `6.3722s` | `21` | `40.0` | `18.8` |
| cap80 20f | `0.1840` | `0.3031` | `10.4824s` | `51` | `78.85` | `39.7` |

Raw summary JSON:

- `outputs/tmp_validation/20260612_sedpp_frontend_replacement_summary.json`

## Interpretation

SED++ inference itself is not the bottleneck. The model forward averages about `0.115s/frame`; semantic component postprocess is below `0.8s/frame`. The expensive part moves downstream:

- cap40: `object_update` dominates at `3.25s/frame`; `runtime_vis` and `association` are manageable.
- cap80: more proposals improve accuracy but drive `runtime_vis`, `association`, and `object_update` up enough that wall time exceeds the YOLOWorld+SAM baseline.

The direct semantic-component replacement is therefore viable as a fast frontend primitive, but not sufficient as a drop-in replacement for accuracy. It needs proposal shaping before entering mapping.

## Next Steps

1. Add class-aware proposal quotas instead of a global cap. Keep structural classes sparse while preserving compact object proposals.
2. Split structural/background handling from object proposals. Large wall/floor/window components should feed structural overlay or background, not object update.
3. Add a mid-confidence rescue path only for compact objects, not full SAM over all frames.
4. Optimize SED++ postprocess with per-class confidence prefiltering and avoid connected components for classes with no pixels above threshold.
5. Re-test a mid cap (`max_proposals=60`) after class quotas; cap40 is too sparse, cap80 is too expensive.
