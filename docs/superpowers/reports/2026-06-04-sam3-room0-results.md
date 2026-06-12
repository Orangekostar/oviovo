# SAM3 Room0 Experiment Results

## Baseline

- Run: `outputs/tmp_validation/20260603_replica_yoloworld_sam_online_s10_200f_room0_s10_200f_fast_eval`
- mIoU: `0.5303457818050147`
- f-mIoU: `0.6062476053140748`
- sec/frame: `4.037710784379999`
- proposal_generation mean: `3.803539853175007`
- sam2_proposals mean: `0.7588200007149704`
- anchor_guided_sam_fusion mean: `2.973407269880024`
- runtime_vis mean: `0.4830470646650338`
- object_update mean: `1.5391627290099974`

## SAM3 Mock 20f

- Run: `outputs/tmp_validation/20260604_sam3_mock_room0_s10_20f`
- status: `complete`
- mIoU: `0.04013631521008214`
- f-mIoU: `0.08687439917522258`
- sec/frame: `0.5763395326001046`
- proposal_generation mean: `0.004075614849716658`
- sam2_proposals mean: not present, as expected for pure `sam3_concept`
- anchor_guided_sam_fusion mean: not present, as expected for pure `sam3_concept`
- runtime_vis mean: `0.00041668229969218375`
- object_update mean: `0.3319589494014508`
- final object count: `7`
- proposal count/frame: `3.0` raw proposals mean, `20/20` frames nonzero
- failure notes: mock path is functionally wired but intentionally not quality-credible.

## SAM3 20f

- Run: `outputs/tmp_validation/20260604_sam3_concept_room0_s10_20f`
- status: blocked before rerun by available GPU memory on 2026-06-05.
- mIoU: not rerun after prompt-budget fix.
- f-mIoU: not rerun after prompt-budget fix.
- sec/frame: not rerun after prompt-budget fix.
- proposal_generation mean: not rerun after prompt-budget fix.
- runtime_vis mean: not rerun after prompt-budget fix.
- final object count: not rerun after prompt-budget fix.
- proposal count/frame: not rerun after prompt-budget fix.
- failure notes: the SAM3 env now exists and imports successfully at `/home/ww/miniconda3/envs/sam3/bin/python` with Python `3.12.13`, torch `2.7.0+cu126`, CUDA `12.6`, and importable `sam3`. The real 20f run was not restarted because `nvidia-smi` showed only about `1.9 GB` free VRAM, with unrelated Python processes using about `9.5 GB` and `12.2 GB`. Do not use the old high-recall full-vocabulary SAM3 command; the current experiment config limits SAM3 to `max_prompts_per_frame: 6`, `max_proposals: 24`, `confidence_threshold: 0.35`, and `candidate_strategy: round_robin`.

## SAM3 200f

- Run: not run: gated off because real SAM3 20f did not complete.
- mIoU: not run: gated off because real SAM3 20f did not complete.
- f-mIoU: not run: gated off because real SAM3 20f did not complete.
- sec/frame: not run: gated off because real SAM3 20f did not complete.
- proposal_generation mean: not run: gated off because real SAM3 20f did not complete.
- runtime_vis mean: not run: gated off because real SAM3 20f did not complete.
- final object count: not run: gated off because real SAM3 20f did not complete.
- proposal count/frame: not run: gated off because real SAM3 20f did not complete.
- failure notes: gated off because the real SAM3 20f run did not complete.

## Decision

- Replace YOLOWorld+SAM2 now: no
- Keep SAM3 as experiment: yes
- Next optimization: rerun 20f worker mode only after GPU memory is available, using sparse prompt budgeting. Only run 200f if real proposals are nonzero on most frames and proposal time is within the plan gate.
