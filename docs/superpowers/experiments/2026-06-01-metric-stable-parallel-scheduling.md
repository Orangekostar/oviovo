# Metric-Stable Parallel Scheduling Validation

## Runs

- Baseline: `outputs/tmp_validation/20260601_room0_fast_high_iou_v2_stride10_200f`
- Candidate: `outputs/tmp_validation/20260601_room0_metric_stable_parallel_s10_200f`
- Candidate smoke: `outputs/tmp_validation/20260601_room0_metric_stable_parallel_s10_20f_smoke`

## Acceptance

- baseline mIoU: `0.522394443344`
- candidate mIoU: `0.563696830841`
- mIoU delta: `+0.041302387498`
- baseline f-mIoU: `0.543366420787`
- candidate f-mIoU: `0.724413257743`
- f-mIoU delta: `+0.181046836955`
- prefetch submitted: `200`
- prefetch hit: `200`
- prefetch miss: `0`
- frontend exceptions: `0`

## Major Classes

- ceiling: baseline `0.368450`, candidate `0.892115`, delta `+0.523664`
- wall: baseline `0.491696`, candidate `0.691007`, delta `+0.199310`
- floor: baseline `0.623722`, candidate `0.790788`, delta `+0.167066`
- sofa: baseline `0.636777`, candidate `0.642279`, delta `+0.005502`
- blinds: baseline `0.582555`, candidate `0.603220`, delta `+0.020664`
- window: baseline `0.250220`, candidate `0.335451`, delta `+0.085231`
- lamp: baseline `0.336039`, candidate `0.774180`, delta `+0.438141`

## Scheduling

- `prefetch_enabled`: `true`
- `prefetch_submitted_count`: `200`
- `prefetch_hit_count`: `200`
- `prefetch_miss_count`: `0`
- `frontend_exception_count`: `0`
- Stage timing summary includes median, p95, max frame, and outlier frame IDs.

## Instrumentation Smoke

- Run: `outputs/tmp_validation/20260602_room0_metric_stable_parallel_instrumentation_s10_20f_smoke`
- `prefetch_submitted_count`: `20`
- `prefetch_hit_count`: `20`
- `prefetch_miss_count`: `0`
- `frontend_exception_count`: `0`
- `association_score_parallel_used_count`: `1382`
- `association_score_parallel_candidate_count_total`: `88056`
- `depth_refinement_parallel_used_count`: `20`
- `depth_refinement_parallel_worker_count_max`: `4`
- Stage timing summary includes top-level `proposal_generation` for prefetched frames.
