# CROVE OVI-MAP Static-Anchor Apartment Result

## Decision

The OVI-MAP static-backbone architecture is implemented and measured on the
complete 1,745-frame TESSE-CD Apartment sequence. The selected implementation
keeps frozen OVI-MAP geometry, applies CROVE lifecycle updates after frame 262,
emits only valid post-cutoff identities from the current checkpoint interval,
and accepts a changed CROVE semantic label only when its current geometry covers
at least 50% of the OVI-MAP anchor voxels.

The final Apartment gate result is `REJECTED_RETAIN_A6`. Office was not run.

## Measured Candidates

| Candidate | Obj. F1 | Dyn. F1 | Chg. F1 | Current mIoU | Ghost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial union | 0.295365 | 0.045866 | 0.071244 | 0.204236 | 0.495922 |
| Current-interval filtering | 0.337883 | 0.069225 | 0.091272 | 0.148771 | 0.443760 |
| Dynamic-only promotion | 0.303029 | 0.069225 | 0.093074 | not run | not run |
| Unconditional semantic transfer | 0.364440 | 0.069225 | 0.087026 | 0.134118 | 0.443760 |
| Geometry-supported semantic transfer | 0.348472 | 0.069225 | 0.088458 | 0.149560 | 0.443760 |

The geometry-supported candidate is retained because it improves both Obj. F1
and current mIoU over current-interval filtering while keeping Dyn., Chg., and
Ghost valid. Its only failed gate is Obj. F1: 0.348472 versus the frozen 0.372762
floor. Current mIoU exceeds its 0.142897 floor, Chg. F1 exceeds its 0.060853
floor, and Ghost is below its 0.646883 ceiling.

## Root Cause

Unconditional semantic transfer changed eight bound anchors at the first
checkpoint. Six changes lacked sufficient current geometric support. The worst
case relabeled a 1,681,524-point OVI-MAP Couch as Chair using a temporal object
that covered only 0.99% of the anchor voxels. This raised official object
matching but reduced point-level mIoU. The retained coverage gate accepts the
two well-supported changes (`Objects -> Books` at 95.1% coverage and
`Unknown -> Books` at 88.3%) and rejects the unsupported changes.

## Evidence

- Composition manifest:
  `/home/ww/oviovo_baseline_runs/20260901_crove_ovimap_anchor/composed_selective/apartment/run_manifest.json`
  (`cb6f5c8858739ff861364a0e5d32735f5001849ca4e083aba430d316f6fe9983`)
- Repeated official metrics:
  `/home/ww/oviovo_baseline_runs/20260901_crove_ovimap_anchor/official_selective/apartment/evaluation/official_metrics.json`
  (`9ceb7024d68857a020f8ac059f5863530d51de3a7e58cf8d592eeda34d8f3855`)
- Gate receipt:
  `/home/ww/oviovo_baseline_runs/20260901_crove_ovimap_anchor/gate_selective/apartment/gate_decision.json`
  (`b1783aa398ca3f7d6db7fce468a4d121c5210c01ce1e4eae4fc37ccc38e6bda1`)
- Official run identity: `crove-ovimap-anchor-selective-apartment-v5`.
- Official state count: 43. Processed frame count: 1,745.

No paper table value is updated from this rejected Apartment candidate.
