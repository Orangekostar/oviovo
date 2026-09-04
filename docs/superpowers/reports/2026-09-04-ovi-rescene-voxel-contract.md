# OVI-MAP x ReScene4D Voxel Contract

Date: 2026-09-04

## Source Identity

| Source | Pinned commit | License | Role |
| --- | --- | --- | --- |
| OVI-MAP | `f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424` | MIT | Per-visit dense mapper |
| ReScene4D | `fb2fe42eb8f1e926567c48eea9acb874e608ee10` | MIT | Optional cross-visit neural reasoner |
| CROVE base | `1e849acdcba46ab92f8704a77695ce7caefa1516` | Project | Legacy state and frozen evaluator |

Every required source file is bound by path, SHA-256, and byte count in
`configs/external/ovi_rescene_sources.json`. The audit compares each working
file byte-for-byte with `git show <commit>:<path>` and rejects missing files,
symlinks, remote mismatches, or mutable source bytes.

## Four Representations

| Name | Resolution | Lifetime | Contents | Owner | Authority |
| --- | ---: | --- | --- | --- | --- |
| OVI mapping voxel | 0.01 m | Persistent within a visit | TSDF/surface and LabelVoxel state | OVI-MAP | Final dense geometry |
| ReScene neural voxel | 0.02 m | One pair/forward pass | Sampled coordinates and learned features | ReScene | Evidence only |
| CROVE legacy temporal voxel | 0.05 m | Persistent historical submap | One weighted object-local representative | CROVE legacy path | Baseline/diagnostic only |
| Evaluator voxel | 0.05 m | One evaluation | Common-v2 comparison cells | TESSE-CD evaluator | Metric domain only |

The resolutions are not interchangeable. OVI's 1 cm state remains immutable
and authoritative. The adapter may spatially resample OVI surface samples to
the 2 cm neural grid, but the visit index is the exact integer `0` or `1`, not
a metric axis. Every neural token must retain a CSR reverse index to all
contributing OVI visit/entity/source samples. ReScene tokens may supply
identity or change evidence but may never be serialized as final geometry.
The common-v2 5 cm evaluation domain remains frozen.

## Checkpoint Gate

The pinned public ReScene tree contains no checkpoint artifact, and its README
states `Checkpoints: Coming soon.` Therefore the source and adapter work may
proceed, but learned B5/B6 inference is not ranking-eligible until an official
or reproducibly trained checkpoint is bound by bytes, configuration, data
split, command, and environment.

Checkpoint status: `BLOCKED_EXTERNAL_PRETRAINED_CHECKPOINT`

VOXEL_CONTRACT_PASS
