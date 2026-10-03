# Compact Tables Results

Implementation: FROZEN. Internal scientific coverage: PARTIAL_WITH_TECHNICAL_BLOCKS (168/172 outputs, 10/14 ordered pools). Timing: PARTIAL_WITH_TECHNICAL_BLOCKS (22/24 actual calls with scientific parity; 2 proved unmeasured leaves). External protocol: AUTHOR_PROTOCOL_AS_REPORTED, NOT_INDEPENDENTLY_VERIFIED. Performance: see raw deltas and descriptive flags below. Publication: PENDING_PUSH_VERIFICATION; the final external publication receipt is authoritative.
Scene office1 A2/A3/A5/G3 are blocked because every preselected FC mask has empty support at the original dense resolution. Their full Replica pools and dependent counts/times are unavailable (--), not zero. All eight scene positions and the shared candidate denominator remain fixed. No seven-scene pool or mean replaces the required complete cohort.

| Cohort | Method | APall (%) | AP50 (%) | AP25 (%) | mIoU (%) | mAcc (%) |
| --- | --- | --- | --- | --- | --- | --- |
| replica8 | CT_A0_NATIVE | 8.63 | 21.48 | 34.59 | 27.24 | 32.67 |
| replica8 | CT_A1_E | 11.74 | 24.50 | 37.97 | 29.69 | 37.43 |
| replica8 | CT_A2_R | -- | -- | -- | -- | -- |
| replica8 | CT_A3_ER | -- | -- | -- | -- | -- |
| replica8 | CT_A4_NATIVE_RECOVERY | 11.74 | 24.50 | 38.15 | 29.82 | 37.86 |
| replica8 | CT_A5_FC_ONLY | -- | -- | -- | -- | -- |
| replica8 | CT_H_U2 | 12.39 | 26.06 | 39.84 | 30.17 | 38.15 |
| replica8 | CT_G3 | -- | -- | -- | -- | -- |
| scannet_cf18 | CT_A0_NATIVE | 7.18 | 15.72 | 21.21 | 16.38 | 25.36 |
| scannet_cf18 | CT_A1_E | 8.31 | 17.96 | 24.98 | 19.00 | 28.38 |
| scannet_cf18 | CT_A2_R | 7.12 | 15.61 | 21.10 | 16.42 | 25.46 |
| scannet_cf18 | CT_A3_ER | 8.23 | 17.84 | 24.77 | 19.03 | 28.48 |
| scannet_cf18 | CT_A4_NATIVE_RECOVERY | 8.30 | 17.93 | 24.92 | 18.99 | 28.44 |
| scannet_cf18 | CT_A5_FC_ONLY | 8.34 | 17.47 | 23.88 | 17.97 | 24.00 |

## Paired Effects

| Cohort | Contrast (pp) | apall | ap50 | ap25 | miou | macc |
| --- | --- | --- | --- | --- | --- | --- |
| replica8 | A3_vs_A0_pp | -- | -- | -- | -- | -- |
| replica8 | A3_vs_A1_pp | -- | -- | -- | -- | -- |
| replica8 | A3_vs_A4_pp | -- | -- | -- | -- | -- |
| replica8 | A3_vs_A5_pp | -- | -- | -- | -- | -- |
| replica8 | E_only_pp | +3.11 | +3.02 | +3.39 | +2.45 | +4.76 |
| replica8 | R_only_pp | -- | -- | -- | -- | -- |
| replica8 | E_given_R_pp | -- | -- | -- | -- | -- |
| replica8 | R_given_E_pp | -- | -- | -- | -- | -- |
| replica8 | interaction_pp | -- | -- | -- | -- | -- |
| scannet_cf18 | A3_vs_A0_pp | +1.05 | +2.12 | +3.56 | +2.64 | +3.12 |
| scannet_cf18 | A3_vs_A1_pp | -0.08 | -0.12 | -0.21 | +0.03 | +0.10 |
| scannet_cf18 | A3_vs_A4_pp | -0.07 | -0.10 | -0.15 | +0.04 | +0.04 |
| scannet_cf18 | A3_vs_A5_pp | -0.11 | +0.37 | +0.89 | +1.05 | +4.48 |
| scannet_cf18 | E_only_pp | +1.13 | +2.24 | +3.78 | +2.62 | +3.02 |
| scannet_cf18 | R_only_pp | -0.07 | -0.11 | -0.10 | +0.04 | +0.10 |
| scannet_cf18 | E_given_R_pp | +1.12 | +2.22 | +3.67 | +2.60 | +3.02 |
| scannet_cf18 | R_given_E_pp | -0.08 | -0.12 | -0.21 | +0.03 | +0.10 |
| scannet_cf18 | interaction_pp | -0.01 | -0.02 | -0.11 | -0.01 | -0.00 |

These are raw differences in percentage points. A3 remains the fixed full method even when a control wins. The E x R interaction is A3-A2-A1+A0; undefined values remain --.

| Cohort | APall > A1 | AP50 drop <= 0.10pp | mIoU drop <= 0.10pp | A3 dominated by |
| --- | --- | --- | --- | --- |
| replica8 | -- | -- | -- | -- |
| scannet_cf18 | False | False | True | none |

Flags are engineering preferences, not significance tests or selection gates. A listed dominator is a simple control with no lower APall/AP50/mIoU and at least one strictly higher metric.

## Recovery Evidence

| Arm | Source-available n/N | Added TP50 | Added FP50 | Ambiguous TP/FP | Seconds/scene |
| --- | --- | --- | --- | --- | --- |
| G1 | -- | -- | -- | -- | -- |
| G3 | -- | -- | -- | -- | -- |
| NONE | 0/53 | 0 | 0 | 0/0 | 0 (by definition) |
| U2 | 9/53 | 3 | 2 | 0/0 | 6.61 |

TP/FP count released matcher score entries, not unique objects. Ambiguous old/new score ties are separate. n/N includes source-available target-small owners and never implies true-positive coverage. Available cold means require all eight actual scene calls, including real empty-view overhead. Model loading, mapping and evaluation are excluded and reported separately.

Model loading total: 7.50 seconds, excluded from all means. GPU: `NVIDIA A40`, compute capability `8.6`, physical UUID `GPU-41ba4cb6-edfe-ed30-c047-a19d9f23d0a8`.

| Comparison | Delta n | Delta TP50 | Delta FP50 | Delta APall (pp) | Delta mIoU (pp) | Delta seconds |
| --- | --- | --- | --- | --- | --- | --- |
| G1_vs_U2 | -- | -- | -- | -- | -- | -- |
| G3_vs_G1 | -- | -- | -- | -- | -- | -- |

G3's additional cost is assessed against its displayed full-map gains and ambiguity, not its recovered count alone.

## Protocol And Boundaries

The common geometry is BB00_NATIVE with the pinned original extension, 0.01m voxels, association4, CropFormer single-label masks and exactly200 original schedule slots without refill. The paper text states0.1m voxels; this reproduction follows the pinned released-code0.01m configuration. Whole predicted meshes use exact Open3D FP32 1NN and strict squared distance<0.05 squared. Positive projected owners need100 target points for instance export; confidence is current-class area divided by the maximum same-class area, written to six decimals. Semantic unknown0 errors remain counted. APall retains the actual original .50--.90 overlap vector, plus separate AP25. Full valid-ID orders, individual scenes/classes, lost old matches, rank changes and failures are in the SI.

External rows: [OVI-MAP final PDF, Table 3](https://openaccess.thecvf.com/content/CVPR2026/papers/Deng_OVI-MAP_Open-Vocabulary_Instance-Semantic_Mapping_CVPR_2026_paper.pdf), also checked against [arXiv v1 PDF](https://arxiv.org/pdf/2603.26541v1). 7 differences from the supplied HTML transcription are preserved in the provenance audit. These are author-reported context, not rerun systems; their raw scorer, rank protocol and ScanNet sequence membership are not independently verified.

Replica is historically exposed. CF18 is18 captures from7 physical ScanNet families, not18 independent rooms or the full ScanNet200 validation set. Known Q fit/calibration families do not overlap CF18, but unrecorded benchmark exposure is unverified. Q and the calibrated N/Q/F scalars are fitted components, frozen for this experiment; no new fit was performed. No untouched-generalization, all-components-training-free, statistical-significance, global SOTA, real-time or end-to-end FPS claim follows.

## Dependency And Paid Costs

| Cohort | Method | Native crop inputs | FC image inputs | FC selected masks | Failed requests | Recorded/required scenes |
| --- | --- | --- | --- | --- | --- | --- |
| replica8 | CT_A0_NATIVE | 25530 | 0 | 0 | 0 | 8/8 |
| replica8 | CT_A1_E | 31854 | 673 | 1067 | 15 | 8/8 |
| replica8 | CT_A2_R | -- | -- | -- | -- | 7/8 |
| replica8 | CT_A3_ER | -- | -- | -- | -- | 7/8 |
| replica8 | CT_A4_NATIVE_RECOVERY | 32106 | 673 | 1067 | 15 | 8/8 |
| replica8 | CT_A5_FC_ONLY | -- | -- | -- | -- | 7/8 |
| replica8 | CT_H_U2 | 31854 | 674 | 1078 | 17 | 8/8 |
| replica8 | CT_G3 | -- | -- | -- | -- | 7/8 |
| scannet_cf18 | CT_A0_NATIVE | 113940 | 0 | 0 | 0 | 18/18 |
| scannet_cf18 | CT_A1_E | 121848 | 2250 | 4700 | 11 | 18/18 |
| scannet_cf18 | CT_A2_R | 113940 | 124 | 132 | 20 | 18/18 |
| scannet_cf18 | CT_A3_ER | 121848 | 2269 | 4832 | 31 | 18/18 |
| scannet_cf18 | CT_A4_NATIVE_RECOVERY | 122640 | 2250 | 4700 | 11 | 18/18 |
| scannet_cf18 | CT_A5_FC_ONLY | 113940 | 2269 | 4832 | 31 | 18/18 |

| Cost origin | Native crop inputs | FC image inputs | CropFormer image inputs | FC pooling calls |
| --- | --- | --- | --- | --- |
| current_task_warm | 122892.00 | 2293.00 | 3600.00 | 4864.00 |
| current_task_cold | 0.00 | 134.00 | 0.00 | 94.00 |
| historical_parent | 25326.00 | 412.00 | 0.00 | 881.00 |

Complete method budgets sum independent per-scene content unions. Incomplete full-cohort budgets are --; their known per-scene costs remain in the SI with explicit blocked dependencies. Shared worker payments are counted once, with historical parent, current acquisition and actual cold replays separate. Selected mask counts include failed requests and are not physical pooling calls. Native-painted support is a required A5 prerequisite. These are input budgets, not measured whole-pipeline standalone wall times. Physical-cost status: COMPLETE. Unknown paid counters are --; known lower bounds, all source hashes and retained failed receipts are in tables/costs.json.

## Same-View Common-Success Diagnostic

| Scene | Status | Both successful | Labels agree | FC-only successful | Native-only successful |
| --- | --- | --- | --- | --- | --- |
| office0 | COMPLETE | 1 | 0 | 0 | 2 |
| office1 | BLOCKED_TECHNICAL | -- | -- | -- | -- |
| office2 | COMPLETE | 4 | 1 | 0 | 3 |
| office3 | COMPLETE | 3 | 2 | 0 | 5 |
| office4 | COMPLETE | 3 | 0 | 0 | 0 |
| room0 | COMPLETE | 3 | 0 | 0 | 5 |
| room1 | COMPLETE | 3 | 0 | 0 | 2 |
| room2 | COMPLETE | 5 | 1 | 0 | 2 |
| scene0011_00 | COMPLETE | 2 | 0 | 0 | 1 |
| scene0011_01 | COMPLETE | 5 | 0 | 0 | 2 |
| scene0050_00 | COMPLETE | 4 | 0 | 0 | 0 |
| scene0050_01 | COMPLETE | 11 | 0 | 0 | 0 |
| scene0050_02 | COMPLETE | 5 | 1 | 0 | 1 |
| scene0084_00 | COMPLETE | 0 | 0 | 0 | 0 |
| scene0084_01 | COMPLETE | 1 | 0 | 0 | 1 |
| scene0084_02 | COMPLETE | 1 | 0 | 0 | 0 |
| scene0168_00 | COMPLETE | 8 | 1 | 0 | 2 |
| scene0168_01 | COMPLETE | 2 | 0 | 0 | 2 |
| scene0168_02 | COMPLETE | 4 | 0 | 0 | 1 |
| scene0231_00 | COMPLETE | 23 | 1 | 0 | 3 |
| scene0231_01 | COMPLETE | 17 | 0 | 0 | 4 |
| scene0231_02 | COMPLETE | 15 | 0 | 0 | 0 |
| scene0378_00 | COMPLETE | 3 | 0 | 0 | 1 |
| scene0378_01 | COMPLETE | 4 | 0 | 0 | 1 |
| scene0378_02 | COMPLETE | 2 | 0 | 0 | 0 |
| scene0518_00 | COMPLETE | 5 | 0 | 0 | 1 |

This post-lock A3/A4 diagnostic uses the same selected G1 frame/mask and saved scores only. Individual scores and class orders are retained in diagnostics/<scene>.json. FC and Native scores belong to different encoders and are not numerically comparable. Label agreement is not accuracy; success-set differences also affect the full-map A3/A4 comparison. No new inference or benchmark row was added.
