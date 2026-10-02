# Recovery Wave 2 Results

Decision: **RETAIN_BASELINE**. Deployment remains **N0_UNCHANGED**. Frozen light: `RW_B_D2`; map nominee: `None`.

Values are percentages; paired deltas are percentage points against each cohort's exact BB00_NATIVE + D2. Development uses the original four scenes; all eight Replica scenes were already exposed. APall uses the actual released .50-.90 overlap vector. Ordered released pooling and summed confusion matrices are used.

Map nomination uses fixed D2 only. Gains in an alternate standard readout are descriptive and cannot nominate a map. Replica gains are regression measurements and cannot replace the committed development choice.

## Table 1: Official Metrics and Coverage

| Cohort | Map / Method | Status / Coverage | Scientific status | APall | AP50 | AP25 | mIoU | mAcc | Delta APall pp | Delta AP50 pp | Delta AP25 pp | Delta mIoU pp | Delta mAcc pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | BB00_NATIVE / RW_B_D2 | COMPLETE 4/4 | NO_NET_GAIN | 12.5818 | 22.5994 | 31.7097 | 23.5623 | 32.4171 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| development | BB00_NATIVE / RW_B_FCEQ | COMPLETE 4/4 | TRADEOFF | 11.2617 | 21.1860 | 32.8271 | 23.9769 | 33.9559 | -1.3201 | -1.4134 | 1.1174 | 0.4146 | 1.5388 |
| development | BB00_NATIVE / RW_W040 | COMPLETE 4/4 | NO_NET_GAIN | 11.7223 | 21.1448 | 30.7573 | 22.3533 | 31.4529 | -0.8595 | -1.4547 | -0.9523 | -1.2090 | -0.9642 |
| development | BB00_NATIVE / RW_W045 | COMPLETE 4/4 | NO_NET_GAIN | 11.7088 | 21.1709 | 30.2811 | 22.0889 | 31.1755 | -0.8730 | -1.4286 | -1.4286 | -1.4733 | -1.2416 |
| development | BB00_NATIVE / RW_U1_NATIVE_SINGLE | COMPLETE 4/4 | NO_NET_GAIN | 12.5818 | 22.5994 | 31.7097 | 23.5566 | 32.4171 | 0.0000 | 0.0000 | 0.0000 | -0.0057 | 0.0000 |
| development | BB00_NATIVE / RW_UQ_PAID_QUERY | COMPLETE 4/4 | NO_NET_GAIN | 12.5818 | 22.5994 | 31.7097 | 23.5623 | 32.4171 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| development | BB00_NATIVE / RW_U2_FC_MATCHED_SINGLE | COMPLETE 4/4 | TRADEOFF | 12.5818 | 22.5994 | 31.6989 | 23.5661 | 32.4400 | 0.0000 | 0.0000 | -0.0108 | 0.0038 | 0.0229 |
| development | BB00_NATIVE / RW_U3_FC_CAPTURED | COMPLETE 4/4 | TRADEOFF | 12.5818 | 22.5994 | 31.6989 | 23.5661 | 32.4400 | 0.0000 | 0.0000 | -0.0108 | 0.0038 | 0.0229 |
| development | BB00_NATIVE / RW_LIGHT_COMBO | COMPLETE 4/4 | NO_NET_GAIN | 12.5818 | 22.5994 | 31.7097 | 23.5623 | 32.4171 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| development | BB00_NATIVE / NATIVE_READOUT | COMPLETE 4/4 | TRADEOFF | 11.2142 | 23.2739 | 33.8526 | 24.1889 | 34.3402 | -1.3675 | 0.6745 | 2.1429 | 0.6266 | 1.9231 |
| development | RW_A1_NATIVE_FALLBACK / NATIVE_READOUT | COMPLETE 4/4 | TRADEOFF | 10.6478 | 24.2779 | 37.0477 | 25.8386 | 33.4634 | -1.9340 | 1.6784 | 5.3381 | 2.2763 | 1.0463 |
| development | RW_A1_NATIVE_FALLBACK / FC_EQ | COMPLETE 4/4 | TRADEOFF | 10.9540 | 23.0235 | 37.2944 | 26.0277 | 34.5050 | -1.6277 | 0.4241 | 5.5847 | 2.4654 | 2.0879 |
| development | RW_A1_NATIVE_FALLBACK / D2 | COMPLETE 4/4 | TRADEOFF | 11.0744 | 23.7344 | 35.5839 | 25.2214 | 31.8513 | -1.5074 | 1.1349 | 3.8742 | 1.6591 | -0.5658 |
| development | RW_A2_MULTI_FREE / NATIVE_READOUT | COMPLETE 4/4 | NET_GAIN_WITH_GUARDRAILS | 12.8184 | 25.4903 | 36.1077 | 25.3006 | 31.4494 | 0.2366 | 2.8909 | 4.3980 | 1.7383 | -0.9677 |
| development | RW_A2_MULTI_FREE / FC_EQ | COMPLETE 4/4 | TRADEOFF | 12.4075 | 26.2225 | 35.8523 | 25.6538 | 32.9578 | -0.1743 | 3.6230 | 4.1427 | 2.0915 | 0.5407 |
| development | RW_A2_MULTI_FREE / D2 | COMPLETE 4/4 | TRADEOFF | 11.9978 | 23.1518 | 32.4439 | 24.0461 | 29.9809 | -0.5839 | 0.5524 | 0.7342 | 0.4838 | -2.4362 |
| development | RW_A3_MULTI_UNION / NATIVE_READOUT | COMPLETE 4/4 | TRADEOFF | 10.5422 | 23.1214 | 34.7595 | 23.8852 | 31.2772 | -2.0396 | 0.5220 | 3.0498 | 0.3229 | -1.1399 |
| development | RW_A3_MULTI_UNION / FC_EQ | COMPLETE 4/4 | TRADEOFF | 10.1120 | 20.6426 | 31.7111 | 22.8402 | 31.6623 | -2.4698 | -1.9568 | 0.0014 | -0.7221 | -0.7548 |
| development | RW_A3_MULTI_UNION / D2 | COMPLETE 4/4 | TRADEOFF | 12.1456 | 22.5169 | 32.3526 | 23.7112 | 29.3952 | -0.4362 | -0.0825 | 0.6429 | 0.1490 | -3.0220 |
| development | RW_S1_CROP_PRIORITY / NATIVE_READOUT | COMPLETE 4/4 | TRADEOFF | 11.2005 | 23.2739 | 34.1342 | 24.2122 | 34.4046 | -1.3813 | 0.6745 | 2.4245 | 0.6499 | 1.9875 |
| development | RW_S1_CROP_PRIORITY / FC_EQ | COMPLETE 4/4 | TRADEOFF | 11.6179 | 23.5669 | 32.8476 | 24.1063 | 34.3293 | -0.9639 | 0.9675 | 1.1379 | 0.5440 | 1.9122 |
| development | RW_S1_CROP_PRIORITY / D2 | COMPLETE 4/4 | TRADEOFF | 12.6920 | 22.5994 | 31.6751 | 23.5242 | 32.4348 | 0.1102 | 0.0000 | -0.0346 | -0.0381 | 0.0177 |
| development | RW_S2_CONFLICT / NATIVE_READOUT | COMPLETE 4/4 | TRADEOFF | 10.8976 | 23.2782 | 34.1334 | 24.1647 | 34.2572 | -1.6842 | 0.6787 | 2.4237 | 0.6024 | 1.8401 |
| development | RW_S2_CONFLICT / FC_EQ | COMPLETE 4/4 | TRADEOFF | 11.6321 | 23.5669 | 32.8476 | 24.0435 | 34.1829 | -0.9497 | 0.9675 | 1.1379 | 0.4812 | 1.7658 |
| development | RW_S2_CONFLICT / D2 | COMPLETE 4/4 | NO_NET_GAIN | 12.5788 | 22.5724 | 31.6751 | 23.4740 | 32.2882 | -0.0030 | -0.0270 | -0.0346 | -0.0883 | -0.1289 |
| replica | BB00_NATIVE / RW_B_D2 | COMPLETE 8/8 | NO_NET_GAIN | 11.7413 | 24.4952 | 37.9746 | 29.6911 | 37.4279 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| replica | BB00_NATIVE / RW_B_FCEQ | COMPLETE 8/8 | TRADEOFF | 10.4068 | 24.3231 | 37.7285 | 29.9804 | 36.6250 | -1.3346 | -0.1721 | -0.2461 | 0.2893 | -0.8029 |
| replica | BB00_NATIVE / RW_W040 | COMPLETE 8/8 | NO_NET_GAIN | 11.8827 | 25.9053 | 39.4395 | 30.6432 | 38.0399 | 0.1413 | 1.4100 | 1.4649 | 0.9521 | 0.6121 |
| replica | BB00_NATIVE / RW_W045 | COMPLETE 8/8 | TRADEOFF | 11.7151 | 24.4518 | 37.9860 | 29.4365 | 37.0621 | -0.0263 | -0.0434 | 0.0114 | -0.2546 | -0.3658 |
| replica | BB00_NATIVE / RW_U1_NATIVE_SINGLE | COMPLETE 8/8 | TRADEOFF | 11.7413 | 24.4952 | 38.1466 | 29.6586 | 37.5643 | 0.0000 | 0.0000 | 0.1721 | -0.0325 | 0.1364 |
| replica | BB00_NATIVE / RW_UQ_PAID_QUERY | COMPLETE 8/8 | TRADEOFF | 11.7413 | 24.4952 | 38.1466 | 29.6597 | 37.5641 | 0.0000 | 0.0000 | 0.1721 | -0.0314 | 0.1363 |
| replica | BB00_NATIVE / RW_U2_FC_MATCHED_SINGLE | COMPLETE 8/8 | NET_GAIN_WITH_GUARDRAILS | 12.3864 | 26.0571 | 39.8413 | 30.1652 | 38.1484 | 0.6450 | 1.5619 | 1.8667 | 0.4741 | 0.7205 |
| replica | BB00_NATIVE / RW_U3_FC_CAPTURED | COMPLETE 8/8 | NET_GAIN_WITH_GUARDRAILS | 12.3864 | 26.0571 | 39.8413 | 30.1652 | 38.1484 | 0.6450 | 1.5619 | 1.8667 | 0.4741 | 0.7205 |
| replica | BB00_NATIVE / NATIVE_READOUT | COMPLETE 8/8 | NO_NET_GAIN | 8.6265 | 21.4769 | 34.5854 | 27.2402 | 32.6695 | -3.1148 | -3.0183 | -3.3892 | -2.4509 | -4.7584 |
| replica | RW_A1_NATIVE_FALLBACK / NATIVE_READOUT | NOT_RUN_NOT_FROZEN_NOMINEE 0/8 | INCONCLUSIVE | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| replica | RW_A1_NATIVE_FALLBACK / FC_EQ | NOT_RUN_NOT_FROZEN_NOMINEE 0/8 | INCONCLUSIVE | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| replica | RW_A1_NATIVE_FALLBACK / D2 | NOT_RUN_NOT_FROZEN_NOMINEE 0/8 | INCONCLUSIVE | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| replica | RW_A2_MULTI_FREE / NATIVE_READOUT | NOT_RUN_NOT_FROZEN_NOMINEE 0/8 | INCONCLUSIVE | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| replica | RW_A2_MULTI_FREE / FC_EQ | NOT_RUN_NOT_FROZEN_NOMINEE 0/8 | INCONCLUSIVE | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| replica | RW_A2_MULTI_FREE / D2 | NOT_RUN_NOT_FROZEN_NOMINEE 0/8 | INCONCLUSIVE | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| replica | RW_A3_MULTI_UNION / NATIVE_READOUT | NOT_RUN_NOT_FROZEN_NOMINEE 0/8 | INCONCLUSIVE | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| replica | RW_A3_MULTI_UNION / FC_EQ | NOT_RUN_NOT_FROZEN_NOMINEE 0/8 | INCONCLUSIVE | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| replica | RW_A3_MULTI_UNION / D2 | NOT_RUN_NOT_FROZEN_NOMINEE 0/8 | INCONCLUSIVE | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| replica | RW_S1_CROP_PRIORITY / NATIVE_READOUT | NOT_RUN_NOT_FROZEN_NOMINEE 0/8 | INCONCLUSIVE | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| replica | RW_S1_CROP_PRIORITY / FC_EQ | NOT_RUN_NOT_FROZEN_NOMINEE 0/8 | INCONCLUSIVE | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| replica | RW_S1_CROP_PRIORITY / D2 | NOT_RUN_NOT_FROZEN_NOMINEE 0/8 | INCONCLUSIVE | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| replica | RW_S2_CONFLICT / NATIVE_READOUT | NOT_RUN_NOT_FROZEN_NOMINEE 0/8 | INCONCLUSIVE | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| replica | RW_S2_CONFLICT / FC_EQ | NOT_RUN_NOT_FROZEN_NOMINEE 0/8 | INCONCLUSIVE | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| replica | RW_S2_CONFLICT / D2 | NOT_RUN_NOT_FROZEN_NOMINEE 0/8 | INCONCLUSIVE | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

## Raw Geometry Screen

| Arm | Status | Best IoU drop pp | R50 count loss | Fragment ratio | Semantic status |
| --- | --- | --- | --- | --- | --- |
| RW_A1_NATIVE_FALLBACK | GEOMETRY_PASS | -0.6949 | -2 | 1.0000 | COMPLETE |
| RW_A2_MULTI_FREE | GEOMETRY_PASS | 3.0879 | 1 | 0.8406 | COMPLETE |
| RW_A3_MULTI_UNION | GEOMETRY_PASS | 3.0487 | 1 | 0.8924 | COMPLETE |
| RW_S1_CROP_PRIORITY | GEOMETRY_PASS | 0.0679 | -1 | 1.0120 | COMPLETE |
| RW_S2_CONFLICT | GEOMETRY_PASS | 0.0679 | -1 | 1.0120 | COMPLETE |

## Table 2: Recovery Funnel

| Scene | Arm | RAW | Painted | Candidates | U1 eligible | Legal views | Source available | Used requests | Added owners | Target min100 | Cap exclusions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scene0056_00 | RW_U1_NATIVE_SINGLE | 107 | 99 | 2 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| scene0056_00 | RW_UQ_PAID_QUERY | 107 | 99 | 2 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| scene0056_00 | RW_U2_FC_MATCHED_SINGLE | 107 | 99 | 2 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| scene0056_00 | RW_U3_FC_CAPTURED | 107 | 99 | 2 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| scene0534_00 | RW_U1_NATIVE_SINGLE | 110 | 93 | 12 | 9 | 9 | 9 | 9 | 9 | 4 | 0 |
| scene0534_00 | RW_UQ_PAID_QUERY | 110 | 93 | 12 | 9 | 9 | 3 | 3 | 3 | 0 | 0 |
| scene0534_00 | RW_U2_FC_MATCHED_SINGLE | 110 | 93 | 12 | 9 | 9 | 9 | 9 | 9 | 4 | 0 |
| scene0534_00 | RW_U3_FC_CAPTURED | 110 | 93 | 12 | 9 | 9 | 9 | 9 | 9 | 4 | 0 |
| scene0445_00 | RW_U1_NATIVE_SINGLE | 16 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0445_00 | RW_UQ_PAID_QUERY | 16 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0445_00 | RW_U2_FC_MATCHED_SINGLE | 16 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0445_00 | RW_U3_FC_CAPTURED | 16 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0626_00 | RW_U1_NATIVE_SINGLE | 31 | 29 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0626_00 | RW_UQ_PAID_QUERY | 31 | 29 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0626_00 | RW_U2_FC_MATCHED_SINGLE | 31 | 29 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0626_00 | RW_U3_FC_CAPTURED | 31 | 29 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office0 | RW_U1_NATIVE_SINGLE | 59 | 45 | 4 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| office0 | RW_UQ_PAID_QUERY | 59 | 45 | 4 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| office0 | RW_U2_FC_MATCHED_SINGLE | 59 | 45 | 4 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| office0 | RW_U3_FC_CAPTURED | 59 | 45 | 4 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| office1 | RW_U1_NATIVE_SINGLE | 40 | 35 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office1 | RW_UQ_PAID_QUERY | 40 | 35 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office1 | RW_U2_FC_MATCHED_SINGLE | 40 | 35 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office1 | RW_U3_FC_CAPTURED | 40 | 35 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office2 | RW_U1_NATIVE_SINGLE | 64 | 51 | 9 | 3 | 4 | 3 | 3 | 3 | 2 | 0 |
| office2 | RW_UQ_PAID_QUERY | 64 | 51 | 9 | 3 | 4 | 2 | 2 | 2 | 1 | 0 |
| office2 | RW_U2_FC_MATCHED_SINGLE | 64 | 51 | 9 | 3 | 4 | 2 | 2 | 2 | 1 | 0 |
| office2 | RW_U3_FC_CAPTURED | 64 | 51 | 9 | 3 | 4 | 2 | 3 | 2 | 1 | 0 |
| office3 | RW_U1_NATIVE_SINGLE | 64 | 51 | 8 | 2 | 2 | 2 | 2 | 2 | 2 | 0 |
| office3 | RW_UQ_PAID_QUERY | 64 | 51 | 8 | 2 | 2 | 2 | 2 | 2 | 2 | 0 |
| office3 | RW_U2_FC_MATCHED_SINGLE | 64 | 51 | 8 | 2 | 2 | 2 | 2 | 2 | 2 | 0 |
| office3 | RW_U3_FC_CAPTURED | 64 | 51 | 8 | 2 | 2 | 2 | 2 | 2 | 2 | 0 |
| office4 | RW_U1_NATIVE_SINGLE | 48 | 43 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office4 | RW_UQ_PAID_QUERY | 48 | 43 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office4 | RW_U2_FC_MATCHED_SINGLE | 48 | 43 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office4 | RW_U3_FC_CAPTURED | 48 | 43 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| room0 | RW_U1_NATIVE_SINGLE | 76 | 54 | 14 | 2 | 2 | 2 | 2 | 2 | 1 | 0 |
| room0 | RW_UQ_PAID_QUERY | 76 | 54 | 14 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| room0 | RW_U2_FC_MATCHED_SINGLE | 76 | 54 | 14 | 2 | 2 | 1 | 1 | 1 | 0 | 0 |
| room0 | RW_U3_FC_CAPTURED | 76 | 54 | 14 | 2 | 2 | 1 | 1 | 1 | 0 | 0 |
| room1 | RW_U1_NATIVE_SINGLE | 53 | 47 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| room1 | RW_UQ_PAID_QUERY | 53 | 47 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| room1 | RW_U2_FC_MATCHED_SINGLE | 53 | 47 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| room1 | RW_U3_FC_CAPTURED | 53 | 47 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| room2 | RW_U1_NATIVE_SINGLE | 64 | 49 | 9 | 3 | 5 | 3 | 3 | 3 | 2 | 0 |
| room2 | RW_UQ_PAID_QUERY | 64 | 49 | 9 | 3 | 5 | 3 | 3 | 3 | 2 | 0 |
| room2 | RW_U2_FC_MATCHED_SINGLE | 64 | 49 | 9 | 3 | 5 | 3 | 3 | 3 | 2 | 0 |
| room2 | RW_U3_FC_CAPTURED | 64 | 49 | 9 | 3 | 5 | 3 | 5 | 3 | 2 | 0 |

Source min100 rows and target min100 points are separate. Candidate clipping retains the exact observed mask; cached pooling was not performed on the smaller residual support.

## Table 3: Actual Matcher and Semantic Changes

| Scene | Method | Added TP50 | Added FP50 | Ambiguous TP/FP50 | Lost old TP entries all overlaps | Old TP displaced by added | Old rank changes | Old class changes | Changed confusion cells | Correct-point delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scene0056_00 | RW_B_D2 | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0056_00 | RW_B_FCEQ | 0 | 0 | 0/0 | 0 | 0 | 20 | 13 | 43 | 1267 |
| scene0056_00 | RW_U1_NATIVE_SINGLE | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0056_00 | RW_U2_FC_MATCHED_SINGLE | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0056_00 | RW_U3_FC_CAPTURED | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0056_00 | RW_UQ_PAID_QUERY | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0056_00 | RW_W040 | 0 | 0 | 0/0 | 0 | 0 | 13 | 9 | 30 | 5774 |
| scene0056_00 | RW_W045 | 0 | 0 | 0/0 | 0 | 0 | 9 | 5 | 23 | 1587 |
| scene0534_00 | RW_B_D2 | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0534_00 | RW_B_FCEQ | 0 | 0 | 0/0 | 0 | 0 | 15 | 9 | 45 | 646 |
| scene0534_00 | RW_U1_NATIVE_SINGLE | 0 | 4 | 0/0 | 0 | 0 | 0 | 0 | 10 | 0 |
| scene0534_00 | RW_U2_FC_MATCHED_SINGLE | 0 | 4 | 0/0 | 0 | 0 | 0 | 0 | 12 | 64 |
| scene0534_00 | RW_U3_FC_CAPTURED | 0 | 4 | 0/0 | 0 | 0 | 0 | 0 | 12 | 64 |
| scene0534_00 | RW_UQ_PAID_QUERY | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 2 | 0 |
| scene0534_00 | RW_W040 | 0 | 0 | 0/0 | 0 | 0 | 3 | 3 | 10 | 0 |
| scene0534_00 | RW_W045 | 0 | 0 | 0/0 | 0 | 0 | 0 | 2 | 4 | 0 |
| scene0445_00 | RW_B_D2 | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0445_00 | RW_B_FCEQ | 0 | 0 | 0/0 | 8 | 0 | 0 | 1 | 6 | -1772 |
| scene0445_00 | RW_U1_NATIVE_SINGLE | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0445_00 | RW_U2_FC_MATCHED_SINGLE | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0445_00 | RW_U3_FC_CAPTURED | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0445_00 | RW_UQ_PAID_QUERY | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0445_00 | RW_W040 | 0 | 0 | 0/0 | 8 | 0 | 0 | 1 | 6 | -1772 |
| scene0445_00 | RW_W045 | 0 | 0 | 0/0 | 8 | 0 | 0 | 1 | 6 | -1772 |
| scene0626_00 | RW_B_D2 | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0626_00 | RW_B_FCEQ | 0 | 0 | 0/0 | 0 | 0 | 0 | 2 | 8 | -377 |
| scene0626_00 | RW_U1_NATIVE_SINGLE | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0626_00 | RW_U2_FC_MATCHED_SINGLE | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0626_00 | RW_U3_FC_CAPTURED | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0626_00 | RW_UQ_PAID_QUERY | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scene0626_00 | RW_W040 | 0 | 0 | 0/0 | 0 | 0 | 0 | 2 | 8 | -377 |
| scene0626_00 | RW_W045 | 0 | 0 | 0/0 | 0 | 0 | 0 | 1 | 4 | -377 |
| office0 | RW_B_D2 | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office0 | RW_B_FCEQ | 0 | 0 | 0/0 | 8 | 0 | 9 | 7 | 48 | -66663 |
| office0 | RW_U1_NATIVE_SINGLE | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 2 | 0 |
| office0 | RW_U2_FC_MATCHED_SINGLE | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 2 | 33 |
| office0 | RW_U3_FC_CAPTURED | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 2 | 33 |
| office0 | RW_UQ_PAID_QUERY | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office0 | RW_W040 | 0 | 0 | 0/0 | 0 | 0 | 1 | 4 | 12 | -60 |
| office0 | RW_W045 | 0 | 0 | 0/0 | 0 | 0 | 1 | 4 | 12 | -60 |
| office1 | RW_B_D2 | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office1 | RW_B_FCEQ | 0 | 0 | 0/0 | 0 | 0 | 5 | 3 | 16 | 11034 |
| office1 | RW_U1_NATIVE_SINGLE | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office1 | RW_U2_FC_MATCHED_SINGLE | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office1 | RW_U3_FC_CAPTURED | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office1 | RW_UQ_PAID_QUERY | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office1 | RW_W040 | 0 | 0 | 0/0 | 0 | 0 | 2 | 2 | 14 | 11034 |
| office1 | RW_W045 | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office2 | RW_B_D2 | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office2 | RW_B_FCEQ | 0 | 0 | 0/0 | 0 | 0 | 6 | 6 | 18 | 3693 |
| office2 | RW_U1_NATIVE_SINGLE | 0 | 1 | 0/0 | 0 | 0 | 0 | 0 | 7 | 86 |
| office2 | RW_U2_FC_MATCHED_SINGLE | 0 | 1 | 0/0 | 0 | 0 | 1 | 0 | 6 | 1611 |
| office2 | RW_U3_FC_CAPTURED | 0 | 1 | 0/0 | 0 | 0 | 1 | 0 | 6 | 1611 |
| office2 | RW_UQ_PAID_QUERY | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 6 | 0 |
| office2 | RW_W040 | 0 | 0 | 0/0 | 0 | 0 | 2 | 2 | 6 | 3693 |
| office2 | RW_W045 | 0 | 0 | 0/0 | 0 | 0 | 2 | 2 | 6 | 3693 |
| office3 | RW_B_D2 | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office3 | RW_B_FCEQ | 0 | 0 | 0/0 | 6 | 0 | 7 | 4 | 14 | -4222 |
| office3 | RW_U1_NATIVE_SINGLE | 0 | 2 | 0/0 | 0 | 0 | 0 | 0 | 6 | 128 |
| office3 | RW_U2_FC_MATCHED_SINGLE | 1 | 1 | 0/0 | 0 | 0 | 0 | 0 | 6 | 230 |
| office3 | RW_U3_FC_CAPTURED | 1 | 1 | 0/0 | 0 | 0 | 0 | 0 | 6 | 230 |
| office3 | RW_UQ_PAID_QUERY | 0 | 2 | 0/0 | 0 | 0 | 0 | 0 | 6 | 128 |
| office3 | RW_W040 | 0 | 0 | 0/0 | 6 | 0 | 6 | 3 | 12 | -4222 |
| office3 | RW_W045 | 0 | 0 | 0/0 | 1 | 0 | 1 | 1 | 6 | -166 |
| office4 | RW_B_D2 | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office4 | RW_B_FCEQ | 0 | 0 | 0/0 | 0 | 0 | 2 | 2 | 6 | 0 |
| office4 | RW_U1_NATIVE_SINGLE | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office4 | RW_U2_FC_MATCHED_SINGLE | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office4 | RW_U3_FC_CAPTURED | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office4 | RW_UQ_PAID_QUERY | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office4 | RW_W040 | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| office4 | RW_W045 | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| room0 | RW_B_D2 | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| room0 | RW_B_FCEQ | 0 | 0 | 0/0 | 0 | 0 | 5 | 3 | 8 | -501 |
| room0 | RW_U1_NATIVE_SINGLE | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 4 | 0 |
| room0 | RW_U2_FC_MATCHED_SINGLE | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 4 | 0 |
| room0 | RW_U3_FC_CAPTURED | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 4 | 0 |
| room0 | RW_UQ_PAID_QUERY | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| room0 | RW_W040 | 0 | 0 | 0/0 | 0 | 0 | 2 | 2 | 6 | -506 |
| room0 | RW_W045 | 0 | 0 | 0/0 | 0 | 0 | 1 | 1 | 2 | 0 |
| room1 | RW_B_D2 | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| room1 | RW_B_FCEQ | 0 | 0 | 0/0 | 17 | 0 | 7 | 7 | 49 | 182426 |
| room1 | RW_U1_NATIVE_SINGLE | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| room1 | RW_U2_FC_MATCHED_SINGLE | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| room1 | RW_U3_FC_CAPTURED | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| room1 | RW_UQ_PAID_QUERY | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| room1 | RW_W040 | 0 | 0 | 0/0 | 0 | 0 | 2 | 4 | 10 | 1526 |
| room1 | RW_W045 | 0 | 0 | 0/0 | 0 | 0 | 1 | 2 | 8 | -174 |
| room2 | RW_B_D2 | 0 | 0 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 |
| room2 | RW_B_FCEQ | 0 | 0 | 0/0 | 0 | 0 | 5 | 6 | 18 | -74 |
| room2 | RW_U1_NATIVE_SINGLE | 0 | 1 | 0/0 | 0 | 0 | 0 | 0 | 12 | 6 |
| room2 | RW_U2_FC_MATCHED_SINGLE | 2 | 0 | 0/0 | 0 | 0 | 3 | 0 | 12 | 1114 |
| room2 | RW_U3_FC_CAPTURED | 2 | 0 | 0/0 | 0 | 0 | 3 | 0 | 12 | 1114 |
| room2 | RW_UQ_PAID_QUERY | 0 | 1 | 0/0 | 0 | 0 | 0 | 0 | 12 | 6 |
| room2 | RW_W040 | 0 | 0 | 0/0 | 0 | 0 | 0 | 2 | 4 | 0 |
| room2 | RW_W045 | 0 | 0 | 0/0 | 0 | 0 | 0 | 1 | 2 | 0 |

TP/FP counts are actual score entries at overlap .50, not unique objects. Duplicate minimum-score FP and later unvisited FP can both contribute. Numeric ties retain ambiguous identity. Full per-overlap losses and confusion deltas are in the compact mechanism artifact.

## Paired U1/U2 Classification

| Scene | Owner | U1 / U2 class | Agreement | Target points | Valid GT points | GT majority / purity | U1 / U2 correct points |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scene0056_00 | 159 | 79 / 79 | True | 0 | 0 | None / NA | 0 / 0 |
| scene0534_00 | 103 | 105 / 58 | False | 131 | 131 | 5 / 0.9924 | 0 / 0 |
| scene0534_00 | 124 | 79 / 154 | False | 64 | 64 | 154 / 1.0000 | 0 / 64 |
| scene0534_00 | 143 | 141 / 79 | False | 407 | 407 | 8 / 1.0000 | 0 / 0 |
| scene0534_00 | 146 | 105 / 27 | False | 203 | 203 | 8 / 1.0000 | 0 / 0 |
| scene0534_00 | 148 | 1163 / 26 | False | 152 | 152 | 8 / 1.0000 | 0 / 0 |
| scene0534_00 | 164 | 13 / 4 | False | 89 | 0 | None / NA | 0 / 0 |
| scene0534_00 | 176 | 141 / 161 | False | 0 | 0 | None / NA | 0 / 0 |
| scene0534_00 | 87 | 105 / 125 | False | 95 | 95 | 8 / 1.0000 | 0 / 0 |
| scene0534_00 | 88 | 105 / 168 | False | 93 | 93 | 8 / 1.0000 | 0 / 0 |
| office0 | 60 | 35 / 38 | False | 33 | 33 | 38 / 1.0000 | 0 / 33 |
| office2 | 114 | 2 / 10 | False | 1697 | 1697 | 10 / 0.9493 | 86 / 1611 |
| office2 | 53 | 35 / 10 | False | 57 | 12 | 7 / 1.0000 | 0 / 0 |
| office3 | 116 | 43 / 43 | True | 128 | 128 | 43 / 1.0000 | 128 / 128 |
| office3 | 97 | 41 / 51 | False | 109 | 109 | 51 / 0.9358 | 0 / 102 |
| room0 | 77 | 50 / 26 | False | 60 | 60 | 3 / 0.8333 | 0 / 0 |
| room2 | 112 | 2 / 21 | False | 917 | 917 | 21 / 0.9924 | 6 / 910 |
| room2 | 47 | 50 / 4 | False | 11 | 11 | 8 / 1.0000 | 0 / 0 |
| room2 | 64 | 50 / 42 | False | 207 | 207 | 42 / 0.9855 | 0 / 204 |

The two arms use the same successful retained request. Technical failures are excluded from this paired diagnostic and remain in full output coverage. Different native relative and FC cosine score scales are not treated as calibrated equivalents. GT majority is post-lock descriptive evidence.

## Same-Covered-Subset Scores

| Cohort | Diagnostic | Common source owners | APall | AP50 | AP25 | mIoU | mAcc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| development | RW_DIAG_U2_COMMON | 10 | 12.5818 | 22.5994 | 31.6989 | 23.5661 | 32.4400 |
| development | RW_DIAG_U3_COMMON | 10 | 12.5818 | 22.5994 | 31.6989 | 23.5661 | 32.4400 |
| replica | RW_DIAG_U2_COMMON | 9 | 12.3864 | 26.0571 | 39.8413 | 30.1652 | 38.1484 |
| replica | RW_DIAG_U3_COMMON | 9 | 12.3864 | 26.0571 | 39.8413 | 30.1652 | 38.1484 |

The scope is the intersection of U2/U3 source successes, selected without GT. These diagnostics do not enter selection.

## Table 4: Planned and Realized Association

| Scene | Arm | Assign / Native | Followers | Candidate / Alias veto visits | Mixed fallback visits | Planned-realized mismatch | Fresh owner rate | Mean best IoU | R50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scene0056_00 | RW_A1_NATIVE_FALLBACK | 1042 / 330 | 0 | 5008159 / 0 | 7005283 | 0 | 15.0875 | 42.9664 | 14 |
| scene0534_00 | RW_A1_NATIVE_FALLBACK | 1235 / 297 | 0 | 5927239 / 0 | 4913402 | 0 | 13.5770 | 35.7501 | 9 |
| scene0445_00 | RW_A1_NATIVE_FALLBACK | 1026 / 84 | 0 | 9061315 / 0 | 3198059 | 0 | 4.3243 | 52.7789 | 8 |
| scene0626_00 | RW_A1_NATIVE_FALLBACK | 1103 / 133 | 0 | 16953020 / 0 | 3159640 | 0 | 7.1197 | 37.6133 | 9 |
| scene0056_00 | RW_A2_MULTI_FREE | 1264 / 108 | 340 | 2393249 / 0 | 3316 | 0 | 7.3615 | 40.6715 | 13 |
| scene0534_00 | RW_A2_MULTI_FREE | 1423 / 109 | 419 | 7466950 / 0 | 849 | 0 | 6.4621 | 29.9850 | 8 |
| scene0445_00 | RW_A2_MULTI_FREE | 1084 / 26 | 189 | 1372967 / 0 | 807 | 0 | 2.0721 | 45.6246 | 7 |
| scene0626_00 | RW_A2_MULTI_FREE | 1196 / 40 | 230 | 3586597 / 0 | 411 | 0 | 3.1553 | 36.1181 | 9 |
| scene0056_00 | RW_A3_MULTI_UNION | 1172 / 200 | 306 | 2160750 / 0 | 122035 | 0 | 8.2362 | 40.4399 | 14 |
| scene0534_00 | RW_A3_MULTI_UNION | 1177 / 355 | 325 | 3889025 / 2 | 3858119 | 0 | 10.8355 | 29.8274 | 7 |
| scene0445_00 | RW_A3_MULTI_UNION | 1036 / 74 | 178 | 632829 / 0 | 78834 | 0 | 2.5225 | 45.8063 | 7 |
| scene0626_00 | RW_A3_MULTI_UNION | 1021 / 215 | 127 | 3189782 / 0 | 984503 | 0 | 3.9644 | 36.8032 | 9 |

Fresh owner rate counts actual positive mode4 group/owner pairs absent from the factor0 preinsert prior set. Candidate and alias vetoes count visits. No explicit CREATE_NEW action is used. Full traces separate unknown count-owner evidence from mismatches.

## Table 5: Cached SAM Completion

| Scene | Arm | Protected crop pixels | S1 additions | Stable known pixels | Self / Other track pixels | Suppressed tracks / pixels | History abstentions | Crop preserved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scene0056_00 | RW_S1_CROP_PRIORITY | 60746577 | 168268 | NA | NA / NA | NA / NA | NA | True |
| scene0534_00 | RW_S1_CROP_PRIORITY | 59632657 | 254505 | NA | NA / NA | NA / NA | NA | True |
| scene0445_00 | RW_S1_CROP_PRIORITY | 61353063 | 40453 | NA | NA / NA | NA / NA | NA | True |
| scene0626_00 | RW_S1_CROP_PRIORITY | 61259193 | 54264 | NA | NA / NA | NA / NA | NA | True |
| scene0056_00 | RW_S2_CONFLICT | 60746577 | 168268 | 29026742 | 14408548 / 857605 | 81 / 9369 | 770 | True |
| scene0534_00 | RW_S2_CONFLICT | 59632657 | 254505 | 43859479 | 20156409 / 1042040 | 52 / 7442 | 392 | True |
| scene0445_00 | RW_S2_CONFLICT | 61353063 | 40453 | 51444760 | 25772049 / 799713 | 7 / 373 | 81 | True |
| scene0626_00 | RW_S2_CONFLICT | 61259193 | 54264 | 48457051 | 20101250 / 1376666 | 45 / 1782 | 302 | True |

S2 uses two preceding completed factor0 snapshots. Unknown self/history is abstention; only additions are suppressed. CropFormer positives and their separate groups are protected. Per-track pixels may overlap and are not unique image-pixel counts.

## Table 6: Actual Physical Operations

| Scene | Map | Operation | Status | Physical image inputs | Native / Q image inputs | Encoder calls | Region poolings | Model load seconds | Worker wall seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| office0 | BB00_NATIVE | RECOVERY_FC | COMPLETE | 0 | NA / NA | None | 1 | 9.4729 | 13.2281 |
| office1 | BB00_NATIVE | RECOVERY_FC | COMPLETE | 0 | NA / NA | None | 0 | 0.0000 | 0.1497 |
| office2 | BB00_NATIVE | RECOVERY_FC | COMPLETE | 0 | NA / NA | None | 2 | 7.7685 | 11.9385 |
| office3 | BB00_NATIVE | RECOVERY_FC | COMPLETE | 0 | NA / NA | None | 2 | 7.5072 | 10.9664 |
| office4 | BB00_NATIVE | RECOVERY_FC | COMPLETE | 0 | NA / NA | None | 0 | 0.0000 | 0.2213 |
| room0 | BB00_NATIVE | RECOVERY_FC | COMPLETE | 0 | NA / NA | None | 1 | 7.4800 | 11.4271 |
| room1 | BB00_NATIVE | RECOVERY_FC | COMPLETE | 0 | NA / NA | None | 0 | 0.0000 | 0.2270 |
| room2 | BB00_NATIVE | RECOVERY_FC | COMPLETE | 0 | NA / NA | None | 6 | 8.4456 | 13.4211 |
| scene0056_00 | BB00_NATIVE | RECOVERY_FC | COMPLETE | 0 | NA / NA | None | 1 | 9.4572 | 13.9049 |
| scene0445_00 | BB00_NATIVE | RECOVERY_FC | COMPLETE | 0 | NA / NA | None | 0 | 0.0000 | 0.1010 |
| scene0534_00 | BB00_NATIVE | RECOVERY_FC | FAILED | 0 | NA / NA | None | 0 | 0.0000 | 0.6183 |
| scene0534_00 | BB00_NATIVE | RECOVERY_FC | COMPLETE | 0 | NA / NA | None | 9 | 8.5050 | 14.8800 |
| scene0626_00 | BB00_NATIVE | RECOVERY_FC | COMPLETE | 0 | NA / NA | None | 0 | 0.0000 | 0.2940 |
| scene0056_00 | RW_A1_NATIVE_FALLBACK | STANDARD_FC | COMPLETE | 1 | NA / NA | None | 82 | 9.2158 | 26.5303 |
| scene0056_00 | RW_A2_MULTI_FREE | STANDARD_FC | COMPLETE | 0 | NA / NA | None | 15 | 8.9677 | 18.3836 |
| scene0056_00 | RW_A3_MULTI_UNION | STANDARD_FC | COMPLETE | 0 | NA / NA | None | 30 | 8.7977 | 17.5507 |
| scene0056_00 | RW_S1_CROP_PRIORITY | STANDARD_FC | COMPLETE | 0 | NA / NA | None | 95 | 7.9681 | 23.9991 |
| scene0056_00 | RW_S2_CONFLICT | STANDARD_FC | COMPLETE | 0 | NA / NA | None | 4 | 9.3100 | 19.6879 |
| scene0445_00 | RW_A1_NATIVE_FALLBACK | STANDARD_FC | COMPLETE | 0 | NA / NA | None | 13 | 9.6610 | 15.5210 |
| scene0445_00 | RW_A2_MULTI_FREE | STANDARD_FC | COMPLETE | 0 | NA / NA | None | 4 | 7.6261 | 9.2530 |
| scene0445_00 | RW_A3_MULTI_UNION | STANDARD_FC | COMPLETE | 0 | NA / NA | None | 3 | 8.1578 | 9.8543 |
| scene0445_00 | RW_S1_CROP_PRIORITY | STANDARD_FC | COMPLETE | 0 | NA / NA | None | 18 | 7.6724 | 11.6064 |
| scene0445_00 | RW_S2_CONFLICT | STANDARD_FC | COMPLETE | 0 | NA / NA | None | 0 | 0.0000 | 1.6983 |
| scene0534_00 | RW_A1_NATIVE_FALLBACK | STANDARD_FC | COMPLETE | 2 | NA / NA | None | 126 | 9.2915 | 38.1195 |
| scene0534_00 | RW_A2_MULTI_FREE | STANDARD_FC | COMPLETE | 1 | NA / NA | None | 53 | 8.5504 | 20.5787 |
| scene0534_00 | RW_A3_MULTI_UNION | STANDARD_FC | COMPLETE | 1 | NA / NA | None | 55 | 8.3352 | 19.7917 |
| scene0534_00 | RW_S1_CROP_PRIORITY | STANDARD_FC | COMPLETE | 1 | NA / NA | None | 129 | 10.4158 | 28.8026 |
| scene0534_00 | RW_S2_CONFLICT | STANDARD_FC | COMPLETE | 0 | NA / NA | None | 13 | 8.4293 | 17.5028 |
| scene0626_00 | RW_A1_NATIVE_FALLBACK | STANDARD_FC | COMPLETE | 0 | NA / NA | None | 27 | 9.7050 | 20.7603 |
| scene0626_00 | RW_A2_MULTI_FREE | STANDARD_FC | COMPLETE | 2 | NA / NA | None | 8 | 7.9221 | 11.6350 |
| scene0626_00 | RW_A3_MULTI_UNION | STANDARD_FC | COMPLETE | 2 | NA / NA | None | 3 | 7.4199 | 10.7367 |
| scene0626_00 | RW_S1_CROP_PRIORITY | STANDARD_FC | COMPLETE | 0 | NA / NA | None | 30 | 9.0956 | 15.9954 |
| scene0626_00 | RW_S2_CONFLICT | STANDARD_FC | COMPLETE | 0 | NA / NA | None | 6 | 8.2112 | 12.2673 |
| scene0056_00 | RW_A1_NATIVE_FALLBACK | NATIVE_Q | COMPLETE | 1638 | 1302 / 336 | 273 | None | 1.1820 | 215.1855 |
| scene0056_00 | RW_A2_MULTI_FREE | NATIVE_Q | COMPLETE | 1014 | 612 / 402 | 169 | None | 1.4365 | 171.7697 |
| scene0056_00 | RW_A3_MULTI_UNION | NATIVE_Q | COMPLETE | 588 | 408 / 180 | 98 | None | 1.1700 | 156.1484 |
| scene0056_00 | RW_S1_CROP_PRIORITY | NATIVE_Q | COMPLETE | 2772 | 2628 / 144 | 462 | None | 1.1710 | 263.3235 |
| scene0056_00 | RW_S2_CONFLICT | NATIVE_Q | COMPLETE | 132 | 114 / 18 | 22 | None | 1.1934 | 180.9425 |
| scene0445_00 | RW_A1_NATIVE_FALLBACK | NATIVE_Q | COMPLETE | 978 | 414 / 564 | 163 | None | 1.1369 | 118.4474 |
| scene0445_00 | RW_A2_MULTI_FREE | NATIVE_Q | COMPLETE | 690 | 168 / 522 | 115 | None | 1.1970 | 98.7694 |
| scene0445_00 | RW_A3_MULTI_UNION | NATIVE_Q | COMPLETE | 510 | 96 / 414 | 85 | None | 1.1656 | 93.2750 |
| scene0445_00 | RW_S1_CROP_PRIORITY | NATIVE_Q | COMPLETE | 1020 | 510 / 510 | 170 | None | 1.1158 | 117.7217 |
| scene0445_00 | RW_S2_CONFLICT | NATIVE_Q | COMPLETE | 24 | 6 / 18 | 4 | None | 1.1882 | 83.5361 |
| scene0534_00 | RW_A1_NATIVE_FALLBACK | NATIVE_Q | COMPLETE | 2226 | 1920 / 306 | 371 | None | 1.1420 | 234.9742 |
| scene0534_00 | RW_A2_MULTI_FREE | NATIVE_Q | COMPLETE | 1200 | 846 / 354 | 200 | None | 1.1664 | 159.9472 |
| scene0534_00 | RW_A3_MULTI_UNION | NATIVE_Q | COMPLETE | 996 | 774 / 222 | 166 | None | 1.3378 | 161.8869 |
| scene0534_00 | RW_S1_CROP_PRIORITY | NATIVE_Q | COMPLETE | 2784 | 2538 / 246 | 464 | None | 1.1065 | 245.6277 |
| scene0534_00 | RW_S2_CONFLICT | NATIVE_Q | COMPLETE | 168 | 150 / 18 | 28 | None | 1.1526 | 169.9950 |
| scene0626_00 | RW_A1_NATIVE_FALLBACK | NATIVE_Q | COMPLETE | 1302 | 654 / 648 | 217 | None | 1.5243 | 144.2176 |
| scene0626_00 | RW_A2_MULTI_FREE | NATIVE_Q | COMPLETE | 798 | 168 / 630 | 133 | None | 1.1931 | 112.3417 |
| scene0626_00 | RW_A3_MULTI_UNION | NATIVE_Q | COMPLETE | 840 | 336 / 504 | 140 | None | 1.1572 | 114.5729 |
| scene0626_00 | RW_S1_CROP_PRIORITY | NATIVE_Q | COMPLETE | 1560 | 1002 / 558 | 260 | None | 1.1341 | 160.5241 |
| scene0626_00 | RW_S2_CONFLICT | NATIVE_Q | COMPLETE | 108 | 72 / 36 | 18 | None | 1.1380 | 103.7441 |
| scene0056_00 | RW_A1_NATIVE_FALLBACK | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 1571.2106 |
| scene0056_00 | RW_A2_MULTI_FREE | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 1579.0374 |
| scene0056_00 | RW_A3_MULTI_UNION | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 1620.1111 |
| scene0056_00 | RW_S1_CROP_PRIORITY | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 1466.4508 |
| scene0056_00 | RW_S2_CONFLICT | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 1657.8612 |
| scene0445_00 | RW_A1_NATIVE_FALLBACK | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 833.1730 |
| scene0445_00 | RW_A2_MULTI_FREE | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 821.3919 |
| scene0445_00 | RW_A3_MULTI_UNION | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 792.4906 |
| scene0445_00 | RW_S1_CROP_PRIORITY | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 707.9340 |
| scene0445_00 | RW_S2_CONFLICT | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 856.7041 |
| scene0534_00 | RW_A1_NATIVE_FALLBACK | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 789.5322 |
| scene0534_00 | RW_A2_MULTI_FREE | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 764.9190 |
| scene0534_00 | RW_A3_MULTI_UNION | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 747.5273 |
| scene0534_00 | RW_S1_CROP_PRIORITY | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 673.7361 |
| scene0534_00 | RW_S2_CONFLICT | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 806.6498 |
| scene0626_00 | RW_A1_NATIVE_FALLBACK | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 871.1651 |
| scene0626_00 | RW_A2_MULTI_FREE | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 858.5278 |
| scene0626_00 | RW_A3_MULTI_UNION | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 828.5711 |
| scene0626_00 | RW_S1_CROP_PRIORITY | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 766.4971 |
| scene0626_00 | RW_S2_CONFLICT | FULL_MAP_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 939.0136 |
| scene0056_00 | S1_FRONTEND | CACHED_SAM_CPU_COMPOSITION | COMPLETE | 0 | NA / NA | NA | 0 | NA | 22.4991 |
| scene0056_00 | S1_FRONTEND | CACHED_SAM_CPU_COMPOSITION | COMPLETE | 0 | NA / NA | NA | 0 | NA | 26.8985 |
| scene0445_00 | S1_FRONTEND | CACHED_SAM_CPU_COMPOSITION | COMPLETE | 0 | NA / NA | NA | 0 | NA | 11.6470 |
| scene0445_00 | S1_FRONTEND | CACHED_SAM_CPU_COMPOSITION | COMPLETE | 0 | NA / NA | NA | 0 | NA | 13.9443 |
| scene0534_00 | S1_FRONTEND | CACHED_SAM_CPU_COMPOSITION | COMPLETE | 0 | NA / NA | NA | 0 | NA | 16.1605 |
| scene0534_00 | S1_FRONTEND | CACHED_SAM_CPU_COMPOSITION | COMPLETE | 0 | NA / NA | NA | 0 | NA | 17.7363 |
| scene0626_00 | S1_FRONTEND | CACHED_SAM_CPU_COMPOSITION | COMPLETE | 0 | NA / NA | NA | 0 | NA | 14.2067 |
| scene0626_00 | S1_FRONTEND | CACHED_SAM_CPU_COMPOSITION | COMPLETE | 0 | NA / NA | NA | 0 | NA | 18.1908 |
| None | native | ISOLATED_NATIVE_BUILD_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 67.9219 |
| None | recovery_native_v2 | ISOLATED_NATIVE_BUILD_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 66.9827 |
| scene0056_00 | A1 | SERIAL_NATIVE_VALIDATION_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 13.8709 |
| scene0056_00 | A2 | SERIAL_NATIVE_VALIDATION_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 14.4810 |
| scene0056_00 | A3 | SERIAL_NATIVE_VALIDATION_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 13.3980 |
| scene0056_00 | ALL_NATIVE | SERIAL_NATIVE_VALIDATION_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 13.5427 |
| scene0056_00 | follower | SERIAL_NATIVE_VALIDATION_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 0.3458 |
| scene0056_00 | follower | SERIAL_NATIVE_VALIDATION_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 0.3614 |
| scene0056_00 | off | SERIAL_NATIVE_VALIDATION_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 11.9905 |
| scene0056_00 | off-parent | SERIAL_NATIVE_VALIDATION_CPU | COMPLETE | 0 | NA / NA | NA | NA | NA | 12.5548 |

### Standalone Logical Requirements

| Cohort | Map / Method | Standalone additional image inputs | Required image inputs | Additional recovery poolings | Required cached-SAM cold inputs |
| --- | --- | --- | --- | --- | --- |
| development | BB00_NATIVE / RW_B_D2 | 0 | 18093 | 0 | 0 |
| development | BB00_NATIVE / RW_B_FCEQ | 0 | 18093 | 0 | 0 |
| development | BB00_NATIVE / RW_W040 | 0 | 18093 | 0 | 0 |
| development | BB00_NATIVE / RW_W045 | 0 | 18093 | 0 | 0 |
| development | BB00_NATIVE / RW_U1_NATIVE_SINGLE | 0 | 18093 | 0 | 0 |
| development | BB00_NATIVE / RW_UQ_PAID_QUERY | 0 | 18093 | 0 | 0 |
| development | BB00_NATIVE / RW_U2_FC_MATCHED_SINGLE | 0 | 18093 | 10 | 0 |
| development | BB00_NATIVE / RW_U3_FC_CAPTURED | 0 | 18093 | 10 | 0 |
| development | RW_A1_NATIVE_FALLBACK | 8479 | 17295 | SEE_PHYSICAL_OPERATION_ROWS | 0 |
| development | RW_A2_MULTI_FREE | 6196 | 13817 | SEE_PHYSICAL_OPERATION_ROWS | 0 |
| development | RW_A3_MULTI_UNION | 5550 | 14750 | SEE_PHYSICAL_OPERATION_ROWS | 0 |
| development | RW_S1_CROP_PRIORITY | 8993 | 18834 | SEE_PHYSICAL_OPERATION_ROWS | 800 |
| development | RW_S2_CONFLICT | 8693 | 18840 | SEE_PHYSICAL_OPERATION_ROWS | 800 |
| replica | BB00_NATIVE / RW_B_D2 | 0 | 32527 | 0 | 0 |
| replica | BB00_NATIVE / RW_B_FCEQ | 0 | 32527 | 0 | 0 |
| replica | BB00_NATIVE / RW_W040 | 0 | 32527 | 0 | 0 |
| replica | BB00_NATIVE / RW_W045 | 0 | 32527 | 0 | 0 |
| replica | BB00_NATIVE / RW_U1_NATIVE_SINGLE | 0 | 32527 | 0 | 0 |
| replica | BB00_NATIVE / RW_UQ_PAID_QUERY | 0 | 32527 | 0 | 0 |
| replica | BB00_NATIVE / RW_U2_FC_MATCHED_SINGLE | 1 | 32528 | 9 | 0 |
| replica | BB00_NATIVE / RW_U3_FC_CAPTURED | 2 | 32529 | 12 | 0 |

Standalone additional image inputs and required baseline contents are recorded separately in `failure_and_costs.json`. Physical cache reuse is not zero standalone compute. Timings are worker wall seconds; separate CUDA-event and CPU timings were not recorded. Concurrent duration sums are not project elapsed time.
