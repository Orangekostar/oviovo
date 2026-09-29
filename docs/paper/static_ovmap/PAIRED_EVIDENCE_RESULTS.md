# Paired evidence: measured results

Offline frozen-evidence semantic refinement on two exposed CAL and eight historical Replica scenes. No new generalization confirmation, online mapping demonstration, or statistical significance claim.

CAL-frozen research nominee: **PE_COMBO_D4_R3**. Deployment: **N0_UNCHANGED**. Original wave-1 nomination remains untouched.

APall/uAP averages the actual released 0.50–0.90 overlap vector; AP25 is separate. All values below are percentages, differences are percentage points. Official pooling calls the released evaluator on ordered scene files and sums semantic confusion matrices.

## CAL / OFFICIAL_CURRENT_CLASS / RELEASED_DATASET_POOL

| Method | apall | ap50 | ap25 | miou | macc | ΔAPall vs B | ΔmIoU vs B | ΔAPall vs N0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N0 | 4.235291 | 14.985804 | 30.183916 | 22.015249 | 28.920037 | 0.010448 | -2.442039 | 0.000000 |
| RV_A7_COS_REFIT | 3.732776 | 14.181265 | 28.535003 | 22.680712 | 30.717638 | -0.492067 | -1.776576 | -0.502515 |
| AW_E03_FC_FROZEN_A7 | 4.224843 | 14.905203 | 31.590608 | 24.457288 | 31.322824 | 0.000000 | 0.000000 | -0.010448 |
| AW_E03_FC_FROZEN_DIRECT | 3.690231 | 13.494268 | 25.322972 | 20.007402 | 26.119578 | -0.534612 | -4.449887 | -0.545060 |
| AW_E03_OVR_DIRECT | 5.119048 | 16.119929 | 29.438933 | 22.411281 | 28.469935 | 0.894204 | -2.046008 | 0.883757 |
| AW_E03_OVR_A7 | 4.234029 | 14.927249 | 31.782474 | 24.884025 | 31.697263 | 0.009186 | 0.426737 | -0.001262 |
| PE_R0_BLEND | 4.234029 | 14.927249 | 31.768445 | 24.863916 | 31.681393 | 0.009186 | 0.406628 | -0.001262 |
| PE_R1_FOURWAY | 4.093180 | 13.622869 | 26.966490 | 21.710783 | 27.699696 | -0.131663 | -2.746505 | -0.142110 |
| PE_R2_GLOBAL | 5.546288 | 18.627646 | 32.214139 | 25.741034 | 30.327686 | 1.321445 | 1.283746 | 1.310997 |
| PE_R3_LOCAL | 5.546288 | 18.627646 | 28.232657 | 24.365021 | 28.924525 | 1.321445 | -0.092268 | 1.310997 |
| PE_R3_LOCAL_CALDELTA | 5.818146 | 19.878013 | 29.687684 | 25.999995 | 30.480335 | 1.593303 | 1.542707 | 1.582855 |
| PE_D1_LOGPOOL | 5.047888 | 18.608907 | 32.558422 | 26.072036 | 32.677062 | 0.823045 | 1.614748 | 0.812598 |
| PE_D1_ANCHORED | 5.044112 | 18.601864 | 32.699769 | 25.793640 | 32.316824 | 0.819269 | 1.336352 | 0.808821 |
| PE_D2_GROUPED | 4.067460 | 13.648589 | 26.945914 | 21.299392 | 26.901827 | -0.157383 | -3.157896 | -0.167831 |
| PE_D3_DIAGONAL | 5.044112 | 18.601864 | 32.711640 | 25.256142 | 31.751952 | 0.819269 | 0.798854 | 0.808821 |
| PE_D4_LINEAGE | 5.014820 | 18.548280 | 32.473799 | 25.725685 | 32.281102 | 0.789976 | 1.268397 | 0.779529 |
| PE_D4_SHUFFLED | 5.014820 | 18.548280 | 32.473799 | 25.725685 | 32.281102 | 0.789976 | 1.268397 | 0.779529 |
| PE_COMBO_D4_R3 | 6.481155 | 22.964433 | 33.080173 | 28.902348 | 33.264997 | 2.256312 | 4.445060 | 2.245864 |

## CAL / FROZEN_N0 / RELEASED_DATASET_POOL

| Method | apall | ap50 | ap25 | miou | macc | ΔAPall vs B | ΔmIoU vs B | ΔAPall vs N0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N0 | 4.235291 | 14.985804 | 30.183916 | 22.015249 | 28.920037 | 0.010448 | -2.442039 | 0.000000 |
| RV_A7_COS_REFIT | 3.931566 | 13.552845 | 29.222514 | 22.680712 | 30.717638 | -0.293277 | -1.776576 | -0.303725 |
| AW_E03_FC_FROZEN_A7 | 4.224843 | 14.905203 | 32.164168 | 24.457288 | 31.322824 | 0.000000 | 0.000000 | -0.010448 |
| AW_E03_FC_FROZEN_DIRECT | 3.402165 | 12.012787 | 23.864638 | 20.007402 | 26.119578 | -0.822678 | -4.449887 | -0.833125 |
| AW_E03_OVR_DIRECT | 4.881932 | 14.343034 | 27.583774 | 22.411281 | 28.469935 | 0.657089 | -2.046008 | 0.646641 |
| AW_E03_OVR_A7 | 4.234029 | 14.927249 | 32.098832 | 24.884025 | 31.697263 | 0.009186 | 0.426737 | -0.001262 |
| PE_R0_BLEND | 4.234029 | 14.927249 | 32.084803 | 24.863916 | 31.681393 | 0.009186 | 0.406628 | -0.001262 |
| PE_R1_FOURWAY | 4.093180 | 13.622869 | 26.951058 | 21.710783 | 27.699696 | -0.131663 | -2.746505 | -0.142110 |
| PE_R2_GLOBAL | 5.537796 | 16.850750 | 30.559597 | 25.741034 | 30.327686 | 1.312953 | 1.283746 | 1.302506 |
| PE_R3_LOCAL | 5.537796 | 16.850750 | 26.608980 | 24.365021 | 28.924525 | 1.312953 | -0.092268 | 1.302506 |
| PE_R3_LOCAL_CALDELTA | 5.818146 | 19.878013 | 29.903733 | 25.999995 | 30.480335 | 1.593303 | 1.542707 | 1.582855 |
| PE_D1_LOGPOOL | 5.039825 | 18.589004 | 33.069885 | 26.072036 | 32.677062 | 0.814982 | 1.614748 | 0.804535 |
| PE_D1_ANCHORED | 5.036825 | 18.583431 | 33.311542 | 25.793640 | 32.316824 | 0.811982 | 1.336352 | 0.801534 |
| PE_D2_GROUPED | 4.067460 | 13.648589 | 26.930482 | 21.299392 | 26.901827 | -0.157383 | -3.157896 | -0.167831 |
| PE_D3_DIAGONAL | 5.036825 | 18.583431 | 33.323413 | 25.256142 | 31.751952 | 0.811982 | 0.798854 | 0.801534 |
| PE_D4_LINEAGE | 5.010533 | 18.535420 | 33.112211 | 25.725685 | 32.281102 | 0.785690 | 1.268397 | 0.775242 |
| PE_D4_SHUFFLED | 5.010533 | 18.535420 | 33.112211 | 25.725685 | 32.281102 | 0.785690 | 1.268397 | 0.775242 |
| PE_COMBO_D4_R3 | 6.538311 | 22.038507 | 32.316285 | 28.902348 | 33.264997 | 2.313468 | 4.445060 | 2.303020 |

## REPLICA / OFFICIAL_CURRENT_CLASS / RELEASED_DATASET_POOL

| Method | apall | ap50 | ap25 | miou | macc | ΔAPall vs B | ΔmIoU vs B | ΔAPall vs N0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N0 | 8.685272 | 21.476863 | 34.585379 | 27.261462 | 32.695265 | -1.805748 | -2.752210 | 0.000000 |
| RV_A7_COS_REFIT | 9.217907 | 21.236029 | 35.201083 | 27.720221 | 33.810573 | -1.273113 | -2.293452 | 0.532635 |
| AW_E03_FC_FROZEN_A7 | 10.491020 | 24.323075 | 37.728463 | 30.013672 | 36.694593 | 0.000000 | 0.000000 | 1.805748 |
| AW_E03_FC_FROZEN_DIRECT | 10.498388 | 20.287022 | 30.235633 | 24.091640 | 33.261039 | 0.007369 | -5.922033 | 1.813117 |
| AW_E03_OVR_DIRECT | 10.939594 | 22.006838 | 33.449918 | 26.180430 | 36.770290 | 0.448575 | -3.833242 | 2.254323 |
| AW_E03_OVR_A7 | 10.232525 | 22.023194 | 36.508624 | 28.623663 | 36.625030 | -0.258494 | -1.390009 | 1.547254 |
| PE_R0_BLEND | 10.441331 | 22.601213 | 36.047207 | 28.151956 | 36.013638 | -0.049688 | -1.861716 | 1.756060 |
| PE_R1_FOURWAY | 11.095742 | 23.026112 | 36.672689 | 28.896876 | 37.456042 | 0.604722 | -1.116796 | 2.410470 |
| PE_R2_GLOBAL | 9.690331 | 21.181997 | 35.073499 | 29.153170 | 36.542929 | -0.800689 | -0.860502 | 1.005059 |
| PE_R3_LOCAL | 9.610518 | 20.983791 | 34.656817 | 28.980842 | 36.540127 | -0.880502 | -1.032830 | 0.925246 |
| PE_R3_LOCAL_CALDELTA | 9.085547 | 20.967106 | 34.244976 | 27.620826 | 36.081609 | -1.405473 | -2.392846 | 0.400276 |
| PE_D1_LOGPOOL | 9.575704 | 23.439305 | 37.463756 | 28.407312 | 35.388174 | -0.915316 | -1.606360 | 0.890433 |
| PE_D1_ANCHORED | 9.525593 | 23.264244 | 36.345774 | 28.147024 | 34.824867 | -0.965427 | -1.866648 | 0.840321 |
| PE_D2_GROUPED | 11.763120 | 24.495210 | 37.974586 | 29.716355 | 37.491652 | 1.272100 | -0.297317 | 3.077848 |
| PE_D3_DIAGONAL | 9.371306 | 22.489842 | 35.943438 | 28.626487 | 35.303407 | -1.119714 | -1.387185 | 0.686034 |
| PE_D4_LINEAGE | 9.261654 | 22.349420 | 35.803015 | 28.632918 | 35.387301 | -1.229366 | -1.380755 | 0.576383 |
| PE_D4_SHUFFLED | 9.261654 | 22.349420 | 35.803015 | 28.632918 | 35.387301 | -1.229366 | -1.380755 | 0.576383 |
| PE_COMBO_D4_R3 | 10.208169 | 23.354851 | 36.953954 | 29.563151 | 37.630020 | -0.282850 | -0.450521 | 1.522898 |

## REPLICA / FROZEN_N0 / RELEASED_DATASET_POOL

| Method | apall | ap50 | ap25 | miou | macc | ΔAPall vs B | ΔmIoU vs B | ΔAPall vs N0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N0 | 8.685272 | 21.476863 | 34.585379 | 27.261462 | 32.695265 | -2.134931 | -2.752210 | 0.000000 |
| RV_A7_COS_REFIT | 9.280876 | 21.473281 | 35.548398 | 27.720221 | 33.810573 | -1.539327 | -2.293452 | 0.595604 |
| AW_E03_FC_FROZEN_A7 | 10.820202 | 25.490282 | 39.683677 | 30.013672 | 36.694593 | 0.000000 | 0.000000 | 2.134931 |
| AW_E03_FC_FROZEN_DIRECT | 11.384098 | 21.411166 | 33.116421 | 24.091640 | 33.261039 | 0.563895 | -5.922033 | 2.698826 |
| AW_E03_OVR_DIRECT | 11.619355 | 22.548895 | 36.286148 | 26.180430 | 36.770290 | 0.799152 | -3.833242 | 2.934083 |
| AW_E03_OVR_A7 | 10.416317 | 22.585022 | 37.521119 | 28.623663 | 36.625030 | -0.403885 | -1.390009 | 1.731045 |
| PE_R0_BLEND | 10.596368 | 23.043481 | 36.888538 | 28.151956 | 36.013638 | -0.223834 | -1.861716 | 1.911096 |
| PE_R1_FOURWAY | 11.367946 | 23.490912 | 37.708611 | 28.896876 | 37.456042 | 0.547744 | -1.116796 | 2.682674 |
| PE_R2_GLOBAL | 10.559471 | 22.406578 | 37.405142 | 29.153170 | 36.542929 | -0.260731 | -0.860502 | 1.874200 |
| PE_R3_LOCAL | 10.479900 | 22.200415 | 36.888999 | 28.980842 | 36.540127 | -0.340303 | -1.032830 | 1.794628 |
| PE_R3_LOCAL_CALDELTA | 9.560226 | 21.941520 | 37.019859 | 27.620826 | 36.081609 | -1.259976 | -2.392846 | 0.874954 |
| PE_D1_LOGPOOL | 9.862060 | 24.557435 | 38.740055 | 28.407312 | 35.388174 | -0.958142 | -1.606360 | 1.176789 |
| PE_D1_ANCHORED | 9.635125 | 24.000585 | 37.509197 | 28.147024 | 34.824867 | -1.185077 | -1.866648 | 0.949853 |
| PE_D2_GROUPED | 12.029257 | 24.975735 | 39.285370 | 29.716355 | 37.491652 | 1.209055 | -0.297317 | 3.343986 |
| PE_D3_DIAGONAL | 9.476479 | 23.479171 | 37.715514 | 28.626487 | 35.303407 | -1.343724 | -1.387185 | 0.791207 |
| PE_D4_LINEAGE | 9.490955 | 23.525406 | 37.763492 | 28.632918 | 35.387301 | -1.329248 | -1.380755 | 0.805683 |
| PE_D4_SHUFFLED | 9.490955 | 23.525406 | 37.763492 | 28.632918 | 35.387301 | -1.329248 | -1.380755 | 0.805683 |
| PE_COMBO_D4_R3 | 10.776896 | 23.793996 | 38.779242 | 29.563151 | 37.630020 | -0.043306 | -0.450521 | 2.091625 |

## Matched mechanism contrasts

| Split | Contrast | apall | ap50 | ap25 | miou | macc |
| --- | --- | --- | --- | --- | --- | --- |
| cal | PE_R3_LOCAL − PE_R2_GLOBAL | 0.000000 | 0.000000 | -3.981481 | -1.376014 | -1.403161 |
| cal | PE_R3_LOCAL − PE_R0_BLEND | 1.312259 | 3.700397 | -3.535788 | -0.498895 | -2.756869 |
| cal | PE_R3_LOCAL − PE_R1_FOURWAY | 1.453108 | 5.004777 | 1.266167 | 2.654237 | 1.224829 |
| cal | PE_R3_LOCAL − PE_R3_LOCAL_CALDELTA | -0.271858 | -1.250367 | -1.455026 | -1.634974 | -1.555811 |
| cal | PE_D4_LINEAGE − PE_D3_DIAGONAL | -0.029292 | -0.053584 | -0.237841 | 0.469543 | 0.529149 |
| cal | PE_D4_LINEAGE − PE_D1_ANCHORED | -0.029292 | -0.053584 | -0.225970 | -0.067955 | -0.035723 |
| cal | PE_D4_LINEAGE − PE_D2_GROUPED | 0.947359 | 4.899691 | 5.527885 | 4.426293 | 5.379274 |
| cal | PE_D4_LINEAGE − PE_D4_SHUFFLED | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| replica | PE_R3_LOCAL − PE_R2_GLOBAL | -0.079813 | -0.198206 | -0.416681 | -0.172328 | -0.002802 |
| replica | PE_R3_LOCAL − PE_R0_BLEND | -0.830813 | -1.617421 | -1.390389 | 0.828886 | 0.526489 |
| replica | PE_R3_LOCAL − PE_R1_FOURWAY | -1.485224 | -2.042320 | -2.015871 | 0.083966 | -0.915915 |
| replica | PE_R3_LOCAL − PE_R3_LOCAL_CALDELTA | 0.524971 | 0.016685 | 0.411842 | 1.360016 | 0.458517 |
| replica | PE_D4_LINEAGE − PE_D3_DIAGONAL | -0.109651 | -0.140422 | -0.140422 | 0.006431 | 0.083894 |
| replica | PE_D4_LINEAGE − PE_D1_ANCHORED | -0.263939 | -0.914824 | -0.542759 | 0.485893 | 0.562434 |
| replica | PE_D4_LINEAGE − PE_D2_GROUPED | -2.501466 | -2.145790 | -2.171571 | -1.083437 | -2.104351 |
| replica | PE_D4_LINEAGE − PE_D4_SHUFFLED | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

The one prespecified D4→R3 interaction was executed. It uses the best active CAL R3 eta 2, the original B candidate set, and D4 mass within that set.

| Split: combo − D4 − active R3 + B | apall | ap50 | ap25 | miou | macc |
| --- | --- | --- | --- | --- | --- |
| cal | 0.144890 | 0.693710 | 3.964325 | 3.268930 | 3.382194 |
| replica | 1.827017 | 4.344714 | 4.222584 | 1.963063 | 2.397186 |

Replica does not receive an extra active-R3 retuning run. The matched interaction is included when its selected eta agrees. If it differs, a matched four-cell Replica interaction is not identifiable from canonical rows and is not fabricated.

## Coverage and degeneracies

| Scene | Owners | Identifiable | Paired F/O | D inputs | N/Q covariance active | Unknown atoms | D4=D3 structural | F/O text max difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| office0 | 45 | 14 | 45 | 45 | 22 | 0 | 23 | 0.0 |
| office1 | 35 | 13 | 34 | 35 | 22 | 0 | 13 | 0.0 |
| office2 | 51 | 21 | 47 | 51 | 26 | 0 | 25 | 0.0 |
| office3 | 51 | 22 | 47 | 51 | 29 | 0 | 22 | 0.0 |
| office4 | 43 | 13 | 42 | 43 | 29 | 0 | 14 | 0.0 |
| room0 | 54 | 35 | 53 | 54 | 34 | 0 | 20 | 0.0 |
| room1 | 47 | 21 | 47 | 47 | 30 | 0 | 17 | 0.0 |
| room2 | 49 | 22 | 46 | 49 | 33 | 0 | 16 | 0.0 |
| scene0056_00 | 99 | 13 | 97 | 99 | 81 | 0 | 18 | 0.0 |
| scene0534_00 | 93 | 9 | 91 | 93 | 71 | 0 | 22 | 0.0 |

Identifiable means unique geometry correspondence with IoU strictly greater than 0.5. All owners, including unidentifiable and fallback owners, remain in complete-map evaluation. Unknown mask support is isolated and labeled unknown; it is not evidence of independence. The proxy models only same-image/same-vision-space support. Cross-model and temporal correlations remain unmodeled.

CAL: shuffled correlation is an identity for 40/192 objects. For the remaining objects it changes probabilities but the measured whole-map D4 and shuffled labels coincide. This is not structural D4=D3 equality for every object.

REPLICA: shuffled correlation is an identity for 150/375 objects. For the remaining objects it changes probabilities but the measured whole-map D4 and shuffled labels coincide. This is not structural D4=D3 equality for every object.

## Dependence weights and graph diagnostics

| Split | Method | Objects | Mean source weights (absent=0) | Mean cycle energy | Mean anchored fit residual | QP active sets / pair counts |
| --- | --- | --- | --- | --- | --- | --- |
| cal | PE_D3_DIAGONAL | 192 | N=0.494614; Q=0.157360; F=0.348026 | 0.00645326 | 0.175392 | {'N+Q+F': 3522300, 'N': 39800, 'N+Q': 39800, 'N+F': 218900} |
| cal | PE_D4_LINEAGE | 192 | N=0.520065; Q=0.097656; F=0.382278 | 0.00676919 | 0.173598 | {'N+F': 439594, 'N+Q+F': 3301606, 'N': 59700, 'N+Q': 19900} |
| cal | PE_D4_SHUFFLED | 192 | N=0.520102; Q=0.097623; F=0.382275 | 0.00674365 | 0.17361 | {'N+F': 436955, 'N+Q+F': 3304245, 'N': 59700, 'N+Q': 19900} |
| replica | PE_D3_DIAGONAL | 375 | N=0.472725; Q=0.177889; F=0.349386 | 0.00661884 | 0.149652 | {'N+Q+F': 455175, 'N+F': 5100, 'N+Q': 12750, 'N': 5100} |
| replica | PE_D4_LINEAGE | 375 | N=0.487107; Q=0.137823; F=0.375070 | 0.00677085 | 0.145281 | {'N+Q+F': 422912, 'N+F': 37363, 'N+Q': 11475, 'N': 6375} |
| replica | PE_D4_SHUFFLED | 375 | N=0.487125; Q=0.137808; F=0.375067 | 0.00674694 | 0.145296 | {'N+Q+F': 422762, 'N+F': 37513, 'N+Q': 11475, 'N': 6375} |

QP pair counts describe numerical problems, not independent training examples.

## Common-identifiable probability calibration

| Split | Method | Same support | NLL | Multiclass Brier |
| --- | --- | --- | --- | --- |
| cal | AW_E03_FC_FROZEN_A7 | 22 | 1.459372 | 0.559222 |
| cal | AW_E03_FC_FROZEN_DIRECT | 22 | 2.033856 | 0.747277 |
| cal | AW_E03_OVR_DIRECT | 22 | 1.961088 | 0.747996 |
| cal | AW_E03_OVR_A7 | 22 | 1.384428 | 0.554179 |
| cal | PE_R0_BLEND | 22 | 1.417136 | 0.555538 |
| cal | PE_R1_FOURWAY | 22 | 1.456750 | 0.579277 |
| cal | PE_R2_GLOBAL | 22 | 1.578173 | 0.595833 |
| cal | PE_R3_LOCAL | 22 | 1.646513 | 0.563091 |
| cal | PE_R3_LOCAL_CALDELTA | 22 | 1.540532 | 0.579074 |
| cal | PE_D1_LOGPOOL | 22 | 1.330757 | 0.545938 |
| cal | PE_D1_ANCHORED | 22 | 1.373016 | 0.548170 |
| cal | PE_D2_GROUPED | 22 | 1.517153 | 0.584685 |
| cal | PE_D3_DIAGONAL | 22 | 1.364142 | 0.538663 |
| cal | PE_D4_LINEAGE | 22 | 1.366746 | 0.539573 |
| cal | PE_D4_SHUFFLED | 22 | 1.366729 | 0.539566 |
| cal | PE_COMBO_D4_R3 | 22 | 1.512477 | 0.523134 |
| replica | AW_E03_FC_FROZEN_A7 | 157 | 1.366568 | 0.598713 |
| replica | AW_E03_FC_FROZEN_DIRECT | 157 | 1.390945 | 0.581581 |
| replica | AW_E03_OVR_DIRECT | 157 | 1.288860 | 0.545325 |
| replica | AW_E03_OVR_A7 | 157 | 1.331764 | 0.587627 |
| replica | PE_R0_BLEND | 157 | 1.343916 | 0.592388 |
| replica | PE_R1_FOURWAY | 157 | 1.300895 | 0.568915 |
| replica | PE_R2_GLOBAL | 157 | 1.453948 | 0.600953 |
| replica | PE_R3_LOCAL | 157 | 1.569856 | 0.589993 |
| replica | PE_R3_LOCAL_CALDELTA | 157 | 1.416774 | 0.590291 |
| replica | PE_D1_LOGPOOL | 157 | 1.383087 | 0.602264 |
| replica | PE_D1_ANCHORED | 157 | 1.363768 | 0.600083 |
| replica | PE_D2_GROUPED | 157 | 1.332928 | 0.579178 |
| replica | PE_D3_DIAGONAL | 157 | 1.345868 | 0.593469 |
| replica | PE_D4_LINEAGE | 157 | 1.340767 | 0.591756 |
| replica | PE_D4_SHUFFLED | 157 | 1.340778 | 0.591758 |
| replica | PE_COMBO_D4_R3 | 157 | 1.558295 | 0.589141 |

N0 canonical-relative confidence and old A7 are not assigned invented probability scales here. Calibration results use the fixed common identifiable support and cannot substitute for official map metrics.

## Actual released matching versus B

| Split | Method | Overlap | Added GT matches | Lost GT matches | Duplicates | Ignored events | Unmatched FP events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cal | N0 | 0.25 | 4 | 3 | 0 | 37 | 77 |
| cal | N0 | 0.5 | 1 | 0 | 0 | 27 | 100 |
| cal | RV_A7_COS_REFIT | 0.25 | 5 | 1 | 0 | 37 | 74 |
| cal | RV_A7_COS_REFIT | 0.5 | 2 | 0 | 0 | 27 | 99 |
| cal | AW_E03_FC_FROZEN_A7 | 0.25 | 0 | 0 | 0 | 37 | 78 |
| cal | AW_E03_FC_FROZEN_A7 | 0.5 | 0 | 0 | 0 | 27 | 101 |
| cal | AW_E03_FC_FROZEN_DIRECT | 0.25 | 2 | 10 | 1 | 38 | 84 |
| cal | AW_E03_FC_FROZEN_DIRECT | 0.5 | 2 | 4 | 0 | 27 | 103 |
| cal | AW_E03_OVR_DIRECT | 0.25 | 3 | 7 | 1 | 38 | 80 |
| cal | AW_E03_OVR_DIRECT | 0.5 | 3 | 4 | 0 | 27 | 102 |
| cal | AW_E03_OVR_A7 | 0.25 | 0 | 0 | 0 | 37 | 78 |
| cal | AW_E03_OVR_A7 | 0.5 | 0 | 0 | 0 | 27 | 101 |
| cal | PE_R0_BLEND | 0.25 | 0 | 0 | 0 | 37 | 78 |
| cal | PE_R0_BLEND | 0.5 | 0 | 0 | 0 | 27 | 101 |
| cal | PE_R1_FOURWAY | 0.25 | 0 | 5 | 0 | 38 | 82 |
| cal | PE_R1_FOURWAY | 0.5 | 0 | 2 | 0 | 27 | 103 |
| cal | PE_R2_GLOBAL | 0.25 | 4 | 4 | 1 | 37 | 77 |
| cal | PE_R2_GLOBAL | 0.5 | 3 | 1 | 0 | 27 | 99 |
| cal | PE_R3_LOCAL | 0.25 | 3 | 5 | 1 | 37 | 79 |
| cal | PE_R3_LOCAL | 0.5 | 3 | 1 | 0 | 27 | 99 |
| cal | PE_R3_LOCAL_CALDELTA | 0.25 | 2 | 4 | 0 | 37 | 80 |
| cal | PE_R3_LOCAL_CALDELTA | 0.5 | 2 | 0 | 0 | 27 | 99 |
| cal | PE_D1_LOGPOOL | 0.25 | 2 | 1 | 0 | 36 | 78 |
| cal | PE_D1_LOGPOOL | 0.5 | 1 | 0 | 0 | 27 | 100 |
| cal | PE_D1_ANCHORED | 0.25 | 2 | 1 | 0 | 36 | 78 |
| cal | PE_D1_ANCHORED | 0.5 | 1 | 0 | 0 | 27 | 100 |
| cal | PE_D2_GROUPED | 0.25 | 0 | 5 | 0 | 38 | 82 |
| cal | PE_D2_GROUPED | 0.5 | 0 | 2 | 0 | 27 | 103 |
| cal | PE_D3_DIAGONAL | 0.25 | 2 | 1 | 0 | 36 | 78 |
| cal | PE_D3_DIAGONAL | 0.5 | 1 | 0 | 0 | 27 | 100 |
| cal | PE_D4_LINEAGE | 0.25 | 2 | 2 | 0 | 36 | 79 |
| cal | PE_D4_LINEAGE | 0.5 | 1 | 1 | 0 | 27 | 101 |
| cal | PE_D4_SHUFFLED | 0.25 | 2 | 2 | 0 | 36 | 79 |
| cal | PE_D4_SHUFFLED | 0.5 | 1 | 1 | 0 | 27 | 101 |
| cal | PE_COMBO_D4_R3 | 0.25 | 4 | 3 | 1 | 37 | 76 |
| cal | PE_COMBO_D4_R3 | 0.5 | 4 | 0 | 0 | 27 | 97 |
| replica | N0 | 0.25 | 10 | 25 | 0 | 40 | 105 |
| replica | N0 | 0.5 | 3 | 17 | 0 | 36 | 151 |
| replica | RV_A7_COS_REFIT | 0.25 | 4 | 13 | 0 | 37 | 98 |
| replica | RV_A7_COS_REFIT | 0.5 | 1 | 10 | 0 | 32 | 146 |
| replica | AW_E03_FC_FROZEN_A7 | 0.25 | 0 | 0 | 0 | 37 | 96 |
| replica | AW_E03_FC_FROZEN_A7 | 0.5 | 0 | 0 | 0 | 33 | 143 |
| replica | AW_E03_FC_FROZEN_DIRECT | 0.25 | 16 | 26 | 0 | 44 | 103 |
| replica | AW_E03_FC_FROZEN_DIRECT | 0.5 | 11 | 16 | 0 | 38 | 148 |
| replica | AW_E03_OVR_DIRECT | 0.25 | 24 | 20 | 0 | 40 | 91 |
| replica | AW_E03_OVR_DIRECT | 0.5 | 17 | 12 | 0 | 36 | 137 |
| replica | AW_E03_OVR_A7 | 0.25 | 7 | 5 | 0 | 38 | 92 |
| replica | AW_E03_OVR_A7 | 0.5 | 1 | 2 | 0 | 35 | 141 |
| replica | PE_R0_BLEND | 0.25 | 2 | 5 | 0 | 39 | 96 |
| replica | PE_R0_BLEND | 0.5 | 1 | 2 | 0 | 34 | 142 |
| replica | PE_R1_FOURWAY | 0.25 | 8 | 4 | 0 | 39 | 95 |
| replica | PE_R1_FOURWAY | 0.5 | 4 | 3 | 0 | 37 | 143 |
| replica | PE_R2_GLOBAL | 0.25 | 29 | 20 | 0 | 32 | 78 |
| replica | PE_R2_GLOBAL | 0.5 | 20 | 14 | 0 | 29 | 127 |
| replica | PE_R3_LOCAL | 0.25 | 30 | 19 | 0 | 33 | 76 |
| replica | PE_R3_LOCAL | 0.5 | 20 | 13 | 0 | 30 | 126 |
| replica | PE_R3_LOCAL_CALDELTA | 0.25 | 23 | 15 | 0 | 34 | 82 |
| replica | PE_R3_LOCAL_CALDELTA | 0.5 | 13 | 10 | 0 | 32 | 132 |
| replica | PE_D1_LOGPOOL | 0.25 | 4 | 6 | 0 | 39 | 95 |
| replica | PE_D1_LOGPOOL | 0.5 | 2 | 3 | 0 | 34 | 142 |
| replica | PE_D1_ANCHORED | 0.25 | 1 | 4 | 0 | 39 | 97 |
| replica | PE_D1_ANCHORED | 0.5 | 1 | 1 | 0 | 34 | 142 |
| replica | PE_D2_GROUPED | 0.25 | 6 | 3 | 0 | 39 | 97 |
| replica | PE_D2_GROUPED | 0.5 | 4 | 2 | 0 | 37 | 143 |
| replica | PE_D3_DIAGONAL | 0.25 | 3 | 5 | 0 | 37 | 97 |
| replica | PE_D3_DIAGONAL | 0.5 | 1 | 3 | 0 | 34 | 143 |
| replica | PE_D4_LINEAGE | 0.25 | 4 | 5 | 0 | 37 | 97 |
| replica | PE_D4_LINEAGE | 0.5 | 2 | 3 | 0 | 34 | 143 |
| replica | PE_D4_SHUFFLED | 0.25 | 4 | 5 | 0 | 37 | 97 |
| replica | PE_D4_SHUFFLED | 0.5 | 2 | 3 | 0 | 34 | 143 |
| replica | PE_COMBO_D4_R3 | 0.25 | 32 | 17 | 0 | 34 | 71 |
| replica | PE_COMBO_D4_R3 | 0.5 | 22 | 12 | 0 | 30 | 123 |

## CAL object outcomes

| Method | changed_B | corrected_B | harmed_B | corrected_N0 | harmed_N0 | unused_correct_source | extra_O_correction_survived | F_correct_O_wrong_harm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N0 | 63 | 1 | 0 | 0 | 0 | 5 | 0 | 1 |
| RV_A7_COS_REFIT | 48 | 2 | 0 | 1 | 0 | 4 | 0 | 1 |
| AW_E03_FC_FROZEN_A7 | 0 | 0 | 0 | 0 | 1 | 6 | 0 | 1 |
| AW_E03_FC_FROZEN_DIRECT | 84 | 2 | 4 | 2 | 5 | 8 | 0 | 0 |
| AW_E03_OVR_DIRECT | 88 | 3 | 4 | 3 | 5 | 7 | 2 | 1 |
| AW_E03_OVR_A7 | 23 | 0 | 0 | 0 | 1 | 6 | 0 | 1 |
| PE_R0_BLEND | 14 | 0 | 0 | 0 | 1 | 6 | 0 | 1 |
| PE_R1_FOURWAY | 31 | 0 | 2 | 0 | 3 | 8 | 0 | 1 |
| PE_R2_GLOBAL | 83 | 3 | 1 | 3 | 2 | 4 | 2 | 1 |
| PE_R3_LOCAL | 78 | 3 | 1 | 3 | 2 | 4 | 2 | 1 |
| PE_R3_LOCAL_CALDELTA | 50 | 2 | 0 | 2 | 1 | 5 | 1 | 1 |
| PE_D1_LOGPOOL | 39 | 1 | 0 | 1 | 1 | 6 | 0 | 1 |
| PE_D1_ANCHORED | 21 | 1 | 0 | 1 | 1 | 6 | 0 | 1 |
| PE_D2_GROUPED | 22 | 0 | 2 | 0 | 3 | 8 | 0 | 1 |
| PE_D3_DIAGONAL | 29 | 1 | 0 | 1 | 1 | 6 | 0 | 1 |
| PE_D4_LINEAGE | 33 | 1 | 1 | 1 | 2 | 7 | 0 | 1 |
| PE_D4_SHUFFLED | 33 | 1 | 1 | 1 | 2 | 7 | 0 | 1 |
| PE_COMBO_D4_R3 | 80 | 4 | 0 | 4 | 1 | 3 | 2 | 1 |

Common-pair identifiable outcomes: `{'F_right/O_right': 8, 'F_right/O_wrong': 1, 'F_wrong/O_wrong': 11, 'F_wrong/O_right': 2}`. Probability-pool blocking margins >1: 1/6 eligible wrong-base objects.

Object corrections are not AP summands. Full all-threshold match-set changes, duplicates, ignored events, false positives and per-class AP changes are saved in matcher_contrasts.json.gz. The first two corrected and first two harmed owner IDs per scene/method are stored in summary.json without manual example selection.

## REPLICA object outcomes

| Method | changed_B | corrected_B | harmed_B | corrected_N0 | harmed_N0 | unused_correct_source | extra_O_correction_survived | F_correct_O_wrong_harm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N0 | 68 | 3 | 19 | 0 | 0 | 40 | 7 | 3 |
| RV_A7_COS_REFIT | 52 | 2 | 10 | 12 | 4 | 33 | 8 | 3 |
| AW_E03_FC_FROZEN_A7 | 0 | 0 | 0 | 19 | 3 | 24 | 8 | 1 |
| AW_E03_FC_FROZEN_DIRECT | 147 | 12 | 18 | 27 | 17 | 30 | 0 | 0 |
| AW_E03_OVR_DIRECT | 150 | 19 | 13 | 33 | 11 | 18 | 16 | 4 |
| AW_E03_OVR_A7 | 29 | 1 | 2 | 19 | 4 | 25 | 8 | 2 |
| PE_R0_BLEND | 14 | 1 | 2 | 19 | 4 | 25 | 8 | 2 |
| PE_R1_FOURWAY | 42 | 4 | 3 | 21 | 4 | 23 | 8 | 2 |
| PE_R2_GLOBAL | 135 | 22 | 14 | 32 | 8 | 22 | 16 | 4 |
| PE_R3_LOCAL | 131 | 22 | 13 | 33 | 8 | 21 | 16 | 4 |
| PE_R3_LOCAL_CALDELTA | 92 | 14 | 10 | 24 | 4 | 23 | 14 | 3 |
| PE_D1_LOGPOOL | 42 | 2 | 4 | 18 | 4 | 26 | 9 | 2 |
| PE_D1_ANCHORED | 26 | 1 | 1 | 19 | 3 | 24 | 8 | 1 |
| PE_D2_GROUPED | 38 | 4 | 2 | 23 | 5 | 22 | 8 | 1 |
| PE_D3_DIAGONAL | 33 | 1 | 3 | 17 | 3 | 26 | 7 | 2 |
| PE_D4_LINEAGE | 35 | 2 | 3 | 18 | 3 | 25 | 8 | 2 |
| PE_D4_SHUFFLED | 35 | 2 | 3 | 18 | 3 | 25 | 8 | 2 |
| PE_COMBO_D4_R3 | 130 | 24 | 12 | 34 | 6 | 20 | 16 | 4 |

Common-pair identifiable outcomes: `{'F_right/O_right': 78, 'F_right/O_wrong': 4, 'F_wrong/O_right': 16, 'F_wrong/O_wrong': 59}`. Probability-pool blocking margins >1: 5/24 eligible wrong-base objects.

Object corrections are not AP summands. Full all-threshold match-set changes, duplicates, ignored events, false positives and per-class AP changes are saved in matcher_contrasts.json.gz. The first two corrected and first two harmed owner IDs per scene/method are stored in summary.json without manual example selection.

## Per-scene changes versus B (official)

| Scene | Method | Δapall | Δap50 | Δap25 | Δmiou | Δmacc |
| --- | --- | --- | --- | --- | --- | --- |
| office0 | N0 | -0.277778 | -2.500000 | -2.500000 | -3.305060 | -1.244454 |
| office0 | RV_A7_COS_REFIT | -0.277778 | -2.500000 | -2.500000 | -0.994822 | 1.115053 |
| office0 | AW_E03_FC_FROZEN_A7 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| office0 | AW_E03_FC_FROZEN_DIRECT | 3.402778 | 5.625000 | 11.250000 | 7.883037 | 7.434102 |
| office0 | AW_E03_OVR_DIRECT | 3.680556 | 7.500000 | 13.750000 | 12.467977 | 10.199270 |
| office0 | AW_E03_OVR_A7 | 3.055556 | 2.500000 | 5.000000 | 4.034527 | 4.540881 |
| office0 | PE_R0_BLEND | 3.055556 | 2.500000 | 2.500000 | 2.608569 | 3.176459 |
| office0 | PE_R1_FOURWAY | 3.055556 | 2.500000 | 5.000000 | 4.398093 | 4.991331 |
| office0 | PE_R2_GLOBAL | 3.680556 | 7.500000 | 13.750000 | 11.964951 | 7.901569 |
| office0 | PE_R3_LOCAL | 3.680556 | 7.500000 | 13.750000 | 11.964951 | 7.901569 |
| office0 | PE_R3_LOCAL_CALDELTA | 3.680556 | 7.500000 | 10.000000 | 6.713227 | 6.811386 |
| office0 | PE_D1_LOGPOOL | -0.277778 | -2.500000 | -2.500000 | -3.069896 | -0.473357 |
| office0 | PE_D1_ANCHORED | 0.000000 | 0.000000 | 0.000000 | -1.083976 | -1.736493 |
| office0 | PE_D2_GROUPED | 3.125000 | 3.125000 | 7.500000 | 4.887855 | 5.577784 |
| office0 | PE_D3_DIAGONAL | -0.277778 | -2.500000 | 0.000000 | 0.417934 | 2.561597 |
| office0 | PE_D4_LINEAGE | -0.277778 | -2.500000 | 0.000000 | 0.417934 | 2.561597 |
| office0 | PE_D4_SHUFFLED | -0.277778 | -2.500000 | 0.000000 | 0.417934 | 2.561597 |
| office0 | PE_COMBO_D4_R3 | 7.569444 | 12.500000 | 18.750000 | 15.441148 | 11.499784 |
| office1 | N0 | -0.326797 | 0.000000 | 0.000000 | -4.032041 | -4.041410 |
| office1 | RV_A7_COS_REFIT | -0.653595 | -5.882353 | 0.000000 | -0.965597 | -0.939927 |
| office1 | AW_E03_FC_FROZEN_A7 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| office1 | AW_E03_FC_FROZEN_DIRECT | 2.941176 | -5.882353 | -5.882353 | -0.823301 | 1.860258 |
| office1 | AW_E03_OVR_DIRECT | 2.941176 | -5.882353 | 0.000000 | 1.896386 | 2.816277 |
| office1 | AW_E03_OVR_A7 | -0.653595 | -5.882353 | 0.000000 | -0.400039 | -0.604564 |
| office1 | PE_R0_BLEND | -0.653595 | -5.882353 | 0.000000 | -0.573627 | -0.604564 |
| office1 | PE_R1_FOURWAY | -0.653595 | -5.882353 | -5.882353 | -3.979482 | -2.451638 |
| office1 | PE_R2_GLOBAL | 0.816993 | -2.941176 | 2.941176 | 0.449230 | 1.144046 |
| office1 | PE_R3_LOCAL | 0.816993 | -2.941176 | 2.941176 | 0.221957 | 0.911872 |
| office1 | PE_R3_LOCAL_CALDELTA | -0.653595 | -5.882353 | 0.000000 | -1.124253 | -0.499487 |
| office1 | PE_D1_LOGPOOL | 0.000000 | 0.000000 | 5.882353 | 1.537857 | 1.511712 |
| office1 | PE_D1_ANCHORED | 0.000000 | 0.000000 | 0.000000 | -0.272236 | -0.335363 |
| office1 | PE_D2_GROUPED | -0.653595 | -5.882353 | -5.882353 | -3.979482 | -2.451638 |
| office1 | PE_D3_DIAGONAL | 0.000000 | 0.000000 | 0.000000 | -0.272223 | -0.335363 |
| office1 | PE_D4_LINEAGE | 0.000000 | 0.000000 | 0.000000 | -0.272223 | -0.335363 |
| office1 | PE_D4_SHUFFLED | 0.000000 | 0.000000 | 0.000000 | -0.272223 | -0.335363 |
| office1 | PE_COMBO_D4_R3 | 0.163399 | -8.823529 | -2.941176 | -1.959593 | -2.885461 |
| office2 | N0 | -1.045752 | -1.176471 | -7.843137 | -4.065370 | -4.108522 |
| office2 | RV_A7_COS_REFIT | -1.045752 | -1.176471 | -7.058824 | -4.763113 | -4.418440 |
| office2 | AW_E03_FC_FROZEN_A7 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| office2 | AW_E03_FC_FROZEN_DIRECT | -0.740741 | -7.450980 | -8.627451 | -2.081638 | -3.673473 |
| office2 | AW_E03_OVR_DIRECT | -0.010893 | 1.568627 | -1.568627 | 2.782113 | 2.184813 |
| office2 | AW_E03_OVR_A7 | 0.000000 | 0.000000 | -7.843137 | -3.650340 | -3.483751 |
| office2 | PE_R0_BLEND | 0.000000 | 0.000000 | -7.843137 | -3.457988 | -3.496640 |
| office2 | PE_R1_FOURWAY | 0.000000 | 0.000000 | -1.960784 | 0.149837 | -0.202067 |
| office2 | PE_R2_GLOBAL | 1.106754 | 4.705882 | 2.745098 | 1.153405 | 2.818468 |
| office2 | PE_R3_LOCAL | 1.106754 | 4.705882 | 2.745098 | 1.153405 | 2.818468 |
| office2 | PE_R3_LOCAL_CALDELTA | 0.032680 | 2.352941 | -5.490196 | -2.808236 | -1.679452 |
| office2 | PE_D1_LOGPOOL | 0.000000 | 0.000000 | -7.843137 | -3.319728 | -3.392692 |
| office2 | PE_D1_ANCHORED | 0.000000 | 0.000000 | -7.843137 | -2.792419 | -3.405125 |
| office2 | PE_D2_GROUPED | 0.000000 | 0.000000 | -1.176471 | 0.237602 | -0.329448 |
| office2 | PE_D3_DIAGONAL | 0.000000 | 0.000000 | -7.843137 | -2.804428 | -3.418014 |
| office2 | PE_D4_LINEAGE | 0.000000 | 0.000000 | -7.843137 | -2.804428 | -3.418014 |
| office2 | PE_D4_SHUFFLED | 0.000000 | 0.000000 | -7.843137 | -2.804428 | -3.418014 |
| office2 | PE_COMBO_D4_R3 | 1.106754 | 4.705882 | -3.137255 | -2.182104 | -0.476106 |
| office3 | N0 | -1.380837 | -7.391304 | -5.652174 | -1.335483 | -2.112088 |
| office3 | RV_A7_COS_REFIT | -0.704509 | -2.608696 | -3.043478 | -1.749979 | -1.645210 |
| office3 | AW_E03_FC_FROZEN_A7 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| office3 | AW_E03_FC_FROZEN_DIRECT | -0.296296 | -4.007246 | -5.637681 | -4.913455 | -2.460222 |
| office3 | AW_E03_OVR_DIRECT | 0.966184 | 3.804348 | -0.905797 | -3.777235 | 1.257035 |
| office3 | AW_E03_OVR_A7 | 0.000000 | 0.000000 | 4.347826 | 3.204671 | 3.453221 |
| office3 | PE_R0_BLEND | 0.000000 | 0.000000 | 0.000000 | 0.116380 | 0.118523 |
| office3 | PE_R1_FOURWAY | 0.483092 | 1.086957 | 5.434783 | 4.191752 | 4.575594 |
| office3 | PE_R2_GLOBAL | 0.062399 | -1.304348 | -0.760870 | 0.597406 | 3.020231 |
| office3 | PE_R3_LOCAL | 0.062399 | -1.304348 | -0.942029 | 0.544724 | 3.000542 |
| office3 | PE_R3_LOCAL_CALDELTA | 0.003019 | -0.099638 | 1.711957 | 1.289192 | 4.403261 |
| office3 | PE_D1_LOGPOOL | -0.124799 | -0.869565 | -0.869565 | -0.458095 | -0.460315 |
| office3 | PE_D1_ANCHORED | 0.000000 | 0.000000 | 0.000000 | 0.002220 | 0.000000 |
| office3 | PE_D2_GROUPED | 0.483092 | 1.086957 | 5.434783 | 4.191752 | 4.575594 |
| office3 | PE_D3_DIAGONAL | -0.483092 | -4.347826 | 0.000000 | 1.446351 | 0.671050 |
| office3 | PE_D4_LINEAGE | -0.483092 | -4.347826 | 0.000000 | 1.446351 | 0.671050 |
| office3 | PE_D4_SHUFFLED | -0.483092 | -4.347826 | 0.000000 | 1.446351 | 0.671050 |
| office3 | PE_COMBO_D4_R3 | 0.545491 | 0.869565 | 1.231884 | 1.700225 | 4.156447 |
| office4 | N0 | 0.000000 | 0.000000 | 0.000000 | -4.796208 | -4.859377 |
| office4 | RV_A7_COS_REFIT | 0.000000 | 0.000000 | 0.000000 | -0.099986 | -0.031606 |
| office4 | AW_E03_FC_FROZEN_A7 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| office4 | AW_E03_FC_FROZEN_DIRECT | -7.692308 | -11.538462 | -28.461538 | -16.568418 | -21.028098 |
| office4 | AW_E03_OVR_DIRECT | -2.564103 | -3.846154 | -19.230769 | -9.034352 | -13.179448 |
| office4 | AW_E03_OVR_A7 | 0.000000 | 0.000000 | 0.769231 | 0.142761 | 0.144132 |
| office4 | PE_R0_BLEND | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| office4 | PE_R1_FOURWAY | 0.000000 | 0.000000 | 0.769231 | 0.150487 | 0.144132 |
| office4 | PE_R2_GLOBAL | 0.000000 | 0.000000 | 0.000000 | 0.023314 | 0.082901 |
| office4 | PE_R3_LOCAL | 0.000000 | 0.000000 | 0.769231 | 0.137016 | 0.207057 |
| office4 | PE_R3_LOCAL_CALDELTA | 0.000000 | 0.000000 | 1.538462 | 0.332974 | 0.326619 |
| office4 | PE_D1_LOGPOOL | 0.000000 | 0.000000 | 0.000000 | -0.002500 | 0.000000 |
| office4 | PE_D1_ANCHORED | 0.000000 | 0.000000 | 0.000000 | -0.001371 | 0.000000 |
| office4 | PE_D2_GROUPED | 0.000000 | 0.000000 | 0.000000 | 0.006355 | 0.000000 |
| office4 | PE_D3_DIAGONAL | 0.000000 | 0.000000 | 0.000000 | -0.001371 | 0.000000 |
| office4 | PE_D4_LINEAGE | 0.000000 | 0.000000 | 0.000000 | 0.006355 | 0.000000 |
| office4 | PE_D4_SHUFFLED | 0.000000 | 0.000000 | 0.000000 | 0.006355 | 0.000000 |
| office4 | PE_COMBO_D4_R3 | 0.000000 | 0.000000 | 1.538462 | 0.414763 | 0.494456 |
| room0 | N0 | -7.396384 | -12.053571 | -9.576720 | -4.200477 | -7.517874 |
| room0 | RV_A7_COS_REFIT | -4.232804 | -4.761905 | -3.095238 | -3.130344 | -3.563242 |
| room0 | AW_E03_FC_FROZEN_A7 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| room0 | AW_E03_FC_FROZEN_DIRECT | -1.105967 | -3.240741 | -3.240741 | -1.404255 | -1.246310 |
| room0 | AW_E03_OVR_DIRECT | -0.863463 | -3.835979 | -4.282407 | -1.540675 | -1.287658 |
| room0 | AW_E03_OVR_A7 | -0.231481 | -0.347222 | 1.296296 | 0.902999 | 0.936864 |
| room0 | PE_R0_BLEND | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| room0 | PE_R1_FOURWAY | -1.388889 | -2.083333 | -2.083333 | -1.016219 | -1.049967 |
| room0 | PE_R2_GLOBAL | -2.985621 | -3.845899 | -0.780423 | -1.628434 | -1.851426 |
| room0 | PE_R3_LOCAL | -2.985621 | -3.845899 | -0.780423 | -1.492343 | -1.792954 |
| room0 | PE_R3_LOCAL_CALDELTA | -5.911045 | -6.594466 | -3.789683 | -3.287726 | -3.861104 |
| room0 | PE_D1_LOGPOOL | -4.166667 | -4.166667 | -4.166667 | -3.459119 | -3.460439 |
| room0 | PE_D1_ANCHORED | -4.166667 | -4.166667 | -4.166667 | -3.459119 | -3.460439 |
| room0 | PE_D2_GROUPED | 0.000000 | 0.000000 | 0.000000 | 0.039031 | 0.058323 |
| room0 | PE_D3_DIAGONAL | -4.166667 | -4.166667 | -4.166667 | -3.459119 | -3.460439 |
| room0 | PE_D4_LINEAGE | -4.166667 | -4.166667 | -4.166667 | -3.394486 | -3.401967 |
| room0 | PE_D4_SHUFFLED | -4.166667 | -4.166667 | -4.166667 | -3.394486 | -3.401967 |
| room0 | PE_COMBO_D4_R3 | -5.300436 | -5.929233 | -2.863757 | -3.198342 | -3.516441 |
| room1 | N0 | 0.000000 | 0.000000 | -5.263158 | -1.363743 | -3.396450 |
| room1 | RV_A7_COS_REFIT | -0.167084 | -0.751880 | -0.751880 | 3.808373 | 4.006827 |
| room1 | AW_E03_FC_FROZEN_A7 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| room1 | AW_E03_FC_FROZEN_DIRECT | 6.168198 | 6.265664 | 2.192982 | -1.630710 | 0.043143 |
| room1 | AW_E03_OVR_DIRECT | 6.649262 | 7.380952 | 10.714286 | 2.527223 | 5.363629 |
| room1 | AW_E03_OVR_A7 | 0.000000 | 0.000000 | 0.000000 | -4.494072 | -3.955544 |
| room1 | PE_R0_BLEND | 0.000000 | 0.000000 | 0.000000 | -4.492030 | -3.955544 |
| room1 | PE_R1_FOURWAY | 6.286550 | 6.798246 | 8.646617 | 1.251232 | 1.910198 |
| room1 | PE_R2_GLOBAL | -0.231830 | 0.100251 | 0.939850 | 2.031446 | 3.731142 |
| room1 | PE_R3_LOCAL | -0.231830 | 0.100251 | 0.939850 | 2.031446 | 3.731142 |
| room1 | PE_R3_LOCAL_CALDELTA | -0.231830 | 0.100251 | 0.939850 | 1.648157 | 3.345403 |
| room1 | PE_D1_LOGPOOL | 1.156363 | 2.117794 | 4.135338 | -1.427428 | -0.855565 |
| room1 | PE_D1_ANCHORED | 1.754386 | 2.631579 | 2.631579 | -2.557722 | -2.026907 |
| room1 | PE_D2_GROUPED | 6.850459 | 7.142857 | 7.142857 | 0.847605 | 1.682196 |
| room1 | PE_D3_DIAGONAL | 1.754386 | 2.631579 | 2.631579 | 1.126159 | 1.106642 |
| room1 | PE_D4_LINEAGE | 1.865776 | 3.383459 | 3.383459 | 1.551960 | 1.553889 |
| room1 | PE_D4_SHUFFLED | 1.865776 | 3.383459 | 3.383459 | 1.551960 | 1.553889 |
| room1 | PE_COMBO_D4_R3 | 0.645363 | 2.731830 | 3.571429 | 3.787895 | 5.915004 |
| room2 | N0 | -1.863426 | -2.708333 | -2.291667 | -4.225292 | -5.122528 |
| room2 | RV_A7_COS_REFIT | 0.567130 | 1.197917 | -0.729167 | 0.524261 | 0.375252 |
| room2 | AW_E03_FC_FROZEN_A7 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| room2 | AW_E03_FC_FROZEN_DIRECT | -6.944444 | -12.500000 | -12.968750 | -8.438477 | -7.717539 |
| room2 | AW_E03_OVR_DIRECT | -2.777778 | -6.250000 | -6.718750 | -4.419009 | -3.893330 |
| room2 | AW_E03_OVR_A7 | 0.000000 | 0.000000 | -1.562500 | -0.443696 | -0.444730 |
| room2 | PE_R0_BLEND | 0.000000 | 0.000000 | -1.562500 | -0.443696 | -0.444730 |
| room2 | PE_R1_FOURWAY | 0.000000 | 0.000000 | 0.000000 | -0.295451 | 0.261704 |
| room2 | PE_R2_GLOBAL | -3.657407 | -8.177083 | -11.875000 | -5.485467 | -5.871350 |
| room2 | PE_R3_LOCAL | -2.615741 | -6.614583 | -10.312500 | -4.819204 | -5.204223 |
| room2 | PE_R3_LOCAL_CALDELTA | 0.162037 | -0.364583 | -4.062500 | -1.264252 | -1.594333 |
| room2 | PE_D1_LOGPOOL | 0.000000 | 0.000000 | -1.562500 | -0.327919 | -0.328953 |
| room2 | PE_D1_ANCHORED | 0.000000 | 0.000000 | -1.562500 | -0.443696 | -0.444730 |
| room2 | PE_D2_GROUPED | 0.000000 | 0.000000 | 0.000000 | -0.153178 | 0.261704 |
| room2 | PE_D3_DIAGONAL | 0.000000 | 0.000000 | 0.000000 | 0.115777 | 0.115777 |
| room2 | PE_D4_LINEAGE | 0.000000 | 0.000000 | 0.000000 | 0.115777 | 0.115777 |
| room2 | PE_D4_SHUFFLED | 0.000000 | 0.000000 | 0.000000 | 0.115777 | 0.115777 |
| room2 | PE_COMBO_D4_R3 | 0.162037 | -0.364583 | -2.500000 | -0.149455 | 0.129877 |

## Seven-scene sensitivity of frozen nominee

| Omitted | Δapall vs B | Δap50 vs B | Δap25 vs B | Δmiou vs B | Δmacc vs B |
| --- | --- | --- | --- | --- | --- |
| office0 | -0.920617 | -1.992899 | -2.174528 | -2.004962 | -0.175669 |
| office1 | 0.084820 | 1.232783 | 0.618745 | 1.665043 | 3.652369 |
| office2 | -0.094447 | -1.029935 | 0.367288 | 0.989942 | 1.441775 |
| office3 | -0.286989 | -1.154942 | -0.407107 | -0.438588 | 0.406660 |
| office4 | -0.179455 | -0.774959 | -0.689226 | -0.346833 | 0.845796 |
| room0 | 1.249929 | 0.528958 | 0.156493 | 1.187403 | 3.040535 |
| room1 | -1.320107 | -2.228663 | -1.257676 | -1.171257 | -0.893863 |
| room2 | 0.126638 | -0.046109 | 1.234933 | 1.055187 | 1.625308 |

These are eight actual seven-scene released pools, not independent trials or bootstrap confidence intervals.

Descriptive Replica best APall: **PE_D2_GROUPED**. Metric-only Pareto frontier across all five metrics: AW_E03_FC_FROZEN_A7, PE_D2_GROUPED, PE_COMBO_D4_R3. These do not replace the CAL-frozen nominee.

The measured result favors the simple D2 grouped-probability control for descriptive Replica APall, while FC fusion retains higher mIoU. The CAL-frozen complex nominee does not transfer its CAL advantage: Replica differences versus B are -0.282850 pp APall and -0.450521 pp mIoU. Keep the original FC research baseline and unchanged deployment; a future independent confirmation of the simple grouping tradeoff is better justified than advancing an unsupported dependence/acquisition mechanism.

## Physical and logical costs

This study invoked **0 image forwards, 0 image-model loads, 0 text-model forwards, 0 training jobs, 0 new temperature fits, 0 geometry reconstructions and 0 downloaded bytes**. Necessary original model work is not free: the following Replica unions deduplicate only exact input/model operations.

| Method | Crop inputs | Dense image encodes | Region pooling/projection |
| --- | --- | --- | --- |
| N0 | 25180 | 0 | 0 |
| RV_A7_COS_REFIT | 37270 | 0 | 0 |
| AW_E03_FC_FROZEN_A7 | 30906 | 673 | 1052 |
| AW_E03_FC_FROZEN_DIRECT | 25180 | 673 | 1052 |
| AW_E03_OVR_DIRECT | 25180 | 673 | 1052 |
| AW_E03_OVR_A7 | 30906 | 673 | 1052 |
| PE_R0_BLEND | 30906 | 1346 | 2104 |
| PE_R1_FOURWAY | 30906 | 1346 | 2104 |
| PE_R2_GLOBAL | 30906 | 1346 | 2104 |
| PE_R3_LOCAL | 30906 | 1346 | 2104 |
| PE_R3_LOCAL_CALDELTA | 30906 | 1346 | 2104 |
| PE_D1_LOGPOOL | 30906 | 673 | 1052 |
| PE_D1_ANCHORED | 30906 | 673 | 1052 |
| PE_D2_GROUPED | 30906 | 673 | 1052 |
| PE_D3_DIAGONAL | 30906 | 673 | 1052 |
| PE_D4_LINEAGE | 30906 | 673 | 1052 |
| PE_D4_SHUFFLED | 30906 | 673 | 1052 |
| PE_COMBO_D4_R3 | 30906 | 1346 | 2104 |

| CPU phase | Recorded jobs | Sum of job seconds (not wall latency) | Largest observed process RSS KiB |
| --- | --- | --- | --- |
| recover | 10 | 190.221 | 3855784 |
| predict | 10 | 664.288 | 1353228 |
| evaluate | 10 | 228.936 | 1051880 |

| Original recovery split | Sum of elapsed seconds | Dependence operator seconds | Remaining recovery/read/validation seconds |
| --- | --- | --- | --- |
| CAL | 108.349 | 66.303 | 42.046 |
| Replica | 129.731 | 11.290 | 118.440 |

Timing includes I/O and cache validation. Concurrent process times are not summed into wall latency; process memory maxima are not summed into a system peak. Historical geometry/text inference costs and unrecorded failed-attempt durations are unknown, not zero. Original model/input manifests are retained as references. Scalar-grid evaluations share identical-output caches with explicit alias records.

CAL sweep prediction/evaluation times are the two CAL scene records in execution/predict_*.json and execution/evaluate_*.json; Replica records are separate. Pool evaluation timers are in each pooled receipt. Report rendering timers are in report/render_*.json; push/publication time is in the external publication/final.json. Recovery totals above include the initial standalone CAL recovery; the process-job table can instead contain its later cache-validation invocation. These are distinct accounting views and must not be added.

## Interpretation limits

Largest measured local outside-probability drift: 0. Mass conservation, PSD and a unique graph solution establish operator properties only; they do not establish accuracy or causal error correction.

The four fixed CAL grids are the only new supervised scalar choices. Source temperatures and original Q policy already used supervision. Pair classes are not independent samples. Common-temperature residuals compare paired class scores, never latent vectors; any measured text difference prevents a visual-only attribution.

Dependence is supported only if D4 exceeds D3, the same-anchor control, and shuffled dependence on relevant metrics. A simple-control win must not be relabeled as a proposed mechanism win. Conditional acquisition, E06, new models and geometry changes remain QUEUED_NOT_EXECUTED.
