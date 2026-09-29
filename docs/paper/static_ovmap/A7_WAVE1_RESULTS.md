# A7 wave-1 measured results

This is development/regression evidence on two previously exposed ScanNet CAL scenes and eight historical Replica scenes. It is not fresh confirmation. All methods retain the same native geometry, owner registry, projection and complete map. E02 changes recognition masks only; it does not improve reconstructed shape.

## A. Performance and matched comparisons

Main results are released dataset pools with OFFICIAL_CURRENT_CLASS ranking. Values are percentages; deltas are percentage points. APall aliases released uAP over actual IoU thresholds 0.50–0.90, with AP25 separate; it is not COCO AP through 0.95.

### CAL official released pool

| Method | APall | AP50 | AP25 | mIoU | mAcc | ΔAPall vs N0 | ΔAPall vs A7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AW_C0_SO400M_A7 | 3.730059 | 14.091911 | 27.998771 | 22.812833 | 30.105945 | -0.505232 | -0.002718 |
| AW_C0_SO400M_DIRECT | 4.291716 | 17.238757 | 26.164756 | 21.910611 | 29.717997 | +0.056425 | +0.558939 |
| AW_COMBO_QR | 3.726239 | 14.013815 | 30.090087 | 23.735309 | 31.434342 | -0.509051 | -0.006537 |
| AW_E01_GMED_A7 | 3.732776 | 14.181265 | 28.512957 | 22.729982 | 30.727156 | -0.502515 | +0.000000 |
| AW_E01_GMED_DIRECT | 3.704418 | 14.095201 | 26.820175 | 17.725941 | 25.068171 | -0.530873 | -0.028358 |
| AW_E01_RAW_EQ_A7 | 3.732776 | 14.181265 | 28.512957 | 22.698982 | 30.746178 | -0.502515 | +0.000000 |
| AW_E01_RAW_EQ_DIRECT | 3.698600 | 14.065623 | 26.377948 | 20.387349 | 28.410039 | -0.536690 | -0.034176 |
| AW_E01_UNIT_EQ_A7 | 3.732776 | 14.181265 | 28.512957 | 22.698982 | 30.746178 | -0.502515 | +0.000000 |
| AW_E01_UNIT_EQ_DIRECT | 3.698600 | 14.065623 | 26.369130 | 20.378726 | 28.410039 | -0.536690 | -0.034176 |
| AW_E02_GLOBAL_A7 | 3.732776 | 14.181265 | 28.512957 | 22.645275 | 30.468596 | -0.502515 | +0.000000 |
| AW_E02_GLOBAL_DIRECT | 3.987769 | 14.496620 | 24.268445 | 20.583726 | 26.463839 | -0.247522 | +0.254992 |
| AW_E02_SAM2_A7 | 3.743512 | 14.230065 | 28.265637 | 22.457145 | 30.325458 | -0.491778 | +0.010736 |
| AW_E02_SAM2_DIRECT | 4.118901 | 16.903660 | 26.447310 | 21.926746 | 27.407505 | -0.116390 | +0.386124 |
| AW_E03_FC_FROZEN_A7 | 4.224843 | 14.905203 | 31.590608 | 24.457288 | 31.322824 | -0.010448 | +0.492067 |
| AW_E03_FC_FROZEN_DIRECT | 3.690231 | 13.494268 | 25.322972 | 20.007402 | 26.119578 | -0.545060 | -0.042545 |
| AW_E03_OVR_A7 | 4.234029 | 14.927249 | 31.782474 | 24.884025 | 31.697263 | -0.001262 | +0.501253 |
| AW_E03_OVR_DIRECT | 5.119048 | 16.119929 | 29.438933 | 22.411281 | 28.469935 | +0.883757 | +1.386271 |
| AW_E04_SHORTLIST | 3.732776 | 14.181265 | 28.535003 | 22.680712 | 30.717638 | -0.502515 | +0.000000 |
| N0 | 4.235291 | 14.985804 | 30.183916 | 22.015249 | 28.920037 | +0.000000 | +0.502515 |
| Q_GAIN | 3.730540 | 14.174906 | 21.543875 | 19.433487 | 23.615675 | -0.504751 | -0.002236 |
| RV_A7_COS_FIXED | 3.732776 | 14.181265 | 28.535003 | 22.743462 | 30.480336 | -0.502515 | +0.000000 |
| RV_A7_COS_REFIT | 3.732776 | 14.181265 | 28.535003 | 22.680712 | 30.717638 | -0.502515 | +0.000000 |
| S_SIGLIP2_AREA | 5.030538 | 18.622869 | 28.983931 | 23.691056 | 31.168455 | +0.795247 | +1.297761 |
| AW_E04_SPATIAL | BLOCKED | — | — | — | — | — | — |
| AW_E04_MIX50 | BLOCKED | — | — | — | — | — | — |

| Matched contrast (new − old) | apall | ap50 | ap25 | miou | macc |
| --- | --- | --- | --- | --- | --- |
| E01_normalization_A7 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| E01_normalization_DIRECT | 0.000000 | 0.000000 | -0.008818 | -0.008623 | 0.000000 |
| E01_robustness_A7 | 0.000000 | 0.000000 | 0.000000 | 0.031000 | -0.019022 |
| E01_robustness_DIRECT | 0.005818 | 0.029578 | 0.451045 | -2.652785 | -3.341868 |
| E02_SAM2_vs_GLOBAL_A7 | 0.010736 | 0.048800 | -0.247320 | -0.188130 | -0.143138 |
| E02_SAM2_vs_GLOBAL_DIRECT | 0.131132 | 2.407040 | 2.178865 | 1.343020 | 0.943666 |
| E03_tuned_vs_frozen_A7 | 0.009186 | 0.022046 | 0.191866 | 0.426737 | 0.374440 |
| E03_tuned_vs_frozen_DIRECT | 1.428816 | 2.625661 | 4.115961 | 2.403879 | 2.350358 |

Interaction is Q+region − Q − region + base A7, using the single CAL-frozen pair.

| Interaction | apall | ap50 | ap25 | miou | macc |
| --- | --- | --- | --- | --- | --- |
| cal | -0.507790 | -0.913433 | -1.670341 | -1.197986 | -0.272438 |

### CAL auxiliary fixed-rank pools and official scene means

| Method | Frozen APall | Frozen AP50 | Frozen AP25 | Mean APall | Mean mIoU | Worst scene ΔAPall vs A7 |
| --- | --- | --- | --- | --- | --- | --- |
| AW_C0_SO400M_A7 | 4.063823 | 14.677162 | 29.005932 | 3.086887 | 22.483909 | -0.005241 |
| AW_C0_SO400M_DIRECT | 4.120615 | 15.454145 | 23.574735 | 3.463141 | 20.870344 | -0.178659 |
| AW_COMBO_QR | 4.069175 | 14.631099 | 31.023729 | 3.083204 | 23.400355 | -0.012607 |
| AW_E01_GMED_A7 | 3.965859 | 13.861487 | 29.509110 | 3.089507 | 22.067949 | +0.000000 |
| AW_E01_GMED_DIRECT | 4.043883 | 14.720569 | 27.309018 | 3.062162 | 17.918470 | -0.054691 |
| AW_E01_RAW_EQ_A7 | 3.965859 | 13.861487 | 29.509110 | 3.089507 | 22.036812 | +0.000000 |
| AW_E01_RAW_EQ_DIRECT | 4.038712 | 14.686581 | 27.011932 | 3.056552 | 19.019821 | -0.065910 |
| AW_E01_UNIT_EQ_A7 | 3.965859 | 13.861487 | 29.509110 | 3.089507 | 22.036812 | +0.000000 |
| AW_E01_UNIT_EQ_DIRECT | 4.038712 | 14.686581 | 27.003113 | 3.056552 | 19.005405 | -0.065910 |
| AW_E02_GLOBAL_A7 | 3.965859 | 13.861487 | 29.509110 | 3.089507 | 22.048123 | +0.000000 |
| AW_E02_GLOBAL_DIRECT | 3.853697 | 13.295121 | 24.137272 | 3.204858 | 19.997271 | -0.500293 |
| AW_E02_SAM2_A7 | 3.983649 | 13.955594 | 29.507921 | 3.099860 | 21.802067 | +0.000000 |
| AW_E02_SAM2_DIRECT | 4.016020 | 15.977734 | 26.897413 | 3.296498 | 21.111194 | -0.511945 |
| AW_E03_FC_FROZEN_A7 | 4.224843 | 14.905203 | 32.164168 | 3.433466 | 23.651214 | -0.043077 |
| AW_E03_FC_FROZEN_DIRECT | 3.402165 | 12.012787 | 23.864638 | 2.827008 | 19.122998 | -0.330067 |
| AW_E03_OVR_A7 | 4.234029 | 14.927249 | 32.098832 | 3.442324 | 24.250254 | -0.025362 |
| AW_E03_OVR_DIRECT | 4.881932 | 14.343034 | 27.583774 | 4.369704 | 21.983822 | -0.997112 |
| AW_E04_SHORTLIST | 3.931566 | 13.552845 | 29.222514 | 3.089507 | 21.865271 | +0.000000 |
| N0 | 4.235291 | 14.985804 | 30.183916 | 3.443540 | 20.656821 | -0.022928 |
| Q_GAIN | 4.062207 | 14.753086 | 22.606435 | 3.087351 | 19.023275 | -0.004312 |
| RV_A7_COS_FIXED | 3.965859 | 13.861487 | 29.531156 | 3.089507 | 22.072704 | +0.000000 |
| RV_A7_COS_REFIT | 3.931566 | 13.552845 | 29.222514 | 3.089507 | 21.865271 | +0.000000 |
| S_SIGLIP2_AREA | 4.756189 | 16.153733 | 27.939692 | 4.001530 | 22.198394 | -0.076540 |

### REPLICA official released pool

| Method | APall | AP50 | AP25 | mIoU | mAcc | ΔAPall vs N0 | ΔAPall vs A7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AW_C0_SO400M_A7 | 9.474653 | 23.207496 | 36.517807 | 28.517334 | 34.862261 | +0.789382 | +0.256747 |
| AW_C0_SO400M_DIRECT | 9.347840 | 21.287828 | 34.161863 | 23.113301 | 32.255714 | +0.662568 | +0.129933 |
| AW_COMBO_QR | 10.286507 | 22.283471 | 36.612652 | 28.413136 | 36.427438 | +1.601235 | +1.068600 |
| AW_E01_GMED_A7 | 8.987528 | 20.336250 | 34.966217 | 27.314416 | 33.497339 | +0.302256 | -0.230379 |
| AW_E01_GMED_DIRECT | 6.680676 | 14.945064 | 24.628503 | 21.558930 | 26.665326 | -2.004595 | -2.537230 |
| AW_E01_RAW_EQ_A7 | 8.997345 | 20.275796 | 34.624782 | 26.780052 | 33.446470 | +0.312074 | -0.220561 |
| AW_E01_RAW_EQ_DIRECT | 6.588633 | 14.903290 | 24.727542 | 21.371873 | 26.794532 | -2.096638 | -2.629273 |
| AW_E01_UNIT_EQ_A7 | 8.997345 | 20.275796 | 34.624782 | 26.780052 | 33.446470 | +0.312074 | -0.220561 |
| AW_E01_UNIT_EQ_DIRECT | 6.588633 | 14.903290 | 24.727542 | 21.363610 | 26.794532 | -2.096638 | -2.629273 |
| AW_E02_GLOBAL_A7 | 9.177206 | 21.098809 | 34.979670 | 27.375432 | 33.597513 | +0.491934 | -0.040701 |
| AW_E02_GLOBAL_DIRECT | 6.765036 | 14.719977 | 28.136152 | 20.430413 | 28.682680 | -1.920236 | -2.452871 |
| AW_E02_SAM2_A7 | 9.124101 | 21.080527 | 34.821928 | 27.570288 | 33.705681 | +0.438829 | -0.093806 |
| AW_E02_SAM2_DIRECT | 7.398191 | 17.111533 | 30.009703 | 20.260541 | 28.747293 | -1.287081 | -1.819716 |
| AW_E03_FC_FROZEN_A7 | 10.491020 | 24.323075 | 37.728463 | 30.013672 | 36.694593 | +1.805748 | +1.273113 |
| AW_E03_FC_FROZEN_DIRECT | 10.498388 | 20.287022 | 30.235633 | 24.091640 | 33.261039 | +1.813117 | +1.280482 |
| AW_E03_OVR_A7 | 10.232525 | 22.023194 | 36.508624 | 28.623663 | 36.625030 | +1.547254 | +1.014619 |
| AW_E03_OVR_DIRECT | 10.939594 | 22.006838 | 33.449918 | 26.180430 | 36.770290 | +2.254323 | +1.721688 |
| AW_E04_SHORTLIST | 9.217907 | 21.236029 | 35.201083 | 27.720221 | 33.810573 | +0.532635 | +0.000000 |
| N0 | 8.685272 | 21.476863 | 34.585379 | 27.261462 | 32.695265 | +0.000000 | -0.532635 |
| Q_GAIN | 7.496819 | 16.911586 | 28.099042 | 23.506454 | 28.957786 | -1.188452 | -1.721087 |
| RV_A7_COS_FIXED | 9.186862 | 21.078259 | 35.516417 | 27.898783 | 33.969385 | +0.501591 | -0.031044 |
| RV_A7_COS_REFIT | 9.217907 | 21.236029 | 35.201083 | 27.720221 | 33.810573 | +0.532635 | +0.000000 |
| S_SIGLIP2_AREA | 7.189961 | 15.909962 | 30.250537 | 21.333148 | 29.543815 | -1.495311 | -2.027946 |
| AW_E04_SPATIAL | BLOCKED | — | — | — | — | — | — |
| AW_E04_MIX50 | BLOCKED | — | — | — | — | — | — |

| Matched contrast (new − old) | apall | ap50 | ap25 | miou | macc |
| --- | --- | --- | --- | --- | --- |
| E01_normalization_A7 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| E01_normalization_DIRECT | 0.000000 | 0.000000 | 0.000000 | -0.008262 | 0.000000 |
| E01_robustness_A7 | -0.009817 | 0.060454 | 0.341435 | 0.534364 | 0.050869 |
| E01_robustness_DIRECT | 0.092043 | 0.041773 | -0.099039 | 0.195320 | -0.129206 |
| E02_SAM2_vs_GLOBAL_A7 | -0.053105 | -0.018282 | -0.157742 | 0.194856 | 0.108168 |
| E02_SAM2_vs_GLOBAL_DIRECT | 0.633154 | 2.391556 | 1.873551 | -0.169872 | 0.064613 |
| E03_tuned_vs_frozen_A7 | -0.258494 | -2.299881 | -1.219839 | -1.390009 | -0.069563 |
| E03_tuned_vs_frozen_DIRECT | 0.441206 | 1.719816 | 3.214285 | 2.088791 | 3.509251 |

Interaction is Q+region − Q − region + base A7, using the single CAL-frozen pair.

| Interaction | apall | ap50 | ap25 | miou | macc |
| --- | --- | --- | --- | --- | --- |
| replica | 0.284360 | 1.160057 | 0.338894 | 0.195278 | 0.115642 |

### REPLICA auxiliary fixed-rank pools and official scene means

| Method | Frozen APall | Frozen AP50 | Frozen AP25 | Mean APall | Mean mIoU | Worst scene ΔAPall vs A7 |
| --- | --- | --- | --- | --- | --- | --- |
| AW_C0_SO400M_A7 | 9.686004 | 23.979742 | 37.630485 | 12.399609 | 31.887847 | -0.289855 |
| AW_C0_SO400M_DIRECT | 9.645186 | 21.869153 | 34.273810 | 10.540986 | 27.422066 | -4.861111 |
| AW_COMBO_QR | 10.481592 | 22.650996 | 37.333955 | 13.366902 | 32.096691 | -0.567130 |
| AW_E01_GMED_A7 | 9.065705 | 20.583873 | 34.954690 | 12.049812 | 31.360473 | -1.620370 |
| AW_E01_GMED_DIRECT | 6.563959 | 14.067591 | 24.755186 | 9.090688 | 24.970988 | -14.988426 |
| AW_E01_RAW_EQ_A7 | 9.060078 | 20.583873 | 34.954690 | 12.049812 | 30.969246 | -1.620370 |
| AW_E01_RAW_EQ_DIRECT | 6.676945 | 14.471286 | 25.411081 | 9.086126 | 24.465983 | -14.988426 |
| AW_E01_UNIT_EQ_A7 | 9.060078 | 20.583873 | 34.954690 | 12.049812 | 30.969246 | -1.620370 |
| AW_E01_UNIT_EQ_DIRECT | 6.676945 | 14.471286 | 25.411081 | 9.086126 | 24.380403 | -14.988426 |
| AW_E02_GLOBAL_A7 | 9.274353 | 21.402406 | 34.978501 | 12.129321 | 31.367374 | -0.694444 |
| AW_E02_GLOBAL_DIRECT | 7.436985 | 16.393806 | 29.450988 | 9.725912 | 25.168928 | -5.439815 |
| AW_E02_SAM2_A7 | 9.171511 | 21.273194 | 35.178389 | 12.013643 | 31.077386 | -1.909722 |
| AW_E02_SAM2_DIRECT | 7.794099 | 17.481998 | 29.917823 | 9.967472 | 25.361433 | -5.439815 |
| AW_E03_FC_FROZEN_A7 | 10.820202 | 25.490282 | 39.683677 | 13.066657 | 32.265157 | -0.567130 |
| AW_E03_FC_FROZEN_DIRECT | 11.384098 | 21.411166 | 33.116421 | 12.533207 | 28.768005 | -7.692308 |
| AW_E03_OVR_A7 | 10.416317 | 22.585022 | 37.521119 | 13.337967 | 32.177259 | -0.567130 |
| AW_E03_OVR_DIRECT | 11.619355 | 22.548895 | 36.286148 | 14.069275 | 32.377961 | -3.344907 |
| AW_E04_SHORTLIST | 9.280876 | 21.473281 | 35.548398 | 12.252358 | 31.343756 | +0.000000 |
| N0 | 8.685272 | 21.476863 | 34.585379 | 11.530285 | 28.849698 | -3.163580 |
| Q_GAIN | 7.738037 | 17.137486 | 28.553737 | 11.278079 | 27.550967 | -4.745370 |
| RV_A7_COS_FIXED | 9.245816 | 21.315511 | 35.863732 | 12.233068 | 31.627932 | -0.154321 |
| RV_A7_COS_REFIT | 9.280876 | 21.473281 | 35.548398 | 12.252358 | 31.343756 | +0.000000 |
| S_SIGLIP2_AREA | 7.854932 | 17.561902 | 31.617720 | 9.922623 | 26.385925 | -5.439815 |

Every per-scene metric, both rank modes, five-metric scene means and deltas remain in comparisons/{cal,replica}.json and the original rows. Scene means are supplementary and never substituted for the released pool.

## B. Effective intervention

### CAL sources

| Source | All owners | Capped targets | Requested views | Usable views | Nonempty final support | Genuine owners | Changed suggestions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AW_C0_SO400M | 192 | 192 | 544 | 544 | 544 | 188 | 69 |
| AW_E01_GMED | 192 | original Q retention | 342 | 342 | cached Q | 179 | 19 |
| AW_E01_RAW_EQ | 192 | original Q retention | 342 | 342 | cached Q | 179 | 20 |
| AW_E01_UNIT_EQ | 192 | original Q retention | 342 | 342 | cached Q | 179 | 20 |
| AW_E02_GLOBAL | 192 | 192 | 544 | 544 | 544 | 188 | 55 |
| AW_E02_SAM2 | 192 | 192 | 544 | 544 | 544 | 188 | 50 |
| AW_E03_FC_FROZEN | 192 | 192 | 544 | 539 | 539 | 188 | 125 |
| AW_E03_OVR | 192 | 192 | 544 | 539 | 539 | 188 | 122 |

| Source | Failed view reasons | Unavailable owner reasons |
| --- | --- | --- |
| AW_C0_SO400M | {} | {"NO_ORIGINAL_STATIC_VIEW": 4} |
| AW_E01_GMED | {} | {"NO_RETAINED_QUERY_EVIDENCE": 13} |
| AW_E01_RAW_EQ | {} | {"NO_RETAINED_QUERY_EVIDENCE": 13} |
| AW_E01_UNIT_EQ | {} | {"NO_RETAINED_QUERY_EVIDENCE": 13} |
| AW_E02_GLOBAL | {} | {"NO_ORIGINAL_STATIC_VIEW": 4} |
| AW_E02_SAM2 | {} | {"NO_ORIGINAL_STATIC_VIEW": 4} |
| AW_E03_FC_FROZEN | {"EMPTY_DENSE_MASK_SUPPORT": 5} | {"NO_ORIGINAL_STATIC_VIEW": 4} |
| AW_E03_OVR | {"EMPTY_DENSE_MASK_SUPPORT": 5} | {"NO_ORIGINAL_STATIC_VIEW": 4} |

### REPLICA sources

| Source | All owners | Capped targets | Requested views | Usable views | Nonempty final support | Genuine owners | Changed suggestions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AW_C0_SO400M | 375 | 375 | 1067 | 1067 | 1067 | 362 | 116 |
| AW_E01_GMED | 375 | original Q retention | 1108 | 1108 | cached Q | 367 | 37 |
| AW_E01_RAW_EQ | 375 | original Q retention | 1108 | 1108 | cached Q | 367 | 42 |
| AW_E01_UNIT_EQ | 375 | original Q retention | 1108 | 1108 | cached Q | 367 | 41 |
| AW_E02_GLOBAL | 375 | 375 | 1067 | 1067 | 1067 | 362 | 58 |
| AW_E02_SAM2 | 375 | 375 | 1067 | 1067 | 1067 | 362 | 40 |
| AW_E03_FC_FROZEN | 375 | 375 | 1067 | 1052 | 1052 | 361 | 196 |
| AW_E03_OVR | 375 | 375 | 1067 | 1052 | 1052 | 361 | 189 |

| Source | Failed view reasons | Unavailable owner reasons |
| --- | --- | --- |
| AW_C0_SO400M | {} | {"NO_ORIGINAL_STATIC_VIEW": 13} |
| AW_E01_GMED | {} | {"NO_RETAINED_QUERY_EVIDENCE": 8} |
| AW_E01_RAW_EQ | {} | {"NO_RETAINED_QUERY_EVIDENCE": 8} |
| AW_E01_UNIT_EQ | {} | {"NO_RETAINED_QUERY_EVIDENCE": 8} |
| AW_E02_GLOBAL | {} | {"NO_ORIGINAL_STATIC_VIEW": 13} |
| AW_E02_SAM2 | {} | {"NO_ORIGINAL_STATIC_VIEW": 13} |
| AW_E03_FC_FROZEN | {"EMPTY_DENSE_MASK_SUPPORT": 15} | {"NO_ORIGINAL_STATIC_VIEW": 13, "NO_SUCCESSFUL_RECOGNITION_VIEW": 1} |
| AW_E03_OVR | {"EMPTY_DENSE_MASK_SUPPORT": 15} | {"NO_ORIGINAL_STATIC_VIEW": 13, "NO_SUCCESSFUL_RECOGNITION_VIEW": 1} |

Unavailable sources never vote in A7; DIRECT falls back to N0. Excluded owners remain in every complete-map evaluation. SHORTLIST is a decision-identical control, not a discovered improvement.

## C. Corrections and damage

Semantic corrections use the original unique geometry correspondence with IoU > 0.5 and are separate from the released AP matcher. Correct-class ranks and source correctness are evaluation-only diagnostics.

### CAL owner outcomes

| Method | Changed vs A7 | Corrected A7 | Harmed A7 | Corrected N0 | Harmed N0 | Unused correct evidence |
| --- | --- | --- | --- | --- | --- | --- |
| AW_C0_SO400M_A7 | 21 | 0 | 1 | 0 | 0 | 2 |
| AW_C0_SO400M_DIRECT | 81 | 1 | 4 | 1 | 3 | 0 |
| AW_COMBO_QR | 52 | 0 | 2 | 0 | 1 | 5 |
| AW_E01_GMED_A7 | 6 | 0 | 0 | 1 | 0 | 1 |
| AW_E01_GMED_DIRECT | 44 | 0 | 1 | 1 | 1 | 0 |
| AW_E01_RAW_EQ_A7 | 7 | 0 | 0 | 1 | 0 | 1 |
| AW_E01_RAW_EQ_DIRECT | 44 | 0 | 1 | 1 | 1 | 0 |
| AW_E01_UNIT_EQ_A7 | 7 | 0 | 0 | 1 | 0 | 1 |
| AW_E01_UNIT_EQ_DIRECT | 44 | 0 | 1 | 1 | 1 | 0 |
| AW_E02_GLOBAL_A7 | 14 | 0 | 0 | 1 | 0 | 0 |
| AW_E02_GLOBAL_DIRECT | 78 | 0 | 3 | 1 | 3 | 0 |
| AW_E02_SAM2_A7 | 17 | 0 | 0 | 1 | 0 | 1 |
| AW_E02_SAM2_DIRECT | 80 | 1 | 5 | 1 | 4 | 0 |
| AW_E03_FC_FROZEN_A7 | 48 | 0 | 2 | 0 | 1 | 4 |
| AW_E03_FC_FROZEN_DIRECT | 125 | 2 | 6 | 2 | 5 | 0 |
| AW_E03_OVR_A7 | 51 | 0 | 2 | 0 | 1 | 5 |
| AW_E03_OVR_DIRECT | 121 | 3 | 6 | 3 | 5 | 0 |
| AW_E04_SHORTLIST | 0 | 0 | 0 | 1 | 0 | 1 |
| N0 | 37 | 0 | 1 | 0 | 0 | 0 |
| Q_GAIN | 53 | 0 | 0 | 1 | 0 | 0 |
| RV_A7_COS_FIXED | 4 | 0 | 0 | 1 | 0 | 1 |
| RV_A7_COS_REFIT | 0 | 0 | 0 | 1 | 0 | 1 |
| S_SIGLIP2_AREA | 69 | 1 | 2 | 1 | 1 | 0 |

| Source | Common identifiable | Old correct | New correct |
| --- | --- | --- | --- |
| AW_C0_SO400M | 22 | 12 | 10 |
| AW_E01_GMED | 22 | 13 | 12 |
| AW_E01_RAW_EQ | 22 | 13 | 12 |
| AW_E01_UNIT_EQ | 22 | 13 | 12 |
| AW_E02_GLOBAL | 22 | 12 | 10 |
| AW_E02_SAM2 | 22 | 12 | 9 |
| AW_E03_FC_FROZEN | 22 | 12 | 9 |
| AW_E03_OVR | 22 | 12 | 10 |

Actual released matcher changes relative to N0 (counts summed across scenes; these are not AP deltas):

| Method | IoU | Added GT matches | Lost GT matches |
| --- | --- | --- | --- |
| AW_C0_SO400M_A7 | 0.25 | 3 | 2 |
| AW_C0_SO400M_A7 | 0.5 | 0 | 0 |
| AW_C0_SO400M_DIRECT | 0.25 | 3 | 9 |
| AW_C0_SO400M_DIRECT | 0.5 | 1 | 3 |
| AW_COMBO_QR | 0.25 | 3 | 5 |
| AW_COMBO_QR | 0.5 | 0 | 1 |
| AW_E01_GMED_A7 | 0.25 | 4 | 1 |
| AW_E01_GMED_A7 | 0.5 | 1 | 0 |
| AW_E01_GMED_DIRECT | 0.25 | 3 | 2 |
| AW_E01_GMED_DIRECT | 0.5 | 1 | 1 |
| AW_E01_RAW_EQ_A7 | 0.25 | 4 | 1 |
| AW_E01_RAW_EQ_A7 | 0.5 | 1 | 0 |
| AW_E01_RAW_EQ_DIRECT | 0.25 | 3 | 2 |
| AW_E01_RAW_EQ_DIRECT | 0.5 | 1 | 1 |
| AW_E01_UNIT_EQ_A7 | 0.25 | 4 | 1 |
| AW_E01_UNIT_EQ_A7 | 0.5 | 1 | 0 |
| AW_E01_UNIT_EQ_DIRECT | 0.25 | 3 | 2 |
| AW_E01_UNIT_EQ_DIRECT | 0.5 | 1 | 1 |
| AW_E02_GLOBAL_A7 | 0.25 | 4 | 1 |
| AW_E02_GLOBAL_A7 | 0.5 | 1 | 0 |
| AW_E02_GLOBAL_DIRECT | 0.25 | 4 | 9 |
| AW_E02_GLOBAL_DIRECT | 0.5 | 1 | 3 |
| AW_E02_SAM2_A7 | 0.25 | 4 | 2 |
| AW_E02_SAM2_A7 | 0.5 | 1 | 0 |
| AW_E02_SAM2_DIRECT | 0.25 | 4 | 10 |
| AW_E02_SAM2_DIRECT | 0.5 | 1 | 4 |
| AW_E03_FC_FROZEN_A7 | 0.25 | 3 | 4 |
| AW_E03_FC_FROZEN_A7 | 0.5 | 0 | 1 |
| AW_E03_FC_FROZEN_DIRECT | 0.25 | 4 | 13 |
| AW_E03_FC_FROZEN_DIRECT | 0.5 | 2 | 5 |
| AW_E03_OVR_A7 | 0.25 | 3 | 4 |
| AW_E03_OVR_A7 | 0.5 | 0 | 1 |
| AW_E03_OVR_DIRECT | 0.25 | 5 | 10 |
| AW_E03_OVR_DIRECT | 0.5 | 3 | 5 |
| AW_E04_SHORTLIST | 0.25 | 4 | 1 |
| AW_E04_SHORTLIST | 0.5 | 1 | 0 |
| N0 | 0.25 | 0 | 0 |
| N0 | 0.5 | 0 | 0 |
| Q_GAIN | 0.25 | 4 | 5 |
| Q_GAIN | 0.5 | 1 | 0 |
| RV_A7_COS_FIXED | 0.25 | 4 | 1 |
| RV_A7_COS_FIXED | 0.5 | 1 | 0 |
| RV_A7_COS_REFIT | 0.25 | 4 | 1 |
| RV_A7_COS_REFIT | 0.5 | 1 | 0 |
| S_SIGLIP2_AREA | 0.25 | 4 | 5 |
| S_SIGLIP2_AREA | 0.5 | 1 | 1 |

### REPLICA owner outcomes

| Method | Changed vs A7 | Corrected A7 | Harmed A7 | Corrected N0 | Harmed N0 | Unused correct evidence |
| --- | --- | --- | --- | --- | --- | --- |
| AW_C0_SO400M_A7 | 30 | 2 | 2 | 12 | 4 | 12 |
| AW_C0_SO400M_DIRECT | 135 | 6 | 18 | 15 | 19 | 0 |
| AW_COMBO_QR | 55 | 9 | 2 | 19 | 4 | 24 |
| AW_E01_GMED_A7 | 14 | 0 | 2 | 12 | 6 | 16 |
| AW_E01_GMED_DIRECT | 84 | 5 | 20 | 9 | 16 | 0 |
| AW_E01_RAW_EQ_A7 | 13 | 0 | 2 | 12 | 6 | 14 |
| AW_E01_RAW_EQ_DIRECT | 86 | 4 | 19 | 7 | 14 | 0 |
| AW_E01_UNIT_EQ_A7 | 13 | 0 | 2 | 12 | 6 | 14 |
| AW_E01_UNIT_EQ_DIRECT | 84 | 4 | 19 | 7 | 14 | 0 |
| AW_E02_GLOBAL_A7 | 17 | 0 | 2 | 11 | 5 | 13 |
| AW_E02_GLOBAL_DIRECT | 144 | 3 | 21 | 12 | 22 | 0 |
| AW_E02_SAM2_A7 | 10 | 0 | 2 | 11 | 5 | 14 |
| AW_E02_SAM2_DIRECT | 140 | 4 | 19 | 13 | 20 | 0 |
| AW_E03_FC_FROZEN_A7 | 52 | 10 | 2 | 19 | 3 | 17 |
| AW_E03_FC_FROZEN_DIRECT | 182 | 20 | 18 | 27 | 17 | 0 |
| AW_E03_OVR_A7 | 52 | 9 | 2 | 19 | 4 | 23 |
| AW_E03_OVR_DIRECT | 171 | 25 | 11 | 33 | 11 | 0 |
| AW_E04_SHORTLIST | 0 | 0 | 0 | 12 | 4 | 13 |
| N0 | 58 | 4 | 12 | 0 | 0 | 0 |
| Q_GAIN | 73 | 4 | 11 | 8 | 7 | 0 |
| RV_A7_COS_FIXED | 2 | 0 | 1 | 11 | 4 | 14 |
| RV_A7_COS_REFIT | 0 | 0 | 0 | 12 | 4 | 13 |
| S_SIGLIP2_AREA | 133 | 5 | 19 | 14 | 20 | 0 |

| Source | Common identifiable | Old correct | New correct |
| --- | --- | --- | --- |
| AW_C0_SO400M | 157 | 66 | 68 |
| AW_E01_GMED | 161 | 75 | 67 |
| AW_E01_RAW_EQ | 161 | 75 | 67 |
| AW_E01_UNIT_EQ | 161 | 75 | 67 |
| AW_E02_GLOBAL | 157 | 66 | 62 |
| AW_E02_SAM2 | 157 | 66 | 65 |
| AW_E03_FC_FROZEN | 157 | 66 | 82 |
| AW_E03_OVR | 157 | 66 | 94 |

Actual released matcher changes relative to N0 (counts summed across scenes; these are not AP deltas):

| Method | IoU | Added GT matches | Lost GT matches |
| --- | --- | --- | --- |
| AW_C0_SO400M_A7 | 0.25 | 14 | 7 |
| AW_C0_SO400M_A7 | 0.5 | 9 | 3 |
| AW_C0_SO400M_DIRECT | 0.25 | 22 | 31 |
| AW_C0_SO400M_DIRECT | 0.5 | 12 | 18 |
| AW_COMBO_QR | 0.25 | 25 | 9 |
| AW_COMBO_QR | 0.5 | 17 | 4 |
| AW_E01_GMED_A7 | 0.25 | 13 | 10 |
| AW_E01_GMED_A7 | 0.5 | 9 | 6 |
| AW_E01_GMED_DIRECT | 0.25 | 13 | 27 |
| AW_E01_GMED_DIRECT | 0.5 | 9 | 16 |
| AW_E01_RAW_EQ_A7 | 0.25 | 13 | 10 |
| AW_E01_RAW_EQ_A7 | 0.5 | 9 | 6 |
| AW_E01_RAW_EQ_DIRECT | 0.25 | 11 | 25 |
| AW_E01_RAW_EQ_DIRECT | 0.5 | 7 | 14 |
| AW_E01_UNIT_EQ_A7 | 0.25 | 13 | 10 |
| AW_E01_UNIT_EQ_A7 | 0.5 | 9 | 6 |
| AW_E01_UNIT_EQ_DIRECT | 0.25 | 11 | 25 |
| AW_E01_UNIT_EQ_DIRECT | 0.5 | 7 | 14 |
| AW_E02_GLOBAL_A7 | 0.25 | 12 | 9 |
| AW_E02_GLOBAL_A7 | 0.5 | 8 | 5 |
| AW_E02_GLOBAL_DIRECT | 0.25 | 17 | 34 |
| AW_E02_GLOBAL_DIRECT | 0.5 | 9 | 21 |
| AW_E02_SAM2_A7 | 0.25 | 13 | 10 |
| AW_E02_SAM2_A7 | 0.5 | 8 | 5 |
| AW_E02_SAM2_DIRECT | 0.25 | 17 | 31 |
| AW_E02_SAM2_DIRECT | 0.5 | 10 | 19 |
| AW_E03_FC_FROZEN_A7 | 0.25 | 25 | 10 |
| AW_E03_FC_FROZEN_A7 | 0.5 | 17 | 3 |
| AW_E03_FC_FROZEN_DIRECT | 0.25 | 36 | 31 |
| AW_E03_FC_FROZEN_DIRECT | 0.5 | 24 | 15 |
| AW_E03_OVR_A7 | 0.25 | 25 | 8 |
| AW_E03_OVR_A7 | 0.5 | 17 | 4 |
| AW_E03_OVR_DIRECT | 0.25 | 42 | 23 |
| AW_E03_OVR_DIRECT | 0.5 | 29 | 10 |
| AW_E04_SHORTLIST | 0.25 | 15 | 9 |
| AW_E04_SHORTLIST | 0.5 | 9 | 4 |
| N0 | 0.25 | 0 | 0 |
| N0 | 0.5 | 0 | 0 |
| Q_GAIN | 0.25 | 12 | 16 |
| Q_GAIN | 0.5 | 8 | 7 |
| RV_A7_COS_FIXED | 0.25 | 14 | 8 |
| RV_A7_COS_FIXED | 0.5 | 8 | 4 |
| RV_A7_COS_REFIT | 0.25 | 15 | 9 |
| RV_A7_COS_REFIT | 0.5 | 9 | 4 |
| S_SIGLIP2_AREA | 0.25 | 20 | 28 |
| S_SIGLIP2_AREA | 0.5 | 11 | 19 |

Full released attribution retains duplicate/ignore events and per-class AP changes; the compact geometry table must not be interpreted as AP true positives.

| Scene | Example method | Corrected owner IDs | Harmed owner IDs |
| --- | --- | --- | --- |
| office0 | AW_COMBO_QR | [3] | [] |
| office1 | AW_COMBO_QR | [] | [] |
| office2 | AW_COMBO_QR | [82] | [] |
| office3 | AW_COMBO_QR | [8, 12] | [] |
| office4 | AW_COMBO_QR | [] | [] |
| room0 | AW_COMBO_QR | [9, 102] | [] |
| room1 | AW_COMBO_QR | [51] | [20] |
| room2 | AW_COMBO_QR | [12] | [6] |
| scene0056_00 | AW_COMBO_QR | [] | [12, 119] |
| scene0534_00 | AW_COMBO_QR | [] | [] |

Examples use owner-ID order, at most two corrected and two harmed owners per scene; none were selected for attractive imagery.

## D. Costs and decisions

### CAL recorded worker work

| Worker | Crop inputs | Dense/SAM images | Region pools/box decodes | Reused raw crops | Text inputs | Load s | Worker wall s | Peak allocated GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AW_C0_SO400M | 3264 | — | — | — | 400 | 11.876 | 241.269 | 4.470 |
| AW_E02_GLOBAL | 1632 | — | — | 1632 | — | 17.981 | 138.742 | 3.379 |
| AW_E02_SAM2 | 1632 | — | — | 1632 | — | 18.593 | 156.397 | 3.379 |
| AW_E03_FC_FROZEN | — | 283 | 539 | — | 5600 | 16.698 | 99.125 | 1.803 |
| AW_E03_OVR | — | 283 | 539 | — | 5600 | 17.756 | 102.192 | 1.803 |
| E01_shared_readout | — | — | — | — | — | unrecorded | 18.596 | — |
| E02_SAM2_masks | — | 283 | 544 | — | — | 6.109 | 83.018 | 1.605 |

Required visual operation unions, summed over scenes (heterogeneous units, not FLOPs or latency):

| Method | image_crop | dense_image | region_pool_projection | sam2_image | sam2_box_decode |
| --- | --- | --- | --- | --- | --- |
| AW_C0_SO400M_A7 | 15724 | 0 | 0 | 0 | 0 |
| AW_C0_SO400M_DIRECT | 15084 | 0 | 0 | 0 | 0 |
| AW_COMBO_QR | 12468 | 283 | 539 | 0 | 0 |
| AW_E01_GMED_A7 | 15724 | 0 | 0 | 0 | 0 |
| AW_E01_GMED_DIRECT | 12468 | 0 | 0 | 0 | 0 |
| AW_E01_RAW_EQ_A7 | 15724 | 0 | 0 | 0 | 0 |
| AW_E01_RAW_EQ_DIRECT | 12468 | 0 | 0 | 0 | 0 |
| AW_E01_UNIT_EQ_A7 | 15724 | 0 | 0 | 0 | 0 |
| AW_E01_UNIT_EQ_DIRECT | 12468 | 0 | 0 | 0 | 0 |
| AW_E02_GLOBAL_A7 | 15724 | 0 | 0 | 0 | 0 |
| AW_E02_GLOBAL_DIRECT | 15084 | 0 | 0 | 0 | 0 |
| AW_E02_SAM2_A7 | 15724 | 0 | 0 | 283 | 544 |
| AW_E02_SAM2_DIRECT | 15084 | 0 | 0 | 283 | 544 |
| AW_E03_FC_FROZEN_A7 | 12468 | 283 | 539 | 0 | 0 |
| AW_E03_FC_FROZEN_DIRECT | 11828 | 283 | 539 | 0 | 0 |
| AW_E03_OVR_A7 | 12468 | 283 | 539 | 0 | 0 |
| AW_E03_OVR_DIRECT | 11828 | 283 | 539 | 0 | 0 |
| AW_E04_SHORTLIST | 15724 | 0 | 0 | 0 | 0 |
| N0 | 11828 | 0 | 0 | 0 | 0 |
| Q_GAIN | 12468 | 0 | 0 | 0 | 0 |
| RV_A7_COS_FIXED | 15724 | 0 | 0 | 0 | 0 |
| RV_A7_COS_REFIT | 15724 | 0 | 0 | 0 | 0 |
| S_SIGLIP2_AREA | 15084 | 0 | 0 | 0 | 0 |

### REPLICA recorded worker work

| Worker | Crop inputs | Dense/SAM images | Region pools/box decodes | Reused raw crops | Text inputs | Load s | Worker wall s | Peak allocated GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AW_C0_SO400M | 6402 | — | — | — | 408 | 64.847 | 568.901 | 4.470 |
| AW_E02_GLOBAL | 3201 | — | — | 3201 | — | 51.158 | 399.808 | 3.379 |
| AW_E02_SAM2 | 3201 | — | — | 3201 | — | 52.405 | 410.345 | 3.379 |
| AW_E03_FC_FROZEN | — | 673 | 1052 | — | 5712 | 64.828 | 296.153 | 1.894 |
| AW_E03_OVR | — | 673 | 1052 | — | 5712 | 67.514 | 300.209 | 1.894 |
| E01_shared_readout | — | — | — | — | — | unrecorded | 41.724 | — |
| E02_SAM2_masks | — | 673 | 1067 | — | — | 25.706 | 244.443 | 1.605 |

Required visual operation unions, summed over scenes (heterogeneous units, not FLOPs or latency):

| Method | image_crop | dense_image | region_pool_projection | sam2_image | sam2_box_decode |
| --- | --- | --- | --- | --- | --- |
| AW_C0_SO400M_A7 | 37270 | 0 | 0 | 0 | 0 |
| AW_C0_SO400M_DIRECT | 31544 | 0 | 0 | 0 | 0 |
| AW_COMBO_QR | 30906 | 673 | 1052 | 0 | 0 |
| AW_E01_GMED_A7 | 37270 | 0 | 0 | 0 | 0 |
| AW_E01_GMED_DIRECT | 30906 | 0 | 0 | 0 | 0 |
| AW_E01_RAW_EQ_A7 | 37270 | 0 | 0 | 0 | 0 |
| AW_E01_RAW_EQ_DIRECT | 30906 | 0 | 0 | 0 | 0 |
| AW_E01_UNIT_EQ_A7 | 37270 | 0 | 0 | 0 | 0 |
| AW_E01_UNIT_EQ_DIRECT | 30906 | 0 | 0 | 0 | 0 |
| AW_E02_GLOBAL_A7 | 37270 | 0 | 0 | 0 | 0 |
| AW_E02_GLOBAL_DIRECT | 31544 | 0 | 0 | 0 | 0 |
| AW_E02_SAM2_A7 | 37270 | 0 | 0 | 673 | 1067 |
| AW_E02_SAM2_DIRECT | 31544 | 0 | 0 | 673 | 1067 |
| AW_E03_FC_FROZEN_A7 | 30906 | 673 | 1052 | 0 | 0 |
| AW_E03_FC_FROZEN_DIRECT | 25180 | 673 | 1052 | 0 | 0 |
| AW_E03_OVR_A7 | 30906 | 673 | 1052 | 0 | 0 |
| AW_E03_OVR_DIRECT | 25180 | 673 | 1052 | 0 | 0 |
| AW_E04_SHORTLIST | 37270 | 0 | 0 | 0 | 0 |
| N0 | 25180 | 0 | 0 | 0 | 0 |
| Q_GAIN | 30906 | 0 | 0 | 0 | 0 |
| RV_A7_COS_FIXED | 37270 | 0 | 0 | 0 | 0 |
| RV_A7_COS_REFIT | 37270 | 0 | 0 | 0 | 0 |
| S_SIGLIP2_AREA | 31544 | 0 | 0 | 0 | 0 |

| Downloaded asset | Verified GiB | Recorded download seconds |
| --- | --- | --- |
| fc_frozen | 1.313877 | 70.92392960493453 |
| ovrcoat | 4.315670 | 275.3797492070589 |
| sam2 | 0.836406 | not recorded |
| so400m | 4.268061 | 226.13072451995686 |

Changed-source scalar fitting took 0.101839 recorded seconds in total. There are 358 distinct reused/new evaluator receipts with 195.954 recorded seconds; this includes historical work and is not new-run latency.

Verified new model/tokenizer/config bytes: 11,525,559,579 (10.734 GiB), within 25 GiB. Model download receipts, real adapter smoke costs, scalar-fit times and distinct evaluation receipts are included in report/data.json.

Required operations are model/input-identified unions in costs/<scene>.json, including original N0 mapping inputs. Physical cache reuse is reported separately and is not free logical work. Worker wall time includes load/I/O; parallel worker sums are not end-to-end latency. Per-worker peaks are maxima, not summed. Historical frontend time, unrecorded failed/interrupted attempts and SAM2 download time are incomplete. Recorded evaluation receipts include historical cache reuse; they do not measure this invocation's new work.

The CAL-frozen research nomination is N0. Its CAL APall is 4.235291%, versus 4.234029% for OVR_A7; the small but non-tied APall difference takes precedence over OVR's higher mIoU. Deployment remains N0_UNCHANGED. Replica results did not change the nomination.

E01 provides no Replica APall gain over original A7. GMED_A7 (8.987528%) is below UNIT_EQ_A7 (8.997345%), so these results do not validate robust aggregation. All E01 variants retain original paid/retained observations and require zero new image forwards.

E02 SAM2_A7 has Replica APall 9.124101%, below GLOBAL_A7 9.177206% and original A7 9.217907%. The additional SAM2 encodings and box decodes are not supported by an APall benefit here. These recognition-mask experiments say nothing about reconstructed 3D shape quality.

C0 SO400M_A7 improves Replica APall to 9.474653% and mIoU to 28.517334%, versus A7 9.217907% and 27.720221%. This is an ordinary model-capacity control, not evidence of a new aggregation algorithm; its CAL APall is slightly below A7.

E03 has a conditional matched-control result. OVR_DIRECT exceeds FC_FROZEN_DIRECT in Replica APall (10.939594% versus 10.498388%), and common-identifiable source correctness improves from 82/157 to 94/157. However, OVR_A7 is below FC_FROZEN_A7 in APall (10.232525% versus 10.491020%) and mIoU (28.623663% versus 30.013672%). Region tuning helps the direct source but is not a uniform fusion gain. OVR_DIRECT's higher APall also coexists with lower mIoU than FC_FROZEN_A7; no single metric establishes universal superiority.

FC_FROZEN_A7 changes 52/375 Replica owner labels relative to A7. On 161 uniquely geometry-identifiable owners it corrects 10 and harms 2, with 17 unused-correct-evidence cases. OVR_A7 corrects 9 and harms 2, with 23 unused-correct-evidence cases. These geometry diagnostics are distinct from released AP match-set changes and cannot be counted as AP true positives.

The sole CAL-frozen GMED+OVR combination has Replica APall 10.286507% and mIoU 28.413136%; it does not exceed FC_FROZEN_A7. Its CAL APall interaction is negative (-0.507790 percentage points). Favorable and adverse interactions are retained; no alternate combination was selected after viewing Replica.

E04 SHORTLIST exactly preserves A7 decisions. SPATIAL and MIX50 remain untested because official original SAM3 access returned 401/GatedRepo. Missing model evidence is not a measured zero or a scientific failure.

Both datasets were previously exposed. These are development/regression findings without a claim of statistical significance or fresh generalization. Visual networks are frozen, but fitting temperature scalars still uses CAL semantic supervision. Pretraining/checkpoint overlap with evaluation datasets is unresolved, not assumed clean.

Next-wave recommendation (not executed): E06: a fixed class text-prototype bank with image features frozen. Among available, uniquely geometry-identifiable Replica source errors, the correct class ranks 2-5 in 60/75 FC_FROZEN errors and 53/63 OVR errors (S2:70/91). This supports testing class-language discrimination as an unresolved capability. Use the roadmap's predeclared raw-name/structure/appearance bank and freeze image features; do not invent GT-specific aliases or tune descriptions on these errors. This is a prospective recommendation, not a proved remedy; fusion conflict remains a limitation. E06 was not executed in wave1.
