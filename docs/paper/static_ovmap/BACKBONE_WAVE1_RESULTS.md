# Backbone wave-1 measured results

Status: `COMPLETE_NO_NET_GAIN`. Successful map-scene configurations: 64/64; primary official rows: 192/192. All data are exposed. Deployment: N0_UNCHANGED.

## Development Official Pools (%)

| Map | Readout | APall | AP50 | AP25 | mIoU | mAcc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BB00_NATIVE | D2 | 12.58 | 22.60 | 31.71 | 23.56 | 32.42 |
| BB00_NATIVE | FC_EQ | 11.26 | 21.19 | 32.83 | 23.98 | 33.96 |
| BB00_NATIVE | NATIVE_READOUT | 11.21 | 23.27 | 33.85 | 24.19 | 34.34 |
| BB01_SYNC | D2 | 13.21 | 25.46 | 35.03 | 25.18 | 34.08 |
| BB01_SYNC | FC_EQ | 11.62 | 23.57 | 33.47 | 24.40 | 34.47 |
| BB01_SYNC | NATIVE_READOUT | 11.20 | 23.28 | 33.86 | 24.50 | 34.54 |
| BB03_SAM2_GEOM | D2 | 9.81 | 20.72 | 33.42 | 22.64 | 33.53 |
| BB03_SAM2_GEOM | FC_EQ | 9.59 | 22.14 | 34.00 | 24.72 | 36.32 |
| BB03_SAM2_GEOM | NATIVE_READOUT | 10.23 | 22.19 | 31.85 | 23.89 | 34.65 |
| BB03_SAM2_RAW | D2 | 9.80 | 20.71 | 33.41 | 22.65 | 33.54 |
| BB03_SAM2_RAW | FC_EQ | 9.58 | 22.13 | 34.00 | 24.73 | 36.32 |
| BB03_SAM2_RAW | NATIVE_READOUT | 10.45 | 22.84 | 32.37 | 23.86 | 34.51 |
| BB05_BIDIR | D2 | 6.09 | 13.91 | 32.70 | 22.97 | 33.20 |
| BB05_BIDIR | FC_EQ | 6.04 | 13.81 | 30.53 | 22.85 | 33.25 |
| BB05_BIDIR | NATIVE_READOUT | 6.73 | 14.70 | 31.36 | 23.20 | 33.68 |
| BB05_FORWARD | D2 | 8.35 | 15.71 | 30.08 | 22.13 | 29.29 |
| BB05_FORWARD | FC_EQ | 8.70 | 16.72 | 31.41 | 23.22 | 30.36 |
| BB05_FORWARD | NATIVE_READOUT | 9.41 | 21.06 | 35.90 | 26.42 | 33.24 |
| BB05_RATIO_GATE | D2 | 12.58 | 22.69 | 31.71 | 23.58 | 32.48 |
| BB05_RATIO_GATE | FC_EQ | 11.26 | 21.31 | 32.83 | 23.99 | 34.01 |
| BB05_RATIO_GATE | NATIVE_READOUT | 11.21 | 23.35 | 33.85 | 24.21 | 34.40 |
| BBX_COMPOSE | D2 | 9.76 | 20.82 | 34.73 | 22.84 | 32.56 |
| BBX_COMPOSE | FC_EQ | 9.77 | 22.49 | 36.96 | 25.08 | 35.25 |
| BBX_COMPOSE | NATIVE_READOUT | 10.45 | 22.87 | 33.87 | 24.32 | 34.55 |

## Replica Official Pools (%)

| Map | Readout | APall | AP50 | AP25 | mIoU | mAcc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BB00_NATIVE | D2 | 11.74 | 24.50 | 37.97 | 29.69 | 37.43 |
| BB00_NATIVE | FC_EQ | 10.41 | 24.32 | 37.73 | 29.98 | 36.62 |
| BB00_NATIVE | NATIVE_READOUT | 8.63 | 21.48 | 34.59 | 27.24 | 32.67 |
| BB01_SYNC | D2 | 11.54 | 23.75 | 37.93 | 30.03 | 37.61 |
| BB01_SYNC | FC_EQ | 10.40 | 24.34 | 38.74 | 31.03 | 37.33 |
| BB01_SYNC | NATIVE_READOUT | 8.56 | 21.35 | 34.46 | 26.47 | 32.14 |
| BB03_SAM2_RAW | D2 | 11.37 | 22.30 | 36.22 | 30.06 | 37.09 |
| BB03_SAM2_RAW | FC_EQ | 11.22 | 22.57 | 37.24 | 29.73 | 36.38 |
| BB03_SAM2_RAW | NATIVE_READOUT | 8.20 | 17.85 | 30.58 | 25.13 | 31.02 |
| BBX_COMPOSE | D2 | 11.12 | 21.96 | 36.33 | 30.43 | 37.18 |
| BBX_COMPOSE | FC_EQ | 10.97 | 22.20 | 36.87 | 29.77 | 36.00 |
| BBX_COMPOSE | NATIVE_READOUT | 7.83 | 17.15 | 29.96 | 24.78 | 31.10 |

## Raw Geometry (%)

| Cohort | Map | Eligible GT | Raw R25 | Raw R50 | Raw R75 | Best GT IoU | Fragments/GT | Substantial impurity | Substantial GT coverage | Surface P5 | Surface C5 | Surface F5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| development | BB00_NATIVE | 108 | 60.185 | 35.185 | 12.963 | 40.265 | 2.324 | 35.051 | 74.801 | 74.870 | 81.265 | 77.937 |
| development | BB01_SYNC | 108 | 60.185 | 36.111 | 12.963 | 40.324 | 2.324 | 34.113 | 74.897 | 74.870 | 81.265 | 77.937 |
| development | BB03_SAM2_GEOM | 108 | 60.185 | 36.111 | 12.963 | 40.828 | 2.333 | 35.278 | 75.321 | 74.870 | 81.266 | 77.937 |
| development | BB03_SAM2_RAW | 108 | 60.185 | 36.111 | 12.963 | 40.828 | 2.333 | 35.278 | 75.321 | 74.870 | 81.266 | 77.937 |
| development | BB05_BIDIR | 108 | 52.778 | 23.148 | 8.333 | 33.578 | 5.481 | 17.618 | 60.770 | 74.870 | 81.265 | 77.937 |
| development | BB05_FORWARD | 108 | 60.185 | 32.407 | 12.963 | 39.165 | 2.759 | 31.443 | 72.501 | 74.870 | 81.265 | 77.937 |
| development | BB05_RATIO_GATE | 108 | 60.185 | 35.185 | 12.963 | 40.265 | 2.324 | 35.051 | 74.801 | 74.870 | 81.265 | 77.937 |
| development | BBX_COMPOSE | 108 | 60.185 | 36.111 | 12.963 | 40.695 | 2.333 | 34.709 | 75.445 | 74.870 | 81.266 | 77.937 |
| replica | BB00_NATIVE | 389 | 62.468 | 41.902 | 18.252 | 42.713 | 1.684 | 37.063 | 74.607 | 97.029 | 83.382 | 89.689 |
| replica | BB01_SYNC | 389 | 62.211 | 41.388 | 18.509 | 42.586 | 1.663 | 37.838 | 74.601 | 97.029 | 83.382 | 89.689 |
| replica | BB03_SAM2_RAW | 389 | 62.211 | 40.874 | 18.252 | 42.621 | 1.704 | 35.381 | 74.158 | 97.029 | 83.382 | 89.689 |
| replica | BBX_COMPOSE | 389 | 61.697 | 40.103 | 18.766 | 42.311 | 1.692 | 36.407 | 74.310 | 97.029 | 83.382 | 89.689 |

## Mechanism and Simple-Control Comparisons

| Cohort | Variant / Control | Readout | APall delta (pp) | mIoU delta (pp) | Raw R50 delta (pp) | Surface F5 delta (pp) | Label |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| development | BB01_SYNC / BB00_NATIVE | NATIVE_READOUT | -0.012 | +0.310 | +0.926 | +0.000 | TRADEOFF |
| development | BB01_SYNC / BB00_NATIVE | FC_EQ | +0.356 | +0.424 | +0.926 | +0.000 | MEASURED_NET_GAIN |
| development | BB01_SYNC / BB00_NATIVE | D2 | +0.626 | +1.617 | +0.926 | +0.000 | MEASURED_NET_GAIN |
| development | BB03_SAM2_GEOM / BB00_NATIVE | NATIVE_READOUT | -0.981 | -0.303 | +0.926 | -0.000 | NO_MEASURED_NET_GAIN |
| development | BB03_SAM2_GEOM / BB00_NATIVE | FC_EQ | -1.667 | +0.746 | +0.926 | -0.000 | TRADEOFF |
| development | BB03_SAM2_GEOM / BB00_NATIVE | D2 | -2.769 | -0.922 | +0.926 | -0.000 | NO_MEASURED_NET_GAIN |
| development | BB03_SAM2_GEOM / BB03_SAM2_RAW | NATIVE_READOUT | -0.216 | +0.029 | +0.000 | -0.000 | TRADEOFF |
| development | BB03_SAM2_GEOM / BB03_SAM2_RAW | FC_EQ | +0.016 | -0.008 | +0.000 | -0.000 | TRADEOFF |
| development | BB03_SAM2_GEOM / BB03_SAM2_RAW | D2 | +0.014 | -0.014 | +0.000 | -0.000 | TRADEOFF |
| development | BB03_SAM2_RAW / BB00_NATIVE | NATIVE_READOUT | -0.765 | -0.331 | +0.926 | -0.000 | NO_MEASURED_NET_GAIN |
| development | BB03_SAM2_RAW / BB00_NATIVE | FC_EQ | -1.683 | +0.754 | +0.926 | -0.000 | TRADEOFF |
| development | BB03_SAM2_RAW / BB00_NATIVE | D2 | -2.783 | -0.908 | +0.926 | -0.000 | NO_MEASURED_NET_GAIN |
| development | BB05_BIDIR / BB00_NATIVE | NATIVE_READOUT | -4.484 | -0.985 | -12.037 | +0.000 | NO_MEASURED_NET_GAIN |
| development | BB05_BIDIR / BB00_NATIVE | FC_EQ | -5.225 | -1.128 | -12.037 | +0.000 | NO_MEASURED_NET_GAIN |
| development | BB05_BIDIR / BB00_NATIVE | D2 | -6.488 | -0.589 | -12.037 | +0.000 | NO_MEASURED_NET_GAIN |
| development | BB05_BIDIR / BB05_FORWARD | NATIVE_READOUT | -2.681 | -3.212 | -9.259 | +0.000 | NO_MEASURED_NET_GAIN |
| development | BB05_BIDIR / BB05_FORWARD | FC_EQ | -2.661 | -0.375 | -9.259 | +0.000 | NO_MEASURED_NET_GAIN |
| development | BB05_BIDIR / BB05_FORWARD | D2 | -2.253 | +0.839 | -9.259 | +0.000 | TRADEOFF |
| development | BB05_FORWARD / BB00_NATIVE | NATIVE_READOUT | -1.803 | +2.226 | -2.778 | +0.000 | TRADEOFF |
| development | BB05_FORWARD / BB00_NATIVE | FC_EQ | -2.564 | -0.753 | -2.778 | +0.000 | NO_MEASURED_NET_GAIN |
| development | BB05_FORWARD / BB00_NATIVE | D2 | -4.236 | -1.428 | -2.778 | +0.000 | NO_MEASURED_NET_GAIN |
| development | BB05_FORWARD / BB05_RATIO_GATE | NATIVE_READOUT | -1.798 | +2.209 | -2.778 | +0.000 | TRADEOFF |
| development | BB05_FORWARD / BB05_RATIO_GATE | FC_EQ | -2.563 | -0.768 | -2.778 | +0.000 | NO_MEASURED_NET_GAIN |
| development | BB05_FORWARD / BB05_RATIO_GATE | D2 | -4.237 | -1.449 | -2.778 | +0.000 | NO_MEASURED_NET_GAIN |
| development | BB05_RATIO_GATE / BB00_NATIVE | NATIVE_READOUT | -0.005 | +0.018 | +0.000 | -0.000 | TRADEOFF |
| development | BB05_RATIO_GATE / BB00_NATIVE | FC_EQ | -0.001 | +0.015 | +0.000 | -0.000 | TRADEOFF |
| development | BB05_RATIO_GATE / BB00_NATIVE | D2 | +0.001 | +0.021 | +0.000 | -0.000 | NO_MEASURED_NET_GAIN |
| development | BBX_COMPOSE / BB00_NATIVE | NATIVE_READOUT | -0.768 | +0.130 | +0.926 | -0.000 | TRADEOFF |
| development | BBX_COMPOSE / BB00_NATIVE | FC_EQ | -1.494 | +1.105 | +0.926 | -0.000 | TRADEOFF |
| development | BBX_COMPOSE / BB00_NATIVE | D2 | -2.820 | -0.724 | +0.926 | -0.000 | NO_MEASURED_NET_GAIN |
| replica | BB01_SYNC / BB00_NATIVE | NATIVE_READOUT | -0.064 | -0.766 | -0.514 | +0.000 | NO_MEASURED_NET_GAIN |
| replica | BB01_SYNC / BB00_NATIVE | FC_EQ | -0.003 | +1.048 | -0.514 | +0.000 | TRADEOFF |
| replica | BB01_SYNC / BB00_NATIVE | D2 | -0.202 | +0.340 | -0.514 | +0.000 | TRADEOFF |
| replica | BB03_SAM2_RAW / BB00_NATIVE | NATIVE_READOUT | -0.428 | -2.115 | -1.028 | -0.000 | NO_MEASURED_NET_GAIN |
| replica | BB03_SAM2_RAW / BB00_NATIVE | FC_EQ | +0.816 | -0.250 | -1.028 | -0.000 | TRADEOFF |
| replica | BB03_SAM2_RAW / BB00_NATIVE | D2 | -0.373 | +0.365 | -1.028 | -0.000 | TRADEOFF |
| replica | BBX_COMPOSE / BB00_NATIVE | NATIVE_READOUT | -0.802 | -2.462 | -1.799 | +0.000 | NO_MEASURED_NET_GAIN |
| replica | BBX_COMPOSE / BB00_NATIVE | FC_EQ | +0.562 | -0.207 | -1.799 | +0.000 | TRADEOFF |
| replica | BBX_COMPOSE / BB00_NATIVE | D2 | -0.620 | +0.740 | -1.799 | +0.000 | TRADEOFF |

These labels describe APall/mIoU measurements only. The raw columns separately expose instance and surface changes; a positive semantic label alone does not establish stronger geometry. BIDIR is compared with FORWARD, FORWARD with the ratio switch, and GEOM with RAW as well as BB00. Missing complete cohorts produce no pooled comparison.

The bound native overlap-ratio threshold is 0.0. At 0.0, its added predicate is redundant for accepted positive-count candidates. Small differences between RATIO_GATE and BB00 may arise from the recorded eight-thread mapping variation; they do not establish an effective ratio-gating mechanism.

## Per-Scene Changes Against the Fresh BB00

| Scene | Map | Readout | APall delta (pp) | AP50 delta (pp) | AP25 delta (pp) | mIoU delta (pp) | mAcc delta (pp) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| office4 | BBX_COMPOSE | NATIVE_READOUT | +0.000 | +0.000 | +0.769 | -0.087 | -0.174 |
| office4 | BBX_COMPOSE | FC_EQ | +0.000 | +0.000 | +0.769 | -4.612 | -4.904 |
| office4 | BBX_COMPOSE | D2 | +0.000 | +0.000 | +0.000 | +0.013 | -0.144 |
| office4 | BB01_SYNC | NATIVE_READOUT | +0.000 | +0.000 | +0.000 | -0.012 | -0.059 |
| office4 | BB01_SYNC | FC_EQ | +0.000 | +0.000 | +0.000 | -0.013 | -0.059 |
| office4 | BB01_SYNC | D2 | +0.000 | +0.000 | +0.000 | -0.013 | -0.059 |
| office4 | BB03_SAM2_RAW | NATIVE_READOUT | +0.000 | +0.000 | +0.769 | -0.094 | -0.193 |
| office4 | BB03_SAM2_RAW | FC_EQ | +0.000 | +0.000 | +0.769 | +0.192 | +0.024 |
| office4 | BB03_SAM2_RAW | D2 | +0.000 | +0.000 | +0.769 | +0.191 | +0.024 |
| room1 | BBX_COMPOSE | NATIVE_READOUT | -0.167 | -0.752 | +1.880 | +0.301 | +2.141 |
| room1 | BBX_COMPOSE | FC_EQ | +6.850 | +7.143 | +4.511 | +4.609 | +4.774 |
| room1 | BBX_COMPOSE | D2 | -0.682 | -0.877 | -1.880 | +4.277 | +3.849 |
| room1 | BB01_SYNC | NATIVE_READOUT | +0.097 | +0.000 | +0.000 | -0.031 | -0.021 |
| room1 | BB01_SYNC | FC_EQ | +0.097 | +0.000 | +0.000 | -0.035 | -0.026 |
| room1 | BB01_SYNC | D2 | +0.097 | +0.000 | +0.000 | +0.056 | -0.013 |
| room1 | BB03_SAM2_RAW | NATIVE_READOUT | +0.272 | +2.632 | +2.632 | +1.283 | +2.274 |
| room1 | BB03_SAM2_RAW | FC_EQ | +6.906 | +7.895 | +5.263 | +4.844 | +4.923 |
| room1 | BB03_SAM2_RAW | D2 | -0.696 | -0.877 | -1.880 | +4.136 | +3.570 |
| office2 | BBX_COMPOSE | NATIVE_READOUT | -2.560 | -10.784 | -2.941 | -4.537 | -5.239 |
| office2 | BBX_COMPOSE | FC_EQ | -2.560 | -8.824 | +0.196 | -0.312 | -1.239 |
| office2 | BBX_COMPOSE | D2 | -2.560 | -8.824 | -0.980 | -0.485 | -1.664 |
| office2 | BB01_SYNC | NATIVE_READOUT | +0.218 | +0.000 | +0.000 | +0.057 | +0.143 |
| office2 | BB01_SYNC | FC_EQ | +0.218 | +0.000 | +0.000 | +0.081 | +0.208 |
| office2 | BB01_SYNC | D2 | +0.218 | +0.000 | +0.000 | +0.050 | +0.209 |
| office2 | BB03_SAM2_RAW | NATIVE_READOUT | -2.560 | -10.784 | -2.941 | -4.583 | -5.293 |
| office2 | BB03_SAM2_RAW | FC_EQ | -2.560 | -8.824 | +0.196 | +0.054 | -1.368 |
| office2 | BB03_SAM2_RAW | D2 | -2.560 | -8.824 | -0.980 | -0.666 | -1.793 |
| scene0056_00 | BB05_RATIO_GATE | NATIVE_READOUT | +0.000 | +0.000 | +0.000 | +0.015 | +0.017 |
| scene0056_00 | BB05_RATIO_GATE | FC_EQ | +0.000 | +0.000 | +0.000 | +0.012 | +0.014 |
| scene0056_00 | BB05_RATIO_GATE | D2 | +0.000 | +0.000 | +0.000 | +0.021 | +0.014 |
| scene0056_00 | BBX_COMPOSE | NATIVE_READOUT | -0.033 | -0.171 | +1.953 | -0.268 | -0.619 |
| scene0056_00 | BBX_COMPOSE | FC_EQ | -0.115 | -2.934 | +2.096 | -2.989 | +1.144 |
| scene0056_00 | BBX_COMPOSE | D2 | +0.810 | +4.613 | +7.837 | +1.647 | +6.036 |
| scene0056_00 | BB01_SYNC | NATIVE_READOUT | +0.005 | +0.024 | +0.000 | +0.212 | +0.048 |
| scene0056_00 | BB01_SYNC | FC_EQ | +0.000 | +0.000 | +0.050 | +0.042 | +0.052 |
| scene0056_00 | BB01_SYNC | D2 | +0.000 | +0.000 | +0.000 | +0.056 | +0.058 |
| scene0056_00 | BB05_BIDIR | NATIVE_READOUT | -0.751 | -6.995 | -14.007 | -0.576 | -0.014 |
| scene0056_00 | BB05_BIDIR | FC_EQ | -0.755 | -6.936 | -20.012 | -5.697 | -4.223 |
| scene0056_00 | BB05_BIDIR | D2 | +0.007 | +0.003 | -12.478 | -0.703 | +1.950 |
| scene0056_00 | BB03_SAM2_RAW | NATIVE_READOUT | -0.033 | -0.171 | +3.738 | -0.144 | -0.625 |
| scene0056_00 | BB03_SAM2_RAW | FC_EQ | -0.115 | -2.934 | +2.096 | -2.963 | +1.224 |
| scene0056_00 | BB03_SAM2_RAW | D2 | +0.810 | +4.613 | +7.837 | +1.683 | +6.074 |
| scene0056_00 | BB05_FORWARD | NATIVE_READOUT | +1.662 | +0.306 | +1.028 | -0.497 | -1.346 |
| scene0056_00 | BB05_FORWARD | FC_EQ | +2.175 | +1.772 | -5.412 | -5.170 | -1.658 |
| scene0056_00 | BB05_FORWARD | D2 | +0.380 | +1.141 | -6.101 | -5.042 | -0.491 |
| scene0056_00 | BB03_SAM2_GEOM | NATIVE_READOUT | -0.033 | -0.171 | +3.738 | -0.197 | -0.655 |
| scene0056_00 | BB03_SAM2_GEOM | FC_EQ | -0.115 | -2.934 | +2.096 | -3.082 | +1.097 |
| scene0056_00 | BB03_SAM2_GEOM | D2 | +0.810 | +4.613 | +7.837 | +1.571 | +5.954 |
| office1 | BBX_COMPOSE | NATIVE_READOUT | -1.797 | -7.353 | -7.353 | -3.421 | -3.921 |
| office1 | BBX_COMPOSE | FC_EQ | -0.654 | -5.882 | -5.882 | -2.934 | -3.307 |
| office1 | BBX_COMPOSE | D2 | +1.144 | +1.471 | +1.471 | +2.143 | +0.166 |
| office1 | BB01_SYNC | NATIVE_READOUT | +0.000 | +0.000 | +0.000 | -0.027 | +0.004 |
| office1 | BB01_SYNC | FC_EQ | +0.000 | +0.000 | +0.000 | -0.317 | -0.452 |
| office1 | BB01_SYNC | D2 | +0.000 | +0.000 | +0.000 | -0.046 | -0.093 |
| office1 | BB03_SAM2_RAW | NATIVE_READOUT | -0.654 | -5.882 | -5.882 | -2.431 | -3.004 |
| office1 | BB03_SAM2_RAW | FC_EQ | -0.654 | -5.882 | -5.882 | -3.026 | -3.461 |
| office1 | BB03_SAM2_RAW | D2 | +1.144 | +1.471 | +1.471 | +2.047 | +0.010 |
| scene0626_00 | BB05_RATIO_GATE | NATIVE_READOUT | -0.337 | +0.000 | +0.000 | -0.009 | +0.009 |
| scene0626_00 | BB05_RATIO_GATE | FC_EQ | -0.337 | +0.000 | +0.000 | +0.017 | -0.052 |
| scene0626_00 | BB05_RATIO_GATE | D2 | -0.337 | +0.000 | +0.000 | +0.007 | -0.063 |
| scene0626_00 | BBX_COMPOSE | NATIVE_READOUT | -2.694 | -3.030 | -2.273 | -3.999 | -3.073 |
| scene0626_00 | BBX_COMPOSE | FC_EQ | -8.923 | -9.091 | -2.273 | -7.427 | -1.856 |
| scene0626_00 | BBX_COMPOSE | D2 | -8.923 | -9.091 | -2.273 | -9.551 | -3.986 |
| scene0626_00 | BB01_SYNC | NATIVE_READOUT | -0.337 | +0.000 | +0.000 | +0.004 | +0.002 |
| scene0626_00 | BB01_SYNC | FC_EQ | -0.337 | +0.000 | +0.000 | +0.016 | -0.023 |
| scene0626_00 | BB01_SYNC | D2 | -0.337 | +0.000 | +0.000 | +0.016 | -0.028 |
| scene0626_00 | BB05_BIDIR | NATIVE_READOUT | -3.213 | -4.419 | +0.126 | -0.838 | -0.196 |
| scene0626_00 | BB05_BIDIR | FC_EQ | -10.031 | -11.237 | -15.783 | -8.290 | -3.679 |
| scene0626_00 | BB05_BIDIR | D2 | -10.031 | -11.237 | -8.965 | -7.709 | -3.870 |
| scene0626_00 | BB03_SAM2_RAW | NATIVE_READOUT | -2.694 | -3.030 | -2.273 | -3.987 | -3.070 |
| scene0626_00 | BB03_SAM2_RAW | FC_EQ | -9.512 | -9.848 | -18.182 | -12.211 | -5.927 |
| scene0626_00 | BB03_SAM2_RAW | D2 | -8.923 | -9.091 | -2.273 | -9.541 | -3.986 |
| scene0626_00 | BB05_FORWARD | NATIVE_READOUT | -4.728 | -6.692 | +13.763 | +7.684 | +0.600 |
| scene0626_00 | BB05_FORWARD | FC_EQ | -11.546 | -13.510 | -4.419 | +2.170 | +4.485 |
| scene0626_00 | BB05_FORWARD | D2 | -11.546 | -13.510 | -4.419 | -0.009 | +2.299 |
| scene0626_00 | BB03_SAM2_GEOM | NATIVE_READOUT | -2.357 | -3.030 | -2.273 | -3.995 | -3.089 |
| scene0626_00 | BB03_SAM2_GEOM | FC_EQ | -9.175 | -9.848 | -18.182 | -12.219 | -5.946 |
| scene0626_00 | BB03_SAM2_GEOM | D2 | -8.586 | -9.091 | -2.273 | -9.549 | -4.003 |
| scene0445_00 | BB05_RATIO_GATE | NATIVE_READOUT | +0.000 | +0.000 | +0.000 | +0.015 | -0.003 |
| scene0445_00 | BB05_RATIO_GATE | FC_EQ | +0.000 | +0.000 | +0.000 | +0.013 | -0.005 |
| scene0445_00 | BB05_RATIO_GATE | D2 | +0.000 | +0.000 | +0.000 | +0.030 | +0.013 |
| scene0445_00 | BBX_COMPOSE | NATIVE_READOUT | -1.111 | +0.000 | +0.000 | -0.145 | -0.197 |
| scene0445_00 | BBX_COMPOSE | FC_EQ | +0.000 | +0.000 | +10.000 | +2.679 | +2.710 |
| scene0445_00 | BBX_COMPOSE | D2 | -7.778 | -10.000 | +0.000 | -4.997 | -5.175 |
| scene0445_00 | BB01_SYNC | NATIVE_READOUT | +0.000 | +0.000 | +0.000 | -0.019 | +0.007 |
| scene0445_00 | BB01_SYNC | FC_EQ | +0.000 | +0.000 | +0.000 | -0.017 | +0.008 |
| scene0445_00 | BB01_SYNC | D2 | +0.000 | +0.000 | +0.000 | -0.013 | -0.014 |
| scene0445_00 | BB05_BIDIR | NATIVE_READOUT | -11.111 | -10.000 | +0.000 | -15.239 | -4.345 |
| scene0445_00 | BB05_BIDIR | FC_EQ | -17.778 | -10.000 | +0.000 | -14.828 | -6.823 |
| scene0445_00 | BB05_BIDIR | D2 | -25.556 | -20.000 | -10.000 | -22.495 | -14.702 |
| scene0445_00 | BB03_SAM2_RAW | NATIVE_READOUT | -1.111 | +0.000 | +0.000 | -0.168 | -0.212 |
| scene0445_00 | BB03_SAM2_RAW | FC_EQ | +0.000 | +0.000 | +10.000 | +2.792 | +2.832 |
| scene0445_00 | BB03_SAM2_RAW | D2 | -7.778 | -10.000 | -10.000 | -7.850 | -8.105 |
| scene0445_00 | BB05_FORWARD | NATIVE_READOUT | -6.944 | -7.500 | -7.500 | -6.675 | +0.772 |
| scene0445_00 | BB05_FORWARD | FC_EQ | -1.111 | +0.000 | +0.000 | -0.185 | +0.337 |
| scene0445_00 | BB05_FORWARD | D2 | -8.889 | -10.000 | -10.000 | -7.853 | -7.540 |
| scene0445_00 | BB03_SAM2_GEOM | NATIVE_READOUT | +0.000 | +0.000 | +0.000 | -0.132 | -0.166 |
| scene0445_00 | BB03_SAM2_GEOM | FC_EQ | +0.000 | +0.000 | +10.000 | +2.861 | +2.909 |
| scene0445_00 | BB03_SAM2_GEOM | D2 | -7.778 | -10.000 | -10.000 | -7.821 | -8.065 |
| office0 | BBX_COMPOSE | NATIVE_READOUT | +0.764 | +1.250 | +0.000 | +1.638 | +1.698 |
| office0 | BBX_COMPOSE | FC_EQ | +0.486 | -1.250 | -2.500 | -0.913 | +1.502 |
| office0 | BBX_COMPOSE | D2 | +0.764 | +1.250 | +0.000 | -1.012 | -1.513 |
| office0 | BB01_SYNC | NATIVE_READOUT | +1.181 | +0.000 | +0.000 | +0.004 | +0.014 |
| office0 | BB01_SYNC | FC_EQ | +1.181 | +0.000 | +0.000 | +0.081 | +0.120 |
| office0 | BB01_SYNC | D2 | +1.181 | +0.000 | +0.000 | +0.136 | +0.128 |
| office0 | BB03_SAM2_RAW | NATIVE_READOUT | +0.764 | +1.250 | +0.000 | +1.638 | +1.724 |
| office0 | BB03_SAM2_RAW | FC_EQ | +0.486 | -1.250 | -2.500 | -0.991 | +1.402 |
| office0 | BB03_SAM2_RAW | D2 | +0.764 | +1.250 | +0.000 | -1.067 | -1.616 |
| office3 | BBX_COMPOSE | NATIVE_READOUT | +0.483 | +0.000 | +0.000 | -0.086 | -0.025 |
| office3 | BBX_COMPOSE | FC_EQ | +0.483 | +0.000 | +4.348 | +3.116 | +3.400 |
| office3 | BBX_COMPOSE | D2 | +1.087 | +0.000 | +0.000 | -0.040 | -0.120 |
| office3 | BB01_SYNC | NATIVE_READOUT | +0.000 | +0.000 | +0.000 | -0.080 | -0.074 |
| office3 | BB01_SYNC | FC_EQ | +0.097 | +0.435 | +5.870 | +4.128 | +4.892 |
| office3 | BB01_SYNC | D2 | -0.386 | -0.652 | +0.435 | -0.204 | +0.186 |
| office3 | BB03_SAM2_RAW | NATIVE_READOUT | +0.000 | +0.000 | +0.000 | -0.174 | -0.176 |
| office3 | BB03_SAM2_RAW | FC_EQ | +0.000 | +0.000 | +4.348 | +3.048 | +3.266 |
| office3 | BB03_SAM2_RAW | D2 | +0.604 | +0.000 | +0.000 | -0.107 | -0.252 |
| room2 | BBX_COMPOSE | NATIVE_READOUT | -0.521 | +0.781 | +0.781 | +0.925 | +0.999 |
| room2 | BBX_COMPOSE | FC_EQ | -0.694 | +0.000 | +0.781 | +0.735 | +1.038 |
| room2 | BBX_COMPOSE | D2 | -0.694 | +0.000 | +2.344 | +1.257 | +1.500 |
| room2 | BB01_SYNC | NATIVE_READOUT | -1.215 | -1.562 | -2.604 | -5.055 | -3.056 |
| room2 | BB01_SYNC | FC_EQ | -0.521 | +0.000 | +1.562 | +0.156 | +0.544 |
| room2 | BB01_SYNC | D2 | -0.521 | +0.000 | +3.125 | +0.619 | +1.006 |
| room2 | BB03_SAM2_RAW | NATIVE_READOUT | +0.174 | +0.781 | +0.781 | +0.917 | +0.984 |
| room2 | BB03_SAM2_RAW | FC_EQ | +0.000 | +0.000 | +0.781 | +0.742 | +1.027 |
| room2 | BB03_SAM2_RAW | D2 | +0.000 | +0.000 | -0.781 | +0.535 | +0.421 |
| scene0534_00 | BB05_RATIO_GATE | NATIVE_READOUT | +0.021 | +0.188 | +0.000 | +0.025 | +0.088 |
| scene0534_00 | BB05_RATIO_GATE | FC_EQ | +0.029 | +0.263 | +0.000 | +0.010 | +0.082 |
| scene0534_00 | BB05_RATIO_GATE | D2 | +0.029 | +0.263 | +0.000 | +0.019 | +0.085 |
| scene0534_00 | BBX_COMPOSE | NATIVE_READOUT | -0.177 | +0.164 | -0.148 | +2.189 | +4.031 |
| scene0534_00 | BBX_COMPOSE | FC_EQ | +1.028 | +5.746 | +3.490 | +3.746 | +0.691 |
| scene0534_00 | BBX_COMPOSE | D2 | -0.068 | +1.140 | +4.335 | +0.519 | -1.564 |
| scene0534_00 | BB01_SYNC | NATIVE_READOUT | +0.000 | +0.000 | +0.061 | +1.336 | +1.422 |
| scene0534_00 | BB01_SYNC | FC_EQ | +0.439 | +3.947 | +1.937 | +1.329 | +0.884 |
| scene0534_00 | BB01_SYNC | D2 | +1.170 | +5.263 | +8.918 | +3.609 | +2.940 |
| scene0534_00 | BB05_BIDIR | NATIVE_READOUT | -1.365 | -10.526 | -0.672 | +2.710 | +2.789 |
| scene0534_00 | BB05_BIDIR | FC_EQ | -1.365 | -10.526 | +2.757 | +0.970 | +0.443 |
| scene0534_00 | BB05_BIDIR | D2 | -1.365 | -10.526 | +8.168 | +3.304 | +3.679 |
| scene0534_00 | BB03_SAM2_RAW | NATIVE_READOUT | -0.164 | +0.139 | -4.139 | +1.455 | +3.231 |
| scene0534_00 | BB03_SAM2_RAW | FC_EQ | +1.029 | +5.561 | +3.785 | +3.196 | +2.911 |
| scene0534_00 | BB03_SAM2_RAW | D2 | +0.008 | +0.970 | +4.598 | +0.056 | +0.764 |
| scene0534_00 | BB05_FORWARD | NATIVE_READOUT | +0.780 | +0.000 | +4.732 | +2.191 | -0.168 |
| scene0534_00 | BB05_FORWARD | FC_EQ | -0.390 | -5.263 | +2.898 | -2.145 | -7.276 |
| scene0534_00 | BB05_FORWARD | D2 | -0.390 | -5.263 | +8.348 | +0.139 | -3.927 |
| scene0534_00 | BB03_SAM2_GEOM | NATIVE_READOUT | -0.147 | +0.274 | -3.761 | +1.326 | +3.505 |
| scene0534_00 | BB03_SAM2_GEOM | FC_EQ | +1.034 | +5.592 | +3.808 | +3.241 | +2.962 |
| scene0534_00 | BB03_SAM2_GEOM | D2 | +0.014 | +1.007 | +4.627 | +0.100 | +0.807 |
| room0 | BBX_COMPOSE | NATIVE_READOUT | +1.903 | +0.000 | -1.280 | -3.577 | -2.710 |
| room0 | BBX_COMPOSE | FC_EQ | -0.693 | -6.696 | -5.655 | -1.263 | -3.522 |
| room0 | BBX_COMPOSE | D2 | -0.874 | -8.333 | -8.333 | -2.439 | -4.975 |
| room0 | BB01_SYNC | NATIVE_READOUT | +0.463 | +0.000 | +0.000 | -3.491 | -3.777 |
| room0 | BB01_SYNC | FC_EQ | +0.463 | +0.000 | +0.000 | +5.029 | +3.239 |
| room0 | BB01_SYNC | D2 | +0.231 | -2.083 | -2.083 | +4.123 | +2.320 |
| room0 | BB03_SAM2_RAW | NATIVE_READOUT | +1.620 | +0.000 | -1.280 | -3.556 | -3.178 |
| room0 | BB03_SAM2_RAW | FC_EQ | +0.413 | -2.530 | -1.488 | -0.589 | -0.631 |
| room0 | BB03_SAM2_RAW | D2 | +0.231 | -4.167 | -4.167 | -1.731 | -2.077 |

Nominee measurement label: `TRADEOFF`. Bridge: `VERIFIED`. APall uses the actual released overlap vector in the evaluation contexts, .50 through .90; AP25 is separate. Pools use the released evaluator over complete ordered cohorts. Semantic metrics sum scene confusion matrices.

Raw numeric geometry and official native triangle-first-color paint are separate. Class-agnostic recall uses strict IoU > .25/.5/.75, maximum-cardinality one-to-one matches and every eligible GT in the denominator. Missed GT best-IoU is zero. Surface precision/completeness use every exported vertex and every valid target vertex at strict 5cm; duplicated native vertices are retained consistently. Detailed raw geometry, GT-indexed gains/losses and predicted overlap associations are in the compact release.

Substantial-fragment impurity is the intersection-row-weighted mean of one minus fragment purity over the fixed intersections of at least 100 points and at least 1% GT coverage. It is conditional on those intersections and does not include unmatched predicted owners. Substantial GT coverage averages summed retained-fragment coverage over every eligible GT, including zero for missed GT.

## Actual Physical Work and Failures

```json
{
  "physical_image_encodings": 227386,
  "physical_image_encoder_calls_known": 39658,
  "physical_FC_region_poolings": 6420,
  "physical_FC_text_inputs": 3514,
  "model_loads_with_recorded_duration": 142,
  "worker_stage_wall_seconds_including_validation": 23595.07795149309,
  "recorded_phase_wall_seconds": 44855.82000000001,
  "recorded_phase_user_cpu_seconds": 247748.45,
  "recorded_phase_system_cpu_seconds": 10857.38
}
```

Recorded failed command attempts: 20. Recorded failed worker receipts: 0. Phase failures: 1. Details and retained logs are in `physical_work.json` and `failures.json`.

FC batch-call count was not separately recorded and remains null in its worker ledger; its actual image-input and region-pooling counts are recorded. The known encoder-call total includes N/Q and SAM only. Costs distinguish actual forwards from content-deduplicated standalone obligations. Common SAM predictions are required by both RAW/GEOM. N/Q/FC costs are required by both fused readouts. Timing is measured stage attribution, not a cold end-to-end rerun; missing timing components are disclosed per job. GPU-worker wall time includes CPU preprocessing, loading and IO; device-only GPU time was not instrumented. Recorded phase CPU totals cover the phases listed in `physical_work.json`, including children, and exclude earlier uninstrumented preparation/bridge work. Concurrent stages overlap and their duration sum is not end-to-end elapsed time. No significance or independent-generalization claim is made.
