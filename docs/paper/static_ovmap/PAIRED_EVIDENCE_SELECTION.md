# Paired evidence CAL selection

Frozen research nominee: **PE_COMBO_D4_R3**; deployment N0_UNCHANGED.

Sequential practical bands: APall 0.05 pp, mIoU 0.10 pp, AP50 0.10 pp; then fixed compute tier, distance from identity, registry order. Bands express preference, not significance.

| Configuration | Parameter | Tier | apall | ap50 | ap25 | miou | macc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N0 | None | 0 | 4.235291 | 14.985804 | 30.183916 | 22.015249 | 28.920037 |
| RV_A7_COS_REFIT | None | 0 | 3.732776 | 14.181265 | 28.535003 | 22.680712 | 30.717638 |
| AW_E03_FC_FROZEN_A7 | None | 0 | 4.224843 | 14.905203 | 31.590608 | 24.457288 | 31.322824 |
| AW_E03_FC_FROZEN_DIRECT | None | 0 | 3.690231 | 13.494268 | 25.322972 | 20.007402 | 26.119578 |
| AW_E03_OVR_DIRECT | None | 0 | 5.119048 | 16.119929 | 29.438933 | 22.411281 | 28.469935 |
| AW_E03_OVR_A7 | None | 0 | 4.234029 | 14.927249 | 31.782474 | 24.884025 | 31.697263 |
| PE_R0_BLEND__0 | 0 | 3 | 4.224843 | 14.905203 | 31.590608 | 24.457288 | 31.322824 |
| PE_R0_BLEND__0.25 | 0.25 | 3 | 4.224843 | 14.905203 | 31.559744 | 24.425299 | 31.320828 |
| PE_R0_BLEND__0.5 | 0.5 | 3 | 4.234029 | 14.927249 | 31.768445 | 24.863916 | 31.681393 |
| PE_R0_BLEND__0.75 | 0.75 | 3 | 4.234029 | 14.927249 | 31.782474 | 24.883935 | 31.697172 |
| PE_R0_BLEND__1 | 1 | 3 | 4.234029 | 14.927249 | 31.782474 | 24.884025 | 31.697263 |
| PE_R1_FOURWAY | None | 3 | 4.093180 | 13.622869 | 26.966490 | 21.710783 | 27.699696 |
| PE_R2_GLOBAL__0 | 0 | 3 | 4.224843 | 14.905203 | 31.590608 | 24.457288 | 31.322824 |
| PE_R2_GLOBAL__0.25 | 0.25 | 3 | 4.234029 | 14.927249 | 28.873524 | 24.206096 | 29.150045 |
| PE_R2_GLOBAL__0.5 | 0.5 | 3 | 4.234029 | 14.927249 | 28.814300 | 23.515967 | 29.515734 |
| PE_R2_GLOBAL__1 | 1 | 3 | 4.995101 | 16.174309 | 25.907922 | 22.940512 | 27.377135 |
| PE_R2_GLOBAL__2 | 2 | 3 | 5.546288 | 18.627646 | 32.214139 | 25.741034 | 30.327686 |
| PE_R3_LOCAL__0 | 0 | 3 | 4.224843 | 14.905203 | 31.590608 | 24.457288 | 31.322824 |
| PE_R3_LOCAL__0.25 | 0.25 | 3 | 4.234029 | 14.927249 | 28.873524 | 24.206096 | 29.150045 |
| PE_R3_LOCAL__0.5 | 0.5 | 3 | 4.234029 | 14.927249 | 28.814300 | 23.515967 | 29.515734 |
| PE_R3_LOCAL__1 | 1 | 3 | 4.995101 | 16.174309 | 25.907922 | 22.940512 | 27.377135 |
| PE_R3_LOCAL__2 | 2 | 3 | 5.546288 | 18.627646 | 28.232657 | 24.365021 | 28.924525 |
| PE_R3_LOCAL_CALDELTA__0 | 0 | 3 | 4.224843 | 14.905203 | 31.590608 | 24.457288 | 31.322824 |
| PE_R3_LOCAL_CALDELTA__0.25 | 0.25 | 3 | 4.234029 | 14.927249 | 31.768445 | 24.847774 | 31.473389 |
| PE_R3_LOCAL_CALDELTA__0.5 | 0.5 | 3 | 4.234029 | 14.927249 | 31.651301 | 25.218909 | 31.787482 |
| PE_R3_LOCAL_CALDELTA__1 | 1 | 3 | 5.818146 | 19.878013 | 29.687684 | 25.999995 | 30.480335 |
| PE_R3_LOCAL_CALDELTA__2 | 2 | 3 | 5.529141 | 19.707892 | 29.405497 | 25.721418 | 29.129408 |
| PE_D1_LOGPOOL | None | 1 | 5.047888 | 18.608907 | 32.558422 | 26.072036 | 32.677062 |
| PE_D1_ANCHORED | None | 1 | 5.044112 | 18.601864 | 32.699769 | 25.793640 | 32.316824 |
| PE_D2_GROUPED | None | 1 | 4.067460 | 13.648589 | 26.945914 | 21.299392 | 26.901827 |
| PE_D3_DIAGONAL | None | 2 | 5.044112 | 18.601864 | 32.711640 | 25.256142 | 31.751952 |
| PE_D4_LINEAGE | None | 2 | 5.014820 | 18.548280 | 32.473799 | 25.725685 | 32.281102 |
| PE_D4_SHUFFLED | None | 2 | 5.014820 | 18.548280 | 32.473799 | 25.725685 | 32.281102 |

## Frozen configurations

| Method | Selected parameter | Selected grid ID | Best active parameter |
| --- | --- | --- | --- |
| PE_D1_ANCHORED | None | PE_D1_ANCHORED | None |
| PE_D1_LOGPOOL | None | PE_D1_LOGPOOL | None |
| PE_D2_GROUPED | None | PE_D2_GROUPED | None |
| PE_D3_DIAGONAL | None | PE_D3_DIAGONAL | None |
| PE_D4_LINEAGE | None | PE_D4_LINEAGE | None |
| PE_D4_SHUFFLED | None | PE_D4_SHUFFLED | None |
| PE_R0_BLEND | 0.5 | PE_R0_BLEND__0.5 | 0.5 |
| PE_R1_FOURWAY | None | PE_R1_FOURWAY | None |
| PE_R2_GLOBAL | 2 | PE_R2_GLOBAL__2 | 2 |
| PE_R3_LOCAL | 2 | PE_R3_LOCAL__2 | 2 |
| PE_R3_LOCAL_CALDELTA | 1 | PE_R3_LOCAL_CALDELTA__1 | 1 |

Composition: `{'component_changed_CAL_labels': {'PE_D4_LINEAGE': 33, 'PE_R3_LOCAL__2': 78}, 'id': 'PE_COMBO_D4_R3', 'interaction_probe_uses_nonselected_eta': False, 'method': 'PE_COMBO_D4_R3', 'metrics': {'ap25': 0.3308017342739565, 'ap50': 0.2296443268665491, 'apall': 0.06481154876216604, 'defined_classes': 28, 'macc': 0.3326499675557527, 'miou': 0.28902347868601164, 'uap': 0.06481154876216604}, 'parameter': 2, 'registry_order': 33, 'status': 'MEASURED', 'tier': 3}`

Transfer identity: `1b5c3dfca410a965249f77b2275a6e54517bbc9e5188e3f767180b2654456198`. Full retained sets at each ranking stage are in transfer_lock.json. Every choice was locked before Replica evaluation; DIRECT controls were eligible, shuffled dependence was not.
