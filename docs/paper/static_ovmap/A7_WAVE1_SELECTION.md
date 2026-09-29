# A7 wave-1 selection

CAL-frozen research nomination: **N0**. Deployment: **N0_UNCHANGED**.

Selection uses CAL released pooled APall, then mIoU, then AP50 (tolerance 1e-10), then fewer unique required visual operations, then fixed registry order. Eligible candidates are the measured _A7 variants, E04_MIX50 and the optional combination, with N0 and A7_REFIT as incumbents. DIRECT rows and standalone Q/S2 are diagnostic controls, excluded by the prespecified nomination rule even when their CAL APall is higher. Replica was not used to nominate, prune prescribed methods, tune temperatures or choose the combination.

The one frozen combination uses AW_E01_GMED and AW_E03_OVR; each retains its own CAL-fitted scalar. No new image inference or scalar search is introduced for composition.

Nomination identity: `f167b38fdd6cb7a7eec33a635c091f7f195a1fc1964c3611a2fbf2c88a0a2559`. Transfer lock: `aebcf15ea870a47c28cc0d0df805fe1f6a2bdbe53cdb710919e36d3a8a2896fb`.

| CAL ranking | APall (%) | mIoU (%) | AP50 (%) | Required operations |
| --- | --- | --- | --- | --- |
| N0 | 4.235290887 | 22.015249283 | 14.985803805 | 11828 |
| AW_E03_OVR_A7 | 4.234029003 | 24.884025276 | 14.927248677 | 13290 |
| AW_E03_FC_FROZEN_A7 | 4.224843229 | 24.457288182 | 14.905202822 | 13290 |
| AW_E02_SAM2_A7 | 3.743512432 | 22.457144877 | 14.230065202 | 16551 |
| AW_E01_GMED_A7 | 3.732776283 | 22.729982300 | 14.181264980 | 15724 |
| AW_E01_RAW_EQ_A7 | 3.732776283 | 22.698982388 | 14.181264980 | 15724 |
| AW_E01_UNIT_EQ_A7 | 3.732776283 | 22.698982388 | 14.181264980 | 15724 |
| RV_A7_COS_REFIT | 3.732776283 | 22.680712299 | 14.181264980 | 15724 |
| AW_E02_GLOBAL_A7 | 3.732776283 | 22.645274628 | 14.181264980 | 15724 |
| AW_C0_SO400M_A7 | 3.730058522 | 22.812832811 | 14.091911175 | 15724 |
| AW_COMBO_QR | 3.726239467 | 23.735308780 | 14.013815403 | 13290 |

Both datasets are previously exposed development/regression evidence. N0 remains the deployment baseline regardless of favorable Replica columns.
