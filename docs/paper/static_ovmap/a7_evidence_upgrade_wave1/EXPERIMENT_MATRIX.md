# Exact first-wave experiment registry

Every source variant creates DIRECT and A7 outputs. Five original controls are shared. E04 uses three complete-map outputs. All rows are evaluated in both rank views.

| Family | Variant | Changed slot | DIRECT | A7 |
|---|---|---|---|---|
| E01 | AW_E01_RAW_EQ | Q_GAIN | AW_E01_RAW_EQ_DIRECT | AW_E01_RAW_EQ_A7 |
| E01 | AW_E01_UNIT_EQ | Q_GAIN | AW_E01_UNIT_EQ_DIRECT | AW_E01_UNIT_EQ_A7 |
| E01 | AW_E01_GMED | Q_GAIN | AW_E01_GMED_DIRECT | AW_E01_GMED_A7 |
| E02 | AW_E02_GLOBAL | S_SIGLIP2_AREA | AW_E02_GLOBAL_DIRECT | AW_E02_GLOBAL_A7 |
| E02 | AW_E02_SAM2 | S_SIGLIP2_AREA | AW_E02_SAM2_DIRECT | AW_E02_SAM2_A7 |
| C0 | AW_C0_SO400M | S_SIGLIP2_AREA | AW_C0_SO400M_DIRECT | AW_C0_SO400M_A7 |
| E03 | AW_E03_FC_FROZEN | S_SIGLIP2_AREA | AW_E03_FC_FROZEN_DIRECT | AW_E03_FC_FROZEN_A7 |
| E03 | AW_E03_OVR | S_SIGLIP2_AREA | AW_E03_OVR_DIRECT | AW_E03_OVR_A7 |

E04: `AW_E04_SHORTLIST`, `AW_E04_SPATIAL`, `AW_E04_MIX50`.

Controls: `N0`, `Q_GAIN`, `S_SIGLIP2_AREA`, `RV_A7_COS_FIXED`, `RV_A7_COS_REFIT`.

**When all listed assets are available:** 19 new +5 controls,10 scenes,240 scene-method records,480 dual-rank records,48 Replica pools. These are not independent statistical trials.

**Optional one compatible combination:** `AW_COMBO_QR`,10 additional predictions,20 rank rows,2 Replica pools.

Every blocked leaf remains listed with its concrete cause. Do not replace missing measurements by zeros or call the maximum matrix measured when assets were inaccessible.
