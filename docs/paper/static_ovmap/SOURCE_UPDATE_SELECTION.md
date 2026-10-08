# Exact exposed-cohort selection

Selection: COMPLETE_NO_TARGET_GAIN; selected SU01_G1; passing candidates []; material target False.

All five metrics in both cohorts must be at least G1 minus 1e-10; CF18 APall must exceed D2 by more than 1e-10, with AP50 at least D2 minus 1e-10. The material marker adds 0.001 fraction. Lexicographic tie metrics and simplicity order are the verbatim protocol.

```json
{
  "deployment": "N0_UNCHANGED",
  "flags": {
    "SU02_HARD_MATCHED": {
      "CF_D2_crossing": true,
      "cf_all5": true,
      "material_target_met": false,
      "replica_all5": false,
      "target_met": false
    },
    "SU03_STABLE_MATCHED": {
      "CF_D2_crossing": false,
      "cf_all5": true,
      "material_target_met": false,
      "replica_all5": false,
      "target_met": false
    },
    "SU04_F_REPLACE": {
      "CF_D2_crossing": false,
      "cf_all5": true,
      "material_target_met": false,
      "replica_all5": false,
      "target_met": false
    },
    "SU05_F_BLEND": {
      "CF_D2_crossing": false,
      "cf_all5": false,
      "material_target_met": false,
      "replica_all5": false,
      "target_met": false
    },
    "SU06_F_COARSE": {
      "CF_D2_crossing": false,
      "cf_all5": false,
      "material_target_met": false,
      "replica_all5": false,
      "target_met": false
    },
    "SU07_GLOBAL_BLEND": {
      "CF_D2_crossing": true,
      "cf_all5": true,
      "material_target_met": false,
      "replica_all5": false,
      "target_met": false
    },
    "SU08_PAIRED_DELTA": {
      "CF_D2_crossing": false,
      "cf_all5": false,
      "material_target_met": false,
      "replica_all5": false,
      "target_met": false
    }
  },
  "identity": "1857a48c3907a626a3849b12ecdedb84c53914fd80025992b5cb86300ab31eb1",
  "independent_confirmation": false,
  "material_target_met": false,
  "passing_candidates": [],
  "selected": "SU01_G1",
  "status": "COMPLETE_NO_TARGET_GAIN",
  "target_met": false
}
```

Replica8 and CF18 were previously exposed. CF18 comprises 18 captures from seven physical families; this is neither untouched confirmation nor full ScanNet200 validation. No significance or monotone-accuracy claim is supported. Deployment: N0_UNCHANGED.
