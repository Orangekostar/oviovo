# Exact exposed-cohort selection

Selected `IR01_G1`; TARGET_MET=False; MATERIAL_TARGET_MET=False.

All five metrics must not decrease versus G1 in both cohorts; CF18 APall must strictly exceed D2 and CF18 AP50 must be at least D2. Epsilon=1e-10; the separate material flag is 0.10 percentage points. No gates were relaxed.

- `IR02_NEAREST_ATTACH`: FAIL; failed checks: replica8.apall.nondecrease, replica8.ap50.nondecrease, replica8.miou.nondecrease, replica8.macc.nondecrease, scannet_cf18.ap25.nondecrease, scannet_cf18.macc.nondecrease, CF18_APall_strictly_above_D2, CF18_AP50_at_least_D2
- `IR03_EVIDENCE_ATTACH`: FAIL; failed checks: CF18_APall_strictly_above_D2, CF18_AP50_at_least_D2
- `IR04_DIRECT_GROUP`: FAIL; failed checks: CF18_APall_strictly_above_D2, CF18_AP50_at_least_D2
- `IR05_VERIFIED_REPAIR`: FAIL; failed checks: CF18_APall_strictly_above_D2, CF18_AP50_at_least_D2
- `IR06_ANYUP_REREAD`: FAIL; failed checks: replica8.apall.nondecrease, replica8.ap50.nondecrease, replica8.ap25.nondecrease, replica8.miou.nondecrease, replica8.macc.nondecrease
- `IR07_BOUNDARY_STABLE`: FAIL; failed checks: replica8.apall.nondecrease, replica8.ap50.nondecrease, replica8.ap25.nondecrease, replica8.miou.nondecrease, replica8.macc.nondecrease
- `IR08_COMBINATION`: FAIL; failed checks: replica8.apall.nondecrease, replica8.ap50.nondecrease, replica8.ap25.nondecrease, replica8.miou.nondecrease, replica8.macc.nondecrease

Deployment remains N0_UNCHANGED.
