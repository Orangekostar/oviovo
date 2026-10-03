# Compact Tables Selection

Fixed primary: `CT_A3_ER`. Deployment: `N0_UNCHANGED`. No retrospective method reselection, parameter sweep or automatic deployment.

| Cohort | APall > A1 | AP50 drop <= 0.10pp | mIoU drop <= 0.10pp | A3 dominated by |
| --- | --- | --- | --- | --- |
| replica8 | -- | -- | -- | -- |
| scannet_cf18 | False | False | True | none |

| Cohort | Metric Pareto frontier | Undefined Pareto input |
| --- | --- | --- |
| replica8 | CT_H_U2 | CT_A2_R, CT_A3_ER, CT_A5_FC_ONLY, CT_G3 |
| scannet_cf18 | CT_A1_E, CT_A3_ER, CT_A5_FC_ONLY | none |

Pareto comparison includes every fixed method in each cohort (eight for Replica, six for CF18), using available APall/AP50/mIoU; missing methods are listed, and the complete fixed-method frontier cannot be established when any input is unavailable. An unavailable A3 comparison is --, not evidence that A3 has no dominator. Any A3 dominator is reported without suppression. A3 versus A4 tests representation under the same G1 frame/mask; A3 versus A5 tests incumbent refinement using identical F and G1 evidence. A5's FC-only readout retains Native-derived support and requires that prerequisite in standalone costs.
