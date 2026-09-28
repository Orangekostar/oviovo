# Released-evaluator diagnosis: scene0056_00 M2 calibration

Observed from the completed `attempt_001/rows/compose_cal/scene0056_00/CP_M2_EQUAL_{RAW,CAL}.json` records and their exact `trace_path` / `matches_path` references. Both released traces report exact AP/PR parity. This comparison does not alter prediction or evaluator settings.

RAW and CAL both have uAP 0.016282711446401924 and AP50 0.08265605921855922. Their AP25 values are 0.2657399644006787 and 0.28359710725782156 respectively. Their mIoU values are 0.18902698718521946 and 0.1920717032609879. The two predictions differ on four owners (their separate changes versus N0 are 26 and 28, which must not be subtracted to infer their pairwise changes).

| Owner | RAW → CAL official class | Projected vertices | Actual released event at IoU 0.5 in both variants |
|---|---|---:|---|
| 39 | 52 whiteboard → 79 paper | 28,372 | `ignore_test`, counted FP; ignore fraction 0.224376145495559 |
| 107 | 118 speaker → 64 computer tower | 377 | `ignore_test`, not counted FP; ignore fraction 0.713527851458886 |
| 113 | 141 windowsill → 242 power outlet | 4,449 | `ignore_test`, counted FP; ignore fraction 0.004495392222971455 |
| 146 | 96 radiator → 408 vent | 719 | `ignore_test`, counted FP; ignore fraction 0.18636995827538247 |

All four owners are present in both released prediction-match registries; none is removed by the small-region filter. Their masks and serialized confidences remain fixed. No affected owner emits a first-match or duplicate event at IoU 0.5; the actual ignore-test outcomes above distinguish ignored predictions from counted false positives.

Comparing every class/threshold AP state shows exactly one changed entry: radiator at IoU 0.25 changes from 0.75 to 1.0. All AP entries at the uAP thresholds remain unchanged. At IoU 0.5, the affected classes with GT (whiteboard, windowsill, radiator) have AP 0 in both variants; paper, computer tower, speaker, power outlet and vent have no evaluable GT and AP is null in both. Their PR/input arrays may change without a changed AP aggregate. Thus equal uAP does not imply equal predictions or no intervention; AP25 and semantic metrics reveal different behavior.

The radiator AP25 change is a ranking-list change, not a new GT match: owner 134 matches GT 96000 at IoU 0.32116596638655465 with confidence 1.0 in both variants. RAW also has owner 146 as an unmatched FP with the same confidence 1.0 (`y_true=[1,0]`, `y_score=[1,1]`); CAL moves that FP to the vent class, leaving radiator `y_true=[1]`, `y_score=[1]`. The released PR states yield 0.75 and 1.0 respectively. There is no radiator duplicate-match event at this threshold. This is why an unchanged match count can still change AP25.

Reproduction uses the two immutable row records, `load_prediction` plus `owner_labels` to find pairwise label differences, the saved `prediction/owner_XXXXXX.npy` masks for projected sizes, and the actual released traces' `events` and `states` keyed by `(class_label, overlap_threshold)`. No evaluator rerun or new model inference is needed.

## Additional actual unchanged-uAP contrasts

These checks use the corresponding `rows/<role>/<scene>/<method>.json` trace and match references with the same reconstruction/trace procedure.

| Scene / comparison | Pairwise changed owners | Actual explanation |
|---|---:|---|
| scene0445_00, N0 → Q_COMBINE | 5: 4, 13, 18, 37, 40 | Every class/threshold AP entry is unchanged. At IoU 0.5 the changed predictions move from 4 counted FP + 1 ignored to 3 counted FP + 1 ignored; owner 40 has 128 projected vertices but exits evaluation because its new label is 0, not because of the small-region filter. No changed-owner first match or duplicate event is emitted at this threshold. |
| scene0445_00, Q_GAIN → M4 | 4: 18, 20, 26, 40 | All four remain eligible. Both variants have 3 counted FP + 1 ignored among these owners at IoU 0.5. Every class/threshold AP entry is unchanged, despite class reassignment. |
| scene0445_00, Q_GAIN → M5 | 2: 3, 40 | Both remain eligible and counted FP at IoU 0.5. Only wall AP25 changes, from 0.2 to 0; every uAP-threshold AP entry is unchanged. |
| scene0534_00, M3 → M4 | 53 | Changed-owner eligibility rises from 13 to 42. At IoU 0.5 the changed-owner events move from 9 counted FP + 4 ignored to 31 counted FP + 10 ignored + 1 first match. The remaining 11 excluded M4 owners each have fewer than 100 projected vertices. Equal uAP here reflects cancellation of class AP changes, not unchanged AP entries. |

For the last contrast, shelf AP decreases from 1/3 to 1/4 at each of IoU 0.50, 0.55, 0.60, 0.65 and 0.70; recycling-bin AP decreases from 1 to 3/4 at 0.50; water-cooler AP increases from 0 to 2/3 at 0.50. These are all changed AP entries in the uAP threshold range, and their sum is exactly `5*(-1/12) - 1/4 + 2/3 = 0`. Additional AP25 changes occur for shelf, backpack, recycling bin, bulletin board and water cooler. These observations come from released per-class PR/AP states; they must not be interpreted as additive object gains or generalization evidence.

## Remaining observed predeclared contrasts

The following completes the audit of equal-uAP, different-prediction cases among the predeclared trajectory/calibration contrasts and comparisons against N0. All 50 rows are complete; the second confirmation scene adds no equal-uAP case against N0, and no additional composition was authorized there. “Small” means fewer than 100 projected vertices, as checked against the saved masks; the counts below concern changed owners only.

| Scene / comparison | Changed owners | Eligibility and actual IoU-0.5 events | Full class/threshold AP comparison |
|---|---:|---|---|
| scene0534_00, M2_RAW → M2_CAL | 7 | Both: 5 eligible (2 counted FP, 3 ignored), 2 small | Only bulletin-board AP25 changes, 0.5 → 0.25 |
| scene0534_00, Q_COMBINE → M3 | 17 | Both: 14 eligible (10 counted FP, 4 ignored), 3 small | Only AP25 changes: floor 1 → 0.75; bulletin board 0.25 → 0.1875 |
| scene0534_00, Q_COMBINE → M5 | 30 | Eligible 7 → 29; counted FP 5 → 21; ignored 2 → 8. Before: 22 zero-label owners plus 1 both zero-label/small. After: 1 small owner | Only AP25 changes: paper 0 → 0.05; bulletin board 0.25 → 1/3 |
| scene0534_00, N0 → M1 | 9 | Both: 9 eligible, all counted FP | Only AP25 changes: floor 1 → 0.75; backpack 1 → 0; paper 1/26 → 0; bulletin board 1/16 → 1/3 |
| scene0553_00, N0 → Q_GAIN | 9 | Both: 8 eligible (4 counted FP, 4 ignored), 1 small | Every AP entry unchanged |
| scene0553_00, N0 → Q_COMBINE | 11 | Both: 5 eligible (3 counted FP, 2 ignored), 6 small. Four small owners additionally become label 0 | Every AP entry unchanged |
| scene0626_00, Q_GAIN → M4 | 13 | Both: 8 eligible, all counted FP; 5 small | Only floor AP25 changes, 1 → 0.75 |

No changed-owner first-match or duplicate event occurs at IoU 0.5 in these seven cases. Each AP-state comparison covers all recorded classes and thresholds, rather than inferring equality from the aggregate uAP. The original released traces retain the complete score/ranking arrays and event details for inspection.
