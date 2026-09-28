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
