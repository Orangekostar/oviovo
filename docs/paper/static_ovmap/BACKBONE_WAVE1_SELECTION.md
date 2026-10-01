# Backbone wave-1 selection

Frozen nominee: `BB01_SYNC`. Transfer map IDs: BB00_NATIVE, BB01_SYNC, BB03_SAM2_RAW, BBX_COMPOSE.

Selection uses D2 four-scene official development pooling only. APall band .05pp, mIoU band .1pp, AP50 band .1pp, then required standalone image encodings, median attributable time, changed block count and method ID. The structural champion is selected from BB01_SYNC/BB05_FORWARD/BB05_BIDIR: BB05_RATIO_GATE remains a mandatory measured simple control and is excluded by the normative noncontrol-champion rule. Strict metric ranking, banded ranking and every tie step are frozen in candidate_freeze.json/selection.json before composition/Replica. Individual positive gain is not required for composition; both families require complete inputs and actual raw partition intervention. Replica results do not refit weights, temperatures or the Q model.

## Frozen Candidates and Composition

Structural champion: `BB01_SYNC`. Frontend champion: `BB03_SAM2_RAW`.
Only composition: `BBX_COMPOSE` with `BB01_SYNC` and `BB03_SAM2_RAW`.
Candidate identity: `8aaa398d2edb6269c53fbf37b2045148d2b98dcc6b755b56ef6ee97e811fc10d`. Transfer selection identity: `0226eb45b0cbbf262773c68080da9e31cb27cc0a8d23ccd523fa2ca980e30bb0`.
Recorded pre-Replica freeze commit: `c94a9f94980332d05dbc21cdef8d1a0ced6475ad`.

## Frozen Development Inputs

| Map | APall (%) | mIoU (%) | AP50 (%) | Required image encodings | Median attributable seconds | Changed blocks | Raw intervention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BB00_NATIVE | 12.581784 | 23.562289 | 22.599433 | 18093 | 964.073825 | 0 | False |
| BB01_SYNC | 13.207886 | 25.178977 | 25.456576 | 18141 | 967.157524 | 1 | True |
| BB03_SAM2_GEOM | 9.813102 | 22.640438 | 20.722676 | 18864 | 1047.259538 | 1 | True |
| BB03_SAM2_RAW | 9.798677 | 22.654527 | 20.709921 | 18798 | 1072.016917 | 1 | True |
| BB05_BIDIR | 6.093536 | 22.973349 | 13.906515 | 82443 | 1869.949477 | 1 | True |
| BB05_FORWARD | 8.346233 | 22.134066 | 15.714172 | 28051 | 1219.364222 | 1 | True |
| BB05_RATIO_GATE | 12.583195 | 23.582972 | 22.691497 | 18093 | 960.712046 | 1 | True |
| BBX_COMPOSE | 9.761640 | 22.838552 | 20.823016 | 18828 | 1060.339183 | 2 | True |

These are frozen four-scene D2 pools and standalone obligations. Attributable time is partially measured stage attribution, not a cold end-to-end benchmark. Both temporal arms include the full shared SAM prerequisite.

## Structural Ranking Trace

Strict metric order: BB01_SYNC > BB05_FORWARD > BB05_BIDIR.
Frozen banded preference: BB01_SYNC > BB05_FORWARD > BB05_BIDIR.

| Step | Selected | APall band | mIoU band | AP50 band | Cost tie order |
| ---: | --- | --- | --- | --- | --- |
| 1 | BB01_SYNC | max 13.207886%, width 0.050000pp: BB01_SYNC | max 25.178977%, width 0.100000pp: BB01_SYNC | max 25.456576%, width 0.100000pp: BB01_SYNC | BB01_SYNC |
| 2 | BB05_FORWARD | max 8.346233%, width 0.050000pp: BB05_FORWARD | max 22.134066%, width 0.100000pp: BB05_FORWARD | max 15.714172%, width 0.100000pp: BB05_FORWARD | BB05_FORWARD |
| 3 | BB05_BIDIR | max 6.093536%, width 0.050000pp: BB05_BIDIR | max 22.973349%, width 0.100000pp: BB05_BIDIR | max 13.906515%, width 0.100000pp: BB05_BIDIR | BB05_BIDIR |

## Frontend Ranking Trace

Strict metric order: BB03_SAM2_GEOM > BB03_SAM2_RAW.
Frozen banded preference: BB03_SAM2_RAW > BB03_SAM2_GEOM.

| Step | Selected | APall band | mIoU band | AP50 band | Cost tie order |
| ---: | --- | --- | --- | --- | --- |
| 1 | BB03_SAM2_RAW | max 9.813102%, width 0.050000pp: BB03_SAM2_RAW, BB03_SAM2_GEOM | max 22.654527%, width 0.100000pp: BB03_SAM2_RAW, BB03_SAM2_GEOM | max 20.722676%, width 0.100000pp: BB03_SAM2_RAW, BB03_SAM2_GEOM | BB03_SAM2_RAW > BB03_SAM2_GEOM |
| 2 | BB03_SAM2_GEOM | max 9.813102%, width 0.050000pp: BB03_SAM2_GEOM | max 22.640438%, width 0.100000pp: BB03_SAM2_GEOM | max 20.722676%, width 0.100000pp: BB03_SAM2_GEOM | BB03_SAM2_GEOM |

## Nominee Ranking Trace

Strict metric order: BB01_SYNC > BB00_NATIVE > BB03_SAM2_RAW > BBX_COMPOSE.
Frozen banded preference: BB01_SYNC > BB00_NATIVE > BBX_COMPOSE > BB03_SAM2_RAW.

| Step | Selected | APall band | mIoU band | AP50 band | Cost tie order |
| ---: | --- | --- | --- | --- | --- |
| 1 | BB01_SYNC | max 13.207886%, width 0.050000pp: BB01_SYNC | max 25.178977%, width 0.100000pp: BB01_SYNC | max 25.456576%, width 0.100000pp: BB01_SYNC | BB01_SYNC |
| 2 | BB00_NATIVE | max 12.581784%, width 0.050000pp: BB00_NATIVE | max 23.562289%, width 0.100000pp: BB00_NATIVE | max 22.599433%, width 0.100000pp: BB00_NATIVE | BB00_NATIVE |
| 3 | BBX_COMPOSE | max 9.798677%, width 0.050000pp: BB03_SAM2_RAW, BBX_COMPOSE | max 22.838552%, width 0.100000pp: BBX_COMPOSE | max 20.823016%, width 0.100000pp: BBX_COMPOSE | BBX_COMPOSE |
| 4 | BB03_SAM2_RAW | max 9.798677%, width 0.050000pp: BB03_SAM2_RAW | max 22.654527%, width 0.100000pp: BB03_SAM2_RAW | max 20.709921%, width 0.100000pp: BB03_SAM2_RAW | BB03_SAM2_RAW |

Every metric filter and cost tie order above comes from the immutable freezes. Cost ties use required image encodings before median attributable time, changed blocks and method ID. Exact recipes, pool identities, readout identities and raw-intervention evidence are preserved in [candidate_freeze.json](../../../artifacts/static_ovmap/backbone_wave1_v1/candidate_freeze.json) and [selection.json](../../../artifacts/static_ovmap/backbone_wave1_v1/selection.json). Replica does not alter these nominations.
