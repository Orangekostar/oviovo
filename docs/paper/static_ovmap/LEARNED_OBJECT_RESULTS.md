# Learned object readout results

TRAIN:24 families, 622 original base objects; 74 observed positive categories within160 base categories. 40 adapter-heldout categories remain in the full200-class prediction vocabulary.

DEV nomination: `LR08_MV_AUX`. Research retained: `LR01_G1`; deployment: `N0_UNCHANGED`.

Actual seed17/seed29 updates:10000/6000. Whole-map coverage:286 scene-method records,22 ordered pools.

[Table1: whole-map regression](../../../artifacts/static_ovmap/learned_object_readout_v1/tables/table1_whole_map.md) · [Table2: paired learning](../../../artifacts/static_ovmap/learned_object_readout_v1/tables/table2_learning.md) · [Table3: robustness and compute](../../../artifacts/static_ovmap/learned_object_readout_v1/tables/table3_robustness_compute.md)

[Table1 supplement: seed29](../../../artifacts/static_ovmap/learned_object_readout_v1/tables/table1_repeat_seed29.md). Parameter counts refer to trainable heads; the common frozen FC backbone is excluded.

H full200-class top1 over all four conditions: FC8 33.66%, MA8 23.27%, DEV-nominated LR08_MV_AUX 23.27%. NLL: FC8 4.4782, nominated 4.2216. Denominator: 101 original objects and 404 correlated condition records per method.

H measures GT-derived proposal recognition on new families; Replica8/CF18 are repeatedly exposed whole-map regressions. Seed29 remains separate. All trained main branches are retained, including negative results. Geometry and class-agnostic matches are unchanged; labels and current-class official area ranks can change.
