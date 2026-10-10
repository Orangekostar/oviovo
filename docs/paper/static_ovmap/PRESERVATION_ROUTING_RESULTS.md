# Preservation routing results

Status: **COMPLETE_NO_2D_FOUNDATION**. Deployment: `N0_UNCHANGED`.

Actual scientific updates: 8000; discarded engineering updates:20. TRAIN:622 base-positive originals /74 positive categories; DEV:69 base originals /4 families. Exact parent24/4 split and160/40 class IDs retained.

FC8 DEV A/M/C (%): 37.29/50.61/42.06.

| Arm | Selected step | A (%) | M (%) | C (%) | Net corrections |
| --- | --- | --- | --- | --- | --- |
| PR_R0_DIRECT | 500 | 31.35 | 43.69 | 31.50 | -21 |
| PR_R1_RESIDUAL | 1000 | 36.38 | 49.37 | 41.43 | 0 |
| PR_R2_KEEP | 1500 | 33.48 | 45.61 | 34.28 | -19 |
| PR_R3_RESIDUAL_KEEP | 500 | 36.17 | 50.96 | 36.78 | -7 |

Rstar: `None`. Checkpoints0 are diagnostics and never nominees. Four R arms have matching fresh MA initialization and32000 original-item draws each. Teacher correctness uses TRAIN clean full200 FC of the same prefix, and never enters inference.

[Table1: full maps](../../../artifacts/static_ovmap/preservation_routing_v1/tables/table1_whole_map.md) · [Seed29 map repeat](../../../artifacts/static_ovmap/preservation_routing_v1/tables/table1_whole_map_repeat.md) · [Table2: learning](../../../artifacts/static_ovmap/preservation_routing_v1/tables/table2_learning.md) · [Direct pairs](../../../artifacts/static_ovmap/preservation_routing_v1/tables/table2_direct_pairs.md) · [Table3: robustness/compute](../../../artifacts/static_ovmap/preservation_routing_v1/tables/table3_robustness_compute.md). JSON retains numeric types/full precision; displayed accuracies use percentages.

The fixed foundation/repeat gate failed. G training, new whole-map expansion, H2 acquisition and SAM2 proposal generation were NOT_TRIGGERED. Null table cells are unmeasured, never zero accuracy. Selected diagnostic R heads, component curves and exposed oldH regression are retained.

OldH is exposed regression after nomination. TRAIN/DEV/H proposal conditions are correlated versions of original objects. TRAIN/DEV use registered depth-grid RGB; map inputs retain their inherited preprocessing domain. No independent map/geometry gain is inferred from proposal recognition.

H2: NOT_TRIGGERED_NO_FOUNDATION.

[Compact Table3: DEV corruptions](../../../artifacts/static_ovmap/preservation_routing_v1/tables/table3_DEV_robustness.md) uses all87 post-lock DEV objects with full200 competition, including18 heldout-class objects; these micro accuracies never select checkpoints. [Training compute](../../../artifacts/static_ovmap/preservation_routing_v1/tables/table3_training_compute.md) records optimizer work plus five DEV checks per arm. Detailed strata, no-op denominators and prior/latest nested stage clocks remain in the full Table3 supplement.

Scalar component curves and effective denominators are published under training/. Parent summed-loss/component availability is documented in parent_audit/receipt.json; unavailable historical CE/consistency values remain null.

[Parent TRAIN/DEV audit](../../../artifacts/static_ovmap/preservation_routing_v1/tables/table2_parent_audit.md) separates the retained historical heads from this fresh experiment. [Stored parameter counts](../../../artifacts/static_ovmap/preservation_routing_v1/tables/table3_parameters.md) separate trained heads, frozen initial MA/base and the external FC backbone.

Runtime is recorded, has no qualification gate, and is not an online30FPS claim. Training clocks include DEV validation; conditional cached-feature inference and output construction have separate boundaries. Residual inference requires the explicitly exported frozen initial MA in addition to trainable MA and frozen FC.
