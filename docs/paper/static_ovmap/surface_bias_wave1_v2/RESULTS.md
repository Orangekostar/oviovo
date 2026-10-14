# FG2 surface bias wave1 v2 results

Execution: **COMPLETE**. Science: **NO_HELDOUT_SIGNAL**. Deployment: N0_UNCHANGED.

See formal phase records

New optimizer updates / RGB encodings / SAM proposals: **0 / 0 / 0**. Missing experiment metrics remain null. Existing FG2 results are separately identified as inherited published context; they do not establish a gain for this new calibration. Neither offline cached replay nor these blocked phases establish online 30 FPS or an efficiency advantage over OVI-MAP.

- [Table 1](../../../../artifacts/static_ovmap/surface_bias_wave1_v2/tables/table1_baseline_checks.md)
- [Table 2](../../../../artifacts/static_ovmap/surface_bias_wave1_v2/tables/table2_diagnostics.md)
- [Table 3](../../../../artifacts/static_ovmap/surface_bias_wave1_v2/tables/table3_selection_repeat.md)
- [Table 4](../../../../artifacts/static_ovmap/surface_bias_wave1_v2/tables/table4_whole_map.md)
- [Table 5](../../../../artifacts/static_ovmap/surface_bias_wave1_v2/tables/table5_cost_scope.md)

All requested lambda candidates and the fixed-repeat status are retained. Seed29 and map labels never select lambda. CF18 contains18 captures from7 physical families and is not the complete ScanNet200 validation set. The parent proposals were derived from GT instances, so label-free bias fitting does not make the entire pipeline GT-free or training-free.

The gauge identifies relative offsets only. Equal view pooling with complete visibility can cancel the correction exactly; phi cannot manufacture a change. Independent-support nulls can degenerate. Spatial folds with bilinear-donor purge are not independent datasets because receptive fields may overlap.
