# Native query-membership repair

The v9 capture accessor used `getInstanceLabel(segment, 0.1f)`, matching final
surface export. The unmodified native raycaster instead uses the overload whose
threshold is `0.0f`. On `scene0534_00`, frame 1064 (scheduled index 133), request
owner 140 was positive in the current raster but absent from the captured
export-threshold membership. Those captures cannot support complete causal Q
ancestry, even where the mismatch does not immediately raise an exception.

The native build option `--query-membership` adds a separate read-only
`raycast_instance_label` for every known segment. It preserves the original
export-threshold field, mapper, raycaster, native semantic readout and evaluator.
Request provenance and Q histories use the raycaster field captured in the same
integration transaction. Final surface reconciliation still uses the native
export owners, after the last acquisition barrier. A capture with the old 0.1
threshold and no raycaster field is rejected for causal Q.

An attempted CPU supplement was rejected: despite identical pre-insertion
inputs, native replay allocated different segment IDs, and a persistent ordered
bijection failed by frame 80. Its files and diagnostics remain external under
`tooling/repairs-20260922/rejected-membership-replay` and
`scannet_study_v1/query/membership`. No supplement is used in scientific rows.

All affected development native captures and dependent scene artifacts are
preserved under explicit `historical_native_v9` / `scenes_native_v9` directories.
The corrected full study reuses the locked raw data, all 12 CropFormer caches,
annotations and text embeddings. S/G/Q are regenerated against one common
corrected native capture per scene; results from different capture versions are
not mixed. CONFIRM remains unopened until selection is frozen.

Validation: the targeted capture/Q tests cover a current positive raycast owner
whose export owner is zero, current-only reads, complete ancestry, paid-feature
merge/split rules, unchanged final geometry/ranks and confirmation restrictions.
The compiled accessor also passed the two-frame native build fixture. Full
development reruns remain required before scientific completion.
