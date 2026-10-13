# Primary-agent requirement audit

Execution: **BLOCKED_DEPENDENCY**. Science: **UNASSESSED_DEPENDENCY**. The full approved prompt remains the acceptance contract.

| Prompt | Requirement | Evidence | Actual status |
| --- | --- | --- | --- |
| T0 / sections0–2 | Fixed BASE, isolated namespace, original package/config bytes | common.py / binding.py / frozen docs | CODE_REVIEWED |
| T0 / section2.2 | P0–P2 exact parents; P3 prerequisites lazy; no AnyUp substitute | binding.minimal_fc_dependencies / test05 | SYNTHETIC_DEPENDENCY_SCOPE_VERIFIED |
| T0 / section2.3 | Verify parent before path rewrite; preserve inherited parent_root | binding.preflight / consumed_manifest_identities | CODE_REVIEWED_REAL_BINDING_BLOCKED |
| T0 / section2.3 | Two fixed head SHAs/steps; FC/text bytes and class order | binding.runtime_registry | REAL_EXPERIMENT_BLOCKED |
| T0 / sections2–3 | Read-only links and exact class_split copy; explicit CLI phases | binding.py / runner.py | CODE_REVIEWED |
| T1 / section4 | Float64 fixed-scale Huber and zero-mean weighted graph KKT | solver.py / original focused tests01–02 | SYNTHETIC_SOLVER_VERIFIED |
| T1 / section4 | C1 uses all valid views; C2 per-site Huber; fixed lambda grid | solver.correction / selection.py | CODE_REVIEWED |
| T1 / section4 | Only local_raw changes; local_unit/query/quality/eta unchanged | adapter.py / parent-head focused test03 | SYNTHETIC_PARENT_HEAD_VERIFIED |
| T1 / section4 | Original nonfinite error; only new numeric correction falls back | adapter._adapt | CODE_REVIEWED |
| T1 / sections4–5 | Same-seed lambda0 and geometric fallback; no FC8 fallback | adapter.py / prediction.parity | REAL_FC_PARITY_BLOCKED |
| T1 / section4 | Relative gauge, balanced-pool cancellation, shuffle degeneracy | solver.py / test02 / CLAIMS | SYNTHETIC_BOUNDARY_VERIFIED |
| T1 / section5 | 2 TRAIN ×2 seeds ×4 conditions ×2/4/8 real FC checks | prediction.parity / parity.json | REAL_EXPERIMENT_BLOCKED |
| T1 / section5 | 69 base DEV keyed historical labels/AMC and embedding tolerance | prediction.compare_history | REAL_EXPERIMENT_BLOCKED |
| T2 / section6.1 | 622/69 base objects,24/4 families, four condition-specific positions | binding.py / diagnostics.spatial_folds / test05 | REAL_EXPERIMENT_BLOCKED |
| T2 / section6.1 | Four spatial folds and actual bilinear-donor global purge | diagnostics.py / test05 | SYNTHETIC_ISOLATION_VERIFIED_REAL_DIAGNOSIS_BLOCKED |
| T2 / section6.1 | Missing diagnostic metadata unsupported; no fabricated positions | diagnostics.diagnostic_metadata / data.ObjectLoader / test05 | SYNTHETIC_PARENT_COMPATIBILITY_VERIFIED |
| T2 / section6.2 | Stable five null seeds; degenerate null separately labeled | diagnostics.independent_bias | CODE_REVIEWED_REAL_DIAGNOSIS_BLOCKED |
| T2 / section6.2 | Sum N/D before R; equal families; paired common support | diagnostics.aggregate / diagnostic_units compact archive | REAL_EXPERIMENT_BLOCKED |
| T2 / section6.2 | Signal coverage floors; signal never gates P2/P3 | diagnostics.run / selection.activate | REAL_EXPERIMENT_BLOCKED |
| T3 / section6.3 | Full200 top1/NLL; inherited base-class CE at0.07; exact AMC | prediction.recognition / inherited summarize | REAL_EXPERIMENT_BLOCKED |
| T3 / section6.3 | All12 seed17 candidates; AMC/lambda tie rule; lock C_STAR | selection.lock_selection / test06 | REAL_EXPERIMENT_BLOCKED |
| T4 / section6.4 | Seed29 locked repeats only; no champion replacement | prediction.repeat / selection.activate / test06 | REAL_EXPERIMENT_BLOCKED |
| T4 / section6.4 | Exact pass thresholds and three activation branches | selection.passes/activate / test06 | REAL_EXPERIMENT_BLOCKED |
| T4 / section6.5 | Both seeds, four conditions,2/4 prefixes; lambda0 aliases | prediction.prefix / test05/07 | REAL_EXPERIMENT_BLOCKED |
| T4 / section6.5 | Group raw/phi/embedding/attention effect and harm audits | adapter.py / prediction.recognition | CODE_REVIEWED_REAL_READOUT_BLOCKED |
| T5 / section7.1 | All26 scenes and all expanded arms locked before GT | prediction.predict / evaluation.run | REAL_MAP_NOT_REACHED |
| T5 / section7.2 | Fixed G1 owner/support; replace F only; retain N/Q/temperature/D2 | prediction.predict / unchanged build_payload | CODE_REVIEWED_REAL_MAP_BLOCKED |
| T5 / section7.3 | Official ordered pools, null class, full200 base/novel, Replica N/A | evaluation.py / compact per-class data | CODE_REVIEWED_REAL_MAP_BLOCKED |
| T5 / section7.3 | Identity-proven aliases; exactly2 additional active-run scorer checks | evaluation.evaluate_scene / dynamic coverage | CODE_REVIEWED_REAL_MAP_BLOCKED |
| T5 / section7.4 | Four-cell AP/mIoU map gate and M1 vs locked C_STAR | evaluation.run / science_status | CODE_REVIEWED_REAL_MAP_BLOCKED |
| T6 / section8 | Exactly8 focused groups; real FC dependency skip is explicit | tests/test_surface_bias_wave1_v2.py / tests.json | FOCUSED_TESTS_RECORDED_REAL_FC_SKIP |
| T6 / section8 | Interrupted readout resumes without repeating completed head | common.cached_unit / actual readout-loop test07 | SYNTHETIC_INTERRUPTION_VERIFIED |
| T6 / section8 | Complete files verified; superseded identities/results retained | common.preserve_unit / test07 | SYNTHETIC_RECOVERY_VERIFIED |
| T6 / sections3,8 | One queued GPU; FP32;32GiB host budget;3 CPU scorers; space reserve | runner.py / PrefixLoader / common.py / binding.py | CODE_REVIEWED_REAL_RESOURCE_MEASUREMENT_BLOCKED |
| T7 / section9.1 | Five tables in MD/CSV/JSON; every finite candidate retained | reporting.table / actual artifact files | GENERATED_FROM_SEALED_JSON |
| T7 / section9.2 | Frozen artifact config; compact readouts, N/D, map/match/per-class data | reporting.compact_results / compact_results_index.json | NO_NEW_SCIENTIFIC_OUTPUTS |
| T7 / section9.2 | 40/150MiB finite task paths; no RGB/GT/grids/base weights | publication.check_files / artifact_index.json | CHECKED_BEFORE_ACTUAL_C1 |
| T7 / section9.3 | Normal C1/C2 push, prior-release retention, unchanged resume reuse | publication.py / test08 / committed receipt | ACTUAL_NEW_C1_PENDING_AT_REPORT_SNAPSHOT |
| T7 / section9.4 | Three status axes; real worker exits; external observed all exit | runner.py / run_summary.json / stage_exits.json | OBSERVED_SCIENTIFIC_FAILURE_RETAINED |
| T7 / section10 | No unsupported mechanism/map/instance/onlineFPS/OVI-MAP claim | CLAIMS.md / RESULTS.md | EVIDENCE_BOUNDARIES_REVIEWED |

The targeted repair preserves the original published release and the frozen scientific spec. Diagnostic-only field compatibility calls the original parent kernels; it does not invent UV, grids or positions. Missing forward observations still require the exact original tensors or caches. Synthetic tests verify engineering behavior and cannot establish FC parity or new scientific gains.

No new science phase is marked complete. Actual publication is proven by the committed C1 receipt and external C2 final receipt; this C1 report never predeclares its future push.
