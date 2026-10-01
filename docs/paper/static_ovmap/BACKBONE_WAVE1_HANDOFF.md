# Backbone wave-1 handoff

Repository: `/home/ww/crove/ovimap-backbone-wave1`
External attempt: `/mnt/shared/ww/ovimap-backbone-wave1-v1/attempt_001`
Isolated upstream: `/home/ww/crove/ovimap-backbone-wave1-upstream`
Isolated native build: `/mnt/shared/ww/ovimap-backbone-wave1-v1/tooling/native`

Apply the original module_validation_v1 patch, then third_party_patches/ovimap/backbone_wave1_v1/backbone_wave1_v1.patch to the pinned upstream. The old native extension remains intact. SAM uses its pinned source checkout and historical environment; optional connected-components CUDA extension absence uses the official loader fallback and is recorded.

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_backbone_wave1.py --phase all --spec docs/paper/static_ovmap/backbone_wave1_v1/PROTOCOL_SPEC.json --output-root /mnt/shared/ww/ovimap-backbone-wave1-v1/attempt_001 --gpu 2 --mapping-workers 2 --mapping-threads 8 --evaluation-workers 3 --resume
```

Report and publish never launch mapping/models. The coordinator, map outputs and GPU use real locks. Interrupted maps restart from frame zero with previous attempts retained. Restore external arrays/data/weights from external_artifacts.json and validate consumed receipts. Implementation and transfer-freeze commits are in execution/freeze receipts; the final publication SHA is external in publication/final.json.

Actual consumed implementation commits: `247aacc227bbeb46e5c0d28fdbf079ef4c318a09`, `26a6c4c698f8517f64e1f52c4933f58f90bd1ad7`, `2b01092f0df6540c7dc323501dfe8db5c10727b8`, `316aaf3cd4ce9512812c5e820da0a66d2949c658`, `452226757bc664b237dee9ac3e19c3cfcc03961d`, `50587252b00f96cb340300a4f43df538b454590d`, `525e62f4049786590b1ad560abd8be9c54e26ca8`, `587d5f8a13376fd023a0c8c90e576e73e69bce07`, `5db5fde1aea2e91c131bd6f9895f8a49c4394616`, `66b4a077f9d86359380a742fa184a3c7f5ab922e`, `7235679c65b38eca0733e93c064a29967fa1744f`, `9abbd926c70775fe034ada93a48a5b3af4a38a8c`, `be946323908cb3392e3cf21f2bc0cc87b64e493a`, `c45ed1ecc83faca64bdfda56341599aeb392e054`, `c94a9f94980332d05dbc21cdef8d1a0ced6475ad`.

Every measured map's raw inputs/outputs are identified by path, bytes and SHA256 in `code_model_output_provenance.json`. The unchanged measured bridge retains its original receipt and explicitly records the validation-only C6 controller alias. Per-class summaries reconstruct the original released AP from locked matches and verify exact pooled AP agreement. Secondary ranks use each new map's own native rank table.

## Production Acceptance

Primary requirement review: `LOCAL_REQUIREMENTS_VERIFIED_PUBLICATION_EXTERNAL`. [Measured review and evidence paths](../../../artifacts/static_ovmap/backbone_wave1_v1/validation/primary_review.json).

| Contract | Review status | Evidence |
| --- | --- | --- |
| C1 | VERIFIED_FULL_STUDY | `resolved_inputs.json`; `matrix.json`; `maps/*/*/map_receipt.json`; `readouts/*/*/receipt.json`; `replica_launch_correction.json`; `review/transfer_launch_regression.json`; `review/full_measurement_audit.json` |
| C2 | VERIFIED_IMPLEMENTATION_AND_DEVELOPMENT | `tests/evaluation/test_backbone_wave1_kernels.py`; `maps/scene*/BB00_NATIVE/diagnostics/*/order_diagnostic.json`; `readouts/scene*/BB01_SYNC/intervention.json` |
| C3 | VERIFIED_REDUNDANT_BOUND_CONTROL | `resolved_inputs.json`; `third_party_patches/ovimap/backbone_wave1_v1/backbone_wave1_v1.patch`; `pools/development/BB05_RATIO_GATE/*/OFFICIAL_CURRENT_CLASS.json` |
| C4 | VERIFIED_ACTUAL_NATIVE_TRACE | `prepare/native_trace/summary.json`; `prepare/native_trace/forward/receipt.json`; `prepare/native_trace/bidir/receipt.json` |
| C5 | VERIFIED_DIRECTIONAL_AND_UNMATCHED | `tests/evaluation/test_backbone_wave1_kernels.py`; `maps/scene*/BB05_FORWARD/diagnostics/*/association_plan.json`; `maps/scene*/BB05_BIDIR/diagnostics/*/association_plan.json` |
| C6 | VERIFIED_ACTUAL_MODE4_INTEGRATION | `prepare/native_trace/summary.json`; `maps/scene*/BB05_FORWARD/map_receipt.json`; `maps/scene*/BB05_BIDIR/map_receipt.json`; `third_party_patches/ovimap/backbone_wave1_v1/backbone_wave1_v1.patch` |
| C7 | VERIFIED_FULL_STUDY | `prepare/sam2_preflight/receipt.json`; `frontend/*/SAM2_PAIRED/receipt.json`; `tests/evaluation/test_backbone_wave1_frontend.py`; `review/full_measurement_audit.json` |
| C8 | VERIFIED_FULL_STUDY | `bridge_parity.json`; `readouts/*/*/native_query/native_query_receipt.json`; `readouts/*/*/fc/receipt.json`; `readouts/*/*/receipt.json`; `review/scoped_tests.json`; `review/full_measurement_audit.json` |
| C9 | VERIFIED_FULL_ORDERED_POOLS_AND_RAW_DIAGNOSTICS | `pools/*/*/*/OFFICIAL_CURRENT_CLASS.json`; `pools/*/*/*/FROZEN_N0.json`; `readouts/*/*/raw_geometry_diagnostics.json`; `readouts/*/*/evaluation_rows.json`; `review/full_measurement_audit.json` |
| C10 | VERIFIED_FROZEN_CHRONOLOGY | `candidate_freeze.json`; `selection.json`; `transfer_freeze_commit.json`; `execution/compose_1790810686395531140.json` |
| C11 | LOCAL_VALIDATION_VERIFIED_PUBLICATION_CHECK_EXTERNAL | `review/scoped_tests.json`; `prepare/native_trace/summary.json`; `bridge_parity.json`; `review/full_measurement_audit.json`; `report/render_receipt.json`; `publication/final.json (required after normal push)` |
| C12 | VERIFIED_FULL_STUDY_WITH_EXPLICIT_TIMING_LIMITS | `execution/screen.resources.json`; `execution/compose.resources.json`; `execution/transfer.resources.json`; `execution/transfer_retry_001.resources.json`; `readouts/*/*/native_query/native_query_receipt.json`; `readouts/*/*/fc/receipt.json`; `frontend/*/SAM2_PAIRED/receipt.json` |
| C13 | VERIFIED_FULL_STUDY | `frontend/*/SAM2_PAIRED/chunks/*`; `frontend/*/SAM2_PAIRED/receipt.json`; `src/static_ovmap/backbone_wave1/frontend_sam2.py`; `review/full_measurement_audit.json` |

Measured maps: 64/64; primary official rows: 192/192; bridge: `VERIFIED`. [Scope and scientific status](../../../artifacts/static_ovmap/backbone_wave1_v1/completion.json), [native trace](../../../artifacts/static_ovmap/backbone_wave1_v1/validation/native_trace.json), [scoped tests](../../../artifacts/static_ovmap/backbone_wave1_v1/validation/scoped_tests.json), [freeze chronology](../../../artifacts/static_ovmap/backbone_wave1_v1/transfer_freeze_commit.json), [physical work](../../../artifacts/static_ovmap/backbone_wave1_v1/physical_work.json), [external restoration](../../../artifacts/static_ovmap/backbone_wave1_v1/external_artifacts.json). Publication is verified separately after the normal push in the external `publication/final.json`; it is not inferred from this report or committed recursively.
