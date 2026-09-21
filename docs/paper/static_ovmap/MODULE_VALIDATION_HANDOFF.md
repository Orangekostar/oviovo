# OVI-MAP Module Validation Handoff

- Branch: `research/ovimap-module-validation-v1`
- Commit: `b8a5917003631408cc0a11e6ed3c4e78e3f26f6f`
- Implementation: `COMPLETE`
- Experiment: `BLOCKED_INDEPENDENT_SCENES`
- Science: `INCONCLUSIVE_PREREQUISITES`
- Confirmation: `NOT_REQUIRED_NO_RETAINED_CANDIDATE`
- Publication: `PENDING_FINAL_COMMIT_AND_PUSH`

Evidence:

- `/mnt/shared/ww/ovimap-module-validation-v1/attempt_001/capture_summary.json`
- `/mnt/shared/ww/ovimap-module-validation-v1/attempt_001/semantic_summary.json`
- `/mnt/shared/ww/ovimap-module-validation-v1/attempt_001/geometry_summary.json`
- `/mnt/shared/ww/ovimap-module-validation-v1/attempt_001/query_summary.json`
- `/mnt/shared/ww/ovimap-module-validation-v1/attempt_001/selection.json`
- `/mnt/shared/ww/ovimap-module-validation-v1/attempt_001/method_matrix.json`

Executed scope:

- These are implementation-boundary smokes on historical Room0, not SELECT measurements.
- Semantic adapters: native SigLIP crop vectors [9, 1024]; legacy six-crop max error 8.120649263448909e-08.
- Geometry smoke: 12288 surface rows, 4262 native leaves, 8 complete bounded hypotheses.
- Query smoke: 35 current-state candidates with exact request/mask/bbox parity and 31 native combine selections with exact request IDs; Q_AREA logical cost {'attempts': 18, 'crop_inputs': 108, 'failures': 0, 'successes': 18}; Q_UNCERTAINTY logical cost {'attempts': 18, 'crop_inputs': 108, 'failures': 0, 'successes': 18}; physical cost {'cache_hits': 36, 'crop_inputs': 0, 'inference_seconds': 0.0, 'model_forwards': 0, 'model_loads': 0, 'tiles': 0}.
- Worktree: /home/ww/crove/ovimap-module-validation.
- Pinned OVI source: /home/ww/crove/ovimap-module-validation-upstream at f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424.
- Loaded native extension: /mnt/shared/ww/ovimap-module-validation-v1/tooling/native_patch_build/devel/lib/consistent_gsm.cpython-311-x86_64-linux-gnu.so (sha256 ffc733804859fbbf5d4afebcc140056e95dffa4a9c0a1f6f6fd78f61bca3b297).
- Historical capture: /mnt/shared/ww/ovimap-module-validation-v1/tooling/historical_two_frame_capture_v3/room0/manifest.json; independent ScanNet root: unbound.
- Bound model roots: /home/ww/vv/paper2/model_cache/google_siglip-large-patch16-384, /mnt/shared/ww/ovimap-module-validation-v1/models/siglip2-large-patch16-384, /mnt/shared/ww/ovimap-module-validation-v1/models/WOW-Seg, /mnt/shared/ww/ovimap-module-validation-v1/models/all-MiniLM-L6-v2.

Reproduction commands:

- `conda run -n ovimap-map python scripts/evaluation/run_ovimap_module_study.py --spec docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json --resolved-config /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/resolved_config.json --output-root /mnt/shared/ww/ovimap-module-validation-v1 --phase bind`
- `conda run -n ovimap-map python scripts/evaluation/run_ovimap_module_study.py --spec docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json --resolved-config /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/resolved_config.json --output-root /mnt/shared/ww/ovimap-module-validation-v1 --phase capture`
- `conda run -n ovimap-map python scripts/evaluation/run_ovimap_module_study.py --spec docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json --resolved-config /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/resolved_config.json --output-root /mnt/shared/ww/ovimap-module-validation-v1 --phase semantic`
- `conda run -n ovimap-map python scripts/evaluation/run_ovimap_module_study.py --spec docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json --resolved-config /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/resolved_config.json --output-root /mnt/shared/ww/ovimap-module-validation-v1 --phase geometry`
- `conda run -n ovimap-map python scripts/evaluation/run_ovimap_module_study.py --spec docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json --resolved-config /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/resolved_config.json --output-root /mnt/shared/ww/ovimap-module-validation-v1 --phase query`
- `conda run -n ovimap-map python scripts/evaluation/run_ovimap_module_study.py --spec docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json --resolved-config /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/resolved_config.json --output-root /mnt/shared/ww/ovimap-module-validation-v1 --phase select`
- `conda run -n ovimap-map python scripts/evaluation/run_ovimap_module_study.py --spec docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json --resolved-config /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/resolved_config.json --output-root /mnt/shared/ww/ovimap-module-validation-v1 --phase confirm`
- `conda run -n ovimap-map python scripts/evaluation/run_ovimap_module_study.py --spec docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json --resolved-config /mnt/shared/ww/ovimap-module-validation-v1/attempt_001/resolved_config.json --output-root /mnt/shared/ww/ovimap-module-validation-v1 --phase report`

Next evidence-based action: Provide at least 14 eligible independent scene families, including the required ScanNet captures, then resume the locked FIT/CAL/SELECT/CONFIRM phases.
