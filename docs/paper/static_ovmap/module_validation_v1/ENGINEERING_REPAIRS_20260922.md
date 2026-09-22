# Module-validation engineering repairs — 2026-09-22

This correction supersedes the previous `implementation: COMPLETE` claim. The previous release implemented module primitives and retained historical smoke receipts, but its public phases did not execute those checks and did not connect the independent-scene scientific pipeline. Scientific results remain inconclusive. Missing ScanNet data and an unimplemented development driver are separate limitations.

## Corrected behavior

- Locked prediction buffers cannot be made writable; locking revalidates edited predictions. Geometry partitions must preserve all native foreground support and background.
- Evaluation reuse binds GT arrays/files and evaluator context, supports the actual released module namespace, and ignores unmatched projection sentinels before indexing.
- CAL/SELECT comparisons use the frozen `1e-10` metric tolerance before cost/complexity tie-breaks. Undefined final metrics do not become a negative scientific result.
- Phase reuse verifies output content, external smoke inputs, code identity, and upstream receipts. Exceptions produce failure receipts while independent reporting continues. Attempt allocation and execution are locked separately.
- Historical capture/S/G/Q boundaries now execute through a checked-in CLI. Native C++ compilation, extension import, and the two-frame mapper replay have a reproducible script. Existing studies and baseline installations remain intact.
- Reports stay inside their attempt directory. Costs, timings, errors, and evidence derive from actual receipts. Export happens after the final report receipt and progress are written, into an explicitly selected new directory.
- Missing scientific prerequisites leave selection unfrozen and confirmation blocked. They no longer imply a measured KEEP-all result or a completed confirmation gate.

## Reproduction

Use the existing ABI-compatible `ovimap-map` environment. The visual smoke uses a task-local environment because that base environment's Transformers lacks SigLIP2:

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python -m venv --system-site-packages /mnt/shared/ww/ovimap-module-validation-v1/tooling/repairs-20260922/semantic-env
/mnt/shared/ww/ovimap-module-validation-v1/tooling/repairs-20260922/semantic-env/bin/python -m pip install --no-deps -r configs/evaluation/ovimap_module_smoke_requirements.txt
```

Rebuild and capture into a new output directory; the pinned patched checkout and existing native ABI dependencies are explicit inputs:

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/rebuild_ovimap_module_native.py \
  --upstream /home/ww/crove/ovimap-module-validation-upstream \
  --baseline-build /home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native/OVI-MAP \
  --generated-ros-headers /home/ww/vv/paper2/OVI-MAP/mapping_ros_ws/devel/include \
  --data-root /home/ww/vv/dataset/Replica \
  --frontend-cache /home/ww/vv/paper2/OVI-MAP/output/benchmark_20260615_ovimap \
  --output-root /mnt/shared/ww/ovimap-module-validation-v1/tooling/repairs-20260922/native-reproduction
```

`configs/evaluation/ovimap_module_historical_smoke.json` binds the successful `native-v7` replay and model locations. Point its capture/replay fields to the new reproduction when validating that replay. The default study command uses this configuration:

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_module_study.py \
  --spec docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json \
  --phase all --output-root /mnt/shared/ww/ovimap-module-validation-v1
```

For one boundary, pass `--phase capture|semantic|geometry|query` to `run_ovimap_module_smoke.py` with `--resolved-config configs/evaluation/ovimap_module_historical_smoke.json --output <new-receipt.json>`. Use the visual environment for `semantic`, and `ovimap-map` for the other boundaries. `capture` verifies the already executed native replay; it does not charge for or pretend to rerun mapping.

The study command accepts `--export-root <new-directory>` to export finalized small artifacts. Report-only execution does not write into the repository. A new code identity creates a new attempt, including when an old `resolved_config.json` is supplied.

## Validation and remaining scope

Run the focused checks:

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python -m pytest -q tests/module_validation
ruff check src/static_ovmap/module_validation scripts/evaluation/run_ovimap_module_study.py scripts/evaluation/run_ovimap_module_smoke.py scripts/evaluation/rebuild_ovimap_module_native.py tests/module_validation
```

The successful native replay processes Room0 frames 0 and 10, captures 1,855,128 surface rows and 35 requests, and records the loaded extension's path/hash. Real adapter checks cover six-crop parity, SigLIP2's own text space, WOW's actual mask-consumption boundary, geometry partition preservation, native query parity, and cached acquisition accounting. These are historical integration checks, not SELECT measurements.

Still outstanding: the full independent-scene capture → direct readout → FIT/CAL → SELECT/combination → frozen confirmation driver, plus authorized physically independent ScanNet scenes. Existing training, feature, selection, and evaluation functions do not by themselves constitute this integration. The public report intentionally remains `implementation: PARTIAL`; adding dataset paths alone will not complete it. No neural backbone training, benchmark substitution, or fabricated metric rows were performed.
