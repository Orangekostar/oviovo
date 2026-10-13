# FG2 calibration handoff

BASE: `bec4176d9a6784a89ea293a9803e5cf6fab178ec`. Branch: `research/fg2-surface-bias-wave1-v2` (actual publication branch is in PUBLICATION_RECEIPT).
Execution: BLOCKED_DEPENDENCY; science: UNASSESSED_DEPENDENCY; deployment: N0_UNCHANGED.
Scientific source digest: `6f1a5d6bc6dcbec380d8da0bb120b307e737184841c346bcaf3b94cda5f49675`.

Original seed17 step1000 and seed29 step2000 heads are referenced by the [selected model registry](../../../../artifacts/static_ovmap/surface_bias_wave1_v2/selected_model_registry.json); no head was trained here. Lambda lock: null. The full parent source binding is verified before any in-memory path migration; parent_root keeps its inherited meaning. FG_ROOT is the CLI parent-root, LR_ROOT comes from that binding. Dependencies: see [dependency manifest](../../../../artifacts/static_ovmap/surface_bias_wave1_v2/dependency_manifest.json).

Use the existing controller environment and FC_python from the verified binding. Inherit one GPU string; the FC worker uses execution_config(binding)['gpu_lock'] and queued_gpu_lock. All writes belong to OUT; parent caches and upstream sources are read-only. Run from the worktree:

```bash
"$PY_CTRL" -u scripts/evaluation/run_fg2_surface_bias_wave1_v2.py --spec configs/static_ovmap/surface_bias_wave1_v2.json --parent-root "$FG_ROOT" --output-root "$OUT" --gpu "$GPU" --phase all --resume
PYTHONPATH=src "$PY_CTRL" -m pytest -q tests/test_surface_bias_wave1_v2.py
"$PY_CTRL" scripts/evaluation/run_fg2_surface_bias_wave1_v2.py --spec configs/static_ovmap/surface_bias_wave1_v2.json --parent-root "$FG_ROOT" --output-root "$OUT" --phase report --resume
```

Table Markdown/CSV/JSON come from sealed runtime JSON; the report phase regenerates them. Actual scientific worker exits are in stage_exits. The final all-process return code is observed by its invoking parent and stored externally, never predeclared in its own result commit.

Recovery point: restore the exact missing parent manifests, referenced grids, FC/text and input arrays; rerun identical spec/paths. Different scientific parameters require a new version. Do not recompute RGB, train, select another checkpoint, use similar caches or shrink cohorts. Blocked dependency: /mnt/shared/capacity/node101/ww/ovimap-frozen-fc-routing-v1/attempt_001/source_binding.json.

Results: [RESULTS](RESULTS.md). Limits and supported claims: [CLAIMS](CLAIMS.md). Requirement audit: [REVIEW](REVIEW.md). Large runtime arrays stay in OUT; Git contains compact tables and summaries. Cached replay performance is not online FPS.

Verified result commit C1: `462d7f704431b6e60d6ec462ea9a7265962bdeb0`. [Immutable results](https://github.com/Orangekostar/oviovo/blob/462d7f704431b6e60d6ec462ea9a7265962bdeb0/docs/paper/static_ovmap/surface_bias_wave1_v2/RESULTS.md).

Publication of C1 was verified; the committed receipt proves C1 only. The final C2/local/remote equality receipt is external at OUT/publication/final.json.
