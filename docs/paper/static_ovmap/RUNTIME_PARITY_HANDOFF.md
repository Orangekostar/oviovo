# Runtime Parity Handoff

Attempt: `/mnt/shared/ww/ovimap-runtime-parity-v1/attempt_001`. Reference binding: `aab44cdbe2fa6fb5aefcdf0296baa1daceb50491e122983de63069412c507f29`. Implementation freeze: `1bb3c1c60a917a349dfb0804332d32feb1baa7acc4dde6446f3f20c0badf2f29`.

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_runtime_parity.py --spec configs/static_ovmap/runtime_parity_v1.json --source-worktree /mnt/shared/ww/ovimap-cvpr-compact-tables-v1/worktree --v2-results-root /home/ww/ovimap-area-fallback-v2/attempt_001 --output-root /mnt/shared/ww/ovimap-runtime-parity-v1/attempt_001 --phase all --resume
```

Phases: bind, profile, screen, verify, freeze, benchmark, tables, publish, all. Profile/screen consume the same bounded pilot series; completed leaves are reused by source/variant/scene/round. Live reservations are observed, never killed. Internal or parity failures are not automatically replayed.

Common cold inputs contain no parent recovery features/views/results. Parent vectors are available only to VerificationContext after timing or in CPU cached export checks. Dense tensors, scans and weights remain outside Git. Original source files, its index and old timing reservations are preserved.

The publication receipt is external at publication/final.json and is written only after ordinary push and exact local/remote SHA equality. Rendering QA and the requirement audit are linked in artifacts/static_ovmap/runtime_parity_v1.
